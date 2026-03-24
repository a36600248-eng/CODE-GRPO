# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import logging as py_logging
import os
import random
import re
import shutil
import gc
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
from accelerate import logging
from datasets import IterableDataset, load_dataset
from transformers import TrainerCallback

warnings.filterwarnings("ignore", message=r"TRL currently supports vLLM versions.*")

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import use_adapter
from trl.extensions.code_grpo.adapters import load_dataset_adapter
from trl.trainer.code_grpo_config import CodeGRPOConfig
from trl.trainer.code_grpo_trainer import CodeGRPOTrainer


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

# Reduce known low-signal runtime warnings from dependencies on the shared server.
warnings.filterwarnings("ignore", message=r"Detected kernel version .* can cause the process to hang\.")


def _safe_slug(value: str, fallback: str, max_len: int = 64) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._").lower()
    if not text:
        text = fallback
    return text[:max_len]


def _first_dataset_tag(script_args, dataset_args) -> str:
    if script_args.dataset_name:
        return _safe_slug(script_args.dataset_name.split("/")[-1], "dataset", 48)
    datasets = getattr(dataset_args, "datasets", None) or []
    if not datasets:
        return "dataset"
    first = datasets[0]
    path_tag = _safe_slug(getattr(first, "path", "dataset"), "dataset", 32)
    data_files = getattr(first, "data_files", None)
    file_tag = ""
    if isinstance(data_files, str):
        file_tag = _safe_slug(os.path.splitext(os.path.basename(data_files))[0], "data", 24)
    elif isinstance(data_files, list) and data_files:
        file_tag = _safe_slug(os.path.splitext(os.path.basename(str(data_files[0])))[0], "data", 24)
    elif isinstance(data_files, dict) and data_files:
        first_key = sorted(data_files.keys())[0]
        file_tag = _safe_slug(os.path.splitext(os.path.basename(str(data_files[first_key])))[0], "data", 24)
    return f"{path_tag}-{file_tag}" if file_tag else path_tag


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _build_resume_signature(script_args, training_args, model_args, dataset_args) -> dict[str, Any]:
    training_keys = (
        "codegrpo_mode",
        "backend",
        "zero_pass_soft_reward_enabled",
        "pseudo_multiround_enabled",
        "question_prior_enabled",
        "K",
        "generation_batch_size",
        "max_completion_length",
        "max_completion_length_code",
        "max_steps",
        "eval_steps",
        "seed",
        "data_seed",
    )
    script_keys = (
        "dataset_name",
        "dataset_train_split",
        "dataset_test_split",
        "dataset_adapter",
    )
    return {
        "dataset_tag": _first_dataset_tag(script_args, dataset_args),
        "model_name_or_path": str(getattr(model_args, "model_name_or_path", "")),
        "training_args": {key: _json_safe(getattr(training_args, key, None)) for key in training_keys},
        "script_args": {key: _json_safe(getattr(script_args, key, None)) for key in script_keys},
    }


def _manifest_matches_resume_signature(manifest_path: str, signature: dict[str, Any]) -> bool:
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return False

    if str(payload.get("mode")) != str(signature["training_args"].get("codegrpo_mode")):
        return False
    if str(payload.get("model_args", {}).get("model_name_or_path", "")) != str(signature["model_name_or_path"]):
        return False
    if str(payload.get("script_args", {}).get("dataset_name", None)) != str(signature["script_args"].get("dataset_name", None)):
        return False
    if str(payload.get("paths", {}).get("mode", "")) != str(signature["training_args"].get("codegrpo_mode")):
        return False

    manifest_dataset_tag = _safe_slug(
        str(payload.get("run_id", "")).split("__", 4)[3] if "__" in str(payload.get("run_id", "")) and len(str(payload.get("run_id", "")).split("__")) >= 5 else "",
        "",
        48,
    )
    if manifest_dataset_tag != _safe_slug(str(signature["dataset_tag"]), "", 48):
        return False

    for key, value in signature["training_args"].items():
        if _json_safe(payload.get("training_args", {}).get(key)) != value:
            return False
    for key, value in signature["script_args"].items():
        if _json_safe(payload.get("script_args", {}).get(key)) != value:
            return False
    return True


def _baseline_eval_is_supported(training_args, trainer) -> tuple[bool, str | None]:
    if str(getattr(training_args, "backend", "hf")) == "vllm":
        return False, "backend='vllm' cannot guarantee a true base-model baseline."
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    if not hasattr(unwrapped_model, "disable_adapter"):
        return False, "baseline_eval requires a PEFT adapter that can be disabled."
    return True, None


def _get_rank() -> int:
    for key in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def _configure_run_layout(script_args, training_args, model_args, dataset_args) -> dict[str, str]:
    mode = _safe_slug(getattr(training_args, "codegrpo_mode", "train"), "train", 16)
    base_output_dir = os.path.abspath(training_args.output_dir)
    resume_from_checkpoint = getattr(training_args, "resume_from_checkpoint", None)
    resume_signature = _build_resume_signature(script_args, training_args, model_args, dataset_args)

    def _checkpoint_step(path: str) -> tuple[int, float]:
        name = os.path.basename(path.rstrip("/\\"))
        try:
            step = int(name.split("-")[-1])
        except Exception:
            step = -1
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = -1.0
        return step, mtime

    def _find_resume_checkpoint() -> str | None:
        if not resume_from_checkpoint:
            return None
        if isinstance(resume_from_checkpoint, str):
            checkpoint_dir = os.path.abspath(resume_from_checkpoint)
            return checkpoint_dir if os.path.isdir(checkpoint_dir) else None
        pattern = os.path.join(base_output_dir, mode, "*", "*_out", "checkpoint-*")
        candidates = []
        for path in glob.glob(pattern):
            if not os.path.isdir(path):
                continue
            run_root = os.path.dirname(os.path.dirname(path))
            manifest_path = os.path.join(run_root, "run_manifest.json")
            if _manifest_matches_resume_signature(manifest_path, resume_signature):
                candidates.append(path)
        if not candidates:
            return None
        candidates.sort(key=_checkpoint_step)
        return os.path.abspath(candidates[-1])

    checkpoint_dir = _find_resume_checkpoint()
    tensorboard_root_dir = getattr(training_args, "tensorboard_root_dir", None)
    if checkpoint_dir:
        artifacts_dir = os.path.dirname(checkpoint_dir)
        run_root = os.path.dirname(artifacts_dir)
        run_id = os.path.basename(run_root)
        logs_dir = os.path.join(run_root, "logs")
        traces_dir = os.path.join(run_root, "traces", "rollout")
        if tensorboard_root_dir:
            tensorboard_dir = os.path.join(os.path.abspath(tensorboard_root_dir), mode, run_id)
        else:
            tensorboard_dir = os.path.join(run_root, "tensorboard")
        for path in (run_root, artifacts_dir, logs_dir, traces_dir, tensorboard_dir):
            os.makedirs(path, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backend = _safe_slug(getattr(training_args, "backend", "hf"), "hf", 16)
        model_tag = _safe_slug(os.path.basename(str(model_args.model_name_or_path).rstrip("/\\")), "model", 48)
        dataset_tag = _first_dataset_tag(script_args, dataset_args)
        run_id = f"{timestamp}__{mode}__{model_tag}__{dataset_tag}__{backend}"
        run_root = os.path.join(base_output_dir, mode, run_id)
        artifacts_dir = os.path.join(run_root, "test_out" if mode == "test" else "train_out")
        logs_dir = os.path.join(run_root, "logs")
        traces_dir = os.path.join(run_root, "traces", "rollout")
        if tensorboard_root_dir:
            tensorboard_dir = os.path.join(os.path.abspath(tensorboard_root_dir), mode, run_id)
        else:
            tensorboard_dir = os.path.join(run_root, "tensorboard")
        for path in (run_root, artifacts_dir, logs_dir, traces_dir, tensorboard_dir):
            os.makedirs(path, exist_ok=True)

    training_args.output_dir = artifacts_dir
    training_args.logging_dir = tensorboard_dir
    training_args.debug_trace_dir = traces_dir
    if not getattr(training_args, "run_name", None):
        training_args.run_name = run_id

    return {
        "run_id": run_id,
        "run_root": run_root,
        "artifacts_dir": artifacts_dir,
        "logs_dir": logs_dir,
        "traces_dir": traces_dir,
        "tensorboard_dir": tensorboard_dir,
        "mode": mode,
        "resume_checkpoint_dir": checkpoint_dir,
    }


def _attach_runtime_file_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root_logger = py_logging.getLogger()
    abs_path = os.path.abspath(log_path)

    def _configure_runtime_logging_levels():
        if root_logger.level > py_logging.INFO:
            root_logger.setLevel(py_logging.INFO)

        logger_levels = {
            "trl": py_logging.INFO,
            "accelerate": py_logging.WARNING,
            "accelerate.utils.other": py_logging.ERROR,
            "transformers": py_logging.WARNING,
            "transformers.trainer": py_logging.WARNING,
            "transformers.tokenization_utils_base": py_logging.WARNING,
            "transformers.configuration_utils": py_logging.WARNING,
            "transformers.modeling_utils": py_logging.WARNING,
            "transformers.generation.configuration_utils": py_logging.WARNING,
            "transformers.image_processing_auto": py_logging.WARNING,
            "datasets": py_logging.WARNING,
            "vllm": py_logging.WARNING,
            "peft": py_logging.WARNING,
            "urllib3": py_logging.WARNING,
            "httpx": py_logging.WARNING,
            "httpcore": py_logging.WARNING,
            "uvicorn": py_logging.WARNING,
            "uvicorn.access": py_logging.ERROR,
            "filelock": py_logging.WARNING,
        }
        for logger_name, level in logger_levels.items():
            named_logger = py_logging.getLogger(logger_name)
            named_logger.propagate = True
            named_logger.setLevel(level)

        try:
            from datasets.utils import logging as datasets_logging

            datasets_logging.set_verbosity_warning()
        except Exception:
            pass

        try:
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_warning()
        except Exception:
            pass

    for handler in root_logger.handlers:
        if isinstance(handler, py_logging.FileHandler) and getattr(handler, "baseFilename", "") == abs_path:
            _configure_runtime_logging_levels()
            return
    file_handler = py_logging.FileHandler(abs_path, encoding="utf-8")
    file_handler.setLevel(py_logging.INFO)
    file_handler.setFormatter(
        py_logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)
    _configure_runtime_logging_levels()


class TextMetricsCallback(TrainerCallback):
    """Persist train/eval metrics to plain text and jsonl files."""

    def __init__(self, text_path: str | None, jsonl_path: str):
        self.text_path = text_path
        self.jsonl_path = jsonl_path
        self._train_started_at_ts: float | None = None
        target_dir = os.path.dirname(self.text_path) if self.text_path else os.path.dirname(self.jsonl_path)
        os.makedirs(target_dir, exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_text(self, line: str):
        if not self.text_path:
            return
        with open(self.text_path, "a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")

    def _append_json(self, payload: dict[str, Any]):
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "unknown"
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _progress_line(self, *, state, args, logs: dict[str, Any]) -> str | None:
        if self._train_started_at_ts is None:
            return None
        current_step = int(state.global_step)
        max_steps = int(getattr(state, "max_steps", 0) or getattr(args, "max_steps", 0) or 0)
        elapsed_seconds = time.time() - self._train_started_at_ts
        if current_step <= 0:
            avg_step_seconds = None
            eta_seconds = None
        else:
            avg_step_seconds = elapsed_seconds / current_step
            remaining_steps = max(0, max_steps - current_step) if max_steps > 0 else 0
            eta_seconds = avg_step_seconds * remaining_steps if max_steps > 0 else None
        if "loss" in logs:
            total_fragment = str(max_steps) if max_steps > 0 else "?"
            return (
                f"[progress] train step {current_step}/{total_fragment} | "
                f"elapsed={self._format_duration(elapsed_seconds)} | "
                f"avg_step={self._format_duration(avg_step_seconds)} | "
                f"eta={self._format_duration(eta_seconds)}"
            )
        if any(str(key).startswith("eval_") for key in logs.keys()):
            total_fragment = str(max_steps) if max_steps > 0 else "?"
            return (
                f"[progress] eval checkpoint at step {current_step}/{total_fragment} | "
                f"elapsed={self._format_duration(elapsed_seconds)}"
            )
        return None

    def on_train_begin(self, args, state, control, **kwargs):
        del args, control, kwargs
        self._train_started_at_ts = time.time()
        self._append_text(f"[{self._now()}] TRAIN_BEGIN step={state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        self._append_text(f"[{self._now()}] TRAIN_END step={state.global_step}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        del control, kwargs
        if not logs:
            return
        payload = {
            "time": self._now(),
            "step": int(state.global_step),
            "epoch": state.epoch,
            "logs": logs,
        }
        self._append_json(payload)
        self._append_text(f"[{payload['time']}] LOG step={payload['step']} epoch={payload['epoch']} logs={payload['logs']}")
        progress_line = self._progress_line(state=state, args=args, logs=logs)
        if progress_line:
            logger.info(progress_line)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        del args, control, kwargs
        payload = {
            "time": self._now(),
            "event": "EVAL",
            "step": int(state.global_step),
            "epoch": state.epoch,
            "metrics": metrics or {},
        }
        self._append_json(payload)
        self._append_text(
            f"[{payload['time']}] EVAL step={payload['step']} epoch={payload['epoch']} metrics={payload['metrics']}"
        )


def _write_run_manifest(path: str, run_layout: dict[str, str], script_args, training_args, model_args, dataset_args):
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_layout["run_id"],
        "mode": run_layout["mode"],
        "paths": run_layout,
        "script_args": _json_safe(vars(script_args)),
        "training_args": _json_safe(vars(training_args)),
        "model_args": _json_safe(vars(model_args)),
        "dataset_args": _json_safe(vars(dataset_args)),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_run_index(path: str, run_layout: dict[str, str], rank: int):
    lines = [
        f"run_id: {run_layout['run_id']}",
        f"mode: {run_layout['mode']}",
        "",
        "Directories:",
        f"- artifacts: {run_layout['artifacts_dir']}",
        f"- logs: {run_layout['logs_dir']}",
        f"- traces: {run_layout['traces_dir']}",
        f"- tensorboard: {run_layout['tensorboard_dir']}",
        f"- review bundle: {os.path.join(run_layout['run_root'], 'review_bundle')}",
        "",
        "Most useful files:",
        f"- trainer jsonl log: {os.path.join(run_layout['logs_dir'], f'trainer_events_rank{rank}.jsonl')}",
        f"- rollout summary: {os.path.join(run_layout['logs_dir'], f'rollout_summary_rank{rank}.jsonl')}",
        f"- traces: {os.path.join(run_layout['traces_dir'], '*.json')}",
        f"- packaged share folder: {os.path.join(run_layout['run_root'], 'review_bundle')}",
        f"- trainer state / checkpoints / saved model: {run_layout['artifacts_dir']}",
        "",
        "Reading order:",
        "1. trainer_events_rank*.jsonl -> step-level loss and metrics",
        "2. rollout_summary_rank*.jsonl -> per-question summary",
        "3. plots/*.png -> compact trend visualization",
        "4. traces/*.json -> sampled detailed traces when needed",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _copy_if_exists(src: str, dst: str):
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _load_jsonl_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return rows

def _last_numeric(rows: list[dict[str, Any]], key: str):
    for row in reversed(rows):
        logs = row.get("logs") if isinstance(row.get("logs"), dict) else row
        if key in logs and isinstance(logs[key], (int, float)):
            return float(logs[key])
    return None


def _best_numeric(rows: list[dict[str, Any]], key: str):
    vals = []
    for row in rows:
        logs = row.get("logs") if isinstance(row.get("logs"), dict) else row
        value = logs.get(key)
        if isinstance(value, (int, float)):
            vals.append(float(value))
    return max(vals) if vals else None


def _build_review_summary(run_layout: dict[str, str], rank: int) -> dict[str, Any]:
    trainer_jsonl = os.path.join(run_layout["logs_dir"], f"trainer_events_rank{rank}.jsonl")
    rollout_jsonl = os.path.join(run_layout["logs_dir"], f"rollout_summary_rank{rank}.jsonl")
    trainer_rows = _load_jsonl_rows(trainer_jsonl)
    rollout_rows = _load_jsonl_rows(rollout_jsonl)
    train_rows = [row for row in trainer_rows if isinstance(row.get("logs"), dict) and "loss" in row["logs"]]
    eval_rows = [row for row in trainer_rows if isinstance(row.get("logs"), dict) and any(str(k).startswith("eval_") for k in row["logs"].keys())]

    summary = {
        "run_id": run_layout.get("run_id"),
        "mode": run_layout.get("mode"),
        "train_last_step": int(train_rows[-1].get("step", 0)) if train_rows else None,
        "eval_last_step": int(eval_rows[-1].get("step", 0)) if eval_rows else None,
        "train_mean_pass_rate_last": _last_numeric(train_rows, "mean_pass_rate"),
        "train_mean_R_code_last": _last_numeric(train_rows, "mean_R_code"),
        "train_advantage_code_zero_rate_last": _last_numeric(train_rows, "advantage/code_zero_rate"),
        "train_soft_reward_trigger_rate_last": _last_numeric(train_rows, "soft_reward_trigger_rate"),
        "train_zero_pass_soft_trigger_rate_last": _last_numeric(train_rows, "zero_pass_soft_trigger_rate"),
        "train_soft_lift_last": _last_numeric(train_rows, "soft_lift"),
        "train_pseudo_iterative_pool_size_last": _last_numeric(train_rows, "pseudo/iterative_pool_size"),
        "eval_pass_at_1_last": _last_numeric(eval_rows, "eval_pass_at_1"),
        "eval_pass_at_1_best": _best_numeric(eval_rows, "eval_pass_at_1"),
        "eval_best_pass_rate_overall_last": _last_numeric(eval_rows, "eval_best_pass_rate_overall"),
        "eval_best_pass_rate_overall_best": _best_numeric(eval_rows, "eval_best_pass_rate_overall"),
        "eval_mean_pass_rate_last": _last_numeric(eval_rows, "eval_mean_pass_rate"),
        "eval_mean_R_code_last": _last_numeric(eval_rows, "eval_mean_R_code"),
        "rollout_row_count": len(rollout_rows),
    }
    return summary


def _generate_review_plots(bundle_root: str, logs_bundle: str, artifacts_bundle: str, rank: int):
    plots_dir = os.path.join(bundle_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional at runtime
        with open(os.path.join(plots_dir, "PLOTS_DISABLED.txt"), "w", encoding="utf-8") as handle:
            handle.write(
                "Plot generation skipped because matplotlib is unavailable.\n"
                f"Reason: {type(exc).__name__}: {exc}\n"
            )
        return

    trainer_jsonl = os.path.join(logs_bundle, f"trainer_events_rank{rank}.jsonl")
    rows = _load_jsonl_rows(trainer_jsonl)
    if not rows:
        with open(os.path.join(plots_dir, "PLOTS_DISABLED.txt"), "w", encoding="utf-8") as handle:
            handle.write("Plot generation skipped because trainer_events_rank*.jsonl was not found or was empty.\n")
        return

    train_rows = [row for row in rows if isinstance(row.get("logs"), dict) and "loss" in row["logs"]]
    eval_rows = [row for row in rows if isinstance(row.get("logs"), dict) and "eval_pass_at_1_within_1" in row["logs"]]

    # 1. Training reward curves
    reward_key_candidates = {
        "R_code": ["window/mean_R_code", "mean_R_code"],
        "R_soft_effective": ["window/mean_R_soft_effective", "mean_R_soft_effective"],
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for label, candidates in reward_key_candidates.items():
        points: list[tuple[int, float]] = []
        for row in train_rows:
            logs = row["logs"]
            value = None
            for key in candidates:
                if key in logs:
                    value = logs[key]
                    break
            if value is None:
                continue
            points.append((int(row.get("step", 0)), float(value)))
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], label=label)
            plotted = True
    if plotted:
        ax.set_title("Training Reward Curves")
        ax.set_xlabel("Trainer step")
        ax.set_ylabel("Reward")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "train_reward_curves.png"), dpi=180)
    plt.close(fig)

    # 2. Eval pass@1 by repair round
    eval_round_keys: dict[int, str] = {}
    for row in eval_rows:
        for key in row["logs"].keys():
            match = re.fullmatch(r"eval_pass_at_1_within_(\d+)", key)
            if match:
                eval_round_keys[int(match.group(1))] = key
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for round_idx in sorted(eval_round_keys.keys()):
        key = eval_round_keys[round_idx]
        std_key = f"{key}_std"
        points = [(int(row.get("step", 0)), float(row["logs"][key])) for row in eval_rows if key in row["logs"]]
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=f"<= round {round_idx}")
        if any(std_key in row["logs"] for row in eval_rows):
            std_points = [
                (int(row.get("step", 0)), float(row["logs"].get(std_key, 0.0)))
                for row in eval_rows
                if key in row["logs"]
            ]
            if len(std_points) == len(points):
                stds = [p[1] for p in std_points]
                lower = [max(0.0, y - s) for y, s in zip(ys, stds)]
                upper = [min(1.0, y + s) for y, s in zip(ys, stds)]
                ax.fill_between(xs, lower, upper, alpha=0.12)
        plotted = True
    if plotted:
        ax.set_title("Eval Pass@1 by Repair Round (mean +/- std)")
        ax.set_xlabel("Trainer step")
        ax.set_ylabel("Solve rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "eval_pass_at_1_by_round.png"), dpi=180)
    plt.close(fig)

    # 3. Optional baseline summary for plotting consumers
    baseline_path = os.path.join(artifacts_bundle, "baseline_eval_results.json")
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as handle:
                baseline = json.load(handle)
            with open(os.path.join(plots_dir, "baseline_plot_points.json"), "w", encoding="utf-8") as handle:
                json.dump(baseline, handle, ensure_ascii=False, indent=2)
        except Exception:
            pass


def _build_review_bundle(run_layout: dict[str, str], rank: int, trace_sample_size: int = 2):
    bundle_root = os.path.join(run_layout["run_root"], "review_bundle")
    logs_bundle = os.path.join(bundle_root, "logs")
    artifacts_bundle = os.path.join(bundle_root, "artifacts")
    samples_bundle = os.path.join(bundle_root, "samples")
    os.makedirs(logs_bundle, exist_ok=True)
    os.makedirs(artifacts_bundle, exist_ok=True)
    os.makedirs(samples_bundle, exist_ok=True)

    # Core metadata
    _copy_if_exists(os.path.join(run_layout["run_root"], "RUN_INDEX.txt"), os.path.join(bundle_root, "RUN_INDEX.txt"))
    _copy_if_exists(
        os.path.join(run_layout["run_root"], "run_manifest.json"),
        os.path.join(bundle_root, "run_manifest.json"),
    )

    # Logs
    for filename in (
        f"trainer_events_rank{rank}.jsonl",
        f"rollout_summary_rank{rank}.jsonl",
        "test_metrics.txt",
    ):
        _copy_if_exists(
            os.path.join(run_layout["logs_dir"], filename),
            os.path.join(logs_bundle, filename),
        )

    # Artifact summaries only. Keep this bundle focused on direct result consumption.
    for filename in (
        "train_results.json",
        "eval_results.json",
        "baseline_eval_results.json",
        "test_results.json",
        "all_results.json",
    ):
        _copy_if_exists(
            os.path.join(run_layout["artifacts_dir"], filename),
            os.path.join(artifacts_bundle, filename),
        )

    _copy_if_exists(
        os.path.join(run_layout["logs_dir"], f"step_samples_rank{rank}.jsonl"),
        os.path.join(samples_bundle, "step_samples.jsonl"),
    )

    summary_payload = _build_review_summary(run_layout, rank)
    with open(os.path.join(bundle_root, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    # Randomly sample a few rollout traces only when explicitly requested.
    trace_candidates = sorted(glob.glob(os.path.join(run_layout["traces_dir"], "*.json")))
    if trace_sample_size > 0 and trace_candidates:
        traces_bundle = os.path.join(bundle_root, "traces")
        os.makedirs(traces_bundle, exist_ok=True)
        rng = random.Random(run_layout["run_id"])
        selected = (
            rng.sample(trace_candidates, min(trace_sample_size, len(trace_candidates)))
            if len(trace_candidates) > trace_sample_size
            else trace_candidates
        )
        for src in selected:
            _copy_if_exists(src, os.path.join(traces_bundle, os.path.basename(src)))

    bundle_note = [
        "This folder is a compact review bundle for quick debugging.",
        "",
        "Included:",
        "- RUN_INDEX.txt, run_manifest.json, summary.json",
        "- compact jsonl logs",
        "- result json files",
        "- compact sample dump (samples/step_samples.jsonl)",
        "- optional sampled rollout traces only when explicitly requested",
        "",
        "Not included on purpose:",
        "- verbose runtime log",
        "- duplicated plain-text trainer log",
        "- trainer_state.json",
        "- generated plots",
        "- full trace dump",
        "",
        "If a requested file is missing here, it was either not produced or intentionally omitted for compactness.",
    ]
    with open(os.path.join(bundle_root, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(bundle_note) + "\n")


@dataclass
class CodeGRPOScriptArguments(ScriptArguments):
    dataset_adapter: str = field(
        default="default",
        metadata={
            "help": "Dataset adapter name/class path. Use 'default' or a dotted path "
            "like 'my_pkg.module:MyDatasetAdapter'."
        },
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={"help": "Optional cap on the number of training examples after dataset adaptation."},
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={"help": "Optional cap on the number of eval/test examples after dataset adaptation."},
    )


def _load_dataset(script_args, dataset_args):
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used and "
            "`dataset_name` will be ignored."
        )
        return get_dataset(dataset_args)
    if dataset_args.datasets and not script_args.dataset_name:
        return get_dataset(dataset_args)
    if not dataset_args.datasets and script_args.dataset_name:
        return load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    raise ValueError("Either `datasets` or `dataset_name` must be provided.")


def _limit_examples(dataset_split, max_samples: int | None):
    if max_samples is None:
        return dataset_split
    max_samples = int(max_samples)
    if max_samples <= 0:
        return dataset_split
    if isinstance(dataset_split, IterableDataset):
        take_fn = getattr(dataset_split, "take", None)
        return take_fn(max_samples) if callable(take_fn) else dataset_split
    select_fn = getattr(dataset_split, "select", None)
    if callable(select_fn):
        return dataset_split.select(range(min(len(dataset_split), max_samples)))
    return dataset_split


def _adapt_splits(
    dataset,
    adapter,
    train_split: str,
    test_split: str,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    *,
    load_train_split: bool = True,
):
    train_dataset = None
    if load_train_split:
        train_dataset = _limit_examples(adapter.adapt_dataset(dataset[train_split]), max_train_samples)
    eval_dataset = (
        _limit_examples(adapter.adapt_dataset(dataset[test_split]), max_eval_samples) if test_split in dataset else None
    )
    return train_dataset, eval_dataset


def main(script_args, training_args, model_args, dataset_args):
    run_layout = _configure_run_layout(script_args, training_args, model_args, dataset_args)
    rank = _get_rank()
    runtime_log_path = os.path.join(run_layout["logs_dir"], f"runtime_rank{rank}.txt")
    trainer_text_log_path = os.path.join(run_layout["logs_dir"], f"trainer_events_rank{rank}.txt")
    trainer_jsonl_path = os.path.join(run_layout["logs_dir"], f"trainer_events_rank{rank}.jsonl")
    _attach_runtime_file_logger(runtime_log_path)
    if rank == 0:
        _write_run_manifest(
            os.path.join(run_layout["run_root"], "run_manifest.json"),
            run_layout,
            script_args,
            training_args,
            model_args,
            dataset_args,
        )
        _write_run_index(os.path.join(run_layout["run_root"], "RUN_INDEX.txt"), run_layout, rank)
    logger.info(
        "[RUN] run_id=%s mode=%s output_dir=%s logs_dir=%s",
        run_layout["run_id"],
        run_layout["mode"],
        training_args.output_dir,
        run_layout["logs_dir"],
    )
    if getattr(training_args, "resume_from_checkpoint", None) and not run_layout.get("resume_checkpoint_dir"):
        logger.warning("resume_from_checkpoint was requested, but no matching checkpoint run was found under %s.", training_args.output_dir)

    adapter = load_dataset_adapter(script_args.dataset_adapter)
    dataset = _load_dataset(script_args, dataset_args)

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = {
        "revision": model_args.model_revision,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
        "device_map": None,
    }
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    train_dataset, eval_dataset = _adapt_splits(
        dataset=dataset,
        adapter=adapter,
        train_split=script_args.dataset_train_split,
        test_split=script_args.dataset_test_split,
        max_train_samples=script_args.max_train_samples,
        max_eval_samples=script_args.max_eval_samples,
        load_train_split=training_args.codegrpo_mode != "test",
    )

    trainer = CodeGRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=[
            TextMetricsCallback(
                trainer_text_log_path if getattr(training_args, "write_trainer_text_log", False) else None,
                trainer_jsonl_path,
            )
        ],
    )

    if (
        training_args.codegrpo_mode == "train"
        and eval_dataset is not None
        and getattr(training_args, "run_base_model_baseline_eval", False)
    ):
        baseline_supported, reason = _baseline_eval_is_supported(training_args, trainer)
        if not baseline_supported:
            logger.warning("Skipping run_base_model_baseline_eval because %s", reason)
        else:
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
            with use_adapter(unwrapped_model, adapter_name=None):
                baseline_metrics = trainer.evaluate(eval_dataset=eval_dataset)
            trainer.log_metrics("baseline_eval", baseline_metrics)
            trainer.save_metrics("baseline_eval", baseline_metrics)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if training_args.codegrpo_mode == "test":
        if eval_dataset is None:
            raise ValueError("codegrpo_mode=test requires an eval/test split; refusing to fall back to train_dataset.")
        test_dataset = eval_dataset
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        public_metrics = trainer.public_metrics(metrics, split="test")
        if rank == 0:
            with open(os.path.join(run_layout["logs_dir"], "test_metrics.txt"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_safe(metrics), ensure_ascii=False) + "\n")
            _build_review_bundle(
                run_layout,
                rank,
                trace_sample_size=int(getattr(training_args, "review_bundle_trace_sample_size", 2)),
            )
        trainer.accelerator.print(public_metrics)
        trainer.accelerator.print("CodeGRPO test mode completed.")
        return

    try:
        train_result = trainer.train(resume_from_checkpoint=run_layout.get("resume_checkpoint_dir"))
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.accelerator.print("CodeGRPO training completed.")
        trainer.save_model(training_args.output_dir)
        trainer.accelerator.print(f"Model saved to {training_args.output_dir}.")
        if rank == 0:
            _build_review_bundle(
                run_layout,
                rank,
                trace_sample_size=int(getattr(training_args, "review_bundle_trace_sample_size", 2)),
            )
    except Exception:
        logger.exception("[RUN] training failed")
        raise

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (CodeGRPOScriptArguments, CodeGRPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("code_grpo", help="Run the Code GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
