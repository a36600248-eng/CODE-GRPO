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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
from accelerate import logging
from datasets import load_dataset
from transformers import TrainerCallback

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = _safe_slug(getattr(training_args, "codegrpo_mode", "train"), "train", 16)
    backend = _safe_slug(getattr(training_args, "backend", "hf"), "hf", 16)
    model_tag = _safe_slug(os.path.basename(str(model_args.model_name_or_path).rstrip("/\\")), "model", 48)
    dataset_tag = _first_dataset_tag(script_args, dataset_args)
    run_id = f"{timestamp}__{mode}__{model_tag}__{dataset_tag}__{backend}"

    base_output_dir = os.path.abspath(training_args.output_dir)
    run_root = os.path.join(base_output_dir, mode, run_id)
    artifacts_dir = os.path.join(run_root, "test_out" if mode == "test" else "train_out")
    logs_dir = os.path.join(run_root, "logs")
    traces_dir = os.path.join(run_root, "traces", "rollout")
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
    }


def _attach_runtime_file_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root_logger = py_logging.getLogger()
    abs_path = os.path.abspath(log_path)
    for handler in root_logger.handlers:
        if isinstance(handler, py_logging.FileHandler) and getattr(handler, "baseFilename", "") == abs_path:
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
    if root_logger.level > py_logging.INFO:
        root_logger.setLevel(py_logging.INFO)
    for logger_name in ("accelerate", "transformers", "trl", "datasets"):
        named_logger = py_logging.getLogger(logger_name)
        named_logger.propagate = True
        if named_logger.level > py_logging.INFO:
            named_logger.setLevel(py_logging.INFO)


class TextMetricsCallback(TrainerCallback):
    """Persist train/eval metrics to plain text and jsonl files."""

    def __init__(self, text_path: str, jsonl_path: str):
        self.text_path = text_path
        self.jsonl_path = jsonl_path
        os.makedirs(os.path.dirname(self.text_path), exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_text(self, line: str):
        with open(self.text_path, "a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")

    def _append_json(self, payload: dict[str, Any]):
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        del args, control, kwargs
        self._append_text(f"[{self._now()}] TRAIN_BEGIN step={state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        self._append_text(f"[{self._now()}] TRAIN_END step={state.global_step}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
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
        f"- runtime log: {os.path.join(run_layout['logs_dir'], f'runtime_rank{rank}.txt')}",
        f"- trainer text log: {os.path.join(run_layout['logs_dir'], f'trainer_events_rank{rank}.txt')}",
        f"- trainer jsonl log: {os.path.join(run_layout['logs_dir'], f'trainer_events_rank{rank}.jsonl')}",
        f"- rollout summary: {os.path.join(run_layout['logs_dir'], f'rollout_summary_rank{rank}.jsonl')}",
        f"- traces: {os.path.join(run_layout['traces_dir'], '*.json')}",
        f"- packaged share folder: {os.path.join(run_layout['run_root'], 'review_bundle')}",
        f"- trainer state / checkpoints / saved model: {run_layout['artifacts_dir']}",
        "",
        "Reading order:",
        "1. trainer_events_rank*.txt  -> step-level loss and metrics",
        "2. rollout_summary_rank*.jsonl -> per-question summary",
        "3. traces/*.json -> full tree and audit details for sampled questions",
        "4. runtime_rank*.txt -> raw runtime log",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _copy_if_exists(src: str, dst: str):
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _build_review_bundle(run_layout: dict[str, str], rank: int, trace_sample_size: int = 3):
    bundle_root = os.path.join(run_layout["run_root"], "review_bundle")
    logs_bundle = os.path.join(bundle_root, "logs")
    traces_bundle = os.path.join(bundle_root, "traces")
    artifacts_bundle = os.path.join(bundle_root, "artifacts")
    os.makedirs(logs_bundle, exist_ok=True)
    os.makedirs(traces_bundle, exist_ok=True)
    os.makedirs(artifacts_bundle, exist_ok=True)

    # Core metadata
    _copy_if_exists(os.path.join(run_layout["run_root"], "RUN_INDEX.txt"), os.path.join(bundle_root, "RUN_INDEX.txt"))
    _copy_if_exists(
        os.path.join(run_layout["run_root"], "run_manifest.json"),
        os.path.join(bundle_root, "run_manifest.json"),
    )

    # Logs
    for filename in (
        f"runtime_rank{rank}.txt",
        f"trainer_events_rank{rank}.txt",
        f"trainer_events_rank{rank}.jsonl",
        f"rollout_summary_rank{rank}.jsonl",
        "test_metrics.txt",
    ):
        _copy_if_exists(
            os.path.join(run_layout["logs_dir"], filename),
            os.path.join(logs_bundle, filename),
        )

    # Artifact summaries, but not heavy checkpoints.
    for filename in (
        "trainer_state.json",
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

    # Randomly sample a few rollout traces for quick review.
    trace_candidates = sorted(glob.glob(os.path.join(run_layout["traces_dir"], "*.json")))
    if trace_candidates:
        rng = random.Random(run_layout["run_id"])
        selected = (
            rng.sample(trace_candidates, min(trace_sample_size, len(trace_candidates)))
            if len(trace_candidates) > trace_sample_size
            else trace_candidates
        )
        for src in selected:
            _copy_if_exists(src, os.path.join(traces_bundle, os.path.basename(src)))

    bundle_note = [
        "This folder is the minimal package to send for debugging.",
        "",
        "Included:",
        "- RUN_INDEX.txt and run_manifest.json",
        "- plain-text and jsonl logs",
        "- trainer state / result json files",
        "- a random sample of rollout trace json files",
        "",
        "If a requested file is missing here, it was not produced in this run.",
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


def _adapt_splits(dataset, adapter, train_split: str, test_split: str):
    train_dataset = adapter.adapt_dataset(dataset[train_split])
    eval_dataset = adapter.adapt_dataset(dataset[test_split]) if test_split in dataset else None
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

    adapter = load_dataset_adapter(script_args.dataset_adapter)
    dataset = _load_dataset(script_args, dataset_args)

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = {
        "revision": model_args.model_revision,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
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
    )

    trainer = CodeGRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=[TextMetricsCallback(trainer_text_log_path, trainer_jsonl_path)],
    )

    if (
        training_args.codegrpo_mode == "train"
        and eval_dataset is not None
        and getattr(training_args, "run_base_model_baseline_eval", False)
    ):
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        if hasattr(unwrapped_model, "disable_adapter"):
            with use_adapter(unwrapped_model, adapter_name=None):
                baseline_metrics = trainer.evaluate(eval_dataset=eval_dataset)
        else:
            logger.warning("run_base_model_baseline_eval requested, but trainer model has no PEFT adapter to disable.")
            baseline_metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("baseline_eval", baseline_metrics)
        trainer.save_metrics("baseline_eval", baseline_metrics)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if training_args.codegrpo_mode == "test":
        test_dataset = eval_dataset if eval_dataset is not None else train_dataset
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        if rank == 0:
            with open(os.path.join(run_layout["logs_dir"], "test_metrics.txt"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_safe(metrics), ensure_ascii=False) + "\n")
            _build_review_bundle(run_layout, rank)
        trainer.accelerator.print(metrics)
        trainer.accelerator.print("CodeGRPO test mode completed.")
        return

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.accelerator.print("CodeGRPO training completed.")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"Model saved to {training_args.output_dir}.")
    if rank == 0:
        _build_review_bundle(run_layout, rank)

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
