import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Any

import torch
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback

from ..extensions.code_grpo import (
    CodeGRPOTreeRunner,
    build_backend,
    build_generation_completion,
    build_token_masks,
)
from ..extras.profiling import profiling_context
from .code_grpo_config import CodeGRPOConfig
from .grpo_trainer import GRPOTrainer
from .utils import get_config_model_id, pad


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logger = get_logger(__name__)


class CodeGRPOTrainer(GRPOTrainer):
    """Tree-search-based code GRPO trainer with code/reason orthogonal updates."""

    _tag_names = ["trl", "code_grpo"]
    _name = "CodeGRPO"

    def __init__(
        self,
        model: "str | PreTrainedModel",
        args: CodeGRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config=None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = CodeGRPOConfig(f"{model_name}-CodeGRPO")

        def _dummy_reward(prompts, completions, **kwargs):
            del completions, kwargs
            return [0.0 for _ in prompts]

        super().__init__(
            model=model,
            reward_funcs=_dummy_reward,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        if isinstance(self.processing_class, PreTrainedTokenizerBase):
            self.code_tokenizer = self.processing_class
        else:
            self.code_tokenizer = self.processing_class.tokenizer

        self.code_max_completion_length = int(
            getattr(self.args, "max_completion_length_code", None) or self.max_completion_length
        )
        self.audit_max_completion_length = int(
            getattr(self.args, "max_completion_length_audit", None) or self.max_completion_length
        )

        generation_defaults = {
            "max_new_tokens": self.code_max_completion_length,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }
        if hasattr(self, "generation_kwargs"):
            generation_defaults.update(self.generation_kwargs)

        vllm_generation = self.vllm_generation if getattr(self.args, "use_vllm", False) else None
        self.code_backend = build_backend(
            backend_name=self.args.backend,
            model=self.model,
            tokenizer=self.code_tokenizer,
            device=self.accelerator.device,
            generation_defaults=generation_defaults,
            vllm_generation=vllm_generation,
        )
        self.tree_runner = CodeGRPOTreeRunner(
            backend=self.code_backend,
            tokenizer=self.code_tokenizer,
            args=self.args,
            logger=logger,
        )
        self._reward_window_interval_steps = self._resolve_reward_window_interval_steps()
        self._reward_window_buffers: dict[str, list[float]] = defaultdict(list)
        self._trace_dump_counter = 0
        self._train_trace_dump_counter = 0
        self._last_eval_metrics: dict[str, float] = {}
        self._last_weight_probe_step: int | None = None
        self._last_weight_probe_metrics: dict[str, float] = {}
        logs_dir = os.path.join(os.path.dirname(self.args.output_dir), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self._rollout_summary_path = os.path.join(logs_dir, f"rollout_summary_rank{self.accelerator.process_index}.jsonl")

    def _resolve_reward_window_interval_steps(self) -> int:
        bins = int(getattr(self.args, "reward_window_bins", 0) or 0)
        max_steps = int(getattr(self.args, "max_steps", 0) or 0)
        if bins > 0 and max_steps > 0:
            return max(1, math.ceil(max_steps / bins))
        eval_steps = getattr(self.args, "eval_steps", 0)
        if isinstance(eval_steps, int) and eval_steps > 0:
            return eval_steps
        return 0

    @staticmethod
    def _window_mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _maybe_sync_vllm_weights(self) -> None:
        if not getattr(self, "use_vllm", False):
            return
        if not hasattr(self, "vllm_generation") or not hasattr(self, "_last_loaded_step"):
            return
        # Standalone eval can route adapters to vLLM via request-level dynamic LoRA.
        # In that mode, syncing HF-side weights into the colocated engine is both
        # unnecessary and harmful because it re-enters the old merge/load_weights path.
        if (
            getattr(self.vllm_generation, "vllm_dynamic_lora_path", None)
            and not getattr(self.vllm_generation, "vllm_dynamic_lora_online_refresh", False)
        ):
            self._last_loaded_step = self.state.global_step
            return
        sync_steps = max(1, int(getattr(self.args, "vllm_sync_steps", 5) or 1))
        should_sync = self._last_loaded_step < 0 or (
            self.state.global_step - self._last_loaded_step >= sync_steps
        )
        if not should_sync:
            return
        with profiling_context(self, "sync_weights"):
            self.vllm_generation.sync_weights()
        self._last_loaded_step = self.state.global_step

    def _update_train_reward_window_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        interval = self._reward_window_interval_steps
        if interval <= 0:
            return {}

        tracked = {
            "mean_R_code": "window/mean_R_code",
            "mean_R_reason": "window/mean_R_reason",
            "mean_R_soft_effective": "window/mean_R_soft_effective",
        }
        for source_key in tracked:
            if source_key in metrics:
                self._reward_window_buffers[source_key].append(float(metrics[source_key]))

        current_step = int(self.state.global_step)
        max_steps = int(getattr(self.args, "max_steps", 0) or 0)
        should_emit = current_step > 0 and (current_step % interval == 0 or (max_steps > 0 and current_step >= max_steps))
        if not should_emit:
            return {}

        payload: dict[str, float] = {
            "window/steps": float(len(self._reward_window_buffers.get("mean_R_code", []))),
            "window/end_step": float(current_step),
        }
        for source_key, target_key in tracked.items():
            values = self._reward_window_buffers.get(source_key, [])
            if values:
                payload[target_key] = self._window_mean(values)
        self._reward_window_buffers.clear()
        return payload

    @staticmethod
    def _select_probe_parameter(named_parameters, preferred_substrings: list[str]):
        for preferred in preferred_substrings:
            for name, parameter in named_parameters:
                if preferred in name:
                    return name, parameter
        return None, None

    @staticmethod
    def _tensor_probe_metrics(parameter: torch.Tensor, prefix: str) -> dict[str, float]:
        flat = parameter.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {
                f"{prefix}_sample_sum": 0.0,
                f"{prefix}_sample_abs_mean": 0.0,
                f"{prefix}_sample_abs_max": 0.0,
            }
        sample = flat[: min(1024, flat.numel())]
        return {
            f"{prefix}_sample_sum": float(sample.sum().item()),
            f"{prefix}_sample_abs_mean": float(sample.abs().mean().item()),
            f"{prefix}_sample_abs_max": float(sample.abs().max().item()),
        }

    def _get_vllm_probe_model(self):
        vllm_generation = getattr(self, "vllm_generation", None)
        llm = getattr(vllm_generation, "llm", None)
        llm_engine = getattr(llm, "llm_engine", None)
        model_executor = getattr(llm_engine, "model_executor", None)
        driver_worker = getattr(model_executor, "driver_worker", None)
        if driver_worker is None:
            workers = getattr(model_executor, "workers", None)
            if workers:
                driver_worker = workers[0]
        model_runner = getattr(driver_worker, "model_runner", None)
        return getattr(model_runner, "model", None)

    def _collect_eval_weight_probe_metrics(self) -> dict[str, float]:
        current_step = int(self.state.global_step)
        if self._last_weight_probe_step == current_step and self._last_weight_probe_metrics:
            return dict(self._last_weight_probe_metrics)

        metrics: dict[str, float] = {
            "probe_hf_lora_found": 0.0,
            "probe_vllm_qproj_found": 0.0,
        }
        hf_probe_name = ""
        vllm_probe_name = ""

        model_for_probe = self.accelerator.unwrap_model(self.model)
        hf_named_parameters = list(model_for_probe.named_parameters())
        hf_probe_name, hf_parameter = self._select_probe_parameter(
            hf_named_parameters,
            [
                "layers.0.self_attn.q_proj.lora_B.default.weight",
                "layers.0.self_attn.q_proj.lora_B.weight",
                "lora_B.default.weight",
                "lora_B.weight",
            ],
        )
        if hf_parameter is not None:
            metrics["probe_hf_lora_found"] = 1.0
            metrics.update(self._tensor_probe_metrics(hf_parameter, "probe_hf_lora"))

        if getattr(self, "use_vllm", False):
            vllm_model = self._get_vllm_probe_model()
            if vllm_model is not None:
                vllm_named_parameters = list(vllm_model.named_parameters())
                vllm_probe_name, vllm_parameter = self._select_probe_parameter(
                    vllm_named_parameters,
                    [
                        "model.layers.0.self_attn.q_proj.weight",
                        "layers.0.self_attn.q_proj.weight",
                        "self_attn.q_proj.weight",
                        "model.layers.0.self_attn.qkv_proj.weight",
                        "layers.0.self_attn.qkv_proj.weight",
                        "self_attn.qkv_proj.weight",
                        "qkv_proj.weight",
                    ],
                )
                if vllm_parameter is not None:
                    metrics["probe_vllm_qproj_found"] = 1.0
                    metrics.update(self._tensor_probe_metrics(vllm_parameter, "probe_vllm_qproj"))
                else:
                    candidate_names = [name for name, _ in vllm_named_parameters[:12]]
                    logger.info("[WEIGHT_PROBE] step=%s vllm_probe_candidates=%s", current_step, candidate_names)

        logger.info(
            "[WEIGHT_PROBE] step=%s hf_name=%s hf_sum=%.8f hf_abs_mean=%.8f vllm_name=%s vllm_sum=%.8f vllm_abs_mean=%.8f",
            current_step,
            hf_probe_name or "<missing>",
            metrics.get("probe_hf_lora_sample_sum", 0.0),
            metrics.get("probe_hf_lora_sample_abs_mean", 0.0),
            vllm_probe_name or "<missing>",
            metrics.get("probe_vllm_qproj_sample_sum", 0.0),
            metrics.get("probe_vllm_qproj_sample_abs_mean", 0.0),
        )

        self._last_weight_probe_step = current_step
        self._last_weight_probe_metrics = dict(metrics)
        return dict(metrics)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["question_id", "prompt", "test_cases"]

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean_v = sum(values) / len(values)
        return (sum((value - mean_v) ** 2 for value in values) / len(values)) ** 0.5

    def _unique_examples(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for idx, example in enumerate(inputs):
            key = str(example.get("question_id", f"q_{idx}"))
            if key not in deduped:
                deduped[key] = example
        return list(deduped.values())

    def _dump_rollout_traces(self, rollouts):
        trace_root = str(self.args.debug_trace_dir)
        trace_dir = trace_root if os.path.isabs(trace_root) else os.path.join(self.args.output_dir, trace_root)
        os.makedirs(trace_dir, exist_ok=True)
        for rollout in rollouts:
            safe_qid = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in rollout.question_id)
            trace_path = os.path.join(
                trace_dir,
                f"{safe_qid}_rank{self.accelerator.process_index}_{self._trace_dump_counter:06d}.json",
            )
            payload = {
                "question_id": rollout.question_id,
                "audit_indices": rollout.audit_indices,
                "rounds": rollout.rounds,
            }
            with open(trace_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            self._trace_dump_counter += 1

    def _select_rollouts_for_trace(self, rollouts, mode: str):
        selected = list(rollouts)
        whitelist = {str(qid) for qid in getattr(self.args, "debug_trace_question_ids", [])}
        if whitelist:
            selected = [rollout for rollout in selected if rollout.question_id in whitelist]
        sample_size = int(getattr(self.args, "debug_trace_sample_size", 0) or 0)
        if sample_size > 0 and len(selected) > sample_size and not whitelist:
            seed = int(self.args.seed + self.state.global_step + (0 if mode == "train" else 10_000_000))
            rng = random.Random(seed)
            selected = rng.sample(selected, sample_size)
        return selected

    def _should_dump_train_traces(self) -> bool:
        if not bool(getattr(self.args, "dump_train_traces", False)):
            return False
        max_files = int(getattr(self.args, "max_train_trace_files", 0) or 0)
        if max_files > 0 and self._train_trace_dump_counter >= max_files:
            return False
        interval = int(getattr(self.args, "dump_train_trace_interval_steps", 1) or 1)
        interval = abs(interval)
        if interval > 1 and int(self.state.global_step) % interval != 0:
            return False
        return True

    def _append_rollout_summaries(self, rollouts, mode: str):
        lines = []
        for rollout in rollouts:
            payload = {
                "mode": mode,
                "global_step": int(self.state.global_step),
                "question_id": rollout.question_id,
                "node_count": rollout.node_count,
                "resample_count": rollout.resample_count,
                "mean_R_code": rollout.mean_R_code,
                "mean_R_reason": rollout.mean_R_reason,
                "mean_pass_rate": rollout.mean_pass_rate,
                **rollout.eval_metrics,
            }
            lines.append(json.dumps(payload, ensure_ascii=False))
        if lines:
            with open(self._rollout_summary_path, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")

    def _build_training_batch(self, train_samples: list) -> dict[str, torch.Tensor]:
        device = self.accelerator.device
        prompt_ids_tensors = []
        prompt_masks = []
        completion_ids_tensors = []
        completion_masks = []
        code_masks = []
        reason_masks = []
        advantages_code = []
        advantages_reason = []
        r_code = []
        pass_rates = []

        for sample in train_samples:
            prompt_ids = self.code_tokenizer(sample.prompt_text, add_special_tokens=False)["input_ids"]
            completion_ids = self.code_tokenizer(sample.completion_text, add_special_tokens=False)["input_ids"]
            if not completion_ids:
                completion_ids = [self.eos_token_id]

            code_mask = list(sample.code_token_mask)
            reason_mask = list(sample.reason_token_mask)
            if len(code_mask) < len(completion_ids):
                code_mask.extend([0] * (len(completion_ids) - len(code_mask)))
            if len(reason_mask) < len(completion_ids):
                reason_mask.extend([0] * (len(completion_ids) - len(reason_mask)))
            code_mask = code_mask[: len(completion_ids)]
            reason_mask = reason_mask[: len(completion_ids)]

            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
            completion_tensor = torch.tensor(completion_ids, dtype=torch.long)
            prompt_ids_tensors.append(prompt_tensor)
            prompt_masks.append(torch.ones_like(prompt_tensor, dtype=torch.long))
            completion_ids_tensors.append(completion_tensor)
            completion_masks.append(torch.ones_like(completion_tensor, dtype=torch.long))
            code_masks.append(torch.tensor(code_mask, dtype=torch.float32))
            reason_masks.append(torch.tensor(reason_mask, dtype=torch.float32))
            advantages_code.append(sample.A_code)
            advantages_reason.append(sample.A_reason)
            r_code.append(sample.R_code)
            pass_rates.append(sample.pass_rate)

        prompt_ids = pad(
            prompt_ids_tensors,
            padding_value=self.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        prompt_mask = pad(
            prompt_masks,
            padding_value=0,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        completion_ids = pad(
            completion_ids_tensors,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        completion_mask = pad(
            completion_masks,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        code_token_mask = pad(code_masks, padding_value=0.0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of).to(
            device
        )
        reason_token_mask = pad(
            reason_masks, padding_value=0.0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "code_token_mask": code_token_mask,
            "reason_token_mask": reason_token_mask,
            "advantages_code": torch.tensor(advantages_code, dtype=torch.float32, device=device),
            "advantages_reason": torch.tensor(advantages_reason, dtype=torch.float32, device=device),
            "r_code": torch.tensor(r_code, dtype=torch.float32, device=device),
            "pass_rate": torch.tensor(pass_rates, dtype=torch.float32, device=device),
            "num_items_in_batch": completion_mask.sum(),
        }

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        mode = "train" if self.model.training else "eval"
        examples = self._unique_examples(inputs)
        rollout_t0 = time.perf_counter()
        self._maybe_sync_vllm_weights()
        probe_metrics = self._collect_eval_weight_probe_metrics() if mode == "eval" else {}

        seed_offset = self.state.global_step if mode == "train" else self.state.global_step + 1_000_000
        base_rng = random.Random(self.args.seed + seed_offset)

        rollouts = []
        for _idx, example in enumerate(examples):
            rollout_seed = base_rng.randint(0, 2**31 - 1)
            if mode == "eval" and getattr(self.args, "eval_code_only_single_trajectory", True):
                repeat_count = max(1, int(getattr(self.args, "eval_repeat_count", 1)))
                for repeat_idx in range(repeat_count):
                    repeat_seed = rollout_seed + repeat_idx * 9973
                    rollout = self.tree_runner.run_question_eval_code_only(
                        example,
                        rng=random.Random(repeat_seed),
                    )
                    rollout.repeat_idx = repeat_idx
                    rollouts.append(rollout)
            else:
                rollout = self.tree_runner.run_question(
                    example,
                    rng=random.Random(rollout_seed),
                    update_exec_baseline=(mode == "train"),
                )
                rollouts.append(rollout)
        rollout_time_s = max(time.perf_counter() - rollout_t0, 1e-8)

        train_samples = [sample for rollout in rollouts for sample in rollout.train_samples]
        if not train_samples and examples:
            fallback_completion = build_generation_completion("")
            _, fallback_code_mask, fallback_reason_mask = build_token_masks(self.code_tokenizer, fallback_completion)
            train_samples = [
                {
                    "question_id": str(examples[0]["question_id"]),
                    "prompt_text": str(examples[0]["prompt"]),
                    "completion_text": fallback_completion,
                    "code_token_mask": fallback_code_mask,
                    "reason_token_mask": fallback_reason_mask,
                    "A_code": 0.0,
                    "A_reason": 0.0,
                    "R_code": 0.0,
                    "pass_rate": 0.0,
                }
            ]
            # Align object shape with TrainSample attributes.
            class Obj:
                def __init__(self, d):
                    self.__dict__.update(d)

            train_samples = [Obj(train_samples[0])]
        if not train_samples:
            raise ValueError("CodeGRPOTrainer received an empty generation batch and could not create fallback samples.")

        batch = self._build_training_batch(train_samples)

        mean_r_code = self._mean([rollout.mean_R_code for rollout in rollouts])
        mean_r_reason = self._mean([rollout.mean_R_reason for rollout in rollouts])
        mean_pass_rate = self._mean([rollout.mean_pass_rate for rollout in rollouts])
        std_r_code = self._mean([rollout.std_R_code for rollout in rollouts])
        std_r_reason = self._mean([rollout.std_R_reason for rollout in rollouts])
        node_count = float(sum(rollout.node_count for rollout in rollouts))
        resample_count = float(sum(rollout.resample_count for rollout in rollouts))

        self._metrics[mode]["mean_R_code"].append(mean_r_code)
        self._metrics[mode]["mean_R_reason"].append(mean_r_reason)
        self._metrics[mode]["mean_pass_rate"].append(mean_pass_rate)
        self._metrics[mode]["std_R_code"].append(std_r_code)
        self._metrics[mode]["std_R_reason"].append(std_r_reason)
        self._metrics[mode]["node_count"].append(node_count)
        self._metrics[mode]["resample_count"].append(resample_count)
        self._metrics[mode]["rollout_time_s"].append(float(rollout_time_s))
        self._metrics[mode]["rollout_nodes_per_s"].append(float(node_count / rollout_time_s))
        self._metrics[mode]["rollout_questions_per_s"].append(float(len(examples) / rollout_time_s))

        metric_keys = sorted({key for rollout in rollouts for key in rollout.eval_metrics.keys()})
        merged_eval_metrics: dict[str, float] = {}
        per_repeat_metric_values: dict[str, list[float]] = {}
        if mode == "eval" and getattr(self.args, "eval_code_only_single_trajectory", True):
            repeat_groups: dict[int, list[Any]] = {}
            for rollout in rollouts:
                repeat_idx = int(getattr(rollout, "repeat_idx", 0) or 0)
                repeat_groups.setdefault(repeat_idx, []).append(rollout)
            for key in metric_keys:
                repeat_means = [
                    self._mean([rollout.eval_metrics.get(key, 0.0) for rollout in group])
                    for _repeat_idx, group in sorted(repeat_groups.items())
                ]
                per_repeat_metric_values[key] = repeat_means
                value = self._mean(repeat_means)
                value_std = self._std(repeat_means)
                self._metrics[mode][key].append(value)
                self._metrics[mode][f"{key}_std"].append(value_std)
                merged_eval_metrics[key] = value
                merged_eval_metrics[f"{key}_std"] = value_std
        else:
            for key in metric_keys:
                value = self._mean([rollout.eval_metrics.get(key, 0.0) for rollout in rollouts])
                self._metrics[mode][key].append(value)
                merged_eval_metrics[key] = value

        if self.args.codegrpo_mode == "test":
            selected_rollouts = self._select_rollouts_for_trace(rollouts, mode="test")
            self._dump_rollout_traces(selected_rollouts)
            if bool(getattr(self.args, "log_trace_dump_events", False)):
                logger.info("[TEST] dumped %d/%d rollout trace files", len(selected_rollouts), len(rollouts))
        elif mode == "eval":
            if self.args.dump_eval_traces:
                selected_rollouts = self._select_rollouts_for_trace(rollouts, mode="eval")
                self._dump_rollout_traces(selected_rollouts)
                if bool(getattr(self.args, "log_trace_dump_events", False)):
                    logger.info("[EVAL] dumped %d/%d rollout trace files", len(selected_rollouts), len(rollouts))
        else:
            if self._should_dump_train_traces():
                selected_rollouts = self._select_rollouts_for_trace(rollouts, mode="train")
                max_files = int(getattr(self.args, "max_train_trace_files", 0) or 0)
                if max_files > 0:
                    remaining = max(0, max_files - self._train_trace_dump_counter)
                    selected_rollouts = selected_rollouts[:remaining]
                self._dump_rollout_traces(selected_rollouts)
                self._train_trace_dump_counter += len(selected_rollouts)
                if bool(getattr(self.args, "log_trace_dump_events", False)):
                    logger.info("[TRAIN] dumped %d/%d rollout trace files", len(selected_rollouts), len(rollouts))
            if bool(getattr(self.args, "log_train_rollout_details", False)):
                logger.info("[TRAIN] built %d training samples from %d questions", len(train_samples), len(examples))

        if mode == "eval":
            for key, value in probe_metrics.items():
                self._metrics[mode][key].append(float(value))
                merged_eval_metrics[key] = float(value)

            self._last_eval_metrics = {
                "mean_R_code": mean_r_code,
                "mean_R_reason": mean_r_reason,
                "mean_pass_rate": mean_pass_rate,
                "std_R_code": std_r_code,
                "std_R_reason": std_r_reason,
                "node_count": node_count,
                "resample_count": resample_count,
                **merged_eval_metrics,
            }
        self._append_rollout_summaries(rollouts, mode=mode)

        return batch

    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # CodeGRPO uses tree rollout per incoming batch directly; avoid GRPO buffered slicing assumptions.
        return self._generate_and_score_completions(generation_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        if return_outputs:
            raise ValueError("CodeGRPOTrainer does not support returning outputs from compute_loss.")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            compute_entropy=False,
        )

        code_mask = inputs["code_token_mask"] * completion_mask
        reason_mask = inputs["reason_token_mask"] * completion_mask
        advantages_code = inputs["advantages_code"]
        advantages_reason = inputs["advantages_reason"]

        loss_code_per_seq = -advantages_code * (per_token_logps * code_mask).sum(dim=1)
        loss_reason_per_seq = -advantages_reason * (per_token_logps * reason_mask).sum(dim=1)
        loss_code = loss_code_per_seq.mean()
        loss_reason = loss_reason_per_seq.mean()
        loss = loss_code + self.args.beta_reason * loss_reason

        mode = "train" if self.model.training else "eval"
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        loss = loss / normalizer

        gathered_loss_code = self.accelerator.gather(loss_code.detach()).float().mean().item()
        gathered_loss_reason = self.accelerator.gather(loss_reason.detach()).float().mean().item()
        self._metrics[mode]["loss_code"].append(gathered_loss_code)
        self._metrics[mode]["loss_reason"].append(gathered_loss_reason)
        if bool(getattr(self.args, "log_reward_losses", False)):
            logger.info(
                "[REWARD] loss_code=%.6f loss_reason=%.6f beta_reason=%.4f",
                gathered_loss_code,
                gathered_loss_reason,
                self.args.beta_reason,
            )
        return loss

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if self._last_eval_metrics:
            normalized_metrics = {}
            for key, value in self._last_eval_metrics.items():
                normalized_key = key if key.startswith("eval_") else f"eval_{key}"
                normalized_metrics[normalized_key] = value
            metrics.update(normalized_metrics)
        return metrics

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        if mode == "train":
            metrics.update(self._update_train_reward_window_metrics(metrics))
        else:
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
            self._last_eval_metrics = dict(metrics)

        logs = {**logs, **metrics}
        self._metrics[mode].clear()
        super().log(logs, start_time)
