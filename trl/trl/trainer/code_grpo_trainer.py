import copy
import json
import math
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import transformers
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
from ..models.utils import disable_gradient_checkpointing
from .code_grpo_config import CodeGRPOConfig
from .grpo_trainer import GRPOTrainer
from .utils import get_config_model_id, pad, use_adapter


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logger = get_logger(__name__)


@dataclass
class _CodeGRPOLookaheadBatch:
    current_batch: Any
    next_batch: Any | None = None

    def __contains__(self, key):
        if isinstance(self.current_batch, dict):
            return key in self.current_batch
        return False

    def __getitem__(self, key):
        if isinstance(self.current_batch, dict):
            return self.current_batch[key]
        raise KeyError(key)

    def __iter__(self):
        if isinstance(self.current_batch, dict):
            return iter(self.current_batch)
        return iter(())

    def __len__(self):
        if isinstance(self.current_batch, dict):
            return len(self.current_batch)
        return 0

    def get(self, key, default=None):
        if isinstance(self.current_batch, dict):
            return self.current_batch.get(key, default)
        return default


@dataclass
class _CodeGRPORolloutResult:
    mode: str
    batch: dict[str, torch.Tensor | Any]
    rollouts: list[Any]
    metric_updates: dict[str, float]
    eval_metric_snapshot: dict[str, float]
    train_sample_count: int
    examples_count: int


class _CodeGRPOLookaheadLoader:
    def __init__(self, base_loader):
        self._base_loader = base_loader

    def __len__(self):
        return len(self._base_loader)

    def __getattr__(self, name):
        return getattr(self._base_loader, name)

    def __iter__(self):
        iterator = iter(self._base_loader)
        try:
            current_batch = next(iterator)
        except StopIteration:
            return
        try:
            next_batch = next(iterator)
        except StopIteration:
            next_batch = None

        while True:
            yield _CodeGRPOLookaheadBatch(current_batch=current_batch, next_batch=next_batch)
            if next_batch is None:
                break
            current_batch = next_batch
            try:
                next_batch = next(iterator)
            except StopIteration:
                next_batch = None


class CodeGRPOTrainer(GRPOTrainer):
    """Tree-search-based code GRPO trainer with code/reason orthogonal updates."""

    _tag_names = ["trl", "code_grpo"]
    _name = "CodeGRPO"
    # --- 控制台只打核心进度指标（保持简洁） ---
    _CONSOLE_TRAIN_KEYS = frozenset(
        {
            "loss",
            "loss_code",
            "loss_reason",
            "learning_rate",
            "grad_norm",
            "mean_R_code",
            "mean_pass_rate",
            "reward/tie_rate",
        }
    )
    # --- TensorBoard 写入全量诊断指标 ---
    _TRAIN_LOG_KEYS = frozenset(
        {
            # 核心进度
            "loss",
            "loss_code",
            "loss_reason",
            "learning_rate",
            "grad_norm",
            "kl",
            "mean_R_code",
            "mean_R_reason",
            "mean_pass_rate",
            "rollout_time_s",
            # reward 诊断
            "reward/tie_rate",
            "reward/R_code_min",
            "reward/R_code_max",
            "std_R_code",
            # advantage 诊断
            "advantage/code_nonzero_rate",
            "advantage/code_std",
            "nonzero_A_code_rate",
            "mean_abs_A_code",
            # ratio 诊断
            "ratio/mean",
            "ratio/max",
            "clip_low_rate",
            "clip_high_rate",
            # batch/length 诊断
            "completion_len/code_mean",
            "tokens_per_update",
            "effective_prompts_per_update",
            "effective_rollouts_per_update",
            # tree-level 诊断
            "sibling_group_zero_std_R_code_rate",
            "sibling_group_both_R_code_1_rate",
            "sibling_group_mean_pass_rate_gap",
            # reward window 滑动均值
            "window/mean_R_code",
            "window/mean_R_reason",
            "window/mean_R_soft_effective",
            "window/steps",
            "window/end_step",
        }
    )
    # --- 控制台 eval 只打核心结果 ---
    _CONSOLE_EVAL_KEYS = frozenset(
        {
            "eval_loss",
            "eval_pass_at_1",
            "eval_best_pass_rate_overall",
            "eval_mean_R_code",
            "eval_mean_pass_rate",
        }
    )
    # --- TensorBoard eval 全量写入 ---
    _EVAL_LOG_KEYS = frozenset(
        {
            "eval_loss",
            "eval_kl",
            "eval_pass_at_1",
            "eval_best_pass_rate_overall",
            "eval_mean_R_code",
            "eval_mean_R_reason",
            "eval_mean_pass_rate",
            "eval_std_R_code",
            "eval_generation_format_ok_rate",
            "eval_compile_ok_rate",
            "eval_syntax_error_rate",
            "eval_timeout_rate",
        }
    )
    _EVAL_LOG_PREFIXES = ("eval_pass_at_1_round_", "eval_best_pass_rate_round_")

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

        # CodeGRPO currently relies on repeated model forwards around tree rollouts. With PEFT models, the
        # non-reentrant checkpoint path has been fragile in practice on Qwen2-class models; keep the safer
        # reentrant behavior unless the caller explicitly overrides it.
        if peft_config is not None and getattr(args, "gradient_checkpointing", False):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", True)

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
        self._async_rollout_executor: ThreadPoolExecutor | None = None
        self._prefetched_rollout_future: Future | None = None
        self._prefetched_rollout_batch_id: int | None = None
        self._async_rollout_prefetch_enabled = self._resolve_async_rollout_prefetch_enabled()
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

    def _resolve_async_rollout_prefetch_enabled(self) -> bool:
        enabled = bool(getattr(self.args, "async_rollout_prefetch", False))
        if not enabled:
            return False
        if not bool(getattr(self.args, "use_vllm", False)) or str(getattr(self.args, "backend", "hf")) != "vllm":
            logger.warning("Disabling async_rollout_prefetch because it requires use_vllm=true and backend='vllm'.")
            return False
        if int(getattr(self.accelerator, "num_processes", 1) or 1) != 1:
            logger.warning("Disabling async_rollout_prefetch because it currently supports single-process training only.")
            return False
        return True

    def _ensure_async_rollout_executor(self) -> ThreadPoolExecutor:
        if self._async_rollout_executor is None:
            self._async_rollout_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="codegrpo-rollout")
        return self._async_rollout_executor

    def _shutdown_async_rollout_prefetch(self, wait: bool = False) -> None:
        future = self._prefetched_rollout_future
        executor = self._async_rollout_executor
        self._prefetched_rollout_future = None
        self._prefetched_rollout_batch_id = None
        self._async_rollout_executor = None
        if future is not None and not future.done():
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)

    def _submit_async_rollout_prefetch(self, raw_batch: list[dict[str, Any]]) -> None:
        if not self._async_rollout_prefetch_enabled:
            return
        self._maybe_sync_vllm_weights()
        executor = self._ensure_async_rollout_executor()
        batch_copy = copy.deepcopy(raw_batch)
        self._prefetched_rollout_future = executor.submit(self._compute_rollout_result, batch_copy, "train")
        self._prefetched_rollout_batch_id = id(raw_batch)

    def _pop_prefetched_rollout_result(self, raw_batch: list[dict[str, Any]]) -> _CodeGRPORolloutResult | None:
        if self._prefetched_rollout_batch_id != id(raw_batch) or self._prefetched_rollout_future is None:
            return None
        future = self._prefetched_rollout_future
        self._prefetched_rollout_future = None
        self._prefetched_rollout_batch_id = None
        try:
            return future.result()
        except Exception:
            logger.exception("Async rollout prefetch failed; falling back to synchronous rollout generation.")
            self._shutdown_async_rollout_prefetch(wait=False)
            return None

    @staticmethod
    def _window_mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _maybe_sync_vllm_weights(self) -> None:
        if not getattr(self, "use_vllm", False):
            return
        if not hasattr(self, "vllm_generation") or not hasattr(self, "_last_loaded_step"):
            return
        if (
            getattr(self.vllm_generation, "mode", None) == "server"
            and not getattr(self.vllm_generation, "enable_server_weight_sync", True)
        ):
            self._last_loaded_step = self.state.global_step
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

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["question_id", "prompt", "test_cases"]

    def get_train_dataloader(self):
        train_loader = super().get_train_dataloader()
        if not self._async_rollout_prefetch_enabled:
            return train_loader
        return _CodeGRPOLookaheadLoader(train_loader)

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            self._shutdown_async_rollout_prefetch(wait=False)

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
                "global_step": int(self.state.global_step),
                "question_id": rollout.question_id,
                "audit_indices": rollout.audit_indices,
                "rounds": rollout.rounds,
            }
            with open(trace_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            self._trace_dump_counter += 1

    @staticmethod
    def _iter_rollout_nodes(rollout):
        for round_item in getattr(rollout, "rounds", []) or []:
            for node in round_item.get("nodes", []) or []:
                yield node

    def _score_rollout_for_trace(self, rollout) -> tuple[float, ...]:
        nodes = list(self._iter_rollout_nodes(rollout))
        if not nodes:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        hard_error_count = 0
        compile_fail_count = 0
        soft_fail_count = 0
        max_soft_fail = 0.0
        unsolved_count = 0
        any_full_text = 0.0

        for node in nodes:
            pass_rate = float(node.get("pass_rate", 0.0) or 0.0)
            compile_score = float(node.get("compile_score", 0.0) or 0.0)
            status_code = str(node.get("status_code", "") or "")
            generation_format_ok = bool(node.get("generation_format_ok", False))
            r_soft_effective = float(node.get("R_soft_effective", 0.0) or 0.0)
            generation_debug = node.get("generation_debug", {}) or {}

            if generation_debug.get("full_prompt_rendered") and generation_debug.get("full_raw_output"):
                any_full_text = 1.0

            if status_code in {"SYNTAX_ERROR", "RUNTIME_ERROR", "TIMEOUT"}:
                hard_error_count += 1
            if compile_score < 1.0 or not generation_format_ok:
                compile_fail_count += 1
            if pass_rate < 1.0:
                unsolved_count += 1
                if r_soft_effective > 0.0:
                    soft_fail_count += 1
                    max_soft_fail = max(max_soft_fail, r_soft_effective)

        return (
            float(hard_error_count),
            float(compile_fail_count),
            float(soft_fail_count),
            float(max_soft_fail),
            float(unsolved_count),
            any_full_text,
        )

    def _select_rollouts_for_trace(self, rollouts, mode: str):
        selected = list(rollouts)
        whitelist = {str(qid) for qid in getattr(self.args, "debug_trace_question_ids", [])}
        if whitelist:
            selected = [rollout for rollout in selected if rollout.question_id in whitelist]
        else:
            selected = sorted(selected, key=self._score_rollout_for_trace, reverse=True)
        sample_size = int(getattr(self.args, "debug_trace_sample_size", 0) or 0)
        if sample_size > 0 and len(selected) > sample_size:
            selected = selected[:sample_size]
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

    def _should_record_kl_metric(self, mode: str) -> bool:
        if not bool(getattr(self.args, "log_kl_metrics", True)):
            return False
        if mode == "eval":
            return True
        logging_strategy = str(getattr(self.args, "logging_strategy", "steps") or "steps")
        if logging_strategy == "no":
            return False
        if logging_strategy != "steps":
            return True
        logging_steps = int(getattr(self.args, "logging_steps", 0) or 0)
        if logging_steps <= 1:
            return True
        next_step = int(self.state.global_step) + 1
        max_steps = int(getattr(self.args, "max_steps", 0) or 0)
        return next_step % logging_steps == 0 or (max_steps > 0 and next_step >= max_steps)

    def _compute_reference_kl_metric(
        self,
        per_token_logps: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> float | None:
        if not self._should_record_kl_metric("train" if self.model.training else "eval"):
            return None

        with torch.no_grad():
            ref_per_token_logps = self._get_reference_per_token_logps(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
            )
            if ref_per_token_logps is None:
                return None

            log_ratio = ref_per_token_logps - per_token_logps.detach()
            per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
            denom = completion_mask.sum().clamp(min=1.0)
            mean_kl = (per_token_kl * completion_mask).sum() / denom
            return self.accelerator.gather(mean_kl).nanmean().item()

    def _get_reference_per_token_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor | None:
        if self.ref_model is not None:
            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.ref_model,
                input_ids,
                attention_mask,
                logits_to_keep=logits_to_keep,
                compute_entropy=False,
            )
            return ref_per_token_logps

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        peft_config = getattr(unwrapped_model, "peft_config", None)
        if peft_config is None:
            return None

        adapter_name = "ref" if "ref" in peft_config else None
        with use_adapter(unwrapped_model, adapter_name=adapter_name):
            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                input_ids,
                attention_mask,
                logits_to_keep=logits_to_keep,
                compute_entropy=False,
            )
        return ref_per_token_logps

    def _filter_console_logs(self, logs: dict[str, float], mode: str) -> dict[str, float]:
        """控制台精简过滤：只保留核心进度指标，避免刷屏。"""
        if not bool(getattr(self.args, "compact_logging", True)):
            return logs
        if mode == "train":
            return {key: value for key, value in logs.items() if key in self._CONSOLE_TRAIN_KEYS}
        return {
            key: value
            for key, value in logs.items()
            if key in self._CONSOLE_EVAL_KEYS or any(key.startswith(prefix) for prefix in self._EVAL_LOG_PREFIXES)
        }

    def _filter_tb_logs(self, logs: dict[str, float], mode: str) -> dict[str, float]:
        """TensorBoard 全量过滤：写入所有诊断指标供离线分析。"""
        if mode == "train":
            return {key: value for key, value in logs.items() if key in self._TRAIN_LOG_KEYS}
        return {
            key: value
            for key, value in logs.items()
            if not key.endswith("_std")
            and (key in self._EVAL_LOG_KEYS or any(key.startswith(prefix) for prefix in self._EVAL_LOG_PREFIXES))
        }

    def public_metrics(self, metrics: dict[str, float], split: str) -> dict[str, float]:
        """eval summary 打印用控制台精简版。"""
        mode = "eval" if split in {"eval", "test", "baseline_eval"} else "train"
        return self._filter_console_logs(dict(metrics), mode=mode)

    def _build_training_batch(self, train_samples: list, to_device: bool = True) -> dict[str, torch.Tensor]:
        prompt_ids_tensors = []
        prompt_masks = []
        completion_ids_tensors = []
        completion_masks = []
        code_masks = []
        reason_masks = []
        old_logprobs = []
        advantages_code = []
        advantages_reason = []
        r_code = []
        pass_rates = []
        has_old_logprobs = True

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
            sample_old_logprobs = getattr(sample, "old_per_token_logps", None)
            if sample_old_logprobs is None:
                has_old_logprobs = False
            else:
                values = list(sample_old_logprobs)[: len(completion_ids)]
                if len(values) < len(completion_ids):
                    values.extend([0.0] * (len(completion_ids) - len(values)))
                old_logprobs.append(torch.tensor(values, dtype=torch.float32))
            advantages_code.append(sample.A_code)
            advantages_reason.append(sample.A_reason)
            r_code.append(sample.R_code)
            pass_rates.append(sample.pass_rate)

        prompt_ids = pad(
            prompt_ids_tensors,
            padding_value=self.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        prompt_mask = pad(
            prompt_masks,
            padding_value=0,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        completion_ids = pad(
            completion_ids_tensors,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        completion_mask = pad(
            completion_masks,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        code_token_mask = pad(code_masks, padding_value=0.0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of)
        reason_token_mask = pad(
            reason_masks, padding_value=0.0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )

        batch = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "code_token_mask": code_token_mask,
            "reason_token_mask": reason_token_mask,
            "advantages_code": torch.tensor(advantages_code, dtype=torch.float32),
            "advantages_reason": torch.tensor(advantages_reason, dtype=torch.float32),
            "r_code": torch.tensor(r_code, dtype=torch.float32),
            "pass_rate": torch.tensor(pass_rates, dtype=torch.float32),
            "num_items_in_batch": completion_mask.sum(),
            **(
                {
                    "old_per_token_logps": pad(
                        old_logprobs,
                        padding_value=0.0,
                        padding_side="right",
                        pad_to_multiple_of=self.pad_to_multiple_of,
                    )
                }
                if has_old_logprobs and old_logprobs
                else {}
            ),
        }
        if to_device:
            batch = self._move_batch_to_device(batch)
        return batch

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        moved: dict[str, torch.Tensor | Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _attach_old_per_token_logps(self, batch: dict[str, torch.Tensor | Any]) -> None:
        if "old_per_token_logps" in batch:
            return

        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        completion_ids = batch["completion_ids"]
        completion_mask = batch["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                input_ids,
                attention_mask,
                logits_to_keep=logits_to_keep,
                compute_entropy=False,
            )

        batch["old_per_token_logps"] = old_per_token_logps.detach()

    def _compute_rollout_result(self, inputs: list[dict[str, Any]], mode: str) -> _CodeGRPORolloutResult:
        examples = self._unique_examples(inputs)
        rollout_t0 = time.perf_counter()
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

        batch = self._build_training_batch(train_samples, to_device=False)

        mean_r_code = self._mean([rollout.mean_R_code for rollout in rollouts])
        mean_r_reason = self._mean([rollout.mean_R_reason for rollout in rollouts])
        mean_pass_rate = self._mean([rollout.mean_pass_rate for rollout in rollouts])
        std_r_code = self._mean([rollout.std_R_code for rollout in rollouts])
        std_r_reason = self._mean([rollout.std_R_reason for rollout in rollouts])
        node_count = float(sum(rollout.node_count for rollout in rollouts))
        resample_count = float(sum(rollout.resample_count for rollout in rollouts))

        metric_updates: dict[str, float] = {
            "mean_R_code": mean_r_code,
            "mean_R_reason": mean_r_reason,
            "mean_pass_rate": mean_pass_rate,
            "std_R_code": std_r_code,
            "std_R_reason": std_r_reason,
            "node_count": node_count,
            "resample_count": resample_count,
            "rollout_time_s": float(rollout_time_s),
            "rollout_nodes_per_s": float(node_count / rollout_time_s),
            "rollout_questions_per_s": float(len(examples) / rollout_time_s),
        }

        # [诊断面板] rollout 层面的 reward / advantage / batch 诊断
        all_r_codes = [s.R_code for s in train_samples if hasattr(s, "R_code")]
        all_pass_rates = [s.pass_rate for s in train_samples if hasattr(s, "pass_rate")]
        all_a_codes = [s.A_code for s in train_samples if hasattr(s, "A_code")]
        if all_r_codes:
            metric_updates["reward/R_code_min"] = min(all_r_codes)
            metric_updates["reward/R_code_max"] = max(all_r_codes)
            # [诊断面板] reward tie rate: 一组 sibling 的 R_code 相同的比例
            pair_diffs = []
            for rollout in rollouts:
                for sample in rollout.train_samples:
                    if hasattr(sample, "R_code"):
                        pair_diffs.append(sample.R_code)
            # tie_rate 从 advantage 视角：A_code == 0 的样本比例
            if all_a_codes:
                tie_count = sum(1 for a in all_a_codes if abs(a) < 1e-8)
                metric_updates["reward/tie_rate"] = tie_count / len(all_a_codes)
        if all_pass_rates:
            metric_updates["reward/pass_rate_min"] = min(all_pass_rates)
            metric_updates["reward/pass_rate_max"] = max(all_pass_rates)
        # [诊断面板] effective prompts per update
        metric_updates["effective_prompts_per_update"] = float(len(examples))
        metric_updates["effective_rollouts_per_update"] = float(len(train_samples))

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
                metric_updates[key] = value
                metric_updates[f"{key}_std"] = value_std
                merged_eval_metrics[key] = value
                merged_eval_metrics[f"{key}_std"] = value_std
        else:
            for key in metric_keys:
                value = self._mean([rollout.eval_metrics.get(key, 0.0) for rollout in rollouts])
                metric_updates[key] = value
                merged_eval_metrics[key] = value

        eval_metric_snapshot = (
            {
                "mean_R_code": mean_r_code,
                "mean_R_reason": mean_r_reason,
                "mean_pass_rate": mean_pass_rate,
                "std_R_code": std_r_code,
                "std_R_reason": std_r_reason,
                "node_count": node_count,
                "resample_count": resample_count,
                **merged_eval_metrics,
            }
            if mode == "eval"
            else {}
        )

        return _CodeGRPORolloutResult(
            mode=mode,
            batch=batch,
            rollouts=rollouts,
            metric_updates=metric_updates,
            eval_metric_snapshot=eval_metric_snapshot,
            train_sample_count=len(train_samples),
            examples_count=len(examples),
        )

    def _finalize_rollout_result(self, result: _CodeGRPORolloutResult) -> dict[str, torch.Tensor | Any]:
        mode = result.mode
        batch = self._move_batch_to_device(result.batch)
        if mode == "train":
            self._attach_old_per_token_logps(batch)

        for key, value in result.metric_updates.items():
            self._metrics[mode][key].append(value)

        if self.args.codegrpo_mode == "test":
            selected_rollouts = self._select_rollouts_for_trace(result.rollouts, mode="test")
            self._dump_rollout_traces(selected_rollouts)
            if bool(getattr(self.args, "log_trace_dump_events", False)):
                logger.info("[TEST] dumped %d/%d rollout trace files", len(selected_rollouts), len(result.rollouts))
        elif mode == "eval":
            if self.args.dump_eval_traces:
                selected_rollouts = self._select_rollouts_for_trace(result.rollouts, mode="eval")
                self._dump_rollout_traces(selected_rollouts)
                if bool(getattr(self.args, "log_trace_dump_events", False)):
                    logger.info("[EVAL] dumped %d/%d rollout trace files", len(selected_rollouts), len(result.rollouts))
            self._last_eval_metrics = dict(result.eval_metric_snapshot)
        else:
            if self._should_dump_train_traces():
                selected_rollouts = self._select_rollouts_for_trace(result.rollouts, mode="train")
                max_files = int(getattr(self.args, "max_train_trace_files", 0) or 0)
                if max_files > 0:
                    remaining = max(0, max_files - self._train_trace_dump_counter)
                    selected_rollouts = selected_rollouts[:remaining]
                self._dump_rollout_traces(selected_rollouts)
                self._train_trace_dump_counter += len(selected_rollouts)
                if bool(getattr(self.args, "log_trace_dump_events", False)):
                    logger.info("[TRAIN] dumped %d/%d rollout trace files", len(selected_rollouts), len(result.rollouts))
            if bool(getattr(self.args, "log_train_rollout_details", False)):
                logger.info(
                    "[TRAIN] built %d training samples from %d questions",
                    result.train_sample_count,
                    result.examples_count,
                )

        self._append_rollout_summaries(result.rollouts, mode=mode)
        return batch

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            self._maybe_sync_vllm_weights()
        result = self._compute_rollout_result(inputs, mode)
        return self._finalize_rollout_result(result)

    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        if self.model.training and isinstance(generation_batch, _CodeGRPOLookaheadBatch):
            current_batch = generation_batch.current_batch
            result = self._pop_prefetched_rollout_result(current_batch)
            if result is None:
                self._maybe_sync_vllm_weights()
                result = self._compute_rollout_result(current_batch, "train")
            if generation_batch.next_batch is not None:
                self._submit_async_rollout_prefetch(generation_batch.next_batch)
            return self._finalize_rollout_result(result)
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

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        advantages_code = advantages_code.unsqueeze(1)
        advantages_reason = advantages_reason.unsqueeze(1)

        per_token_code_loss = -torch.min(coef_1 * advantages_code, coef_2 * advantages_code)
        per_token_reason_loss = -torch.min(coef_1 * advantages_reason, coef_2 * advantages_reason)

        # [降低 length bias] code_grpo_loss_type 选择 loss 聚合方式
        # seq_mean: 原始实现，每个序列按长度归一化再取 batch mean → 短序列与长序列等权 → length bias
        # token_mean: 所有 token loss 之和 / 总 active token 数 → 每个 token 等权 → 无 length bias（DAPO/Dr.GRPO）
        loss_type = getattr(self.args, "code_grpo_loss_type", "seq_mean")
        if loss_type == "token_mean":
            code_active_tokens = code_mask.sum().clamp(min=1.0)
            reason_active_tokens = reason_mask.sum().clamp(min=1.0)
            loss_code = (per_token_code_loss * code_mask).sum() / code_active_tokens
            loss_reason = (per_token_reason_loss * reason_mask).sum() / reason_active_tokens
        else:
            loss_code_per_seq = (per_token_code_loss * code_mask).sum(dim=1) / code_mask.sum(dim=1).clamp(min=1.0)
            loss_reason_per_seq = (per_token_reason_loss * reason_mask).sum(dim=1) / reason_mask.sum(dim=1).clamp(min=1.0)
            loss_code = loss_code_per_seq.mean()
            loss_reason = loss_reason_per_seq.mean()
        loss = loss_code + self.args.beta_reason * loss_reason
        if self.beta != 0.0:
            ref_per_token_logps = self._get_reference_per_token_logps(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
            )
            if ref_per_token_logps is not None:
                log_ratio = ref_per_token_logps - per_token_logps
                per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
                seq_kl = (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
                loss = loss + self.beta * seq_kl.mean()

        mode = "train" if self.model.training else "eval"
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        loss = loss / normalizer

        gathered_loss_code = self.accelerator.gather(loss_code.detach()).float().mean().item()
        gathered_loss_reason = self.accelerator.gather(loss_reason.detach()).float().mean().item()
        self._metrics[mode]["loss_code"].append(gathered_loss_code)
        self._metrics[mode]["loss_reason"].append(gathered_loss_reason)

        # [诊断面板] importance ratio 方差监控
        with torch.no_grad():
            flat_ratio = coef_1[completion_mask.bool()]
            if flat_ratio.numel() > 0:
                self._metrics[mode]["ratio/mean"].append(flat_ratio.mean().item())
                self._metrics[mode]["ratio/std"].append(flat_ratio.std().item())
                self._metrics[mode]["ratio/max"].append(flat_ratio.max().item())
                # [诊断面板] clip 比率：正 advantage 被 high clip，负 advantage 被 low clip
                is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages_code < 0)
                is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages_code > 0)
                clip_mask = completion_mask.bool()
                if clip_mask.any():
                    self._metrics[mode]["clip_low_rate"].append(
                        is_low_clipped[clip_mask].float().mean().item()
                    )
                    self._metrics[mode]["clip_high_rate"].append(
                        is_high_clipped[clip_mask].float().mean().item()
                    )
            # [诊断面板] 完成长度分布：code vs reason
            code_lens = code_mask.sum(dim=1)
            reason_lens = reason_mask.sum(dim=1)
            code_active = code_lens[code_lens > 0]
            reason_active = reason_lens[reason_lens > 0]
            if code_active.numel() > 0:
                self._metrics[mode]["completion_len/code_mean"].append(code_active.float().mean().item())
            if reason_active.numel() > 0:
                self._metrics[mode]["completion_len/reason_mean"].append(reason_active.float().mean().item())
            # [诊断面板] advantage 分布
            adv_code_flat = advantages_code.squeeze()
            if adv_code_flat.numel() > 0:
                self._metrics[mode]["advantage/code_mean"].append(adv_code_flat.mean().item())
                self._metrics[mode]["advantage/code_std"].append(adv_code_flat.std().item())
                self._metrics[mode]["advantage/code_nonzero_rate"].append(
                    (adv_code_flat.abs() > 1e-8).float().mean().item()
                )
            adv_reason_flat = advantages_reason.squeeze()
            if adv_reason_flat.numel() > 0:
                self._metrics[mode]["advantage/reason_mean"].append(adv_reason_flat.mean().item())
                self._metrics[mode]["advantage/reason_std"].append(adv_reason_flat.std().item())
            # [诊断面板] effective tokens per update
            self._metrics[mode]["tokens_per_update"].append(completion_mask.sum().item())
        kl_value = self._compute_reference_kl_metric(
            per_token_logps=per_token_logps,
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            logits_to_keep=logits_to_keep,
        )
        if kl_value is not None:
            self._metrics[mode]["kl"].append(kl_value)
        if bool(getattr(self.args, "log_reward_losses", False)):
            logger.info(
                "[REWARD] loss_code=%.6f loss_reason=%.6f beta_reason=%.4f",
                gathered_loss_code,
                gathered_loss_reason,
                self.args.beta_reason,
            )
        return loss

    def evaluate(self, *args, **kwargs):
        if self._async_rollout_prefetch_enabled:
            self._shutdown_async_rollout_prefetch(wait=True)
        metrics = super().evaluate(*args, **kwargs)
        if self._last_eval_metrics:
            normalized_metrics = {}
            for key, value in self._last_eval_metrics.items():
                normalized_key = key if key.startswith("eval_") else f"eval_{key}"
                normalized_metrics[normalized_key] = value
            metrics.update(normalized_metrics)
        return metrics

    def _get_tb_writer(self):
        """懒获取 TensorBoard SummaryWriter，缓存结果。"""
        if hasattr(self, "_tb_writer_cache"):
            return self._tb_writer_cache
        writer = None
        for cb in self.callback_handler.callbacks:
            if hasattr(cb, "tb_writer") and cb.tb_writer is not None:
                writer = cb.tb_writer
                break
        self._tb_writer_cache = writer
        return writer

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        if mode == "train":
            metrics.update(self._update_train_reward_window_metrics(metrics))
        else:
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
            self._last_eval_metrics = dict(metrics)

        all_logs = {**logs, **metrics}
        self._metrics[mode].clear()

        # --- 分层输出：TensorBoard 全量诊断，控制台只打核心进度 ---
        # 1) 手动写 TensorBoard 全量指标（绕过 callback 链路）
        tb_logs = self._filter_tb_logs(all_logs, mode=mode)
        tb_writer = self._get_tb_writer()
        if tb_writer is not None:
            step = self.state.global_step
            for key, value in tb_logs.items():
                if isinstance(value, (int, float)):
                    tb_writer.add_scalar(key, value, step)
            tb_writer.flush()

        # 2) Console: 精简版走 super().log()（ProgressCallback 打印这个）
        #    注意：super().log() 内部也会触发 TensorBoardCallback.on_log()，
        #    但只传精简版 key，不会覆盖已写入的全量指标（TB 是 append 不是 replace）
        console_logs = self._filter_console_logs(all_logs, mode=mode)
        super().log(console_logs, start_time)

    def log_metrics(self, split: str, metrics: dict[str, float]) -> None:
        public_metrics = self.public_metrics(metrics, split=split)
        if not public_metrics:
            return
        super().log_metrics(split, public_metrics)
