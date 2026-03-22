import copy
import difflib
import json
import math
import os
from pathlib import Path
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
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
from .utils import RepeatSampler, get_config_model_id, pad, use_adapter


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logger = get_logger(__name__)


_RUNTIME_STATE_FILE = "codegrpo_runtime_state.json"


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


@dataclass
class _PseudoQuestionState:
    no_pass_streak: int = 0
    total_original_attempts: int = 0
    total_successes: int = 0
    last_original_rollout_step: int = -1


@dataclass
class _QuestionPriorState:
    ema_code_success: float = 0.0
    ema_reason_signal: float = 0.0
    ema_learning_value: float = 1.0
    seen_count: int = 0


@dataclass
class _PseudoIterativeNode:
    question_id: str
    base_question_id: str
    prompt: str
    source_prompt: str
    test_cases: list[dict[str, Any]]
    selection_tag: str
    priority: float
    raw_soft_reward: float
    final_reward: float
    pass_rate: float
    created_step: int


class _WeightedRepeatSampler(RepeatSampler):
    """RepeatSampler with per-index weights for signal-weighted question sampling.

    Question sampling weight: controls which questions are drawn for rollout.
    Does NOT affect reward, advantage, or loss computation — only determines
    which questions appear more often in subsequent training steps.

    Replaces uniform randperm with torch.multinomial weighted sampling.
    All other RepeatSampler behavior (mini_repeat_count, batch_size, repeat_count) is preserved.

    When refresh_steps > 0, weights are re-read from self._weights between
    segments of training steps (pseudo-epoch refresh). The trainer updates
    self._weights externally based on accumulated signal states.
    """

    def __init__(self, data_source, weights: list[float], refresh_steps: int = 0, **kwargs):
        super().__init__(data_source=data_source, shuffle=True, **kwargs)
        self._weights = torch.tensor(weights, dtype=torch.float64)
        self._refresh_steps = refresh_steps  # 0 = no refresh, >0 = refresh every N steps
        self.refresh_count = 0  # actual segment-boundary refresh count (for diagnostics)

    def _sample_one_pass(self):
        """Sample all indices once using current weights, return list of chunks."""
        indexes = torch.multinomial(
            self._weights, self.num_samples, replacement=False, generator=self.generator
        ).tolist()
        chunks = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        return [c for c in chunks if len(c) == self.batch_size]

    def __iter__(self):
        if self._refresh_steps <= 0:
            # No refresh: sample all indices once (original behavior)
            for chunk in self._sample_one_pass():
                for _ in range(self.repeat_count):
                    for index in chunk:
                        for _ in range(self.mini_repeat_count):
                            yield index
            return

        # Pseudo-epoch refresh: yield chunks in segments, re-sample between segments.
        # Each chunk produces (repeat_count * batch_size * mini_repeat_count) items.
        # The DL consumes (per_device_train_batch_size * steps_per_generation) items per batch,
        # but we approximate: each chunk ≈ repeat_count training steps consumed by the DL.
        # So chunks_per_segment ≈ ceil(refresh_steps / repeat_count).
        steps_per_chunk = max(self.repeat_count, 1)
        chunks_per_segment = max(1, (self._refresh_steps + steps_per_chunk - 1) // steps_per_chunk)
        total_yielded = 0
        total_needed = len(self)  # __len__ from RepeatSampler
        is_first_segment = True
        while total_yielded < total_needed:
            all_chunks = self._sample_one_pass()
            if not is_first_segment:
                self.refresh_count += 1  # only count actual re-reads, not the initial sample
            is_first_segment = False
            for chunk in all_chunks[:chunks_per_segment]:
                for _ in range(self.repeat_count):
                    for index in chunk:
                        for _ in range(self.mini_repeat_count):
                            yield index
                            total_yielded += 1
                            if total_yielded >= total_needed:
                                return


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
    """Single-round code GRPO trainer with optional pseudo-multiround and auxiliary SFT."""

    _tag_names = ["trl", "code_grpo"]
    _name = "CodeGRPO"
    # --- 控制台只打核心进度指标（保持简洁） ---
    _CONSOLE_TRAIN_KEYS = frozenset(
        {
            "loss",
            "loss_code",
            "loss_sft",
            "learning_rate",
            "grad_norm",
            "mean_R_code",
            "mean_pass_rate",
            "advantage/code_zero_rate",
            "zero_pass_soft_trigger_rate",
            "soft_lift",
            "pseudo/original_coverage_remaining",
        }
    )
    _TRAIN_LOG_KEYS = frozenset(
        {
            "loss",
            "loss_code",
            "loss_sft",
            "learning_rate",
            "grad_norm",
            "kl",
            "mean_R_code",
            "mean_pass_rate",
            "rollout_time_s",
            "advantage/code_zero_rate",
            "reward/R_code_min",
            "reward/R_code_max",
            "reward/pass_rate_min",
            "reward/pass_rate_max",
            "std_R_code",
            "advantage/code_nonzero_rate",
            "advantage/code_std",
            "advantage/code_mean",
            "ratio/mean",
            "ratio/max",
            "ratio/std",
            "clip_low_rate",
            "clip_high_rate",
            "completion_len/code_mean",
            "completion_len/sft_mean",
            "tokens_per_update",
            "effective_prompts_per_update",
            "effective_rollouts_per_update",
            "sampling_refresh/event_count",
            "audit_count/code_io_aux_per_main_sample",
            "pseudo/source_original_count",
            "pseudo/source_iterative_count",
            "pseudo/iterative_pool_size",
            "pseudo/iterative_nodes_added",
            "pseudo/iterative_nodes_pruned",
            "pseudo/original_no_pass_questions",
            "pseudo/max_original_no_pass_streak",
            "pseudo/source_forced_original_count",
            "pseudo/source_warmstart_original_count",
            "pseudo/original_coverage_remaining",
            "pseudo/original_coverage_completed",
            "question_prior/ema_code_success_mean",
            "question_prior/ema_reason_signal_mean",
            "question_prior/weight_mean",
            "question_prior/high_value_question_count",
            "question_prior/low_value_question_count",
            "question_prior/updated_question_count",
            "zero_pass_soft_trigger_rate",
            "soft_lift",
            "mean_hard_reward",
            "mean_raw_soft_reward",
            "mean_normalized_soft_reward",
            "mean_soft_reward_beta",
            "sibling_group_zero_std_R_code_rate",
            "pair_same_R_code_rate",
            "window/mean_R_code",
            "window/mean_R_soft_effective",
            "window/steps",
            "window/end_step",
        }
    )
    _CONSOLE_EVAL_KEYS = frozenset(
        {
            "eval_loss",
            "eval_pass_at_1",
            "eval_best_pass_rate_overall",
            "eval_mean_R_code",
            "eval_mean_pass_rate",
        }
    )
    _EVAL_LOG_KEYS = frozenset(
        {
            "eval_loss",
            "eval_kl",
            "eval_pass_at_1",
            "eval_best_pass_rate_overall",
            "eval_mean_R_code",
            "eval_mean_pass_rate",
            "eval_std_R_code",
            "eval_generation_format_ok_rate",
            "eval_compile_ok_rate",
            "eval_syntax_error_rate",
            "eval_timeout_rate",
            "eval_zero_pass_soft_trigger_rate",
            "eval_mean_hard_reward",
            "eval_mean_raw_soft_reward",
            "eval_mean_normalized_soft_reward",
            "eval_mean_soft_reward_beta",
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
        self._pseudo_multiround_enabled = bool(getattr(self.args, "pseudo_multiround_enabled", False))
        self._pseudo_question_states: dict[str, _PseudoQuestionState] = {}
        self._pseudo_original_cover_once_enabled = bool(
            self._pseudo_multiround_enabled and getattr(self.args, "pseudo_original_cover_once_before_iterative", False)
        )
        self._pseudo_original_pending_qids: deque[str] = deque()
        self._pseudo_original_example_by_qid: dict[str, dict[str, Any]] = {}
        self._question_prior_enabled = bool(getattr(self.args, "question_prior_enabled", False))
        self._question_prior_states: dict[str, _QuestionPriorState] = {}
        if (
            self._question_prior_enabled
            and not self._pseudo_multiround_enabled
            and int(getattr(self.args, "sampling_refresh_steps", 0) or 0) <= 0
        ):
            logger.warning(
                "question_prior_enabled is set, but pseudo_multiround_enabled is false and "
                "sampling_refresh_steps <= 0. Question prior stats will be tracked, but they will not "
                "meaningfully affect future sampling."
            )
        self._pseudo_iterative_pool: list[_PseudoIterativeNode] = []
        self._pseudo_node_serial = 0
        self._async_rollout_executor: ThreadPoolExecutor | None = None
        self._prefetched_rollout_future: Future | None = None
        self._prefetched_rollout_batch_id: int | None = None
        self._async_rollout_prefetch_enabled = self._resolve_async_rollout_prefetch_enabled()
        if self._pseudo_multiround_enabled and self._async_rollout_prefetch_enabled:
            logger.warning("Disabling async_rollout_prefetch because pseudo_multiround_enabled mutates in-memory pools.")
            self._async_rollout_prefetch_enabled = False
        logs_dir = os.path.join(os.path.dirname(self.args.output_dir), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self._rollout_summary_path = os.path.join(logs_dir, f"rollout_summary_rank{self.accelerator.process_index}.jsonl")
        self._step_samples_path = os.path.join(logs_dir, f"step_samples_rank{self.accelerator.process_index}.jsonl")
        self._initialize_original_coverage_state()
        self._runtime_state_loaded = False

    @staticmethod
    def _resolve_resume_checkpoint_path(output_dir: str, resume_from_checkpoint) -> str | None:
        if not resume_from_checkpoint:
            return None
        if isinstance(resume_from_checkpoint, str):
            return resume_from_checkpoint
        if resume_from_checkpoint is True:
            base = Path(output_dir)
            candidates = [p for p in base.glob("checkpoint-*") if p.is_dir()]
            if not candidates:
                return None
            def _step(path: Path) -> int:
                try:
                    return int(path.name.split("-")[-1])
                except Exception:
                    return -1
            candidates.sort(key=_step)
            return str(candidates[-1])
        return None

    def _runtime_state_payload(self) -> dict[str, Any]:
        return {
            "pseudo_node_serial": int(self._pseudo_node_serial),
            "pseudo_original_pending_qids": list(self._pseudo_original_pending_qids),
            "pseudo_iterative_pool": [asdict(record) for record in self._pseudo_iterative_pool],
            "pseudo_question_states": {qid: asdict(state) for qid, state in self._pseudo_question_states.items()},
            "question_prior_states": {qid: asdict(state) for qid, state in self._question_prior_states.items()},
        }

    def _save_runtime_state(self, checkpoint_dir: str) -> None:
        if not checkpoint_dir:
            return
        os.makedirs(checkpoint_dir, exist_ok=True)
        payload = self._runtime_state_payload()
        path = os.path.join(checkpoint_dir, _RUNTIME_STATE_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def _load_runtime_state(self, checkpoint_dir: str) -> None:
        if not checkpoint_dir:
            return
        path = os.path.join(checkpoint_dir, _RUNTIME_STATE_FILE)
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self._pseudo_node_serial = int(payload.get("pseudo_node_serial", 0) or 0)
        pending = list(payload.get("pseudo_original_pending_qids", []) or [])
        if self._pseudo_original_cover_once_enabled:
            self._pseudo_original_pending_qids = deque(
                [qid for qid in pending if qid in self._pseudo_original_example_by_qid]
            )
        else:
            self._pseudo_original_pending_qids = deque()

        self._pseudo_iterative_pool = [
            _PseudoIterativeNode(**item) for item in list(payload.get("pseudo_iterative_pool", []) or [])
        ]
        self._pseudo_question_states = {
            str(qid): _PseudoQuestionState(**state)
            for qid, state in dict(payload.get("pseudo_question_states", {}) or {}).items()
        }
        self._question_prior_states = {
            str(qid): _QuestionPriorState(**state)
            for qid, state in dict(payload.get("question_prior_states", {}) or {}).items()
        }
        self._runtime_state_loaded = True

    def _resolve_reward_window_interval_steps(self) -> int:
        bins = int(getattr(self.args, "reward_window_bins", 0) or 0)
        max_steps = int(getattr(self.args, "max_steps", 0) or 0)
        if bins > 0 and max_steps > 0:
            return max(1, math.ceil(max_steps / bins))
        eval_steps = getattr(self.args, "eval_steps", 0)
        if isinstance(eval_steps, int) and eval_steps > 0:
            return eval_steps
        return 0

    def _initialize_original_coverage_state(self) -> None:
        if not self._pseudo_original_cover_once_enabled:
            return
        dataset = self.train_dataset
        if dataset is None:
            self._pseudo_original_cover_once_enabled = False
            return
        try:
            dataset_len = len(dataset)
        except Exception:
            logger.warning("Disabling pseudo_original_cover_once_before_iterative because the training dataset is not indexable.")
            self._pseudo_original_cover_once_enabled = False
            return
        for idx in range(dataset_len):
            example = dataset[idx]
            qid = str(example.get("question_id", f"q_{idx}"))
            if qid in self._pseudo_original_example_by_qid:
                continue
            copied = copy.deepcopy(dict(example))
            copied.setdefault("base_question_id", qid)
            copied["source_kind"] = "original_problem"
            self._pseudo_original_example_by_qid[qid] = copied
            self._pseudo_original_pending_qids.append(qid)

    def _prepare_original_coverage_examples(
        self,
        fallback_examples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, float]] | None:
        if not self._pseudo_original_cover_once_enabled:
            return None
        if not self._pseudo_original_pending_qids:
            return None

        prepared: list[dict[str, Any]] = []
        needed = len(fallback_examples)
        while needed > 0 and self._pseudo_original_pending_qids:
            qid = self._pseudo_original_pending_qids.popleft()
            example = copy.deepcopy(self._pseudo_original_example_by_qid[qid])
            example["source_kind"] = "original_problem"
            example.setdefault("base_question_id", qid)
            prepared.append(example)
            needed -= 1

        if needed > 0:
            for example in fallback_examples[:needed]:
                copied = copy.deepcopy(example)
                copied["source_kind"] = "original_problem"
                copied.setdefault("base_question_id", self._base_question_id(example))
                prepared.append(copied)

        remaining = len(self._pseudo_original_pending_qids)
        metrics = {
            "pseudo/source_original_count": float(len(prepared)),
            "pseudo/source_iterative_count": 0.0,
            "pseudo/source_forced_original_count": 0.0,
            "pseudo/source_warmstart_original_count": float(len(prepared)),
            "pseudo/original_coverage_remaining": float(remaining),
            "pseudo/original_coverage_completed": 1.0 if remaining == 0 else 0.0,
            "pseudo/iterative_nodes_pruned": 0.0,
        }
        return prepared, metrics

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
            self._signature_columns = ["question_id", "prompt", "test_cases", "io_mode"]

    def _get_train_sampler(self, dataset=None):
        """CodeGRPO sampler: repeat each prompt only within the current rollout group.

        Unlike base GRPO, CodeGRPO does not buffer one generated batch across
        multiple optimizer steps. Each incoming batch is rolled out directly in
        `_prepare_inputs`, so repeating the same prompt across
        `num_iterations * steps_per_generation` would incorrectly cause the same
        question to be re-rolled for many consecutive training steps.

        Therefore the sampler only repeats each sampled prompt inside its local
        sibling group (`mini_repeat_count=self.num_generations`) and uses
        `repeat_count=1`.
        """
        if dataset is None:
            dataset = self.train_dataset

        if not self._question_prior_enabled:
            return RepeatSampler(
                data_source=dataset,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=1,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )

        # Compute per-index weights from signal states
        weights = []
        for idx in range(len(dataset)):
            qid = str(dataset[idx].get("question_id", f"q_{idx}"))
            w = self._get_question_prior_weight(qid)
            weights.append(w)

        refresh_steps = int(getattr(self.args, "sampling_refresh_steps", 0))
        sampler = _WeightedRepeatSampler(
            data_source=dataset,
            weights=weights,
            refresh_steps=refresh_steps,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=1,
            seed=self.args.seed,
        )
        # Keep reference so we can update weights after signal classification
        self._active_weighted_sampler = sampler
        return sampler

    def get_train_dataloader(self):
        train_loader = super().get_train_dataloader()
        if not self._async_rollout_prefetch_enabled:
            return train_loader
        return _CodeGRPOLookaheadLoader(train_loader)

    def train(self, *args, **kwargs):
        resume_from_checkpoint = kwargs.get("resume_from_checkpoint")
        if resume_from_checkpoint is None and args:
            resume_from_checkpoint = args[0]
        checkpoint_dir = self._resolve_resume_checkpoint_path(self.args.output_dir, resume_from_checkpoint)
        if checkpoint_dir and not self._runtime_state_loaded:
            self._load_runtime_state(checkpoint_dir)
        try:
            return super().train(*args, **kwargs)
        finally:
            self._shutdown_async_rollout_prefetch(wait=False)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_runtime_state(checkpoint_dir)
        self.accelerator.wait_for_everyone()

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

    @staticmethod
    def _base_question_id(example: dict[str, Any]) -> str:
        return str(example.get("base_question_id") or example.get("question_id", "unknown"))

    def _attach_rollout_source_metadata(self, example: dict[str, Any], rollout):
        rollout.source_kind = str(example.get("source_kind", "original_problem"))
        rollout.base_question_id = self._base_question_id(example)
        return rollout

    @staticmethod
    def _code_novelty_scores(nodes: list[dict[str, Any]]) -> list[float]:
        codes = [str(node.get("code_text", "") or "") for node in nodes]
        if len(codes) <= 1:
            return [1.0 for _ in codes]
        scores: list[float] = []
        for idx, code in enumerate(codes):
            similarities: list[float] = []
            for other_idx, other_code in enumerate(codes):
                if idx == other_idx:
                    continue
                similarities.append(difflib.SequenceMatcher(None, code, other_code).ratio())
            novelty = 1.0 - (sum(similarities) / len(similarities) if similarities else 0.0)
            scores.append(float(novelty))
        return scores

    def _make_iterative_prompt(self, source_example: dict[str, Any], node_payload: dict[str, Any], selection_tag: str) -> str:
        base_prompt = str(source_example.get("source_prompt") or source_example.get("prompt", "")).strip()
        code_text = str(node_payload.get("code_text", "") or "").strip()
        pass_cnt = int(node_payload.get("pass_cnt", 0) or 0)
        test_count = int(node_payload.get("test_count", 0) or 0)
        hard_reward = float(node_payload.get("hard_reward", 0.0) or 0.0)
        raw_soft_reward = float(node_payload.get("raw_soft_reward", 0.0) or 0.0)
        normalized_soft_reward = float(node_payload.get("normalized_soft_reward", 0.0) or 0.0)
        final_reward = float(node_payload.get("final_reward", node_payload.get("R_code", 0.0)) or 0.0)
        error_summary = str(node_payload.get("error_summary", "") or "").strip()
        history = list(node_payload.get("history", []) or [])
        latest = history[-1] if history else {}
        failed_input = latest.get("failed_input")
        failed_actual = latest.get("failed_actual")
        selection_reason = {
            "best_pass": "This candidate was closest to solving the task in the previous rollout.",
            "best_soft": "This candidate most increased confidence in the correct answers under the soft evaluator.",
            "novel": "This candidate was the most novel among the zero-pass samples and is kept for exploration.",
        }.get(selection_tag, "This candidate was selected for iterative refinement.")

        lines = [
            "You are revising a previous Python solution attempt.",
            "Return exactly one fenced Python code block.",
            "Do not output reasoning or explanations.",
            "Original problem:",
            base_prompt,
            "Previous candidate code:",
            code_text,
            "Previous rollout summary:",
            f"- selection_tag={selection_tag}",
            f"- selection_reason={selection_reason}",
            f"- pass_cnt={pass_cnt}/{test_count}",
            f"- hard_reward={hard_reward:.6f}",
            f"- raw_soft_reward={raw_soft_reward:.6f}",
            f"- normalized_soft_reward={normalized_soft_reward:.6f}",
            f"- final_reward={final_reward:.6f}",
        ]
        if error_summary:
            lines.append(f"- error_summary={error_summary}")
        if failed_input is not None:
            lines.append(f"- failed_input={failed_input!r}")
        if failed_actual is not None:
            lines.append(f"- failed_actual_or_error={failed_actual!r}")
        lines.extend(
            [
                "Task:",
                "Revise the previous code using the summary above. Keep the same solve(x) interface.",
                "Answer with one fenced Python code block only.",
            ]
        )
        return "\n".join(lines)

    def _cleanup_iterative_pool(self, solved_base_qids: set[str] | None = None) -> int:
        if not self._pseudo_multiround_enabled or not self._pseudo_iterative_pool:
            return 0

        ttl_steps = int(getattr(self.args, "pseudo_iterative_ttl_steps", 0) or 0)
        current_step = int(self.state.global_step)
        solved_base_qids = solved_base_qids or set()
        kept: list[_PseudoIterativeNode] = []
        removed = 0
        for record in self._pseudo_iterative_pool:
            too_old = bool(ttl_steps > 0 and (current_step - int(record.created_step)) >= ttl_steps)
            solved_prune = bool(record.base_question_id in solved_base_qids)
            if too_old or solved_prune:
                removed += 1
                continue
            kept.append(record)
        self._pseudo_iterative_pool = kept
        return removed

    def _append_iterative_nodes_from_rollout(self, rollout, source_example: dict[str, Any]) -> int:
        if not self._pseudo_multiround_enabled:
            return 0
        rounds = list(getattr(rollout, "rounds", []) or [])
        if not rounds:
            return 0
        search_round = next((item for item in rounds if str(item.get("stage", "search")) == "search"), None)
        if search_round is None:
            return 0
        nodes = [node for node in list(search_round.get("nodes", []) or []) if bool(node.get("main_sample_active", False))]
        if not nodes:
            return 0
        if any(float(node.get("pass_rate", 0.0) or 0.0) >= 1.0 for node in nodes):
            return 0

        select_count = max(0, int(getattr(self.args, "pseudo_iterative_select_count", 3) or 0))
        if select_count == 0:
            return 0

        novelty_scores = self._code_novelty_scores(nodes)
        for node, novelty in zip(nodes, novelty_scores, strict=True):
            node["_novelty_score"] = novelty

        ranked_pass = sorted(
            range(len(nodes)),
            key=lambda i: (
                float(nodes[i].get("pass_rate", 0.0) or 0.0),
                float(nodes[i].get("final_reward", nodes[i].get("R_code", 0.0)) or 0.0),
            ),
            reverse=True,
        )
        ranked_soft = sorted(
            range(len(nodes)),
            key=lambda i: (
                float(nodes[i].get("raw_soft_reward", 0.0) or 0.0),
                float(nodes[i].get("normalized_soft_reward", 0.0) or 0.0),
                float(nodes[i].get("final_reward", nodes[i].get("R_code", 0.0)) or 0.0),
            ),
            reverse=True,
        )
        ranked_novel = sorted(
            range(len(nodes)),
            key=lambda i: (
                float(nodes[i].get("_novelty_score", 0.0) or 0.0),
                float(nodes[i].get("raw_soft_reward", 0.0) or 0.0),
            ),
            reverse=True,
        )

        picks: list[tuple[int, str]] = []
        used: set[int] = set()
        for ranked, tag in ((ranked_pass, "best_pass"), (ranked_soft, "best_soft"), (ranked_novel, "novel")):
            for idx in ranked:
                if idx in used:
                    continue
                picks.append((idx, tag))
                used.add(idx)
                break
        if len(picks) < select_count:
            ranked_remaining = sorted(
                range(len(nodes)),
                key=lambda i: float(nodes[i].get("final_reward", nodes[i].get("R_code", 0.0)) or 0.0),
                reverse=True,
            )
            for idx in ranked_remaining:
                if idx in used:
                    continue
                picks.append((idx, "fallback"))
                used.add(idx)
                if len(picks) >= select_count:
                    break

        added = 0
        base_qid = self._base_question_id(source_example)
        question_prior_weight = self._get_question_prior_weight(base_qid)
        capacity = max(0, int(getattr(self.args, "pseudo_iterative_pool_capacity", 0) or 0))
        for idx, tag in picks[:select_count]:
            node_payload = nodes[idx]
            self._pseudo_node_serial += 1
            iterative_qid = f"{base_qid}__iter_{self._pseudo_node_serial}"
            prompt = self._make_iterative_prompt(source_example, node_payload, tag)
            base_priority = max(
                1e-6,
                float(node_payload.get("final_reward", node_payload.get("R_code", 0.0)) or 0.0)
                + max(0.0, float(node_payload.get("raw_soft_reward", 0.0) or 0.0)),
            )
            record = _PseudoIterativeNode(
                question_id=iterative_qid,
                base_question_id=base_qid,
                prompt=prompt,
                source_prompt=str(source_example.get("source_prompt") or source_example.get("prompt", "")),
                test_cases=copy.deepcopy(list(source_example.get("test_cases", []) or [])),
                selection_tag=tag,
                priority=max(1e-6, base_priority * question_prior_weight),
                raw_soft_reward=float(node_payload.get("raw_soft_reward", 0.0) or 0.0),
                final_reward=float(node_payload.get("final_reward", node_payload.get("R_code", 0.0)) or 0.0),
                pass_rate=float(node_payload.get("pass_rate", 0.0) or 0.0),
                created_step=int(self.state.global_step),
            )
            self._pseudo_iterative_pool.append(record)
            added += 1

        if capacity > 0 and len(self._pseudo_iterative_pool) > capacity:
            self._pseudo_iterative_pool = self._pseudo_iterative_pool[-capacity:]
        return added

    def _sample_iterative_examples(self, count: int, rng: random.Random) -> list[dict[str, Any]]:
        if count <= 0 or not self._pseudo_iterative_pool:
            return []
        available = list(self._pseudo_iterative_pool)
        weights = [max(1e-6, record.priority) for record in available]
        chosen: list[dict[str, Any]] = []
        local_count = min(count, len(available))
        for _ in range(local_count):
            total = sum(weights)
            threshold = rng.random() * total
            acc = 0.0
            pick_idx = 0
            for idx, weight in enumerate(weights):
                acc += weight
                if acc >= threshold:
                    pick_idx = idx
                    break
            record = available.pop(pick_idx)
            weights.pop(pick_idx)
            chosen.append(
                {
                    "question_id": record.question_id,
                    "base_question_id": record.base_question_id,
                    "prompt": record.prompt,
                    "source_prompt": record.source_prompt,
                    "test_cases": copy.deepcopy(record.test_cases),
                    "source_kind": "iterative_node",
                    "source_selection_tag": record.selection_tag,
                }
            )
        return chosen

    def _original_keep_probability(self, base_question_id: str) -> float:
        prior_weight = self._get_question_prior_weight(base_question_id)
        state = self._pseudo_question_states.get(base_question_id)
        age_bonus_per_step = float(getattr(self.args, "pseudo_original_age_bonus_per_step", 0.0) or 0.0)
        age_bonus_max = float(getattr(self.args, "pseudo_original_age_bonus_max", 0.0) or 0.0)
        age_bonus = 0.0
        if state is not None and state.last_original_rollout_step >= 0 and age_bonus_per_step > 0.0:
            age_steps = max(0, int(self.state.global_step) - int(state.last_original_rollout_step))
            age_bonus = min(age_bonus_max, age_steps * age_bonus_per_step)
        if state is None:
            return max(
                float(getattr(self.args, "question_prior_keep_prob_floor", 0.05)),
                min(1.0, prior_weight + age_bonus),
            )
        threshold = int(getattr(self.args, "pseudo_original_downweight_after", 2) or 0)
        if state.no_pass_streak <= threshold:
            return max(
                float(getattr(self.args, "question_prior_keep_prob_floor", 0.05)),
                min(1.0, prior_weight + age_bonus),
            )
        decay_steps = state.no_pass_streak - threshold
        decay = float(getattr(self.args, "pseudo_original_keep_prob_decay", 0.5))
        floor = max(
            float(getattr(self.args, "pseudo_original_keep_prob_floor", 0.1)),
            float(getattr(self.args, "question_prior_keep_prob_floor", 0.05)),
        )
        return max(floor, min(1.0, (decay**decay_steps) * prior_weight + age_bonus))

    def _get_question_prior_state(self, base_question_id: str) -> _QuestionPriorState:
        state = self._question_prior_states.get(base_question_id)
        if state is None:
            state = _QuestionPriorState()
            self._question_prior_states[base_question_id] = state
        return state

    def _update_question_prior_state(self, base_question_id: str, rollout) -> _QuestionPriorState:
        state = self._get_question_prior_state(base_question_id)
        rollout_samples = list(getattr(rollout, "train_samples", []) or [])
        solved = bool(float(getattr(rollout, "mean_pass_rate", 0.0) or 0.0) >= 1.0)
        if not solved:
            solved = bool(any(float(getattr(sample, "pass_rate", 0.0) or 0.0) >= 1.0 for sample in rollout_samples))
        code_signal = 1.0 if solved else 0.0
        zero_pass_trigger_rate = float(getattr(rollout, "eval_metrics", {}).get("zero_pass_soft_trigger_rate", 0.0) or 0.0)
        mean_hard_reward = float(getattr(rollout, "eval_metrics", {}).get("mean_hard_reward", 0.0) or 0.0)
        mean_normalized_soft_reward = float(
            getattr(rollout, "eval_metrics", {}).get("mean_normalized_soft_reward", 0.0) or 0.0
        )
        if zero_pass_trigger_rate > 0.0:
            reason_signal = mean_normalized_soft_reward
        else:
            reason_signal = mean_hard_reward
        reason_signal = max(0.0, min(1.0, float(reason_signal)))

        if state.seen_count == 0:
            state.ema_code_success = code_signal
            state.ema_reason_signal = reason_signal
        else:
            momentum = float(getattr(self.args, "question_prior_ema_momentum", 0.9))
            state.ema_code_success = momentum * state.ema_code_success + (1.0 - momentum) * code_signal
            state.ema_reason_signal = momentum * state.ema_reason_signal + (1.0 - momentum) * reason_signal
        state.seen_count += 1
        state.ema_learning_value = self._compute_question_prior_weight_from_state(state)
        return state

    def _compute_question_prior_weight_from_state(self, state: _QuestionPriorState) -> float:
        if not self._question_prior_enabled:
            return 1.0
        code_value = float(state.ema_code_success)
        reason_value = float(state.ema_reason_signal)
        high = float(getattr(self.args, "question_prior_high_threshold", 0.7))
        low = float(getattr(self.args, "question_prior_low_threshold", 0.3))
        gap = float(getattr(self.args, "question_prior_gap_threshold", 0.2))
        if code_value >= high and reason_value >= high:
            return float(getattr(self.args, "question_prior_weight_mastered", 0.4))
        if code_value <= low and reason_value <= low:
            return float(getattr(self.args, "question_prior_weight_too_hard", 0.2))
        if (reason_value - code_value) >= gap:
            return float(getattr(self.args, "question_prior_weight_high_value", 1.0))
        if (code_value - reason_value) >= gap:
            return float(getattr(self.args, "question_prior_weight_mid_negative_gap", 0.7))
        return float(getattr(self.args, "question_prior_weight_default", 0.8))

    def _get_question_prior_weight(self, base_question_id: str) -> float:
        if not self._question_prior_enabled:
            return 1.0
        state = self._question_prior_states.get(base_question_id)
        if state is None or int(state.seen_count) <= 0:
            return 1.0
        return max(0.0, float(state.ema_learning_value))

    def _question_prior_metrics(self) -> dict[str, float]:
        if not self._question_prior_enabled or not self._question_prior_states:
            return {}
        states = [state for state in self._question_prior_states.values() if int(state.seen_count) > 0]
        if not states:
            return {}
        weights = [float(state.ema_learning_value) for state in states]
        code_means = [float(state.ema_code_success) for state in states]
        reason_means = [float(state.ema_reason_signal) for state in states]
        high = float(getattr(self.args, "question_prior_high_threshold", 0.7))
        low = float(getattr(self.args, "question_prior_low_threshold", 0.3))
        return {
            "question_prior/ema_code_success_mean": sum(code_means) / len(code_means),
            "question_prior/ema_reason_signal_mean": sum(reason_means) / len(reason_means),
            "question_prior/weight_mean": sum(weights) / len(weights),
            "question_prior/high_value_question_count": float(
                sum(
                    1
                    for state in states
                    if state.ema_reason_signal >= high and state.ema_code_success <= low
                )
            ),
            "question_prior/low_value_question_count": float(
                sum(
                    1
                    for state in states
                    if state.ema_reason_signal <= low and state.ema_code_success <= low
                )
            ),
            "question_prior/updated_question_count": float(len(states)),
        }

    def _prepare_examples_for_rollout(
        self,
        examples: list[dict[str, Any]],
        mode: str,
        rng: random.Random,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if mode != "train" or not self._pseudo_multiround_enabled:
            return examples, {}

        coverage_payload = self._prepare_original_coverage_examples(examples)
        if coverage_payload is not None:
            return coverage_payload

        pruned = self._cleanup_iterative_pool()

        if not self._pseudo_iterative_pool:
            return examples, {
                "pseudo/source_original_count": float(len(examples)),
                "pseudo/source_iterative_count": 0.0,
                "pseudo/source_forced_original_count": 0.0,
                "pseudo/source_warmstart_original_count": 0.0,
                "pseudo/original_coverage_remaining": float(len(self._pseudo_original_pending_qids)),
                "pseudo/original_coverage_completed": 1.0 if not self._pseudo_original_pending_qids else 0.0,
                "pseudo/iterative_nodes_pruned": float(pruned),
            }

        prepared: list[dict[str, Any]] = []
        original_count = 0
        iterative_count = 0
        forced_original_count = 0
        for slot_idx, example in enumerate(examples):
            force_original = False
            forced_original_fraction = float(getattr(self.args, "pseudo_forced_original_fraction", 0.0) or 0.0)
            if forced_original_fraction > 0.0 and rng.random() < forced_original_fraction:
                force_original = True
            prefer_iterative = ((self.state.global_step + slot_idx) % 2) == 1
            selected: dict[str, Any] | None = None
            if prefer_iterative and not force_original:
                sampled = self._sample_iterative_examples(1, rng)
                if sampled:
                    selected = sampled[0]
            if selected is None and not force_original:
                base_qid = self._base_question_id(example)
                keep_prob = self._original_keep_probability(base_qid)
                if keep_prob < 1.0 and rng.random() > keep_prob:
                    sampled = self._sample_iterative_examples(1, rng)
                    if sampled:
                        selected = sampled[0]
            if selected is None:
                selected = copy.deepcopy(example)
                selected["source_kind"] = "original_problem"
                selected.setdefault("base_question_id", self._base_question_id(example))
                original_count += 1
                if force_original:
                    forced_original_count += 1
            else:
                iterative_count += 1
            prepared.append(selected)

        metrics = {
            "pseudo/source_original_count": float(original_count),
            "pseudo/source_iterative_count": float(iterative_count),
            "pseudo/source_forced_original_count": float(forced_original_count),
            "pseudo/source_warmstart_original_count": 0.0,
            "pseudo/original_coverage_remaining": float(len(self._pseudo_original_pending_qids)),
            "pseudo/original_coverage_completed": 1.0 if not self._pseudo_original_pending_qids else 0.0,
            "pseudo/iterative_pool_size": float(len(self._pseudo_iterative_pool)),
            "pseudo/iterative_nodes_pruned": float(pruned),
        }
        return prepared, metrics

    def _update_pseudo_multiround_state(self, examples: list[dict[str, Any]], rollouts: list[Any]) -> dict[str, float]:
        if not self._pseudo_multiround_enabled:
            return self._question_prior_metrics()

        added_nodes = 0
        solved_base_qids: set[str] = set()
        # Train mode currently guarantees one rollout per prepared example.
        # If train-time repeated eval-style rollouts are introduced later, this zip(strict=True)
        # must be revisited together with pseudo-state updates.
        for example, rollout in zip(examples, rollouts, strict=True):
            base_qid = self._base_question_id(example)
            source_kind = str(example.get("source_kind", "original_problem"))
            round_nodes = list(getattr(rollout, "train_samples", []) or [])
            succeeded = bool(float(getattr(rollout, "mean_pass_rate", 0.0) or 0.0) >= 1.0)
            if not succeeded:
                succeeded = bool(any(float(getattr(sample, "pass_rate", 0.0) or 0.0) >= 1.0 for sample in round_nodes))

            state = self._pseudo_question_states.get(base_qid)
            if state is None:
                state = _PseudoQuestionState()
                self._pseudo_question_states[base_qid] = state

            if source_kind == "original_problem":
                state.total_original_attempts += 1
                state.last_original_rollout_step = int(self.state.global_step)
                if succeeded:
                    state.no_pass_streak = 0
                    state.total_successes += 1
                    solved_base_qids.add(base_qid)
                else:
                    state.no_pass_streak += 1
            elif succeeded:
                # A successful iterative refinement clears original no-pass pressure.
                state.no_pass_streak = 0
                state.total_successes += 1
                solved_base_qids.add(base_qid)

            added_nodes += self._append_iterative_nodes_from_rollout(rollout, example)

        pruned = self._cleanup_iterative_pool(solved_base_qids=solved_base_qids)
        max_streak = max((state.no_pass_streak for state in self._pseudo_question_states.values()), default=0)
        original_no_pass_questions = sum(
            1 for state in self._pseudo_question_states.values() if int(state.no_pass_streak) > 0
        )
        return {
            "pseudo/iterative_nodes_added": float(added_nodes),
            "pseudo/iterative_nodes_pruned": float(pruned),
            "pseudo/iterative_pool_size": float(len(self._pseudo_iterative_pool)),
            "pseudo/original_no_pass_questions": float(original_no_pass_questions),
            "pseudo/max_original_no_pass_streak": float(max_streak),
            **self._question_prior_metrics(),
        }

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
                "base_question_id": getattr(rollout, "base_question_id", rollout.question_id),
                "source_kind": getattr(rollout, "source_kind", "original_problem"),
                "node_count": rollout.node_count,
                "resample_count": rollout.resample_count,
                "mean_R_code": rollout.mean_R_code,
                "mean_pass_rate": rollout.mean_pass_rate,
                **rollout.eval_metrics,
            }
            lines.append(json.dumps(payload, ensure_ascii=False))
        if lines:
            with open(self._rollout_summary_path, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")

    def _append_step_samples(self, rollouts, mode: str):
        lines = []
        for rollout in rollouts:
            for round_item in getattr(rollout, "rounds", []) or []:
                for candidate_idx, node in enumerate(round_item.get("nodes", []) or []):
                    if not bool(node.get("main_sample_active", False)):
                        continue
                    code = str(node.get("code", "") or "")
                    payload = {
                        "mode": mode,
                        "global_step": int(self.state.global_step),
                        "question_id": rollout.question_id,
                        "base_question_id": getattr(rollout, "base_question_id", rollout.question_id),
                        "source_kind": getattr(rollout, "source_kind", "original_problem"),
                        "candidate_idx": int(candidate_idx),
                        "node_id": node.get("node_id"),
                        "round_idx": int(node.get("round_idx", 0) or 0),
                        "pass_rate": float(node.get("pass_rate", 0.0) or 0.0),
                        "hard_reward": float(node.get("hard_reward", 0.0) or 0.0),
                        "raw_soft_reward": float(node.get("raw_soft_reward", 0.0) or 0.0),
                        "normalized_soft_reward": float(node.get("normalized_soft_reward", 0.0) or 0.0),
                        "final_reward": float(node.get("final_reward", node.get("R_code", 0.0)) or 0.0),
                        "A_code": float(node.get("A_code", 0.0) or 0.0),
                        "compile_ok": float(node.get("compile_score", 0.0) or 0.0) > 0.0,
                        "generation_format_ok": bool(node.get("generation_format_ok", False)),
                        "status_code": str(node.get("status_code", "FAIL")),
                        "code_preview": code[:240],
                    }
                    lines.append(json.dumps(payload, ensure_ascii=False))
        if lines:
            with open(self._step_samples_path, "a", encoding="utf-8") as handle:
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
        """????????????????????????"""
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
        sft_masks = []
        old_logprobs = []
        advantages_code = []
        sft_weights = []
        r_code = []
        pass_rates = []
        has_old_logprobs = True

        for sample in train_samples:
            prompt_ids = self.code_tokenizer(sample.prompt_text, add_special_tokens=False)["input_ids"]
            completion_ids = self.code_tokenizer(sample.completion_text, add_special_tokens=False)["input_ids"]
            if not completion_ids:
                completion_ids = [self.eos_token_id]

            code_mask = list(sample.code_token_mask)
            sft_mask = list(sample.sft_token_mask)
            if len(code_mask) < len(completion_ids):
                code_mask.extend([0] * (len(completion_ids) - len(code_mask)))
            if len(sft_mask) < len(completion_ids):
                sft_mask.extend([0] * (len(completion_ids) - len(sft_mask)))
            code_mask = code_mask[: len(completion_ids)]
            sft_mask = sft_mask[: len(completion_ids)]

            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
            completion_tensor = torch.tensor(completion_ids, dtype=torch.long)
            prompt_ids_tensors.append(prompt_tensor)
            prompt_masks.append(torch.ones_like(prompt_tensor, dtype=torch.long))
            completion_ids_tensors.append(completion_tensor)
            completion_masks.append(torch.ones_like(completion_tensor, dtype=torch.long))
            code_masks.append(torch.tensor(code_mask, dtype=torch.float32))
            sft_masks.append(torch.tensor(sft_mask, dtype=torch.float32))
            sample_old_logprobs = getattr(sample, "old_per_token_logps", None)
            if sample_old_logprobs is None:
                has_old_logprobs = False
            else:
                values = list(sample_old_logprobs)[: len(completion_ids)]
                if len(values) < len(completion_ids):
                    values.extend([0.0] * (len(completion_ids) - len(values)))
                old_logprobs.append(torch.tensor(values, dtype=torch.float32))
            advantages_code.append(sample.A_code)
            sft_weights.append(sample.sft_weight)
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
        sft_token_mask = pad(
            sft_masks, padding_value=0.0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )

        batch = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "code_token_mask": code_token_mask,
            "sft_token_mask": sft_token_mask,
            "advantages_code": torch.tensor(advantages_code, dtype=torch.float32),
            "sft_weights": torch.tensor(sft_weights, dtype=torch.float32),
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
        examples, pseudo_prepare_metrics = self._prepare_examples_for_rollout(examples, mode, base_rng)

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
                    self._attach_rollout_source_metadata(example, rollout)
                    rollouts.append(rollout)
            else:
                rollout = self.tree_runner.run_question(example, rng=random.Random(rollout_seed))
                self._attach_rollout_source_metadata(example, rollout)
                rollouts.append(rollout)
        rollout_time_s = max(time.perf_counter() - rollout_t0, 1e-8)

        train_samples = [sample for rollout in rollouts for sample in rollout.train_samples]
        if not train_samples and examples:
            fallback_completion = build_generation_completion("")
            _, fallback_code_mask, _ = build_token_masks(self.code_tokenizer, fallback_completion)
            train_samples = [
                {
                    "question_id": str(examples[0]["question_id"]),
                    "prompt_text": str(examples[0]["prompt"]),
                    "completion_text": fallback_completion,
                    "code_token_mask": fallback_code_mask,
                    "A_code": 0.0,
                    "sft_token_mask": [0] * len(fallback_code_mask),
                    "sft_weight": 0.0,
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
        mean_pass_rate = self._mean([rollout.mean_pass_rate for rollout in rollouts])
        std_r_code = self._mean([rollout.std_R_code for rollout in rollouts])
        node_count = float(sum(rollout.node_count for rollout in rollouts))
        resample_count = float(sum(rollout.resample_count for rollout in rollouts))

        metric_updates: dict[str, float] = {
            "mean_R_code": mean_r_code,
            "mean_pass_rate": mean_pass_rate,
            "std_R_code": std_r_code,
            "node_count": node_count,
            "resample_count": resample_count,
            "rollout_time_s": float(rollout_time_s),
            "rollout_nodes_per_s": float(node_count / rollout_time_s),
            "rollout_questions_per_s": float(len(examples) / rollout_time_s),
        }
        metric_updates.update(pseudo_prepare_metrics)

        # [诊断面板] rollout 层面的 reward / advantage / batch 诊断
        all_r_codes = [s.R_code for s in train_samples if hasattr(s, "R_code")]
        all_pass_rates = [s.pass_rate for s in train_samples if hasattr(s, "pass_rate")]
        all_a_codes = [s.A_code for s in train_samples if hasattr(s, "A_code")]
        if all_r_codes:
            metric_updates["reward/R_code_min"] = min(all_r_codes)
            metric_updates["reward/R_code_max"] = max(all_r_codes)
            # advantage/code_zero_rate: A_code == 0 的样本比例（sibling reward 相同或 std<=eps 时为 0）
            if all_a_codes:
                zero_count = sum(1 for a in all_a_codes if abs(a) < 1e-8)
                metric_updates["advantage/code_zero_rate"] = zero_count / len(all_a_codes)
        if all_pass_rates:
            metric_updates["reward/pass_rate_min"] = min(all_pass_rates)
            metric_updates["reward/pass_rate_max"] = max(all_pass_rates)
        # [诊断面板] effective prompts per update
        metric_updates["effective_prompts_per_update"] = float(len(examples))
        metric_updates["effective_rollouts_per_update"] = float(len(train_samples))

        # [诊断面板] audit count per main sample — from rollout eval_metrics
        total_main = sum(r.eval_metrics.get("main_sample_count", 0.0) for r in rollouts)
        total_code_io_aux = sum(r.eval_metrics.get("code_io_aux_sample_count", 0.0) for r in rollouts)
        if total_main > 0:
            metric_updates["audit_count/code_io_aux_per_main_sample"] = total_code_io_aux / total_main

        if mode == "train":
            # Train mode currently guarantees one rollout per prepared example.
            # If train-time repeated eval-style rollouts are introduced later, this zip(strict=True)
            # must be revisited together with question-prior updates.
            for example, rollout in zip(examples, rollouts, strict=True):
                base_qid = self._base_question_id(example)
                self._update_question_prior_state(base_qid, rollout)

            # --- Update sampler weights for pseudo-epoch refresh ---
            # Only write weights when refresh is enabled (sampling_refresh_steps > 0).
            # When refresh_steps=0, sampler._weights stays frozen at build time — truly static.
            # Question sampling weight: only affects which questions are drawn for rollout.
            # Does NOT change reward, advantage, or loss computation.
            refresh_steps = int(getattr(self.args, "sampling_refresh_steps", 0))
            sampler = getattr(self, "_active_weighted_sampler", None)
            if (
                self._question_prior_enabled
                and refresh_steps > 0
                and sampler is not None
                and hasattr(sampler, "_weights")
                and self.train_dataset is not None
            ):
                new_weights = []
                for idx in range(len(self.train_dataset)):
                    qid = str(self.train_dataset[idx].get("question_id", f"q_{idx}"))
                    w = self._get_question_prior_weight(qid)
                    new_weights.append(w)
                sampler._weights = torch.tensor(new_weights, dtype=torch.float64)
                # Read actual refresh count from sampler (incremented at segment boundaries)
                metric_updates["sampling_refresh/event_count"] = float(getattr(sampler, "refresh_count", 0))

            metric_updates.update(self._question_prior_metrics())

        if mode == "train":
            metric_updates.update(self._update_pseudo_multiround_state(examples, rollouts))

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
                "mean_pass_rate": mean_pass_rate,
                "std_R_code": std_r_code,
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
        self._append_step_samples(result.rollouts, mode=mode)
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
        sft_mask = inputs["sft_token_mask"] * completion_mask
        advantages_code = inputs["advantages_code"]
        sft_weights = inputs["sft_weights"]

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        advantages_code = advantages_code.unsqueeze(1)
        sft_weights = sft_weights.unsqueeze(1)

        per_token_code_loss = -torch.min(coef_1 * advantages_code, coef_2 * advantages_code)
        per_token_sft_loss = -per_token_logps

        # [降低 length bias] code_grpo_loss_type 选择 loss 聚合方式
        # seq_mean: 原始实现，每个序列按长度归一化再取 batch mean → 短序列与长序列等权 → length bias
        # token_mean: 所有 token loss 之和 / 总 active token 数 → 每个 token 等权 → 无 length bias（DAPO/Dr.GRPO）
        loss_type = getattr(self.args, "code_grpo_loss_type", "seq_mean")
        if loss_type == "token_mean":
            code_active_tokens = code_mask.sum().clamp(min=1.0)
            sft_active_weight = (sft_mask * sft_weights).sum().clamp(min=1.0)
            loss_code = (per_token_code_loss * code_mask).sum() / code_active_tokens
            loss_sft = (per_token_sft_loss * sft_mask * sft_weights).sum() / sft_active_weight
        else:
            loss_code_per_seq = (per_token_code_loss * code_mask).sum(dim=1) / code_mask.sum(dim=1).clamp(min=1.0)
            sft_denom = sft_mask.sum(dim=1).clamp(min=1.0)
            loss_sft_per_seq = ((per_token_sft_loss * sft_mask).sum(dim=1) / sft_denom) * sft_weights.squeeze(1)
            loss_code = loss_code_per_seq.mean()
            loss_sft = loss_sft_per_seq.mean()
        loss = loss_code + self.args.beta_sft * loss_sft
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
        gathered_loss_sft = self.accelerator.gather(loss_sft.detach()).float().mean().item()
        self._metrics[mode]["loss_code"].append(gathered_loss_code)
        self._metrics[mode]["loss_sft"].append(gathered_loss_sft)

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
            sft_lens = sft_mask.sum(dim=1)
            code_active = code_lens[code_lens > 0]
            sft_active = sft_lens[sft_lens > 0]
            if code_active.numel() > 0:
                self._metrics[mode]["completion_len/code_mean"].append(code_active.float().mean().item())
            if sft_active.numel() > 0:
                self._metrics[mode]["completion_len/sft_mean"].append(sft_active.float().mean().item())
            # [诊断面板] advantage 分布
            adv_code_flat = advantages_code.squeeze()
            if adv_code_flat.numel() > 0:
                self._metrics[mode]["advantage/code_mean"].append(adv_code_flat.mean().item())
                self._metrics[mode]["advantage/code_std"].append(adv_code_flat.std().item())
                self._metrics[mode]["advantage/code_nonzero_rate"].append(
                    (adv_code_flat.abs() > 1e-8).float().mean().item()
                )
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
