import json
import os
import random
import time
from typing import Any

import torch
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback

from ..extensions.code_grpo import (
    CodeGRPOTreeRunner,
    build_backend,
    build_canonical_completion,
    build_token_masks,
)
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

        generation_defaults = {
            "max_new_tokens": self.max_completion_length,
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
        self._trace_dump_counter = 0
        self._last_eval_metrics: dict[str, float] = {}
        logs_dir = os.path.join(os.path.dirname(self.args.output_dir), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self._rollout_summary_path = os.path.join(logs_dir, f"rollout_summary_rank{self.accelerator.process_index}.jsonl")

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["question_id", "prompt", "test_cases"]

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

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
                "eval_metrics": rollout.eval_metrics,
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

        seed_offset = self.state.global_step if mode == "train" else self.state.global_step + 1_000_000
        base_rng = random.Random(self.args.seed + seed_offset)

        rollouts = []
        for _idx, example in enumerate(examples):
            rollout_seed = base_rng.randint(0, 2**31 - 1)
            rollout = self.tree_runner.run_question(
                example,
                rng=random.Random(rollout_seed),
                update_exec_baseline=(mode == "train"),
            )
            rollouts.append(rollout)
        rollout_time_s = max(time.perf_counter() - rollout_t0, 1e-8)

        train_samples = [sample for rollout in rollouts for sample in rollout.train_samples]
        if not train_samples and examples:
            fallback_completion = build_canonical_completion(
                code="",
                reasoning="",
                logic_prediction="",
                exec_prediction="",
                include_predictions=False,
                include_reason=False,
            )
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
        for key in metric_keys:
            value = self._mean([rollout.eval_metrics.get(key, 0.0) for rollout in rollouts])
            self._metrics[mode][key].append(value)
            merged_eval_metrics[key] = value
        if "pass_at_k_round_n" in metric_keys:
            merged_pass_at_k_round_n = self._mean([rollout.eval_metrics.get("pass_at_k_round_n", 0.0) for rollout in rollouts])
            self._metrics[mode]["pass_at_k_round_n"].append(merged_pass_at_k_round_n)
            merged_eval_metrics["pass_at_k_round_n"] = merged_pass_at_k_round_n

        if self.args.codegrpo_mode == "test":
            selected_rollouts = self._select_rollouts_for_trace(rollouts, mode="test")
            self._dump_rollout_traces(selected_rollouts)
            logger.info("[TEST] dumped %d/%d rollout trace files", len(selected_rollouts), len(rollouts))
        elif mode == "eval":
            selected_rollouts = self._select_rollouts_for_trace(rollouts, mode="eval")
            self._dump_rollout_traces(selected_rollouts)
            logger.info("[EVAL] dumped %d/%d rollout trace files", len(selected_rollouts), len(rollouts))
        else:
            if self.args.dump_train_traces:
                selected_rollouts = self._select_rollouts_for_trace(rollouts, mode="train")
                self._dump_rollout_traces(selected_rollouts)
                logger.info("[TRAIN] dumped %d/%d rollout trace files", len(selected_rollouts), len(rollouts))
            logger.info("[TRAIN] built %d training samples from %d questions", len(train_samples), len(examples))

        if mode == "eval":
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
            metrics.update({f"eval_{key}": value for key, value in self._last_eval_metrics.items()})
        return metrics
