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
import json
import os
import warnings

import torch
from accelerate import logging
from peft import PeftConfig

warnings.filterwarnings("ignore", message=r"TRL currently supports vLLM versions.*")

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.extensions.code_grpo.adapters import load_dataset_adapter
from trl.trainer.code_grpo_config import CodeGRPOConfig
from trl.trainer.code_grpo_trainer import CodeGRPOTrainer

from trl.scripts.code_grpo import (
    CodeGRPOScriptArguments,
    TextMetricsCallback,
    _adapt_splits,
    _attach_runtime_file_logger,
    _build_review_bundle,
    _configure_run_layout,
    _get_rank,
    _json_safe,
    _load_dataset,
    _write_run_index,
    _write_run_manifest,
)


logger = logging.get_logger(__name__)


def main(script_args, training_args, model_args, dataset_args):
    training_args.codegrpo_mode = "test"
    training_args.run_base_model_baseline_eval = False

    effective_model_name_or_path = model_args.model_name_or_path
    eval_peft_config = get_peft_config(model_args)
    adapter_path = model_args.model_name_or_path if os.path.exists(os.path.join(model_args.model_name_or_path, "adapter_config.json")) else None
    if adapter_path is not None:
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        if training_args.use_vllm and training_args.backend == "vllm":
            if getattr(training_args, "vllm_mode", "server") == "colocate":
                # Keep the HF-side trainer on a normal base+LoRA construction so transformers.Trainer
                # does not reject the model as a purely quantized read-only checkpoint. The actual
                # adapter under evaluation is applied only through vLLM dynamic LoRA requests below.
                effective_model_name_or_path = adapter_config.base_model_name_or_path
                training_args.vllm_dynamic_lora_path = adapter_path
                training_args.vllm_dynamic_lora_name = os.path.basename(os.path.abspath(adapter_path)) or "adapter"
                training_args.vllm_dynamic_lora_int_id = 1
                logger.info(
                    "[EVAL] using vLLM dynamic LoRA: hf_model=%s base_model=%s adapter=%s",
                    effective_model_name_or_path,
                    adapter_config.base_model_name_or_path,
                    adapter_path,
                )
            else:
                # Server-mode vLLM does not support dynamic LoRA requests in the current implementation.
                # Fall back to normal HF-side adapter loading for correctness.
                effective_model_name_or_path = adapter_path
                eval_peft_config = None
                training_args.use_vllm = False
                training_args.backend = "hf"
                logger.warning(
                    "[EVAL] adapter checkpoint with vLLM server mode is not supported via dynamic LoRA; "
                    "falling back to HF backend for standalone eval: adapter=%s base_model=%s",
                    adapter_path,
                    adapter_config.base_model_name_or_path,
                )
        else:
            effective_model_name_or_path = adapter_path
            eval_peft_config = None
            logger.info(
                "[EVAL] loading adapter checkpoint directly on HF backend: adapter=%s base_model=%s",
                adapter_path,
                adapter_config.base_model_name_or_path,
            )

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
    )
    if eval_dataset is None:
        raise ValueError("Standalone code_grpo_eval requires an eval/test split.")

    trainer = CodeGRPOTrainer(
        model=effective_model_name_or_path,
        args=training_args,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        peft_config=eval_peft_config,
        callbacks=[
            TextMetricsCallback(
                trainer_text_log_path if getattr(training_args, "write_trainer_text_log", False) else None,
                trainer_jsonl_path,
            )
        ],
    )

    metrics = trainer.evaluate(eval_dataset=eval_dataset)
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
    trainer.accelerator.print("Standalone CodeGRPO eval completed.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (CodeGRPOScriptArguments, CodeGRPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "code_grpo_eval", help="Run standalone CodeGRPO code-only evaluation", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
