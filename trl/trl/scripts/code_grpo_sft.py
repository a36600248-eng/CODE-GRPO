# Copyright 2020-2026 The HuggingFace Team. All rights reserved.

import argparse
import os
from dataclasses import dataclass, field
from typing import Any

from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.extensions.code_grpo.parser import build_generation_completion


logger = logging.get_logger(__name__)
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

FORMAT_INSTRUCTION = (
    "You are training for code generation. "
    "Return exactly one fenced Python code block and no extra explanation."
)


@dataclass
class CodeGRPOSFTScriptArguments(ScriptArguments):
    add_format_instruction: bool = field(
        default=True,
        metadata={"help": "Prepend a fixed code-format instruction to every prompt."},
    )
    format_instruction: str = field(
        default=FORMAT_INSTRUCTION,
        metadata={"help": "Instruction text prepended when add_format_instruction=True."},
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


def _pick_first(example: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in example and example[key] not in (None, ""):
            return example[key]
    return None


def _normalize_example(example: dict[str, Any], idx: int, script_args: CodeGRPOSFTScriptArguments) -> dict[str, str]:
    prompt = _pick_first(example, ["prompt", "question", "instruction", "problem", "query"])
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"SFT example #{idx} is missing non-empty prompt/question text.")
    prompt = prompt.strip()
    if script_args.add_format_instruction:
        prompt = f"{script_args.format_instruction.strip()}\n\nQuestion:\n{prompt}"

    completion = _pick_first(example, ["completion", "answer"])
    if isinstance(completion, str) and completion.strip():
        return {"prompt": prompt, "completion": completion.strip()}

    code = _pick_first(example, ["code", "solution", "program", "answer_code"])
    if not isinstance(code, str) or not code.strip():
        raise ValueError(
            f"SFT example #{idx} is missing completion and code fields. "
            "Provide `completion` or at least `code`."
        )
    return {"prompt": prompt, "completion": build_generation_completion(code)}


def _prepare_split(dataset_split, script_args: CodeGRPOSFTScriptArguments):
    remove_columns = list(dataset_split.column_names) if hasattr(dataset_split, "column_names") else None
    return dataset_split.map(
        lambda example, idx: _normalize_example(example, idx, script_args),
        with_indices=True,
        remove_columns=remove_columns,
    )


def main(script_args, training_args, model_args, dataset_args):
    dataset = _load_dataset(script_args, dataset_args)

    model_kwargs: dict[str, Any] = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    train_dataset = _prepare_split(dataset[script_args.dataset_train_split], script_args)
    eval_dataset = None
    if training_args.eval_strategy != "no" and script_args.dataset_test_split in dataset:
        eval_dataset = _prepare_split(dataset[script_args.dataset_test_split], script_args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.accelerator.print("CodeGRPO-SFT training completed.")

    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (CodeGRPOSFTScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("code_grpo_sft", help="Run code-only SFT for CodeGRPO", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
