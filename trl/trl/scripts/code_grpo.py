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
import os
from dataclasses import dataclass, field

import torch
from accelerate import logging
from datasets import load_dataset

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
from trl.extensions.code_grpo.adapters import load_dataset_adapter
from trl.trainer.code_grpo_config import CodeGRPOConfig
from trl.trainer.code_grpo_trainer import CodeGRPOTrainer


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


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
    )

    if training_args.codegrpo_mode == "test":
        test_dataset = eval_dataset if eval_dataset is not None else train_dataset
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.accelerator.print(metrics)
        trainer.accelerator.print("CodeGRPO test mode completed.")
        return

    trainer.train()
    trainer.accelerator.print("CodeGRPO training completed.")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"Model saved to {training_args.output_dir}.")

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

