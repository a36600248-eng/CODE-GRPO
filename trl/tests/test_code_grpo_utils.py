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

from types import SimpleNamespace

from trl.extensions.code_grpo.error_utils import summarize_error
from trl.extensions.code_grpo.matcher import is_match, values_equal
from trl.extensions.code_grpo.parser import (
    build_canonical_completion,
    parse_exec_prediction_only,
    parse_exec_response,
    parse_generation_output,
    parse_logic_prediction_only,
    parse_logic_response,
    parse_prediction_only,
)
from trl.extensions.code_grpo.tree import _compute_code_reward, _is_double_zero_node
from trl.extensions.code_grpo.types import ExecResult, Node
from trl.trainer.code_grpo_config import CodeGRPOConfig
from trl.trainer.code_grpo_trainer import CodeGRPOTrainer
from trl.trainer import utils as trainer_utils


def test_parse_generation_output():
    text = (
        "<CODE>\nprint('x')\n</CODE>\n<REASON>\nthink\n<LOGIC_PREDICTION>\n'x'\n</LOGIC_PREDICTION>\n"
        "<EXEC_PREDICTION>\n'x'\n</EXEC_PREDICTION>\n</REASON>"
    )
    code, reason, logic_prediction, exec_prediction = parse_generation_output(text)
    assert code == "print('x')"
    assert reason == "think"
    assert logic_prediction == "'x'"
    assert exec_prediction == "'x'"


def test_parse_generation_output_prefers_last_tag_block():
    text = (
        "<CODE>\nprint('old')\n</CODE>\n"
        "<REASON>\nold\n<LOGIC_PREDICTION>\n1\n</LOGIC_PREDICTION>\n<EXEC_PREDICTION>\n1\n</EXEC_PREDICTION>\n</REASON>\n"
        "<CODE>\nprint('new')\n</CODE>\n"
        "<REASON>\nnew\n<LOGIC_PREDICTION>\n2\n</LOGIC_PREDICTION>\n<EXEC_PREDICTION>\n2\n</EXEC_PREDICTION>\n</REASON>"
    )
    code, reason, logic_prediction, exec_prediction = parse_generation_output(text)
    assert code == "print('new')"
    assert reason == "new"
    assert logic_prediction == "2"
    assert exec_prediction == "2"


def test_parse_prediction_only_fallback():
    raw = "ValueError"
    assert parse_prediction_only(raw) == "ValueError"


def test_parse_logic_and_exec_prediction_only():
    text = "<LOGIC_PREDICTION>2</LOGIC_PREDICTION><EXEC_PREDICTION>TypeError</EXEC_PREDICTION>"
    assert parse_logic_prediction_only(text) == "2"
    assert parse_exec_prediction_only(text) == "TypeError"


def test_parse_logic_response_format_ok():
    text = "<REASON>because x+1</REASON><LOGIC_PREDICTION>2</LOGIC_PREDICTION>"
    reason, pred, format_ok = parse_logic_response(text, require_reason_before_prediction=True)
    assert reason == "because x+1"
    assert pred == "2"
    assert format_ok


def test_parse_exec_response_format_bad_order():
    text = "<EXEC_PREDICTION>TypeError</EXEC_PREDICTION><REASON>arg mismatch</REASON>"
    reason, pred, format_ok = parse_exec_response(text, require_reason_before_prediction=True)
    assert reason == "arg mismatch"
    assert pred == "TypeError"
    assert not format_ok


def test_parse_logic_response_prefers_last_match():
    text = (
        "<REASON>first</REASON><LOGIC_PREDICTION>1</LOGIC_PREDICTION>"
        "<REASON>second</REASON><LOGIC_PREDICTION>2</LOGIC_PREDICTION>"
    )
    reason, pred, format_ok = parse_logic_response(text, require_reason_before_prediction=True)
    assert reason == "second"
    assert pred == "2"
    assert format_ok


def test_canonical_completion_contains_tags():
    completion = build_canonical_completion("a=1", "reason", "1", "1")
    assert "<CODE>" in completion
    assert "<REASON>" in completion
    assert "<LOGIC_PREDICTION>" in completion
    assert "<EXEC_PREDICTION>" in completion


def test_values_equal_structured():
    assert values_equal("[1, 2, 3]", (1, 2, 3))
    assert values_equal({"a": 1.0}, {"a": 1.0 + 1e-7})
    assert values_equal(" a   b ", "a b")


def test_is_match_error_type():
    actual = ExecResult(kind="RUNTIME_ERROR", error_type="ValueError", error_msg="x")
    assert is_match("ValueError: bad value", actual)
    assert not is_match("TypeError", actual)


def test_summarize_error_limits():
    text = "line1\nline2\nline3"
    summary = summarize_error(text, max_chars=8, max_lines=2)
    assert summary.count("\n") <= 1
    assert len(summary) <= 8


def test_code_reward_margin_scaling():
    # Soft term should not overshadow hard pass-rate term.
    assert _compute_code_reward(pass_rate=1.0, r_soft=1.0, lambda_soft=1.0) == 1.0
    assert _compute_code_reward(pass_rate=0.5, r_soft=1.0, lambda_soft=0.2) == 0.6
    assert _compute_code_reward(pass_rate=0.0, r_soft=1.0, lambda_soft=0.2) == 0.2


def test_lambda_soft_range_validation():
    CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, lambda_soft=0.0)
    CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, lambda_soft=1.0)
    try:
        CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, lambda_soft=1.1)
        assert False, "Expected ValueError for lambda_soft > 1"
    except ValueError:
        pass


def test_format_penalty_range_validation():
    CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, format_penalty_logic=0.0, format_penalty_exec=1.0)
    try:
        CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, format_penalty_logic=1.1)
        assert False, "Expected ValueError for format_penalty_logic > 1"
    except ValueError:
        pass


def test_double_zero_node_detection():
    node = Node(node_id="n1", parent_id="root", round_idx=1, code="x", reasoning="y", R_code=0.0, R_reason=0.0)
    assert _is_double_zero_node(node)
    node.R_code = 1e-6
    assert not _is_double_zero_node(node)


def test_code_grpo_trainer_syncs_vllm_weights_once_per_step():
    class DummyVLLMGeneration:
        def __init__(self):
            self.sync_calls = 0

        def sync_weights(self):
            self.sync_calls += 1

    trainer = object.__new__(CodeGRPOTrainer)
    trainer.use_vllm = True
    trainer.vllm_generation = DummyVLLMGeneration()
    trainer._last_loaded_step = -1
    trainer.state = SimpleNamespace(global_step=7)
    trainer.args = SimpleNamespace(report_to=[])
    trainer.accelerator = SimpleNamespace(is_main_process=True)

    CodeGRPOTrainer._maybe_sync_vllm_weights(trainer)
    assert trainer.vllm_generation.sync_calls == 1


def test_create_model_from_path_loads_adapter_checkpoint(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "checkpoint-1"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    calls = []

    class DummyArch:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("base", model_id, kwargs))
            return "base-model"

    class DummyPeftConfig:
        base_model_name_or_path = "base-model-id"

        @classmethod
        def from_pretrained(cls, model_id):
            assert model_id == str(adapter_dir)
            return cls()

    class DummyPeftModel:
        @classmethod
        def from_pretrained(cls, base_model, model_id, is_trainable=False):
            calls.append(("adapter", base_model, model_id, is_trainable))
            return "adapter-model"

    monkeypatch.setattr(trainer_utils, "PeftConfig", DummyPeftConfig)
    monkeypatch.setattr(trainer_utils, "PeftModel", DummyPeftModel)

    model = trainer_utils.create_model_from_path(str(adapter_dir), architecture=DummyArch, dtype="float32")

    assert model == "adapter-model"
    assert calls == [
        ("base", "base-model-id", {"dtype": trainer_utils.torch.float32, "device_map": "auto"}),
        ("adapter", "base-model", str(adapter_dir), False),
    ]
    assert trainer._last_loaded_step == 7

    CodeGRPOTrainer._maybe_sync_vllm_weights(trainer)
    assert trainer.vllm_generation.sync_calls == 1
