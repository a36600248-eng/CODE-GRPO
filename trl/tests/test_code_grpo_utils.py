# Copyright 2020-2026 The HuggingFace Team. All rights reserved.

from types import SimpleNamespace

from trl.extensions.code_grpo.error_utils import summarize_error
from trl.extensions.code_grpo.matcher import is_match, values_equal
from trl.extensions.code_grpo.parser import build_generation_completion, parse_generation_output, parse_generation_response
from trl.extensions.code_grpo.soft_reward import compute_soft_reward
from trl.extensions.code_grpo.tree import _compute_code_reward, _is_double_zero_node, _is_soft_reward_eligible
from trl.extensions.code_grpo.types import ExecResult, Node
from trl.trainer.code_grpo_config import CodeGRPOConfig
from trl.trainer.code_grpo_trainer import CodeGRPOTrainer
from trl.trainer import utils as trainer_utils


def test_parse_generation_output_prefers_last_code_block():
    text = "```python\nprint('old')\n```\nnoise\n```python\nprint('new')\n```"
    code, reason, logic_prediction, exec_prediction = parse_generation_output(text)
    assert code == "print('new')"
    assert reason == ""
    assert logic_prediction == ""
    assert exec_prediction == ""


def test_parse_generation_response_requires_single_block_for_format_ok():
    text = "```python\nprint('x')\n```"
    code, _, _, _, format_ok = parse_generation_response(text)
    assert code == "print('x')"
    assert format_ok

    text = "```python\nprint('a')\n```\n```python\nprint('b')\n```"
    code, _, _, _, format_ok = parse_generation_response(text)
    assert code == "print('b')"
    assert not format_ok


def test_parse_generation_response_recovers_incomplete_fenced_block():
    text = "```python\nprint('x')\n"
    code, _, _, _, format_ok = parse_generation_response(text)
    assert code == "print('x')"
    assert not format_ok


def test_build_generation_completion_contains_fence():
    completion = build_generation_completion("a=1")
    assert completion.startswith("```python")
    assert "a=1" in completion


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
    assert _compute_code_reward(
        pass_rate=1.0,
        r_soft=1.0,
        lambda_soft=1.0,
        compile_score=0.0,
        compile_scale=0.0,
        generation_format_score=0.0,
        generation_format_scale=0.0,
    ) == 1.0
    assert _compute_code_reward(
        pass_rate=0.5,
        r_soft=1.0,
        lambda_soft=0.2,
        compile_score=0.0,
        compile_scale=0.0,
        generation_format_score=0.0,
        generation_format_scale=0.0,
    ) == 0.7
    assert _compute_code_reward(
        pass_rate=0.0,
        r_soft=1.0,
        lambda_soft=0.2,
        compile_score=0.0,
        compile_scale=0.0,
        generation_format_score=0.0,
        generation_format_scale=0.0,
    ) == 0.2


def test_lambda_soft_range_validation():
    CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, lambda_soft=0.0)
    CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, lambda_soft=1.0)
    try:
        CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, lambda_soft=1.1)
        assert False, "Expected ValueError for lambda_soft > 1"
    except ValueError:
        pass


def test_double_zero_node_detection():
    node = Node(node_id="n1", parent_id="root", round_idx=1, code="x", R_code=0.0)
    assert _is_double_zero_node(node)
    node.R_code = 1e-6
    assert not _is_double_zero_node(node)


def test_soft_reward_eligible_accepts_stdio_without_solve():
    stdio_code = "import sys\nprint(sys.stdin.read(), end='')\n"
    assert _is_soft_reward_eligible(stdio_code, "stdio")
    assert not _is_soft_reward_eligible(stdio_code, "call")
    assert _is_soft_reward_eligible("def solve(x):\n    return x\n", "call")


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
    assert trainer._last_loaded_step == 7

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


def test_compute_soft_reward_reuses_problem_logprob_cache():
    calls = []

    class DummyEvaluator:
        def logprob(self, prompt, target_text):
            calls.append((prompt, target_text))
            if "code view" in prompt.lower():
                return 1.0
            return 0.25

    problem = {
        "prompt": "Add two numbers",
        "diagnostic_inputs": ["1 2\n", "3 4\n"],
        "diagnostic_outputs": ["3\n", "7\n"],
    }
    cache = {}

    reward1, details1 = compute_soft_reward(
        problem=problem,
        code="print(3)",
        diagnostic_inputs=problem["diagnostic_inputs"],
        oracle_outputs=problem["diagnostic_outputs"],
        evaluator=DummyEvaluator(),
        problem_logprob_cache=cache,
    )
    assert len(details1) == 2
    assert len(cache) == 2
    assert len(calls) == 4

    reward2, details2 = compute_soft_reward(
        problem=problem,
        code="print(7)",
        diagnostic_inputs=problem["diagnostic_inputs"],
        oracle_outputs=problem["diagnostic_outputs"],
        evaluator=DummyEvaluator(),
        problem_logprob_cache=cache,
    )
    assert len(details2) == 2
    assert reward1 == reward2
    assert len(calls) == 6
