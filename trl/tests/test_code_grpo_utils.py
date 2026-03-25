# Copyright 2020-2026 The HuggingFace Team. All rights reserved.

import math
import random
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

from trl.extensions.code_grpo import tree as tree_module
from trl.extensions.code_grpo.error_utils import summarize_error
from trl.extensions.code_grpo.matcher import is_match, values_equal
from trl.extensions.code_grpo.parser import build_generation_completion, parse_generation_output, parse_generation_response
from trl.extensions.code_grpo.soft_reward import build_diagnostic_inputs, compute_soft_reward, get_oracle_outputs
from trl.scripts import code_grpo as code_grpo_script
from trl.extensions.code_grpo.tree import (
    CodeGRPOTreeRunner,
    _compute_code_reward,
    _is_double_zero_node,
    _is_soft_reward_eligible,
)
from trl.extensions.code_grpo.types import ExecResult, Node
from trl.trainer.code_grpo_config import CodeGRPOConfig
from trl.trainer.code_grpo_trainer import CodeGRPOTrainer, _DeferredCodeIOSample, _PseudoIterativeNode
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



def test_code_reward_keeps_compile_and_format_shaping_for_zero_pass():
    assert _compute_code_reward(
        pass_rate=0.0,
        r_soft=1.0,
        lambda_soft=0.125,
        compile_score=1.0,
        compile_scale=0.1,
        generation_format_score=1.0,
        generation_format_scale=0.05,
    ) == 0.275


def test_code_reward_applies_bounded_soft_reward_to_partial_pass_samples():
    assert _compute_code_reward(
        pass_rate=0.25,
        r_soft=1.0,
        lambda_soft=0.125,
        compile_score=0.0,
        compile_scale=0.0,
        generation_format_score=0.0,
        generation_format_scale=0.0,
    ) == 0.375


def test_maybe_truncate_generated_code_caps_train_side_tokens():
    runner = object.__new__(CodeGRPOTreeRunner)

    class DummyTokenizer:
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            del add_special_tokens, return_offsets_mapping
            return {"input_ids": [ord(ch) for ch in text]}

        def decode(self, token_ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(chr(token_id) for token_id in token_ids)

    runner.tokenizer = DummyTokenizer()

    code, truncated, original_token_count = runner._maybe_truncate_generated_code("abcdef", token_cap=3)
    assert code == "abc"
    assert truncated is True
    assert original_token_count == 6

    code2, truncated2, original_token_count2 = runner._maybe_truncate_generated_code("abc", token_cap=3)
    assert code2 == "abc"
    assert truncated2 is False
    assert original_token_count2 == 3


def test_lambda_soft_range_validation():
    CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, lambda_soft=0.0)
    CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, lambda_soft=1.0)
    try:
        CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, lambda_soft=1.1)
        assert False, "Expected ValueError for lambda_soft > 1"
    except ValueError:
        pass
    CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, train_generation_truncate_tokens=0)
    try:
        CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, train_generation_truncate_tokens=-1)
        assert False, "Expected ValueError for negative train_generation_truncate_tokens"
    except ValueError:
        pass
    CodeGRPOConfig(
        use_cpu=True,
        bf16=False,
        fp16=False,
        train_generation_total_token_cap=512,
        train_generation_completion_reserve_tokens=128,
    )
    try:
        CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, train_generation_total_token_cap=-1)
        assert False, "Expected ValueError for negative train_generation_total_token_cap"
    except ValueError:
        pass
    try:
        CodeGRPOConfig(use_cpu=True, bf16=False, fp16=False, train_generation_completion_reserve_tokens=-1)
        assert False, "Expected ValueError for negative train_generation_completion_reserve_tokens"
    except ValueError:
        pass
    try:
        CodeGRPOConfig(
            use_cpu=True,
            bf16=False,
            fp16=False,
            train_generation_total_token_cap=128,
            train_generation_completion_reserve_tokens=256,
        )
        assert False, "Expected ValueError when reserve exceeds total cap"
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


def test_compute_soft_reward_skips_nonfinite_logprob():
    class DummyEvaluator:
        def __init__(self):
            self.calls = 0

        def logprob(self, prompt, target_text):
            del prompt, target_text
            self.calls += 1
            if self.calls == 1:
                return float("nan")
            return 1.0

    problem = {
        "prompt": "Add two numbers",
        "diagnostic_inputs": ["1 2\n"],
        "diagnostic_outputs": ["3\n"],
    }

    reward, details = compute_soft_reward(
        problem=problem,
        code="print(3)",
        diagnostic_inputs=problem["diagnostic_inputs"],
        oracle_outputs=problem["diagnostic_outputs"],
        evaluator=DummyEvaluator(),
        problem_logprob_cache={},
    )

    assert math.isnan(reward)
    assert len(details) == 1
    assert details[0]["skipped"] == "problem_logprob_unavailable"


def test_build_diagnostic_inputs_prefers_shorter_paired_cases():
    problem = {
        "prompt": "Echo",
        "diagnostic_inputs": ["99999999999999999999\n", "1\n"],
        "diagnostic_outputs": ["99999999999999999999\n", "1\n"],
    }

    selected_inputs = build_diagnostic_inputs(problem, max_count=1)
    selected_outputs = get_oracle_outputs(problem, selected_inputs)

    assert selected_inputs == ["1\n"]
    assert selected_outputs == ["1\n"]


def test_build_diagnostic_inputs_preserves_explicit_inputs_when_outputs_are_missing():
    problem = {
        "prompt": "Echo",
        "diagnostic_inputs": ["99999999999999999999\n", "1\n"],
        "diagnostic_outputs": [],
        "test_cases": [
            {"input": "99999999999999999999\n", "output": "99999999999999999999\n"},
            {"input": "1\n", "output": "1\n"},
        ],
    }

    selected_inputs = build_diagnostic_inputs(problem, max_count=1)
    selected_outputs = get_oracle_outputs(problem, selected_inputs)

    assert selected_inputs == ["1\n"]
    assert selected_outputs == ["1\n"]


def test_get_oracle_outputs_aligns_to_requested_input_order():
    problem = {
        "prompt": "Echo",
        "diagnostic_inputs": ["99999999999999999999\n", "1\n"],
        "diagnostic_outputs": ["99999999999999999999\n", "1\n"],
    }

    oracle_outputs = get_oracle_outputs(problem, ["1\n", "99999999999999999999\n"])

    assert oracle_outputs == ["1\n", "99999999999999999999\n"]


def test_build_diagnostic_inputs_skips_explicit_cases_without_available_output():
    problem = {
        "prompt": "Echo",
        "diagnostic_inputs": ["missing\n", "1\n"],
        "diagnostic_outputs": [],
        "test_cases": [{"input": "1\n", "output": "1\n"}],
    }

    selected_inputs = build_diagnostic_inputs(problem, max_count=2)
    selected_outputs = get_oracle_outputs(problem, selected_inputs)

    assert selected_inputs == ["1\n"]
    assert selected_outputs == ["1\n"]


def test_compute_soft_reward_skips_missing_oracle_output():
    class DummyEvaluator:
        def logprob(self, prompt, target_text):
            del prompt, target_text
            raise AssertionError("logprob should not be called when oracle output is missing")

    reward, details = compute_soft_reward(
        problem={"prompt": "Echo"},
        code="print(input())",
        diagnostic_inputs=["missing\n"],
        oracle_outputs=[None],
        evaluator=DummyEvaluator(),
        problem_logprob_cache={},
    )

    assert math.isnan(reward)
    assert details[0]["skipped"] == "oracle_output_unavailable"


def test_build_round_record_preserves_iterative_prompt_fields():
    runner = object.__new__(CodeGRPOTreeRunner)
    node = Node(
        node_id="n1",
        parent_id="root",
        round_idx=1,
        code="print(input())",
        pass_rate=0.25,
        R_code=0.2,
        exec_summary={
            "pass_cnt": 2,
            "test_count": 8,
            "normalized_soft_reward": 0.4,
            "soft_reward_beta": 0.05,
            "history": [],
            "generation_debug": {},
        },
    )

    record = runner._build_round_record(1, [node], stage="search")
    payload = record["nodes"][0]

    assert payload["code_text"] == "print(input())"
    assert payload["pass_cnt"] == 2
    assert payload["test_count"] == 8
    assert math.isclose(payload["R_soft_effective"], 0.02)


def test_run_question_eval_code_only_uses_single_generation(monkeypatch):
    class DummyBackend:
        def __init__(self):
            self.generate_calls = 0
            self.generate_many_calls = 0

        def generate(self, prompt_text, **kwargs):
            del prompt_text, kwargs
            self.generate_calls += 1
            return "```python\nprint('ok')\n```"

        def generate_many(self, prompts, **kwargs):
            del prompts, kwargs
            self.generate_many_calls += 1
            return ["```python\nprint('wrong')\n```"]

        def logprob(self, prompt, target_text, **kwargs):
            del prompt, target_text, kwargs
            return 0.0

    class DummyTokenizer:
        is_fast = False

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            del tokenize, add_generation_prompt
            return messages[0]["content"]

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            del add_special_tokens, return_offsets_mapping
            return {"input_ids": [1] * max(1, len(text))}

    def fake_execute_batch(code, case_inputs, timeout_s, error_max_chars, error_max_lines, io_mode):
        del code, case_inputs, timeout_s, error_max_chars, error_max_lines, io_mode
        return [ExecResult(kind="OK", value="ok\n")]

    monkeypatch.setattr(tree_module, "execute_batch", fake_execute_batch)

    args = SimpleNamespace(
        use_chat_template_for_codegrpo=True,
        max_completion_length=32,
        max_completion_length_code=32,
        generation_min_new_tokens_code=0,
        generation_temperature_code=0.7,
        eval_generation_temperature_code=0.0,
        generation_top_p_code=0.95,
        generation_empty_retry_count=0,
        generation_outside_noise_chars=0,
        context_round_window=1,
        K=8,
        N_max=8,
        M_retry=0,
        error_max_chars=200,
        error_max_lines=20,
        code_timeout_seconds=1.0,
        zero_pass_soft_reward_enabled=False,
        zero_pass_soft_reward_diag_count=0,
        zero_pass_soft_reward_clip_low=-2.0,
        zero_pass_soft_reward_clip_high=2.0,
        zero_pass_soft_reward_beta_scale=0.5,
        soft_reward_ineligible_scale=0.3,
        code_compile_reward_scale=0.0,
        code_format_reward_scale=0.0,
        code_aux_reward_without_format_scale=1.0,
        code_io_aux_training_enabled=False,
        code_io_aux_case_count=0,
        code_io_aux_include_correct=True,
        code_io_aux_include_incorrect=True,
        code_io_aux_include_errors=True,
        code_io_aux_sft_weight_correct=1.0,
        code_io_aux_sft_weight_incorrect=1.0,
        advantage_base="R_code",
        advantage_mode="mean_only",
        trace_store_full_text=False,
        eval_round_n=1,
        eval_k_list=[1],
        eval_T_max_override=1,
        T_max=1,
        num_generations_eval=1,
    )
    backend = DummyBackend()
    runner = CodeGRPOTreeRunner(backend=backend, tokenizer=DummyTokenizer(), args=args, logger=SimpleNamespace(info=lambda *a, **k: None))
    sample = {
        "question_id": "q1",
        "prompt": "Print ok",
        "test_cases": [{"input": "", "output": "ok\n"}],
        "diagnostic_inputs": [],
        "diagnostic_outputs": [],
        "io_mode": "stdio",
    }

    rollout = runner.run_question_eval_code_only(sample, rng=None)

    assert backend.generate_calls == 1
    assert backend.generate_many_calls == 0
    assert rollout.node_count == 1
    assert len(rollout.rounds) == 1
    assert len(rollout.rounds[0]["nodes"]) == 1
    assert rollout.eval_metrics["pass_at_1"] == 1.0


def test_run_question_eval_code_only_supports_multiple_single_trajectory_rounds(monkeypatch):
    class DummyBackend:
        def __init__(self):
            self.generate_calls = 0
            self.generate_many_calls = 0

        def generate(self, prompt_text, **kwargs):
            del kwargs
            self.generate_calls += 1
            if self.generate_calls == 1:
                assert "Latest feedback:" not in prompt_text
                return "```python\nprint('wrong')\n```"
            assert "Latest feedback:" in prompt_text
            assert "actual_result_or_error" in prompt_text
            return "```python\nprint('ok')\n```"

        def generate_many(self, prompts, **kwargs):
            del prompts, kwargs
            self.generate_many_calls += 1
            return ["```python\nprint('wrong')\n```"]

        def logprob(self, prompt, target_text, **kwargs):
            del prompt, target_text, kwargs
            return 0.0

    class DummyTokenizer:
        is_fast = False

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            del tokenize, add_generation_prompt
            return messages[0]["content"]

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            del add_special_tokens, return_offsets_mapping
            return {"input_ids": [1] * max(1, len(text))}

    def fake_execute_batch(code, case_inputs, timeout_s, error_max_chars, error_max_lines, io_mode):
        del case_inputs, timeout_s, error_max_chars, error_max_lines, io_mode
        if "wrong" in code:
            return [ExecResult(kind="OK", value="wrong\n")]
        return [ExecResult(kind="OK", value="ok\n")]

    monkeypatch.setattr(tree_module, "execute_batch", fake_execute_batch)

    args = SimpleNamespace(
        use_chat_template_for_codegrpo=True,
        max_completion_length=32,
        max_completion_length_code=32,
        generation_min_new_tokens_code=0,
        generation_temperature_code=0.7,
        eval_generation_temperature_code=0.0,
        generation_top_p_code=0.95,
        generation_empty_retry_count=0,
        generation_outside_noise_chars=0,
        context_round_window=1,
        K=8,
        N_max=8,
        M_retry=0,
        error_max_chars=200,
        error_max_lines=20,
        code_timeout_seconds=1.0,
        zero_pass_soft_reward_enabled=False,
        zero_pass_soft_reward_diag_count=0,
        zero_pass_soft_reward_clip_low=-2.0,
        zero_pass_soft_reward_clip_high=2.0,
        zero_pass_soft_reward_beta_scale=0.5,
        soft_reward_ineligible_scale=0.3,
        code_compile_reward_scale=0.0,
        code_format_reward_scale=0.0,
        code_aux_reward_without_format_scale=1.0,
        code_io_aux_training_enabled=False,
        code_io_aux_case_count=0,
        code_io_aux_include_correct=True,
        code_io_aux_include_incorrect=True,
        code_io_aux_include_errors=True,
        code_io_aux_sft_weight_correct=1.0,
        code_io_aux_sft_weight_incorrect=1.0,
        advantage_base="R_code",
        advantage_mode="mean_only",
        trace_store_full_text=False,
        eval_round_n=2,
        eval_k_list=[1],
        eval_T_max_override=2,
        T_max=1,
        num_generations_eval=1,
    )
    backend = DummyBackend()
    runner = CodeGRPOTreeRunner(backend=backend, tokenizer=DummyTokenizer(), args=args, logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None))
    sample = {
        "question_id": "q2",
        "prompt": "Print ok",
        "test_cases": [{"input": "", "output": "ok\n"}],
        "diagnostic_inputs": [],
        "diagnostic_outputs": [],
        "io_mode": "stdio",
    }

    rollout = runner.run_question_eval_code_only(sample, rng=None)

    assert backend.generate_calls == 2
    assert backend.generate_many_calls == 0
    assert rollout.node_count == 2
    assert len(rollout.rounds) == 2
    assert rollout.eval_metrics["pass_at_1_within_1"] == 0.0
    assert rollout.eval_metrics["pass_at_1_within_2"] == 1.0
    assert rollout.eval_metrics["best_pass_rate_within_2"] == 1.0
    assert rollout.eval_metrics["pass_at_1"] == 1.0


def test_resolve_training_generation_kwargs_caps_completion_by_total_budget():
    runner = object.__new__(CodeGRPOTreeRunner)

    class DummyTokenizer:
        is_fast = False

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            del add_special_tokens, return_offsets_mapping
            return {"input_ids": [1] * len(text)}

    runner.tokenizer = DummyTokenizer()
    runner.args = SimpleNamespace(
        max_completion_length=512,
        max_completion_length_code=512,
        generation_min_new_tokens_code=16,
        generation_temperature_code=0.7,
        generation_top_p_code=0.95,
        train_generation_total_token_cap=100,
        train_generation_completion_reserve_tokens=24,
    )

    kwargs, meta = runner._resolve_training_generation_kwargs("x" * 80)

    assert kwargs["max_new_tokens"] == 20
    assert kwargs["min_new_tokens"] == 16
    assert meta["prompt_token_count"] == 80
    assert meta["budget_capped"] is True
    assert meta["reserve_met"] is False


def test_build_train_prompt_and_generation_kwargs_trims_history_to_preserve_completion_budget():
    runner = object.__new__(CodeGRPOTreeRunner)

    class DummyTokenizer:
        is_fast = False

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            del tokenize, add_generation_prompt
            return messages[0]["content"]

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            del add_special_tokens, return_offsets_mapping
            return {"input_ids": [1] * len(text)}

    runner.tokenizer = DummyTokenizer()
    runner.args = SimpleNamespace(
        use_chat_template_for_codegrpo=True,
        context_round_window=1,
        max_completion_length=512,
        max_completion_length_code=512,
        generation_min_new_tokens_code=16,
        generation_temperature_code=0.7,
        generation_top_p_code=0.95,
        train_generation_total_token_cap=260,
        train_generation_completion_reserve_tokens=120,
    )

    history = [
        {
            "status_code": "RUNTIME_ERROR",
            "failed_input": "1 2 3",
            "failed_actual": "ValueError",
            "error_summary": "x" * 180,
            "compile_score": 1.0,
        }
    ]

    _, rendered_prompt, kwargs, meta, used_history = runner._build_train_prompt_and_generation_kwargs(
        question_prompt="Print ok",
        history=history,
        parent_code=None,
    )

    assert meta["history_trimmed_for_budget"] is True
    assert meta["history_items_used"] == 0
    assert used_history == []
    assert kwargs["max_new_tokens"] >= 120
    assert "Earlier history summary:" not in rendered_prompt


def test_make_iterative_prompt_uses_stdio_specific_instruction():
    trainer = object.__new__(CodeGRPOTrainer)

    stdio_prompt = trainer._make_iterative_prompt(
        source_example={"prompt": "Solve it", "io_mode": "stdio"},
        node_payload={
            "code_text": "print(input())",
            "pass_cnt": 2,
            "test_count": 8,
            "error_summary": "NameError: x",
            "raw_soft_reward": 0.1,
            "normalized_soft_reward": 0.4,
            "final_reward": 0.25,
            "history": [{"failed_input": "3 5\n", "failed_actual": "NameError"}],
        },
        selection_tag="best_pass",
    )
    assert "standard input" in stdio_prompt
    assert "standard output" in stdio_prompt
    assert "solve(x)" not in stdio_prompt
    assert "Test result: passed 2 of 8 tests." in stdio_prompt
    assert "Error: NameError: x" in stdio_prompt
    assert "First failing input:" in stdio_prompt
    assert "Actual output or error:" in stdio_prompt
    assert "selection_tag" not in stdio_prompt
    assert "selection_reason" not in stdio_prompt
    assert "raw_soft_reward" not in stdio_prompt
    assert "normalized_soft_reward" not in stdio_prompt
    assert "final_reward" not in stdio_prompt

    call_prompt = trainer._make_iterative_prompt(
        source_example={"prompt": "Solve it", "io_mode": "call"},
        node_payload={"code_text": "def solve(x):\n    return x", "history": []},
        selection_tag="best_pass",
    )
    assert "solve(x)" in call_prompt
    assert "function interface" in call_prompt

    fallback_prompt = trainer._make_iterative_prompt(
        source_example={
            "prompt": "Solve it",
            "io_mode": "stdio",
            "test_cases": [{"input": "x", "output": "y"} for _ in range(8)],
        },
        node_payload={"code": "print(input())", "pass_rate": 0.25, "history": []},
        selection_tag="best_pass",
    )
    assert "Previous code:\nprint(input())" in fallback_prompt
    assert "Test result: passed 2 of 8 tests." in fallback_prompt


def test_append_iterative_nodes_from_rollout_preserves_stdio_metadata():
    trainer = object.__new__(CodeGRPOTrainer)
    trainer._pseudo_multiround_enabled = True
    trainer._pseudo_iterative_pool = []
    trainer._pseudo_node_serial = 0
    trainer.state = SimpleNamespace(global_step=7)
    trainer.args = SimpleNamespace(pseudo_iterative_select_count=1, pseudo_iterative_pool_capacity=8)
    trainer._code_novelty_scores = lambda nodes: [0.0 for _ in nodes]
    trainer._get_question_prior_weight = lambda qid: 1.0
    trainer._base_question_id = lambda example: str(example.get("base_question_id", example["question_id"]))

    rollout = SimpleNamespace(
        rounds=[
            {
                "stage": "search",
                "nodes": [
                    {
                        "main_sample_active": True,
                        "pass_rate": 0.25,
                        "final_reward": 0.25,
                        "R_code": 0.25,
                        "raw_soft_reward": 0.1,
                        "normalized_soft_reward": 0.4,
                        "history": [],
                        "code_text": "print(input())",
                    }
                ],
            }
        ]
    )
    source_example = {
        "question_id": "apps_1",
        "base_question_id": "apps_1",
        "prompt": "Echo input",
        "source_prompt": "Echo input",
        "test_cases": [{"input": "a\n", "output": "a\n"}],
        "diagnostic_inputs": ["1\n"],
        "diagnostic_outputs": ["1\n"],
        "io_mode": "stdio",
    }

    added = trainer._append_iterative_nodes_from_rollout(rollout, source_example)

    assert added == 1
    assert len(trainer._pseudo_iterative_pool) == 1
    record = trainer._pseudo_iterative_pool[0]
    assert record.io_mode == "stdio"
    assert record.diagnostic_inputs == ["1\n"]
    assert record.diagnostic_outputs == ["1\n"]


def test_sample_iterative_examples_preserves_io_mode_and_diagnostics():
    trainer = object.__new__(CodeGRPOTrainer)
    trainer._pseudo_iterative_pool = [
        _PseudoIterativeNode(
            question_id="apps_1__iter_1",
            base_question_id="apps_1",
            prompt="Refine",
            source_prompt="Echo input",
            test_cases=[{"input": "a\n", "output": "a\n"}],
            selection_tag="best_pass",
            priority=1.0,
            raw_soft_reward=0.1,
            final_reward=0.2,
            pass_rate=0.25,
            created_step=3,
            io_mode="stdio",
            diagnostic_inputs=["1\n"],
            diagnostic_outputs=["1\n"],
        )
    ]

    sampled = trainer._sample_iterative_examples(1, rng=random.Random(0))

    assert len(sampled) == 1
    assert sampled[0]["io_mode"] == "stdio"
    assert sampled[0]["diagnostic_inputs"] == ["1\n"]
    assert sampled[0]["diagnostic_outputs"] == ["1\n"]


def test_append_iterative_nodes_from_rollout_skips_novelty_when_disabled():
    trainer = object.__new__(CodeGRPOTrainer)
    trainer._pseudo_multiround_enabled = True
    trainer.args = SimpleNamespace(
        pseudo_iterative_select_count=3,
        pseudo_iterative_pool_capacity=16,
        pseudo_iterative_include_novelty=False,
        pseudo_iterative_soft_priority_bonus_scale=0.1,
    )
    trainer.state = SimpleNamespace(global_step=5)
    trainer._pseudo_iterative_pool = []
    trainer._pseudo_node_serial = 0
    trainer._get_question_prior_weight = lambda qid: 1.0
    trainer._base_question_id = lambda example: str(example.get("base_question_id") or example["question_id"])

    rollout = SimpleNamespace(
        rounds=[
            {
                "stage": "search",
                "nodes": [
                    {
                        "node_id": "n1",
                        "main_sample_active": True,
                        "code_text": "print(1)",
                        "pass_rate": 0.5,
                        "final_reward": 0.5,
                        "raw_soft_reward": 0.1,
                    },
                    {
                        "node_id": "n2",
                        "main_sample_active": True,
                        "code_text": "print(2)",
                        "pass_rate": 0.1,
                        "final_reward": 0.1,
                        "raw_soft_reward": 0.9,
                    },
                ],
            }
        ]
    )
    source_example = {
        "question_id": "q1",
        "prompt": "Print something",
        "test_cases": [{"input": "", "output": "1\n"}],
        "io_mode": "stdio",
    }

    added = trainer._append_iterative_nodes_from_rollout(rollout, source_example)

    assert added == 2
    assert {record.selection_tag for record in trainer._pseudo_iterative_pool} == {"best_pass", "best_soft"}


def test_append_iterative_nodes_from_rollout_does_not_compute_novelty_when_disabled():
    trainer = object.__new__(CodeGRPOTrainer)
    trainer._pseudo_multiround_enabled = True
    trainer.args = SimpleNamespace(
        pseudo_iterative_select_count=2,
        pseudo_iterative_pool_capacity=16,
        pseudo_iterative_include_novelty=False,
        pseudo_iterative_soft_priority_bonus_scale=0.1,
    )
    trainer.state = SimpleNamespace(global_step=5)
    trainer._pseudo_iterative_pool = []
    trainer._pseudo_node_serial = 0
    trainer._get_question_prior_weight = lambda qid: 1.0
    trainer._base_question_id = lambda example: str(example.get("base_question_id") or example["question_id"])
    trainer._code_novelty_scores = lambda nodes: (_ for _ in ()).throw(AssertionError("novelty should not be computed"))

    rollout = SimpleNamespace(
        rounds=[
            {
                "stage": "search",
                "nodes": [
                    {
                        "node_id": "n1",
                        "main_sample_active": True,
                        "code_text": "print(1)",
                        "pass_rate": 0.5,
                        "final_reward": 0.5,
                        "raw_soft_reward": 0.1,
                    },
                    {
                        "node_id": "n2",
                        "main_sample_active": True,
                        "code_text": "print(2)",
                        "pass_rate": 0.1,
                        "final_reward": 0.1,
                        "raw_soft_reward": 0.9,
                    },
                ],
            }
        ]
    )
    source_example = {
        "question_id": "q1",
        "prompt": "Print something",
        "test_cases": [{"input": "", "output": "1\n"}],
        "io_mode": "stdio",
    }

    added = trainer._append_iterative_nodes_from_rollout(rollout, source_example)

    assert added == 2


def test_deferred_code_io_ce_selects_best_pass_and_random_executable():
    trainer = object.__new__(CodeGRPOTrainer)
    trainer.args = SimpleNamespace(
        code_io_ce_select_best_pass=True,
        code_io_ce_select_random_executable=True,
        code_io_ce_sample_count_per_question=2,
    )

    nodes = [
        {"node_id": "a", "compile_score": 1.0, "pass_rate": 0.8, "final_reward": 0.8},
        {"node_id": "b", "compile_score": 1.0, "pass_rate": 0.2, "final_reward": 0.2},
        {"node_id": "c", "compile_score": 0.0, "pass_rate": 0.0, "final_reward": 0.0},
    ]

    chosen = trainer._choose_deferred_code_io_nodes(nodes, random.Random(0))

    assert len(chosen) == 2
    assert chosen[0][1] == "best_pass"
    assert chosen[0][0]["node_id"] == "a"
    assert chosen[1][1] == "random_executable"
    assert chosen[1][0]["node_id"] == "b"


def test_should_run_deferred_code_io_ce_step_uses_periodic_schedule():
    trainer = object.__new__(CodeGRPOTrainer)
    trainer.model = SimpleNamespace(training=True)
    trainer.args = SimpleNamespace(code_io_ce_buffer_enabled=True, code_io_ce_every_n_steps=20)
    trainer.state = SimpleNamespace(global_step=19)
    trainer._code_io_ce_buffer = deque([SimpleNamespace()])

    assert trainer._should_run_deferred_code_io_ce_step() is True


def test_build_deferred_code_io_ce_batch_uses_batch_size_limit():
    trainer = object.__new__(CodeGRPOTrainer)
    trainer.args = SimpleNamespace(code_io_ce_batch_size=3)
    trainer.code_tokenizer = lambda text, add_special_tokens=False: {"input_ids": [1] * max(1, len(text))}
    trainer._metrics = {"train": defaultdict(list)}
    trainer._build_training_batch = lambda train_samples, to_device=False: {"sample_count": len(train_samples), "to_device": to_device}
    trainer._code_io_ce_buffer = deque(
        [
            _DeferredCodeIOSample(
                question_id=f"q{i}",
                prompt_text="p",
                completion_text="c",
                sft_weight=1.0,
                pass_rate=0.0,
                source_tag="best_pass",
            )
            for i in range(5)
        ]
    )

    batch = trainer._build_deferred_code_io_ce_batch()

    assert batch == {"sample_count": 3, "to_device": False}
    assert len(trainer._code_io_ce_buffer) == 2
    assert trainer._metrics["train"]["code_io_ce_buffer/consumed"][-1] == 3.0
    assert trainer._metrics["train"]["code_io_ce_buffer/size"][-1] == 2.0


def test_compute_code_only_eval_metrics_carries_forward_cumulative_metrics_to_target_rounds():
    runner = object.__new__(CodeGRPOTreeRunner)
    runner.args = SimpleNamespace(eval_round_n=3, eval_T_max_override=4, T_max=4)

    metrics = runner._compute_code_only_eval_metrics(
        [
            {
                "stage": "search",
                "nodes": [{"pass_rate": 0.25}],
            }
        ]
    )

    assert metrics["pass_at_1_within_1"] == 0.0
    assert metrics["best_pass_rate_within_1"] == 0.25
    assert metrics["pass_at_1_within_2"] == 0.0
    assert metrics["pass_at_1_within_4"] == 0.0
    assert metrics["best_pass_rate_within_4"] == 0.25
    assert metrics["pass_at_1"] == 0.0
    assert metrics["pass_at_k_within_n"] == 0.0


def test_compute_code_only_eval_metrics_preserves_early_solved_within_later_rounds():
    runner = object.__new__(CodeGRPOTreeRunner)
    runner.args = SimpleNamespace(eval_round_n=2, eval_T_max_override=2, T_max=2)

    metrics = runner._compute_code_only_eval_metrics(
        [
            {
                "stage": "search",
                "nodes": [{"pass_rate": 1.0}],
            }
        ]
    )

    assert metrics["pass_at_1_within_1"] == 1.0
    assert metrics["pass_at_1_within_2"] == 1.0
    assert metrics["best_pass_rate_within_2"] == 1.0
    assert metrics["pass_at_1"] == 1.0
    assert metrics["pass_at_k_within_n"] == 1.0


def test_generate_node_skips_soft_reward_scoring_for_fully_passing_candidate(monkeypatch):
    class DummyBackend:
        def generate(self, prompt_text, **kwargs):
            del prompt_text, kwargs
            raise AssertionError("generate should not be called")

        def logprob(self, prompt, target_text, **kwargs):
            del prompt, target_text, kwargs
            raise AssertionError("logprob should not be called for pass_rate=1.0")

    class DummyTokenizer:
        is_fast = False

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            del add_special_tokens, return_offsets_mapping
            return {"input_ids": [1] * max(1, len(text))}

    def fake_execute_batch(code, case_inputs, timeout_s, error_max_chars, error_max_lines, io_mode):
        del code, case_inputs, timeout_s, error_max_chars, error_max_lines, io_mode
        return [ExecResult(kind="OK", value="ok\n")]

    monkeypatch.setattr(tree_module, "execute_batch", fake_execute_batch)

    runner = object.__new__(CodeGRPOTreeRunner)
    runner.backend = DummyBackend()
    runner.tokenizer = DummyTokenizer()
    runner.args = SimpleNamespace(
        context_round_window=1,
        generation_outside_noise_chars=0,
        error_max_chars=256,
        error_max_lines=8,
        code_timeout_seconds=1.0,
        zero_pass_soft_reward_enabled=True,
        zero_pass_soft_reward_diag_count=2,
        zero_pass_soft_reward_clip_low=-2.0,
        zero_pass_soft_reward_clip_high=2.0,
        zero_pass_soft_reward_beta_scale=0.5,
        code_compile_reward_scale=0.0,
        code_format_reward_scale=0.0,
        code_aux_reward_without_format_scale=1.0,
        code_io_aux_training_enabled=False,
        code_io_ce_buffer_enabled=False,
        code_io_aux_case_count=1,
        code_io_aux_include_correct=True,
        code_io_aux_include_incorrect=True,
        code_io_aux_include_errors=True,
        code_io_aux_sft_weight_correct=1.0,
        code_io_aux_sft_weight_incorrect=1.0,
    )
    runner._code_completion_length = lambda: 64
    runner._count_tokens = lambda text: len(text)
    runner._retry_empty_generation_outputs = lambda prompt_text, outputs, generation_kwargs=None: outputs
    runner._maybe_truncate_generated_code = lambda code, token_cap=0: (code, False, len(code))

    node = runner._generate_node(
        question_id="q1",
        node_id="n1",
        parent=Node(node_id="root", parent_id=None, round_idx=0, code="", exec_summary={"history": []}),
        prompt="Print ok",
        test_cases=[{"input": "", "output": "ok\n"}],
        diagnostic_inputs=[""],
        diagnostic_outputs=["ok\n"],
        io_mode="stdio",
        round_idx=1,
        prompt_text_override="Print ok",
        prompt_text_raw_override="Print ok",
        raw_output_override="```python\nprint('ok')\n```",
        generation_kwargs_override={"max_new_tokens": 32},
    )

    assert node.pass_rate == 1.0
    assert node.exec_summary["soft_reward_eligible"] is False
    assert node.exec_summary["raw_soft_reward"] == 0.0
    assert node.exec_summary["soft_reward_beta"] == 0.0


def test_build_code_io_aux_samples_prefers_shorter_cases():
    samples = tree_module._build_code_io_aux_samples(
        question_id="q1",
        code="print(input())",
        test_cases=[
            {"input": "99999999999999999999\n", "output": "99999999999999999999\n"},
            {"input": "1\n", "output": "1\n"},
        ],
        case_inputs=["99999999999999999999\n", "1\n"],
        exec_results=[
            ExecResult(kind="OK", value="99999999999999999999\n"),
            ExecResult(kind="OK", value="1\n"),
        ],
        pass_flags=[True, True],
        enabled=True,
        case_count=1,
        include_correct=True,
        include_incorrect=True,
        include_errors=True,
        sft_weight_correct=1.0,
        sft_weight_incorrect=1.0,
        prompt_renderer=lambda text: text,
    )

    assert len(samples) == 1
    assert samples[0]["case_index"] == 1
    assert samples[0]["completion_text"] == "1"


def test_question_prior_too_hard_requires_min_seen_count():
    trainer = object.__new__(CodeGRPOTrainer)
    trainer._question_prior_enabled = True
    trainer.args = SimpleNamespace(
        question_prior_high_threshold=0.7,
        question_prior_low_threshold=0.3,
        question_prior_gap_threshold=0.2,
        question_prior_weight_mastered=0.4,
        question_prior_weight_too_hard=0.2,
        question_prior_weight_high_value=1.0,
        question_prior_weight_mid_negative_gap=0.7,
        question_prior_weight_default=0.8,
        question_prior_min_seen_before_too_hard=3,
    )

    early_state = SimpleNamespace(
        ema_code_success=0.1,
        ema_reason_signal=0.1,
        ema_learning_value=1.0,
        seen_count=2,
    )
    mature_state = SimpleNamespace(
        ema_code_success=0.1,
        ema_reason_signal=0.1,
        ema_learning_value=1.0,
        seen_count=3,
    )

    assert trainer._compute_question_prior_weight_from_state(early_state) == 0.8
    assert trainer._compute_question_prior_weight_from_state(mature_state) == 0.2


def test_iterative_node_priority_prefers_mid_pass_rate_over_nearly_solved():
    trainer = object.__new__(CodeGRPOTrainer)
    trainer.args = SimpleNamespace(pseudo_iterative_soft_priority_bonus_scale=0.1)
    trainer._get_question_prior_weight = lambda qid: 1.0

    mid_priority = trainer._compute_iterative_node_priority(
        {"pass_rate": 0.5, "raw_soft_reward": 0.0},
        "q1",
    )
    near_solved_priority = trainer._compute_iterative_node_priority(
        {"pass_rate": 0.875, "raw_soft_reward": 0.0},
        "q1",
    )
    zero_pass_soft_priority = trainer._compute_iterative_node_priority(
        {"pass_rate": 0.0, "raw_soft_reward": 0.3},
        "q1",
    )

    assert mid_priority > near_solved_priority
    assert zero_pass_soft_priority > 1e-6
    assert zero_pass_soft_priority < mid_priority


def test_compute_reference_kl_metric_reuses_precomputed_ref_logps():
    trainer = object.__new__(CodeGRPOTrainer)
    trainer.model = SimpleNamespace(training=True)
    trainer.args = SimpleNamespace(log_kl_metrics=True)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.accelerator = SimpleNamespace(gather=lambda tensor: tensor.unsqueeze(0))
    trainer._should_record_kl_metric = lambda mode: True

    def fail_if_called(**kwargs):
        raise AssertionError("ref forward should be reused, not recomputed")

    trainer._get_reference_per_token_logps = fail_if_called

    per_token_logps = trainer_utils.torch.tensor([[0.1, 0.2]], dtype=trainer_utils.torch.float32)
    ref_per_token_logps = trainer_utils.torch.tensor([[0.15, 0.25]], dtype=trainer_utils.torch.float32)
    completion_mask = trainer_utils.torch.tensor([[1.0, 1.0]], dtype=trainer_utils.torch.float32)

    value = trainer._compute_reference_kl_metric(
        per_token_logps=per_token_logps,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
    )

    assert isinstance(value, float)


def test_configure_run_layout_reuses_existing_checkpoint_run():
    base_output_dir = Path("trl/tests/tmp_resume_layout") / f"run_{random.randint(0, 1_000_000)}" / "runs"
    checkpoint_dir = base_output_dir / "train" / "existing-run" / "train_out" / "checkpoint-40"
    checkpoint_dir.mkdir(parents=True)
    manifest_path = checkpoint_dir.parent.parent / "run_manifest.json"
    manifest_path.write_text(
        """
{
  "run_id": "20260324_000000__train__qwen2.5-coder-7b-instruct__dataset__vllm",
  "mode": "train",
  "paths": {"mode": "train"},
  "script_args": {
    "dataset_name": null,
    "dataset_train_split": null,
    "dataset_test_split": null,
    "dataset_adapter": null
  },
  "training_args": {
    "codegrpo_mode": "train",
    "backend": "vllm",
    "zero_pass_soft_reward_enabled": null,
    "pseudo_multiround_enabled": null,
    "question_prior_enabled": null,
    "K": null,
    "generation_batch_size": null,
    "max_completion_length": null,
    "max_completion_length_code": null,
    "max_steps": null,
    "eval_steps": null,
    "seed": null,
    "data_seed": null
  },
  "model_args": {"model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct"}
}
""".strip(),
        encoding="utf-8",
    )

    script_args = SimpleNamespace(dataset_name=None)
    training_args = SimpleNamespace(
        output_dir=str(base_output_dir),
        codegrpo_mode="train",
        backend="vllm",
        tensorboard_root_dir=None,
        logging_dir=None,
        debug_trace_dir=None,
        run_name=None,
        resume_from_checkpoint=True,
    )
    model_args = SimpleNamespace(model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct")
    dataset_args = SimpleNamespace(datasets=[])

    layout = code_grpo_script._configure_run_layout(script_args, training_args, model_args, dataset_args)

    assert layout["resume_checkpoint_dir"] == str(checkpoint_dir.resolve())
    assert layout["run_id"] == "existing-run"
    assert training_args.output_dir == str(checkpoint_dir.parent.resolve())


def test_configure_run_layout_ignores_mismatched_checkpoint_run():
    base_output_dir = Path("trl/tests/tmp_resume_layout") / f"run_{random.randint(0, 1_000_000)}" / "runs"
    checkpoint_dir = base_output_dir / "train" / "old-run" / "train_out" / "checkpoint-40"
    checkpoint_dir.mkdir(parents=True)
    manifest_path = checkpoint_dir.parent.parent / "run_manifest.json"
    manifest_path.write_text(
        """
{
  "run_id": "old-run",
  "mode": "train",
  "paths": {"mode": "train"},
  "script_args": {
    "dataset_name": null,
    "dataset_train_split": "train",
    "dataset_test_split": "validation",
    "dataset_adapter": "default"
  },
  "training_args": {
    "codegrpo_mode": "train",
    "backend": "vllm",
    "zero_pass_soft_reward_enabled": false,
    "pseudo_multiround_enabled": false,
    "question_prior_enabled": false,
    "K": 8,
    "generation_batch_size": 16,
    "max_completion_length": 512,
    "max_completion_length_code": 512,
    "max_steps": 120,
    "eval_steps": 20,
    "seed": 42,
    "data_seed": 42
  },
  "model_args": {"model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct"}
}
""".strip(),
        encoding="utf-8",
    )

    script_args = SimpleNamespace(
        dataset_name=None,
        dataset_train_split="train",
        dataset_test_split="validation",
        dataset_adapter="default",
    )
    training_args = SimpleNamespace(
        output_dir=str(base_output_dir),
        codegrpo_mode="train",
        backend="vllm",
        tensorboard_root_dir=None,
        logging_dir=None,
        debug_trace_dir=None,
        run_name=None,
        resume_from_checkpoint=True,
        zero_pass_soft_reward_enabled=True,
        pseudo_multiround_enabled=False,
        question_prior_enabled=False,
        K=8,
        generation_batch_size=16,
        max_completion_length=512,
        max_completion_length_code=512,
        max_steps=120,
        eval_steps=20,
        seed=42,
        data_seed=42,
    )
    model_args = SimpleNamespace(model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct")
    dataset_args = SimpleNamespace(datasets=[])

    layout = code_grpo_script._configure_run_layout(script_args, training_args, model_args, dataset_args)

    assert layout["resume_checkpoint_dir"] is None
    assert layout["run_id"] != "old-run"


def test_adapt_splits_can_skip_train_split_for_test_mode():
    class DummyAdapter:
        def adapt_dataset(self, split):
            return split

    dataset = {"validation": [1, 2, 3]}
    train_dataset, eval_dataset = code_grpo_script._adapt_splits(
        dataset=dataset,
        adapter=DummyAdapter(),
        train_split="train",
        test_split="validation",
        load_train_split=False,
    )

    assert train_dataset is None
    assert eval_dataset == [1, 2, 3]


def test_baseline_eval_support_requires_hf_and_peft():
    training_args = SimpleNamespace(backend="vllm")
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
        model=SimpleNamespace(disable_adapter=lambda: None),
    )
    supported, reason = code_grpo_script._baseline_eval_is_supported(training_args, trainer)
    assert not supported
    assert "vllm" in reason

    training_args = SimpleNamespace(backend="hf")
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(unwrap_model=lambda model: object()),
        model=object(),
    )
    supported, reason = code_grpo_script._baseline_eval_is_supported(training_args, trainer)
    assert not supported
    assert "PEFT" in reason
