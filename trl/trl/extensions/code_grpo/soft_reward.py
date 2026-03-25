import json
import math
from typing import Any

from .prompts import build_zero_pass_code_view_prompt, build_zero_pass_problem_view_prompt


def _normalize_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(value).strip()


def _case_complexity_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(value).strip()


def _case_complexity_key(case_input: Any, case_output: Any, index: int) -> tuple[int, int, int, int, int]:
    input_text = _case_complexity_text(case_input)
    output_text = _case_complexity_text(case_output)
    total_len = len(input_text) + len(output_text)
    max_side_len = max(len(input_text), len(output_text))
    line_count = input_text.count("\n") + output_text.count("\n") + 2
    digit_count = sum(ch.isdigit() for ch in input_text) + sum(ch.isdigit() for ch in output_text)
    return (total_len, max_side_len, line_count, digit_count, index)


def select_simple_diagnostic_pairs(problem: dict[str, Any], max_count: int = 0) -> list[tuple[Any, Any]]:
    explicit_inputs = list(problem.get("diagnostic_inputs", []) or [])
    explicit_outputs = list(problem.get("diagnostic_outputs", []) or [])
    candidates: list[tuple[Any, Any]] = []
    if explicit_inputs and explicit_outputs:
        candidates = list(zip(explicit_inputs, explicit_outputs, strict=False))
    else:
        test_cases = list(problem.get("test_cases", []) or [])
        candidates = [(case.get("input"), case.get("output")) for case in test_cases]

    ranked = sorted(
        enumerate(candidates),
        key=lambda item: _case_complexity_key(item[1][0], item[1][1], item[0]),
    )
    selected = [pair for _idx, pair in ranked]
    if max_count > 0:
        selected = selected[:max_count]
    return selected


def build_diagnostic_inputs(problem: dict[str, Any], max_count: int = 0) -> list[Any]:
    return [case_input for case_input, _case_output in select_simple_diagnostic_pairs(problem, max_count=max_count)]


def get_oracle_outputs(problem: dict[str, Any], diagnostic_inputs: list[Any]) -> list[Any]:
    if not diagnostic_inputs:
        return []
    return [
        case_output
        for _case_input, case_output in select_simple_diagnostic_pairs(problem, max_count=len(diagnostic_inputs))
    ]


def normalize_soft_reward_to_unit_interval(raw_value: float, clip_low: float, clip_high: float) -> float:
    clipped = max(float(clip_low), min(float(clip_high), float(raw_value)))
    width = float(clip_high) - float(clip_low)
    if width <= 0:
        return 0.5
    return (clipped - float(clip_low)) / width


def compute_zero_pass_beta(test_count: int, beta_scale: float) -> float:
    if test_count <= 0:
        return 0.0
    safe_cap = (1.0 - 1e-6) / float(test_count)
    return min(float(beta_scale) / float(test_count), safe_cap)


def compute_soft_reward(
    problem: dict[str, Any],
    code: str,
    diagnostic_inputs: list[Any],
    oracle_outputs: list[Any],
    evaluator,
    problem_logprob_cache: dict[tuple[str, str], float] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    question_prompt = str(problem.get("prompt", ""))
    details: list[dict[str, Any]] = []
    deltas: list[float] = []

    for case_input, oracle_output in zip(diagnostic_inputs, oracle_outputs, strict=False):
        target_text = _normalize_output_text(oracle_output)
        problem_prompt = build_zero_pass_problem_view_prompt(question_prompt=question_prompt, case_input=case_input)
        code_prompt = build_zero_pass_code_view_prompt(code=code, case_input=case_input)
        cache_key = (str(case_input), target_text)
        s_prob = None
        if problem_logprob_cache is not None:
            s_prob = problem_logprob_cache.get(cache_key)
        if s_prob is None:
            s_prob = float(evaluator.logprob(problem_prompt, target_text))
            if problem_logprob_cache is not None and math.isfinite(s_prob):
                problem_logprob_cache[cache_key] = s_prob
        if not math.isfinite(float(s_prob)):
            details.append(
                {
                    "input": case_input,
                    "oracle_output": oracle_output,
                    "target_text": target_text,
                    "s_prob": s_prob,
                    "s_code": float("nan"),
                    "delta": 0.0,
                    "skipped": "problem_logprob_unavailable",
                }
            )
            continue
        s_code = float(evaluator.logprob(code_prompt, target_text))
        if not math.isfinite(s_code):
            details.append(
                {
                    "input": case_input,
                    "oracle_output": oracle_output,
                    "target_text": target_text,
                    "s_prob": s_prob,
                    "s_code": s_code,
                    "delta": 0.0,
                    "skipped": "code_logprob_unavailable",
                }
            )
            continue
        delta = s_code - s_prob
        deltas.append(delta)
        details.append(
            {
                "input": case_input,
                "oracle_output": oracle_output,
                "target_text": target_text,
                "s_prob": s_prob,
                "s_code": s_code,
                "delta": delta,
            }
        )

    raw_soft_reward = (sum(deltas) / len(deltas)) if deltas else float("nan")
    return raw_soft_reward, details
