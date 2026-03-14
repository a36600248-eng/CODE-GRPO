import math
import random
import ast
import difflib
from typing import Any

from .error_utils import summarize_error
from .executor import execute_batch
from .matcher import is_match, values_equal
from .parser import (
    build_generation_completion,
    build_token_masks,
    parse_exec_response,
    parse_generation_response,
    parse_logic_response,
)
from .prompts import (
    build_exec_prompt,
    build_frozen_reason_prompt,
    build_generation_prompt,
    build_logic_prompt,
    summarize_generation_history,
)
from .types import ExecResult, Node, QuestionRollout, TrainSample


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def _code_status(exec_results: list[ExecResult], pass_all: bool) -> str:
    if pass_all:
        return "CORRECT"
    if any(res.kind == "SYNTAX_ERROR" for res in exec_results):
        return "SYNTAX_ERROR"
    if any(res.kind == "TIMEOUT" for res in exec_results):
        return "TIMEOUT"
    if any(res.kind == "RUNTIME_ERROR" for res in exec_results):
        return "RUNTIME_ERROR"
    return "FAIL"


def _compute_code_reward(
    pass_rate: float,
    r_soft: float,
    lambda_soft: float,
    compile_score: float,
    compile_scale: float,
    generation_format_score: float,
    generation_format_scale: float,
) -> float:
    """Linear additive main-generation reward."""
    return _clamp01(
        pass_rate
        + _clamp01(lambda_soft) * _clamp01(r_soft)
        + _clamp01(compile_scale) * _clamp01(compile_score)
        + _clamp01(generation_format_scale) * _clamp01(generation_format_score)
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _apply_format_shaping(correct_score: float, format_ok: bool, penalty: float, bonus: float = 0.0) -> float:
    # Keep format as an independent additive term:
    # - valid format always gets a reward bump
    # - invalid format gets an additive deduction
    # This avoids coupling format compliance to task correctness via multiplication.
    format_term = _clamp01(bonus) if format_ok else -_clamp01(penalty)
    return _clamp01(correct_score + format_term)


def _safe_preview(value: Any, max_chars: int = 200) -> str:
    text = repr(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _compute_advantage_diagnostics(nodes: list[Node], eps: float = 1e-12) -> dict[str, float]:
    code_nodes = [node for node in nodes if node.completion_text]
    sibling_groups: dict[str, list[Node]] = {}
    for node in code_nodes:
        if node.parent_id is None:
            continue
        sibling_groups.setdefault(str(node.parent_id), []).append(node)

    groups = list(sibling_groups.values())
    group_stds = [_std([node.R_code for node in group]) for group in groups]
    group_unique_counts = [
        float(len({round(float(node.R_code), 12) for node in group}))
        for group in groups
    ]
    group_sizes = [float(len(group)) for group in groups]
    group_spans = [
        max((float(node.R_code) for node in group), default=0.0) - min((float(node.R_code) for node in group), default=0.0)
        for group in groups
    ]

    return {
        "nonzero_A_code_rate": _mean([1.0 if abs(float(node.A_code)) > eps else 0.0 for node in code_nodes]),
        "nonzero_A_reason_rate": _mean([1.0 if abs(float(node.A_reason)) > eps else 0.0 for node in nodes]),
        "sibling_group_count": float(len(groups)),
        "sibling_group_mean_size": _mean(group_sizes),
        "sibling_group_zero_std_R_code_rate": _mean([1.0 if std <= eps else 0.0 for std in group_stds]),
        "sibling_group_nonzero_std_R_code_rate": _mean([1.0 if std > eps else 0.0 for std in group_stds]),
        "sibling_group_mean_std_R_code": _mean(group_stds),
        "sibling_group_mean_unique_R_code_count": _mean(group_unique_counts),
        "sibling_group_mean_reward_span": _mean(group_spans),
        "sibling_group_all_pass_rate": _mean(
            [1.0 if all(float(node.pass_rate) == 1.0 for node in group) else 0.0 for group in groups]
        ),
        "sibling_group_all_fail_rate": _mean(
            [1.0 if all(float(node.pass_rate) < 1.0 for node in group) else 0.0 for group in groups]
        ),
    }


def _is_double_zero_node(node: Node, eps: float = 1e-12) -> bool:
    """Return True when both rewards are (numerically) zero."""
    return abs(node.R_code) <= eps and abs(node.R_reason) <= eps


def _parse_case_input(value: Any) -> Any:
    """Parse serialized case input while keeping plain strings safe."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return ast.literal_eval(text)
    except Exception:  # noqa: BLE001
        return value


def _has_solve_entrypoint(code: str) -> bool:
    """Return True when code defines a top-level `solve` function."""
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        module = ast.parse(code)
    except Exception:  # noqa: BLE001
        return False
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "solve":
            return True
    return False


def _build_exec_audit_completion(reasoning: str, exec_prediction: str) -> str:
    return (
        "<REASON>\n"
        f"{(reasoning or '').strip()}\n"
        "</REASON>\n"
        "<EXEC_PREDICTION>\n"
        f"{(exec_prediction or '').strip()}\n"
        "</EXEC_PREDICTION>"
    )


def _build_logic_audit_completion(reasoning: str, logic_prediction: str) -> str:
    return (
        "<REASON>\n"
        f"{(reasoning or '').strip()}\n"
        "</REASON>\n"
        "<LOGIC_PREDICTION>\n"
        f"{(logic_prediction or '').strip()}\n"
        "</LOGIC_PREDICTION>"
    )


def _canonical_code_repr(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        return ""
    try:
        tree = ast.parse(code)
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except Exception:  # noqa: BLE001
        return "\n".join(line.strip() for line in code.splitlines() if line.strip())


def _code_similarity(code_a: str, code_b: str) -> float:
    a = _canonical_code_repr(code_a)
    b = _canonical_code_repr(code_b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


class CodeGRPOTreeRunner:
    def __init__(self, backend, tokenizer, args, logger):
        self.backend = backend
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger
        self._exec_case_baseline_ema: dict[str, float] = {}

    def _render_prompt_with_chat_template(self, prompt_text: str) -> str:
        if not getattr(self.args, "use_chat_template_for_codegrpo", True):
            return prompt_text
        apply_fn = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_fn is None:
            return prompt_text
        try:
            rendered = apply_fn(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:  # noqa: BLE001
            pass
        return prompt_text

    def _code_completion_length(self) -> int:
        return int(getattr(self.args, "max_completion_length_code", None) or self.args.max_completion_length)

    def _audit_completion_length(self) -> int:
        return int(getattr(self.args, "max_completion_length_audit", None) or self.args.max_completion_length)

    def _code_generation_kwargs(self, *, eval_mode: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_new_tokens": self._code_completion_length()}
        min_new_tokens = int(getattr(self.args, "generation_min_new_tokens_code", 0) or 0)
        if min_new_tokens > 0:
            kwargs["min_new_tokens"] = min_new_tokens
        temperature_attr = "eval_generation_temperature_code" if eval_mode else "generation_temperature_code"
        temperature = getattr(self.args, temperature_attr, None)
        if temperature is None:
            temperature = getattr(self.args, "generation_temperature_code", None)
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        top_p = getattr(self.args, "generation_top_p_code", None)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        return kwargs

    def _retry_empty_generation_outputs(
        self,
        prompt_text: str,
        raw_outputs: list[str],
        *,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        retry_count = int(getattr(self.args, "generation_empty_retry_count", 0) or 0)
        if retry_count <= 0 or not raw_outputs:
            return raw_outputs
        fixed = list(raw_outputs)
        gen_kwargs = dict(generation_kwargs or self._code_generation_kwargs())
        for idx, raw_output in enumerate(fixed):
            if (raw_output or "").strip():
                continue
            for _ in range(retry_count):
                retried = self.backend.generate(prompt_text, **gen_kwargs)
                if (retried or "").strip():
                    fixed[idx] = retried
                    break
        return fixed

    def _apply_terminal_logic_backprop(self, nodes: list[Node], node_by_id: dict[str, Node]) -> None:
        solved_nodes = [node for node in nodes if node.pass_rate == 1.0]
        if not solved_nodes or self.args.terminal_logic_backprop_bonus <= 0.0:
            return

        threshold = self.args.terminal_backprop_code_similarity_threshold
        max_depth = self.args.terminal_logic_backprop_max_depth
        decay = self.args.terminal_logic_backprop_decay
        bonus_scale = self.args.terminal_logic_backprop_bonus

        bonus_by_node: dict[str, float] = {}
        rewarded_ancestors: set[str] = set()
        for solved in solved_nodes:
            child = solved
            depth = 0
            while child.parent_id and child.parent_id in node_by_id and depth < max_depth:
                parent = node_by_id[child.parent_id]
                similarity = _code_similarity(parent.code or "", child.code or "")
                if similarity < threshold:
                    parent.exec_summary["terminal_backprop_stop_reason"] = "code_drift"
                    parent.exec_summary["terminal_backprop_stop_similarity"] = similarity
                    break

                case_bonus = bonus_scale * (decay**depth)
                logic_samples = list(parent.exec_summary.get("logic_train_samples", []))
                if logic_samples and parent.node_id not in rewarded_ancestors:
                    node_bonus = 0.0
                    hit_count = 0
                    for item in logic_samples:
                        if not bool(item.get("match", False)):
                            continue
                        bonus = _clamp01(case_bonus)
                        item["terminal_bonus"] = _clamp01(float(item.get("terminal_bonus", 0.0)) + bonus)
                        item["confirmed"] = True
                        node_bonus += bonus
                        hit_count += 1
                    parent.exec_summary["logic_train_samples"] = logic_samples
                    if hit_count > 0:
                        node_bonus = node_bonus / max(1, len(logic_samples))
                        bonus_by_node[parent.node_id] = bonus_by_node.get(parent.node_id, 0.0) + node_bonus
                        rewarded_ancestors.add(parent.node_id)

                parent.exec_summary["terminal_backprop_last_similarity"] = similarity
                child = parent
                depth += 1

        for node in nodes:
            bonus = _clamp01(bonus_by_node.get(node.node_id, 0.0))
            if bonus <= 0.0:
                continue
            node.exec_summary["terminal_backprop_bonus"] = float(node.exec_summary.get("terminal_backprop_bonus", 0.0)) + bonus

    def _recompute_all_advantages(self, nodes: list[Node]) -> None:
        sibling_groups: dict[str, list[Node]] = {}
        for node in nodes:
            if node.parent_id is None:
                continue
            sibling_groups.setdefault(node.parent_id, []).append(node)
        for siblings in sibling_groups.values():
            self._assign_group_advantages(siblings, update_exec_baseline=False)

    def _refresh_round_payloads(self, rounds: list[dict[str, Any]], node_by_id: dict[str, Node]) -> None:
        for round_item in rounds:
            for node_payload in round_item.get("nodes", []):
                node = node_by_id.get(str(node_payload.get("node_id", "")))
                if node is None:
                    continue
                node_payload["R_code"] = node.R_code
                node_payload["R_reason"] = node.R_reason
                node_payload["R_soft_raw"] = float(node.exec_summary.get("R_soft_raw", 0.0))
                node_payload["R_soft_match_raw"] = float(node.exec_summary.get("R_soft_match_raw", 0.0))
                node_payload["R_soft_effective"] = float(node.exec_summary.get("R_soft_effective", 0.0))
                node_payload["soft_reward_eligible"] = bool(node.exec_summary.get("soft_reward_eligible", False))
                node_payload["compile_score"] = float(node.exec_summary.get("compile_score", 0.0))
                node_payload["generation_format_ok"] = bool(node.exec_summary.get("generation_format_ok", False))
                node_payload["generation_format_score"] = float(node.exec_summary.get("generation_format_score", 0.0))
                node_payload["terminal_backprop_bonus"] = float(node.exec_summary.get("terminal_backprop_bonus", 0.0))
                node_payload["final_reason_stage"] = bool(node.exec_summary.get("final_reason_stage", False))
                node_payload["reason_only"] = bool(node.exec_summary.get("reason_only", False))
                node_payload["main_sample_active"] = bool(node.completion_text)
                node_payload["logic_sample_count"] = len(node.exec_summary.get("logic_train_samples", []))
                node_payload["exec_sample_count"] = len(node.exec_summary.get("exec_train_samples", []))

    def _build_round_record(self, round_idx: int, nodes: list[Node], stage: str = "search") -> dict[str, Any]:
        return {
            "round": round_idx,
            "stage": stage,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "pass_rate": node.pass_rate,
                    "status_code": node.status_code,
                    "R_code": node.R_code,
                    "R_reason": node.R_reason,
                    "R_soft_raw": float(node.exec_summary.get("R_soft_raw", 0.0)),
                    "R_soft_match_raw": float(node.exec_summary.get("R_soft_match_raw", 0.0)),
                    "R_soft_effective": float(node.exec_summary.get("R_soft_effective", 0.0)),
                    "soft_reward_eligible": bool(node.exec_summary.get("soft_reward_eligible", False)),
                    "compile_score": float(node.exec_summary.get("compile_score", 0.0)),
                    "generation_format_ok": bool(node.exec_summary.get("generation_format_ok", False)),
                    "generation_format_score": float(node.exec_summary.get("generation_format_score", 0.0)),
                    "terminal_backprop_bonus": float(node.exec_summary.get("terminal_backprop_bonus", 0.0)),
                    "final_reason_stage": bool(node.exec_summary.get("final_reason_stage", False)),
                    "reason_only": bool(node.exec_summary.get("reason_only", False)),
                    "main_sample_active": bool(node.completion_text),
                    "logic_sample_count": len(node.exec_summary.get("logic_train_samples", [])),
                    "exec_sample_count": len(node.exec_summary.get("exec_train_samples", [])),
                    "frozen_code": node.frozen_code,
                    "frozen_reason": node.frozen_reason,
                    "pruned_double_zero": node.exec_summary.get("pruned_reason", "") == "double_zero_rewards",
                    "logic_format_ok_rate": node.exec_summary.get("logic_format_ok_rate", 0.0),
                    "exec_format_ok_rate": node.exec_summary.get("exec_format_ok_rate", 0.0),
                    "generation_debug": node.exec_summary.get("generation_debug", {}),
                    "logic_audit": node.exec_summary.get("logic_audit", []),
                    "exec_audit": node.exec_summary.get("exec_audit", []),
                    "code_preview": summarize_error(
                        node.code or "",
                        max_chars=self.args.error_max_chars,
                        max_lines=self.args.error_max_lines,
                    ),
                    "error_summary": node.exec_summary.get("error_summary", ""),
                }
                for node in nodes
            ],
        }

    @staticmethod
    def _case_baseline_key(question_id: str, case_index: int) -> str:
        return f"{question_id}::{case_index}"

    def _get_exec_case_baseline(self, question_id: str, case_index: int) -> float:
        return float(self._exec_case_baseline_ema.get(self._case_baseline_key(question_id, case_index), 0.5))

    def _update_exec_case_baseline(self, question_id: str, case_index: int, score: float) -> None:
        key = self._case_baseline_key(question_id, case_index)
        alpha = _clamp01(float(getattr(self.args, "exec_case_baseline_ema_alpha", 0.1)))
        old = float(self._exec_case_baseline_ema.get(key, 0.5))
        self._exec_case_baseline_ema[key] = _clamp01((1.0 - alpha) * old + alpha * _clamp01(score))

    def _run_final_reason_stage(
        self,
        *,
        question_id: str,
        parents: list[Node],
        prompt: str,
        test_cases: list[dict[str, Any]],
        audit_indices: list[int],
        round_idx: int,
        node_serial: int,
        update_exec_baseline: bool,
    ) -> tuple[list[Node], int, int]:
        final_nodes: list[Node] = []
        produced_total = 0

        for parent in parents:
            siblings: list[Node] = []
            for _ in range(max(1, int(self.args.K_reason))):
                node_serial += 1
                siblings.append(
                    self._generate_reason_only_node(
                        question_id=question_id,
                        node_id=f"n{node_serial}",
                        parent=parent,
                        prompt=prompt,
                        test_cases=test_cases,
                        audit_indices=audit_indices,
                        round_idx=round_idx,
                    )
                )
            if siblings:
                self._assign_group_advantages(siblings, update_exec_baseline=update_exec_baseline)
            produced = len(siblings)
            for node in siblings:
                node.exec_summary["final_reason_stage"] = True
            final_nodes.extend(siblings)
            produced_total += produced

        return final_nodes, produced_total, node_serial

    def _generate_reason_only_node(
        self,
        *,
        question_id: str,
        node_id: str,
        parent: Node,
        prompt: str,
        test_cases: list[dict[str, Any]],
        audit_indices: list[int],
        round_idx: int,
    ) -> Node:
        code = parent.code or ""
        reasoning = parent.reasoning or ""
        trace_max_chars = max(self.args.error_max_chars * 4, self.args.error_max_chars)
        trace_max_lines = max(self.args.error_max_lines * 4, self.args.error_max_lines)
        trace_store_full_text = bool(getattr(self.args, "trace_store_full_text", False))

        logic_scores: list[float] = []
        logic_match_scores: list[float] = []
        logic_format_flags: list[float] = []
        logic_audit_details: list[dict[str, Any]] = []
        logic_train_samples: list[dict[str, Any]] = []

        exec_scores: list[float] = []
        exec_format_flags: list[float] = []
        exec_raw_correct_flags: list[float] = []
        exec_audit_details: list[dict[str, Any]] = []
        exec_train_samples: list[dict[str, Any]] = []
        reason_only_prompt_previews: list[str] = []

        if audit_indices:
            case_inputs = [_parse_case_input(test_cases[idx]["input"]) for idx in audit_indices]
            actual_results = execute_batch(
                code=code,
                case_inputs=case_inputs,
                timeout_s=self.args.code_timeout_seconds,
                error_max_chars=self.args.error_max_chars,
                error_max_lines=self.args.error_max_lines,
            )
            unified_prompts_raw = [build_frozen_reason_prompt(code, case_input) for case_input in case_inputs]
            unified_prompts = [self._render_prompt_with_chat_template(p) for p in unified_prompts_raw]
            unified_outputs = self.backend.generate_many(
                unified_prompts,
                max_new_tokens=self._audit_completion_length(),
            )

            for idx, case_input, case, actual, unified_prompt, unified_output in zip(
                audit_indices,
                case_inputs,
                [test_cases[idx] for idx in audit_indices],
                actual_results,
                unified_prompts,
                unified_outputs,
                strict=True,
            ):
                prompt_preview = summarize_error(unified_prompt, trace_max_chars, trace_max_lines)
                reason_only_prompt_previews.append(prompt_preview)
                exec_reason, exec_prediction, format_ok = parse_exec_response(
                    unified_output,
                    require_reason_before_prediction=self.args.require_reason_before_prediction,
                    prediction_max_chars=self.args.prediction_max_chars,
                    reason_max_chars=self.args.reasoning_max_chars,
                    disallow_code_in_reasoning=self.args.disallow_code_in_reasoning,
                    allow_outside_noise_chars=self.args.format_outside_noise_chars,
                )

                logic_correct = 1.0 if values_equal(exec_prediction, case["output"]) else 0.0
                logic_format_term = self.args.format_bonus_logic if format_ok else -self.args.format_penalty_logic
                logic_score = _apply_format_shaping(
                    logic_correct,
                    format_ok,
                    penalty=self.args.format_penalty_logic,
                    bonus=self.args.format_bonus_logic,
                )
                logic_scores.append(logic_score)
                logic_match_scores.append(logic_correct)
                logic_format_flags.append(1.0 if format_ok else 0.0)
                logic_completion_text = unified_output.strip() or _build_logic_audit_completion(exec_reason, exec_prediction)
                logic_train_samples.append(
                    {
                        "question_id": question_id,
                        "case_index": idx,
                        "prompt_text": unified_prompt,
                        "completion_text": logic_completion_text,
                        "score": logic_score,
                        "match": bool(logic_correct),
                        "format_ok": bool(format_ok),
                        "confirmed": bool(logic_correct),
                        "terminal_bonus": 0.0,
                        "advantage": 0.0,
                    }
                )
                logic_audit_details.append(
                    {
                        "case_index": idx,
                        "input": _safe_preview(case_input, max_chars=400),
                        "expected_output": _safe_preview(case["output"], max_chars=400),
                        "raw_output": summarize_error(unified_output, trace_max_chars, trace_max_lines),
                        "parsed_reason": summarize_error(exec_reason, trace_max_chars, trace_max_lines),
                        "parsed_prediction": _safe_preview(exec_prediction, max_chars=400),
                        "prompt_preview": prompt_preview,
                        "format_ok": bool(format_ok),
                        "match": bool(logic_correct),
                        "correct_raw": logic_correct,
                        "format_term": logic_format_term,
                        "score_after_penalty": logic_score,
                    }
                )

                exec_correct = 1.0 if is_match(exec_prediction, actual) else 0.0
                exec_format_term = self.args.format_bonus_exec if format_ok else -self.args.format_penalty_exec
                exec_raw_correct_flags.append(exec_correct)
                exec_score = _apply_format_shaping(
                    exec_correct,
                    format_ok,
                    penalty=self.args.format_penalty_exec,
                    bonus=self.args.format_bonus_exec,
                )
                exec_scores.append(exec_score)
                exec_format_flags.append(1.0 if format_ok else 0.0)
                exec_completion_text = unified_output.strip() or _build_exec_audit_completion(exec_reason, exec_prediction)
                exec_train_samples.append(
                    {
                        "question_id": question_id,
                        "case_index": idx,
                        "prompt_text": unified_prompt,
                        "completion_text": exec_completion_text,
                        "score": exec_score,
                        "match": bool(exec_correct),
                        "format_ok": bool(format_ok),
                        "advantage": 0.0,
                        "advantage_source": "",
                    }
                )
                exec_audit_details.append(
                    {
                        "case_index": idx,
                        "input": _safe_preview(case_input, max_chars=400),
                        "actual_kind": actual.kind,
                        "actual_value": _safe_preview(actual.value, max_chars=400) if actual.kind == "OK" else "",
                        "actual_error_type": actual.error_type or "",
                        "actual_error_msg": summarize_error(actual.error_msg or "", trace_max_chars, trace_max_lines),
                        "raw_output": summarize_error(unified_output, trace_max_chars, trace_max_lines),
                        "parsed_reason": summarize_error(exec_reason, trace_max_chars, trace_max_lines),
                        "parsed_prediction": _safe_preview(exec_prediction, max_chars=400),
                        "prompt_preview": prompt_preview,
                        "format_ok": bool(format_ok),
                        "match": bool(exec_correct),
                        "correct_raw": exec_correct,
                        "format_term": exec_format_term,
                        "score_after_penalty": exec_score,
                    }
                )

        R_soft = _mean(logic_scores) if logic_scores else 0.0
        R_soft_match_raw = _mean(logic_match_scores) if logic_match_scores else 0.0
        logic_format_ok_rate = _mean(logic_format_flags) if logic_format_flags else 0.0
        R_reason = _mean(exec_scores) if exec_scores else 0.0
        exec_format_ok_rate = _mean(exec_format_flags) if exec_format_flags else 0.0
        status_reason = "NOT_RUN" if not exec_scores else ("HONEST" if all(score == 1.0 for score in exec_raw_correct_flags) else "HALLUCINATION")
        logic_mismatch_count = int(sum(1 for item in logic_audit_details if not item.get("match", False)))
        exec_mismatch_count = int(sum(1 for item in exec_audit_details if not item.get("match", False)))
        logic_failed = next((item for item in logic_audit_details if not item.get("match", False)), None)
        exec_failed = next((item for item in exec_audit_details if not item.get("match", False)), None)

        inherited_history = list(parent.exec_summary.get("history", []))
        child = Node(
            node_id=node_id,
            parent_id=parent.node_id,
            round_idx=round_idx,
            code=code,
            reasoning=reasoning,
            pass_rate=parent.pass_rate,
            R_soft=R_soft,
            R_code=parent.R_code,
            R_reason=R_reason,
            status_code=parent.status_code,
            status_reason=status_reason,  # type: ignore[arg-type]
            frozen_code=True,
            frozen_reason=R_reason == 1.0,
            exec_summary={
                "error_summary": str(parent.exec_summary.get("error_summary", "")),
                "logic_format_ok_rate": logic_format_ok_rate,
                "exec_format_ok_rate": exec_format_ok_rate,
                "generation_format_ok": False,
                "generation_format_score": 0.0,
                "compile_score": float(parent.exec_summary.get("compile_score", 1.0)),
                "soft_reward_eligible": True,
                "R_soft_raw": R_soft,
                "R_soft_match_raw": R_soft_match_raw,
                "R_soft_effective": R_soft,
                "generation_debug": {
                    "raw_output": "",
                    "parsed_reason": "",
                    "parsed_logic_prediction": "",
                    "parsed_exec_prediction": "",
                    "reason_only": True,
                    "prompt_preview": summarize_error(
                        "\n\n".join(reason_only_prompt_previews[:2]),
                        trace_max_chars,
                        trace_max_lines,
                    ),
                },
                "logic_audit": logic_audit_details,
                "logic_train_samples": logic_train_samples,
                "exec_audit": exec_audit_details,
                "exec_train_samples": exec_train_samples,
                "history": inherited_history
                + [
                    {
                        "round": str(round_idx),
                        "status_code": parent.status_code,
                        "status_reason": status_reason,
                        "error_summary": str(parent.exec_summary.get("error_summary", "")),
                        "code_preview": summarize_error(
                            code,
                            max_chars=self.args.error_max_chars,
                            max_lines=self.args.error_max_lines,
                        ),
                        "logic_mismatch_count": logic_mismatch_count,
                        "exec_mismatch_count": exec_mismatch_count,
                        "logic_format_ok_rate": logic_format_ok_rate,
                        "exec_format_ok_rate": exec_format_ok_rate,
                        "generation_format_ok": False,
                        "compile_score": float(parent.exec_summary.get("compile_score", 1.0)),
                        "logic_failed_input": logic_failed.get("input") if logic_failed else None,
                        "logic_failed_prediction": logic_failed.get("parsed_prediction") if logic_failed else None,
                        "exec_failed_input": exec_failed.get("input") if exec_failed else None,
                        "exec_failed_prediction": exec_failed.get("parsed_prediction") if exec_failed else None,
                        "exec_actual_kind": exec_failed.get("actual_kind") if exec_failed else None,
                        "soft_reward_eligible": True,
                        "R_soft_raw": R_soft,
                        "R_soft_match_raw": R_soft_match_raw,
                        "R_soft_effective": R_soft,
                        "reason_only": True,
                    }
                ],
                "reason_only": True,
            },
            completion_text="",
            code_token_mask=[],
            reason_token_mask=[],
            prompt_text="",
        )
        return child

    def run_question(self, sample: dict[str, Any], rng: random.Random, update_exec_baseline: bool = True) -> QuestionRollout:
        question_id = str(sample["question_id"])
        prompt = str(sample["prompt"])
        test_cases = list(sample["test_cases"])
        test_size = len(test_cases)

        audit_count = min(self.args.M_audit, test_size)
        audit_indices = sorted(rng.sample(list(range(test_size)), k=audit_count)) if audit_count > 0 else []

        root = Node(node_id="root", parent_id=None, round_idx=0, code=None, reasoning=None, exec_summary={"history": []})
        frontier: list[Node] = [root]
        rounds: list[dict[str, Any]] = []
        collected_nodes: list[Node] = []
        node_by_id: dict[str, Node] = {root.node_id: root}

        node_count = 0
        resample_count = 0
        node_serial = 0
        solved = False
        final_reason_node_count = 0

        for round_idx in range(1, self.args.T_max + 1):
            if node_count >= self.args.N_max:
                break

            round_nodes: list[Node] = []
            next_frontier: list[Node] = []

            for parent in frontier:
                if node_count >= self.args.N_max:
                    break
                if parent.frozen_code and parent.frozen_reason:
                    continue

                siblings, produced, resampled, node_serial = self._expand_parent(
                    question_id=question_id,
                    parent=parent,
                    prompt=prompt,
                    test_cases=test_cases,
                    audit_indices=audit_indices,
                    round_idx=round_idx,
                    node_serial=node_serial,
                    budget=self.args.N_max - node_count,
                    update_exec_baseline=update_exec_baseline,
                )
                node_count += produced
                resample_count += resampled

                if not siblings:
                    continue

                round_nodes.extend(siblings)
                collected_nodes.extend(siblings)
                for node in siblings:
                    node_by_id[node.node_id] = node
                # Node-level pruning: double-zero nodes are recorded but do not enter the next round.
                for node in siblings:
                    if _is_double_zero_node(node):
                        node.exec_summary["pruned_reason"] = "double_zero_rewards"
                        continue
                    next_frontier.append(node)
                if any(node.pass_rate == 1.0 and node.R_reason == 1.0 for node in siblings):
                    solved = True

            if not round_nodes:
                break

            rounds.append(self._build_round_record(round_idx, round_nodes, stage="search"))

            code_pass_nodes = [node for node in next_frontier if node.pass_rate == 1.0]
            if code_pass_nodes:
                for node in next_frontier:
                    if node.pass_rate != 1.0:
                        node.exec_summary["pruned_reason"] = "stop_after_passed_code_found"

                final_reason_parents = [node for node in code_pass_nodes if not node.frozen_reason]
                if self.args.frozen_reason_one_shot:
                    final_round_idx = len(rounds) + 1
                    final_round_nodes: list[Node] = []
                    if final_reason_parents:
                        final_round_nodes, produced_final, node_serial = self._run_final_reason_stage(
                            question_id=question_id,
                            parents=final_reason_parents,
                            prompt=prompt,
                            test_cases=test_cases,
                            audit_indices=audit_indices,
                            round_idx=final_round_idx,
                            node_serial=node_serial,
                            update_exec_baseline=update_exec_baseline,
                        )
                        final_reason_node_count += produced_final
                        collected_nodes.extend(final_round_nodes)
                        for node in final_round_nodes:
                            node_by_id[node.node_id] = node
                        if final_round_nodes:
                            rounds.append(self._build_round_record(final_round_idx, final_round_nodes, stage="final_reason"))
                        if any(node.pass_rate == 1.0 and node.R_reason == 1.0 for node in final_round_nodes):
                            solved = True
                    if bool(getattr(self.args, "log_train_rollout_details", False)):
                        self.logger.info(
                            "[TREE] final_reason_stage search_round=%d pass_nodes=%d reason_parents=%d generated=%d",
                            round_idx,
                            len(code_pass_nodes),
                            len(final_reason_parents),
                            len(final_round_nodes),
                        )
                    break

                frontier = [node for node in code_pass_nodes if not (node.frozen_code and node.frozen_reason)]
                if bool(getattr(self.args, "log_train_rollout_details", False)):
                    self.logger.info(
                        "[TREE] passed_code_continue search_round=%d pass_nodes=%d next_frontier=%d",
                        round_idx,
                        len(code_pass_nodes),
                        len(frontier),
                    )
                if not frontier:
                    break
                continue

            frontier = [node for node in next_frontier if not (node.frozen_code and node.frozen_reason)]
            if solved:
                break

        # Terminal backprop: if a descendant solves the code task, propagate
        # additional reason credit to aligned ancestors with good logic prediction.
        self._apply_terminal_logic_backprop(collected_nodes, node_by_id)
        # Recompute all group advantages after terminal backprop so A values are consistent.
        self._recompute_all_advantages(collected_nodes)
        # Refresh per-round payload snapshots with post-processed rewards.
        self._refresh_round_payloads(rounds, node_by_id)

        train_samples: list[TrainSample] = []
        for node in collected_nodes:
            if node.completion_text:
                train_samples.append(
                    TrainSample(
                        question_id=question_id,
                        prompt_text=node.prompt_text,
                        completion_text=node.completion_text,
                        code_token_mask=node.code_token_mask,
                        reason_token_mask=node.reason_token_mask,
                        A_code=node.A_code,
                        A_reason=0.0,
                        R_code=node.R_code,
                        pass_rate=node.pass_rate,
                    )
                )

            # Logic-audit outputs are optimized by A_reason so logic-branch
            # format/terminal signals can improve reasoning behavior.
            for item in node.exec_summary.get("logic_train_samples", []):
                logic_prompt_text = str(item.get("prompt_text", ""))
                logic_completion_text = str(item.get("completion_text", ""))
                if not logic_prompt_text or not logic_completion_text:
                    continue
                token_ids = self.tokenizer(logic_completion_text, add_special_tokens=False)["input_ids"]
                if not token_ids:
                    continue
                logic_adv = float(item.get("advantage", 0.0))
                train_samples.append(
                    TrainSample(
                        question_id=question_id,
                        prompt_text=logic_prompt_text,
                        completion_text=logic_completion_text,
                        code_token_mask=[0] * len(token_ids),
                        reason_token_mask=[1] * len(token_ids),
                        A_code=0.0,
                        A_reason=logic_adv,
                        R_code=node.R_code,
                        pass_rate=node.pass_rate,
                    )
                )

            # Execution-audit outputs are optimized by A_reason (reason branch).
            for item in node.exec_summary.get("exec_train_samples", []):
                exec_prompt_text = str(item.get("prompt_text", ""))
                exec_completion_text = str(item.get("completion_text", ""))
                if not exec_prompt_text or not exec_completion_text:
                    continue
                token_ids = self.tokenizer(exec_completion_text, add_special_tokens=False)["input_ids"]
                if not token_ids:
                    continue
                exec_adv = float(item.get("advantage", 0.0))
                train_samples.append(
                    TrainSample(
                        question_id=question_id,
                        prompt_text=exec_prompt_text,
                        completion_text=exec_completion_text,
                        code_token_mask=[0] * len(token_ids),
                        reason_token_mask=[1] * len(token_ids),
                        A_code=0.0,
                        A_reason=exec_adv,
                        R_code=node.R_code,
                        pass_rate=node.pass_rate,
                    )
                )

        search_nodes = [node for node in collected_nodes if not bool(node.exec_summary.get("final_reason_stage", False))]
        final_reason_nodes = [node for node in collected_nodes if bool(node.exec_summary.get("final_reason_stage", False))]
        metric_nodes = search_nodes or collected_nodes

        r_code = [node.R_code for node in metric_nodes]
        r_reason = [node.R_reason for node in metric_nodes]
        pass_rates = [node.pass_rate for node in metric_nodes]
        eval_metrics = self._compute_eval_metrics(rounds)
        if collected_nodes:
            eval_metrics.update(_compute_advantage_diagnostics(metric_nodes))
            logic_items = [
                item for node in metric_nodes for item in node.exec_summary.get("logic_train_samples", [])
            ]
            exec_items = [
                item for node in metric_nodes for item in node.exec_summary.get("exec_train_samples", [])
            ]
            final_logic_items = [
                item for node in final_reason_nodes for item in node.exec_summary.get("logic_train_samples", [])
            ]
            final_exec_items = [
                item for node in final_reason_nodes for item in node.exec_summary.get("exec_train_samples", [])
            ]
            logic_format_ok_rate = _mean(
                [float(node.exec_summary.get("logic_format_ok_rate", 0.0)) for node in metric_nodes]
            )
            exec_format_ok_rate = _mean(
                [float(node.exec_summary.get("exec_format_ok_rate", 0.0)) for node in metric_nodes]
            )
            generation_format_ok_rate = _mean(
                [float(node.exec_summary.get("generation_format_ok", 0.0)) for node in metric_nodes if node.completion_text]
            )
            compile_ok_rate = _mean([float(node.exec_summary.get("compile_score", 0.0)) for node in metric_nodes])
            syntax_error_rate = _mean([1.0 if node.status_code == "SYNTAX_ERROR" else 0.0 for node in metric_nodes])
            timeout_rate = _mean([1.0 if node.status_code == "TIMEOUT" else 0.0 for node in metric_nodes])
            soft_lift = _mean([max(0.0, node.R_code - node.pass_rate) for node in metric_nodes])
            mean_R_soft_raw = _mean([float(node.exec_summary.get("R_soft_raw", 0.0)) for node in metric_nodes])
            mean_R_soft_match_raw = _mean(
                [float(node.exec_summary.get("R_soft_match_raw", 0.0)) for node in metric_nodes]
            )
            mean_R_soft_effective = _mean(
                [float(node.exec_summary.get("R_soft_effective", 0.0)) for node in metric_nodes]
            )
            soft_reward_eligible_rate = _mean(
                [1.0 if bool(node.exec_summary.get("soft_reward_eligible", False)) else 0.0 for node in metric_nodes]
            )
            mean_terminal_backprop_bonus = _mean(
                [float(node.exec_summary.get("terminal_backprop_bonus", 0.0)) for node in metric_nodes]
            )
            logic_confirmed_rate = _mean([1.0 if bool(item.get("confirmed", False)) else 0.0 for item in logic_items])
            main_sample_count = float(sum(1 for node in metric_nodes if node.completion_text))
            logic_sample_count = float(len(logic_items))
            exec_sample_count = float(len(exec_items))
            eval_metrics.update(
                {
                    "search_node_count": float(len(search_nodes)),
                    "logic_format_ok_rate": logic_format_ok_rate,
                    "exec_format_ok_rate": exec_format_ok_rate,
                    "generation_format_ok_rate": generation_format_ok_rate,
                    "compile_ok_rate": compile_ok_rate,
                    "syntax_error_rate": syntax_error_rate,
                    "timeout_rate": timeout_rate,
                    "soft_lift": soft_lift,
                    "mean_R_soft_raw": mean_R_soft_raw,
                    "mean_R_soft_match_raw": mean_R_soft_match_raw,
                    "mean_R_soft_effective": mean_R_soft_effective,
                    "soft_reward_eligible_rate": soft_reward_eligible_rate,
                    "mean_terminal_backprop_bonus": mean_terminal_backprop_bonus,
                    "logic_confirmed_rate": logic_confirmed_rate,
                    "main_sample_count": main_sample_count,
                    "logic_sample_count": logic_sample_count,
                    "exec_sample_count": exec_sample_count,
                    "final_reason_mean_R_code": _mean([node.R_code for node in final_reason_nodes]),
                    "final_reason_mean_R_reason": _mean([node.R_reason for node in final_reason_nodes]),
                    "final_reason_mean_pass_rate": _mean([node.pass_rate for node in final_reason_nodes]),
                    "final_reason_logic_format_ok_rate": _mean(
                        [float(node.exec_summary.get("logic_format_ok_rate", 0.0)) for node in final_reason_nodes]
                    ),
                    "final_reason_exec_format_ok_rate": _mean(
                        [float(node.exec_summary.get("exec_format_ok_rate", 0.0)) for node in final_reason_nodes]
                    ),
                    "final_reason_logic_confirmed_rate": _mean(
                        [1.0 if bool(item.get("confirmed", False)) else 0.0 for item in final_logic_items]
                    ),
                    "final_reason_logic_sample_count": float(len(final_logic_items)),
                    "final_reason_exec_sample_count": float(len(final_exec_items)),
                }
            )

        if bool(getattr(self.args, "log_train_rollout_details", False)):
            self.logger.info(
                "[TREE] question_id=%s rounds=%d node_count=%d final_reason_node_count=%d resample_count=%d solved=%s",
                question_id,
                len(rounds),
                node_count,
                final_reason_node_count,
                resample_count,
                solved,
            )

        eval_metrics["final_reason_node_count"] = float(final_reason_node_count)

        return QuestionRollout(
            question_id=question_id,
            audit_indices=audit_indices,
            rounds=rounds,
            node_count=node_count,
            resample_count=resample_count,
            train_samples=train_samples,
            mean_R_code=_mean(r_code),
            mean_R_reason=_mean(r_reason),
            mean_pass_rate=_mean(pass_rates),
            std_R_code=_std(r_code),
            std_R_reason=_std(r_reason),
            eval_metrics=eval_metrics,
        )

    def run_question_eval_code_only(self, sample: dict[str, Any], rng: random.Random) -> QuestionRollout:
        question_id = str(sample["question_id"])
        prompt = str(sample["prompt"])
        test_cases = list(sample["test_cases"])
        eval_max_rounds = int(getattr(self.args, "eval_T_max_override", 0) or self.args.T_max)

        root = Node(node_id="root", parent_id=None, round_idx=0, code=None, reasoning=None, exec_summary={"history": []})
        parent = root
        rounds: list[dict[str, Any]] = []
        collected_nodes: list[Node] = []
        node_count = 0
        node_serial = 0

        for round_idx in range(1, eval_max_rounds + 1):
            node_serial += 1
            child = self._generate_node(
                question_id=question_id,
                node_id=f"n{node_serial}",
                parent=parent,
                prompt=prompt,
                test_cases=test_cases,
                audit_indices=[],
                round_idx=round_idx,
                generation_kwargs_override=self._code_generation_kwargs(eval_mode=True),
                log_reward=bool(getattr(self.args, "log_eval_trajectories", False)),
            )
            collected_nodes.append(child)
            rounds.append(self._build_round_record(round_idx, [child], stage="search"))
            node_count += 1
            parent = child
            if child.pass_rate == 1.0:
                break

        train_samples: list[TrainSample] = []
        for node in collected_nodes:
            if not node.completion_text:
                continue
            train_samples.append(
                TrainSample(
                    question_id=question_id,
                    prompt_text=node.prompt_text,
                    completion_text=node.completion_text,
                    code_token_mask=node.code_token_mask,
                    reason_token_mask=node.reason_token_mask,
                    A_code=0.0,
                    A_reason=0.0,
                    R_code=node.R_code,
                    pass_rate=node.pass_rate,
                )
            )

        metric_nodes = collected_nodes
        r_code = [node.R_code for node in metric_nodes]
        r_reason = [node.R_reason for node in metric_nodes]
        pass_rates = [node.pass_rate for node in metric_nodes]
        eval_metrics = self._compute_code_only_eval_metrics(rounds)
        if collected_nodes:
            eval_metrics.update(
                {
                    "search_node_count": float(len(metric_nodes)),
                    "generation_format_ok_rate": _mean(
                        [float(node.exec_summary.get("generation_format_ok", 0.0)) for node in metric_nodes if node.completion_text]
                    ),
                    "compile_ok_rate": _mean([float(node.exec_summary.get("compile_score", 0.0)) for node in metric_nodes]),
                    "syntax_error_rate": _mean([1.0 if node.status_code == "SYNTAX_ERROR" else 0.0 for node in metric_nodes]),
                    "timeout_rate": _mean([1.0 if node.status_code == "TIMEOUT" else 0.0 for node in metric_nodes]),
                    "main_sample_count": float(sum(1 for node in metric_nodes if node.completion_text)),
                }
            )

        if getattr(self.args, "log_eval_trajectories", False):
            self.logger.info(
                "[EVAL_TRAJECTORY] question_id=%s rounds=%d node_count=%d solved=%s",
                question_id,
                len(rounds),
                node_count,
                any(node.pass_rate == 1.0 for node in collected_nodes),
            )

        return QuestionRollout(
            question_id=question_id,
            audit_indices=[],
            rounds=rounds,
            node_count=node_count,
            resample_count=0,
            train_samples=train_samples,
            mean_R_code=_mean(r_code),
            mean_R_reason=_mean(r_reason),
            mean_pass_rate=_mean(pass_rates),
            std_R_code=_std(r_code),
            std_R_reason=_std(r_reason),
            eval_metrics=eval_metrics,
        )

    def _expand_parent(
        self,
        question_id: str,
        parent: Node,
        prompt: str,
        test_cases: list[dict[str, Any]],
        audit_indices: list[int],
        round_idx: int,
        node_serial: int,
        budget: int,
        update_exec_baseline: bool,
    ) -> tuple[list[Node], int, int, int]:
        produced_total = 0
        resample_count = 0
        accepted: list[Node] = []

        for retry_idx in range(self.args.M_retry + 1):
            local_budget = budget - produced_total
            if local_budget <= 0:
                break
            branch_k = self.args.K_reason if (parent.frozen_code and not parent.frozen_reason) else self.args.K
            to_generate = min(branch_k, local_budget)

            history = list(parent.exec_summary.get("history", []))
            context_history = history[-self.args.context_round_window :]
            need_code = not parent.frozen_code
            need_reason = not (parent.frozen_reason and parent.frozen_code)
            prompt_text = build_generation_prompt(
                question_prompt=prompt,
                history=context_history,
                need_code=need_code,
                need_reason=need_reason,
                parent_code=parent.code,
            )
            rendered_prompt_text = self._render_prompt_with_chat_template(prompt_text)

            generation_kwargs = self._code_generation_kwargs()
            raw_outputs = self.backend.generate_many(
                [rendered_prompt_text],
                num_generations=to_generate,
                **generation_kwargs,
            )
            raw_outputs = self._retry_empty_generation_outputs(
                rendered_prompt_text,
                raw_outputs,
                generation_kwargs=generation_kwargs,
            )
            siblings: list[Node] = []
            for raw_output in raw_outputs:
                node_serial += 1
                child = self._generate_node(
                    question_id=question_id,
                    node_id=f"n{node_serial}",
                    parent=parent,
                    prompt=prompt,
                    test_cases=test_cases,
                    audit_indices=audit_indices,
                    round_idx=round_idx,
                    prompt_text_override=rendered_prompt_text,
                    prompt_text_raw_override=prompt_text,
                    raw_output_override=raw_output,
                )
                siblings.append(child)
            produced_total += len(siblings)

            # Group-level retry: if all siblings are double-zero, drop the whole group and resample.
            if siblings and all(_is_double_zero_node(node) for node in siblings) and retry_idx < self.args.M_retry:
                resample_count += 1
                if bool(getattr(self.args, "log_train_rollout_details", False)):
                    self.logger.info(
                        "[TREE] resample parent=%s round=%d retry=%d reason=all_double_zero",
                        parent.node_id,
                        round_idx,
                        retry_idx + 1,
                    )
                continue

            accepted = siblings
            break

        if accepted:
            self._assign_group_advantages(accepted, update_exec_baseline=update_exec_baseline)

        return accepted, produced_total, resample_count, node_serial

    def _generate_node(
        self,
        question_id: str,
        node_id: str,
        parent: Node,
        prompt: str,
        test_cases: list[dict[str, Any]],
        audit_indices: list[int],
        round_idx: int,
        prompt_text_override: str | None = None,
        prompt_text_raw_override: str | None = None,
        raw_output_override: str | None = None,
        generation_kwargs_override: dict[str, Any] | None = None,
        log_reward: bool = False,
    ) -> Node:
        history = list(parent.exec_summary.get("history", []))
        context_history = history[-self.args.context_round_window :]
        need_code = not parent.frozen_code
        need_reason = not (parent.frozen_reason and parent.frozen_code)
        can_reuse_reason = parent.frozen_reason and parent.frozen_code
        prompt_history_debug = summarize_generation_history(context_history)

        if prompt_text_override is None:
            prompt_text_raw = build_generation_prompt(
                question_prompt=prompt,
                history=context_history,
                need_code=need_code,
                need_reason=need_reason,
                parent_code=parent.code,
            )
            prompt_text = self._render_prompt_with_chat_template(prompt_text_raw)
        else:
            prompt_text_raw = prompt_text_raw_override or prompt_text_override
            prompt_text = prompt_text_override
        raw_output = (
            raw_output_override
            if raw_output_override is not None
            else self.backend.generate(prompt_text, **(generation_kwargs_override or self._code_generation_kwargs()))
        )
        if raw_output_override is None and not (raw_output or "").strip():
            retried_outputs = self._retry_empty_generation_outputs(
                prompt_text,
                [raw_output],
                generation_kwargs=generation_kwargs_override,
            )
            raw_output = retried_outputs[0] if retried_outputs else raw_output
        parsed_code, parsed_reason, parsed_logic_prediction, parsed_exec_prediction, generation_format_ok = (
            parse_generation_response(
                raw_output,
                allow_outside_noise_chars=self.args.generation_outside_noise_chars,
                prefilled_code=False,
            )
        )
        trace_max_chars = max(self.args.error_max_chars * 4, self.args.error_max_chars)
        trace_max_lines = max(self.args.error_max_lines * 4, self.args.error_max_lines)
        trace_store_full_text = bool(getattr(self.args, "trace_store_full_text", False))

        code = parent.code if parent.frozen_code else parsed_code
        reasoning = parent.reasoning if can_reuse_reason else ""
        if code is None:
            code = ""
        if reasoning is None:
            reasoning = ""

        completion_text = build_generation_completion(code)
        token_ids, _, _ = build_token_masks(self.tokenizer, completion_text)
        code_mask = [1] * len(token_ids)
        reason_mask = [0] * len(token_ids)

        case_inputs = [_parse_case_input(case["input"]) for case in test_cases]
        exec_results = execute_batch(
            code=code,
            case_inputs=case_inputs,
            timeout_s=self.args.code_timeout_seconds,
            error_max_chars=self.args.error_max_chars,
            error_max_lines=self.args.error_max_lines,
        )
        pass_flags: list[bool] = [
            self._is_test_pass(case["output"], actual) for case, actual in zip(test_cases, exec_results, strict=True)
        ]

        pass_rate = _mean([1.0 if ok else 0.0 for ok in pass_flags])
        status_code = _code_status(exec_results, pass_all=all(pass_flags))
        compile_score = 0.0 if status_code == "SYNTAX_ERROR" else 1.0
        generation_format_score = 1.0 if generation_format_ok else 0.0
        error_summary = next((res.error_msg for res in exec_results if res.kind != "OK" and res.error_msg), "")
        failed_case_input: Any | None = None
        failed_case_actual: str | None = None
        for case, actual, passed, parsed_input in zip(test_cases, exec_results, pass_flags, case_inputs, strict=True):
            if passed:
                continue
            failed_case_input = parsed_input
            if actual.kind == "OK":
                failed_case_actual = _safe_preview(actual.value)
            else:
                failed_case_actual = actual.error_type or actual.kind
            break

        logic_scores: list[float] = []
        logic_match_scores: list[float] = []
        logic_format_flags: list[float] = []
        logic_audit_details: list[dict[str, Any]] = []
        logic_train_samples: list[dict[str, Any]] = []

        exec_scores: list[float] = []
        exec_format_flags: list[float] = []
        exec_raw_correct_flags: list[float] = []
        exec_audit_details: list[dict[str, Any]] = []
        exec_train_samples: list[dict[str, Any]] = []

        unify_reason_mode = bool(
            self.args.unify_reason_when_code_frozen and parent.frozen_code and parent.pass_rate == 1.0 and not parent.frozen_reason
        )
        if audit_indices:
            if unify_reason_mode:
                unified_prompts_raw = [build_frozen_reason_prompt(code, case_inputs[idx]) for idx in audit_indices]
                unified_prompts = [self._render_prompt_with_chat_template(p) for p in unified_prompts_raw]
                unified_outputs = self.backend.generate_many(
                    unified_prompts,
                    max_new_tokens=self._audit_completion_length(),
                )
                for idx, unified_prompt, unified_output in zip(audit_indices, unified_prompts, unified_outputs, strict=True):
                    case = test_cases[idx]
                    actual = exec_results[idx]
                    prompt_preview = summarize_error(unified_prompt, trace_max_chars, trace_max_lines)
                    exec_reason, exec_prediction, format_ok = parse_exec_response(
                        unified_output,
                        require_reason_before_prediction=self.args.require_reason_before_prediction,
                        prediction_max_chars=self.args.prediction_max_chars,
                        reason_max_chars=self.args.reasoning_max_chars,
                        disallow_code_in_reasoning=self.args.disallow_code_in_reasoning,
                        allow_outside_noise_chars=self.args.format_outside_noise_chars,
                    )

                    logic_correct = 1.0 if values_equal(exec_prediction, case["output"]) else 0.0
                    logic_format_term = self.args.format_bonus_logic if format_ok else -self.args.format_penalty_logic
                    logic_score = _apply_format_shaping(
                        logic_correct,
                        format_ok,
                        penalty=self.args.format_penalty_logic,
                        bonus=self.args.format_bonus_logic,
                    )
                    logic_scores.append(logic_score)
                    logic_match_scores.append(logic_correct)
                    logic_format_flags.append(1.0 if format_ok else 0.0)
                    logic_completion_text = unified_output.strip() or _build_logic_audit_completion(exec_reason, exec_prediction)
                    logic_train_samples.append(
                        {
                            "question_id": question_id,
                            "case_index": idx,
                            "prompt_text": unified_prompt,
                            "completion_text": logic_completion_text,
                            "score": logic_score,
                            "match": bool(logic_correct),
                            "format_ok": bool(format_ok),
                            "confirmed": bool(pass_rate == 1.0 and logic_correct),
                            "terminal_bonus": 0.0,
                            "advantage": 0.0,
                        }
                    )
                    logic_audit_details.append(
                        {
                            "case_index": idx,
                            "input": _safe_preview(case_inputs[idx], max_chars=400),
                            "expected_output": _safe_preview(case["output"], max_chars=400),
                            "raw_output": summarize_error(unified_output, trace_max_chars, trace_max_lines),
                            "parsed_reason": summarize_error(exec_reason, trace_max_chars, trace_max_lines),
                            "parsed_prediction": _safe_preview(exec_prediction, max_chars=400),
                            "prompt_preview": prompt_preview,
                            "format_ok": bool(format_ok),
                            "match": bool(logic_correct),
                            "correct_raw": logic_correct,
                            "format_term": logic_format_term,
                            "score_after_penalty": logic_score,
                        }
                    )

                    exec_correct = 1.0 if is_match(exec_prediction, actual) else 0.0
                    exec_format_term = self.args.format_bonus_exec if format_ok else -self.args.format_penalty_exec
                    exec_raw_correct_flags.append(exec_correct)
                    exec_score = _apply_format_shaping(
                        exec_correct,
                        format_ok,
                        penalty=self.args.format_penalty_exec,
                        bonus=self.args.format_bonus_exec,
                    )
                    exec_scores.append(exec_score)
                    exec_format_flags.append(1.0 if format_ok else 0.0)
                    exec_completion_text = unified_output.strip() or _build_exec_audit_completion(exec_reason, exec_prediction)
                    exec_train_samples.append(
                        {
                            "question_id": question_id,
                            "case_index": idx,
                            "prompt_text": unified_prompt,
                            "completion_text": exec_completion_text,
                            "score": exec_score,
                            "match": bool(exec_correct),
                            "format_ok": bool(format_ok),
                            "advantage": 0.0,
                            "advantage_source": "",
                        }
                    )
                    exec_audit_details.append(
                        {
                            "case_index": idx,
                            "input": _safe_preview(case_inputs[idx], max_chars=400),
                            "actual_kind": actual.kind,
                            "actual_value": _safe_preview(actual.value, max_chars=400) if actual.kind == "OK" else "",
                            "actual_error_type": actual.error_type or "",
                            "actual_error_msg": summarize_error(actual.error_msg or "", trace_max_chars, trace_max_lines),
                            "raw_output": summarize_error(unified_output, trace_max_chars, trace_max_lines),
                            "parsed_reason": summarize_error(exec_reason, trace_max_chars, trace_max_lines),
                            "parsed_prediction": _safe_preview(exec_prediction, max_chars=400),
                            "prompt_preview": prompt_preview,
                            "format_ok": bool(format_ok),
                            "match": bool(exec_correct),
                            "correct_raw": exec_correct,
                            "format_term": exec_format_term,
                            "score_after_penalty": exec_score,
                        }
                    )
            else:
                logic_prompts_raw = [build_logic_prompt(code, case_inputs[idx], question_prompt=prompt) for idx in audit_indices]
                logic_prompts = [self._render_prompt_with_chat_template(p) for p in logic_prompts_raw]
                logic_outputs = self.backend.generate_many(
                    logic_prompts,
                    max_new_tokens=self._audit_completion_length(),
                )
                for idx, logic_prompt, logic_output in zip(audit_indices, logic_prompts, logic_outputs, strict=True):
                    case = test_cases[idx]
                    logic_prompt_preview = summarize_error(logic_prompt, trace_max_chars, trace_max_lines)
                    logic_reason, logic_prediction, logic_format_ok = parse_logic_response(
                        logic_output,
                        require_reason_before_prediction=self.args.require_reason_before_prediction,
                        prediction_max_chars=self.args.prediction_max_chars,
                        reason_max_chars=self.args.reasoning_max_chars,
                        disallow_code_in_reasoning=self.args.disallow_code_in_reasoning,
                        allow_outside_noise_chars=self.args.format_outside_noise_chars,
                    )
                    logic_correct = 1.0 if values_equal(logic_prediction, case["output"]) else 0.0
                    logic_format_term = self.args.format_bonus_logic if logic_format_ok else -self.args.format_penalty_logic
                    logic_score = _apply_format_shaping(
                        logic_correct,
                        logic_format_ok,
                        penalty=self.args.format_penalty_logic,
                        bonus=self.args.format_bonus_logic,
                    )
                    logic_scores.append(logic_score)
                    logic_match_scores.append(logic_correct)
                    logic_format_flags.append(1.0 if logic_format_ok else 0.0)
                    logic_completion_text = logic_output.strip() or _build_logic_audit_completion(logic_reason, logic_prediction)
                    logic_train_samples.append(
                        {
                            "question_id": question_id,
                            "case_index": idx,
                            "prompt_text": logic_prompt,
                            "completion_text": logic_completion_text,
                            "score": logic_score,
                            "match": bool(logic_correct),
                            "format_ok": bool(logic_format_ok),
                            "confirmed": bool(pass_rate == 1.0 and logic_correct),
                            "terminal_bonus": 0.0,
                            "advantage": 0.0,
                        }
                    )
                    logic_audit_details.append(
                        {
                            "case_index": idx,
                            "input": _safe_preview(case_inputs[idx], max_chars=400),
                            "expected_output": _safe_preview(case["output"], max_chars=400),
                            "raw_output": summarize_error(logic_output, trace_max_chars, trace_max_lines),
                            "parsed_reason": summarize_error(logic_reason, trace_max_chars, trace_max_lines),
                            "parsed_prediction": _safe_preview(logic_prediction, max_chars=400),
                            "prompt_preview": logic_prompt_preview,
                            "format_ok": logic_format_ok,
                            "match": bool(logic_correct),
                            "correct_raw": logic_correct,
                            "format_term": logic_format_term,
                            "score_after_penalty": logic_score,
                        }
                    )

                exec_prompts_raw = [build_exec_prompt(code, case_inputs[idx]) for idx in audit_indices]
                exec_prompts = [self._render_prompt_with_chat_template(p) for p in exec_prompts_raw]
                exec_outputs = self.backend.generate_many(
                    exec_prompts,
                    max_new_tokens=self._audit_completion_length(),
                )
                for idx, exec_prompt, exec_output in zip(audit_indices, exec_prompts, exec_outputs, strict=True):
                    actual = exec_results[idx]
                    exec_prompt_preview = summarize_error(exec_prompt, trace_max_chars, trace_max_lines)
                    exec_reason, exec_prediction, exec_format_ok = parse_exec_response(
                        exec_output,
                        require_reason_before_prediction=self.args.require_reason_before_prediction,
                        prediction_max_chars=self.args.prediction_max_chars,
                        reason_max_chars=self.args.reasoning_max_chars,
                        disallow_code_in_reasoning=self.args.disallow_code_in_reasoning,
                        allow_outside_noise_chars=self.args.format_outside_noise_chars,
                    )
                    exec_correct = 1.0 if is_match(exec_prediction, actual) else 0.0
                    exec_format_term = self.args.format_bonus_exec if exec_format_ok else -self.args.format_penalty_exec
                    exec_raw_correct_flags.append(exec_correct)
                    exec_score = _apply_format_shaping(
                        exec_correct,
                        exec_format_ok,
                        penalty=self.args.format_penalty_exec,
                        bonus=self.args.format_bonus_exec,
                    )
                    exec_scores.append(exec_score)
                    exec_format_flags.append(1.0 if exec_format_ok else 0.0)
                    exec_completion_text = exec_output.strip() or _build_exec_audit_completion(exec_reason, exec_prediction)
                    exec_train_samples.append(
                        {
                            "question_id": question_id,
                            "case_index": idx,
                            "prompt_text": exec_prompt,
                            "completion_text": exec_completion_text,
                            "score": exec_score,
                            "match": bool(exec_correct),
                            "format_ok": bool(exec_format_ok),
                            "advantage": 0.0,
                            "advantage_source": "",
                        }
                    )
                    exec_audit_details.append(
                        {
                            "case_index": idx,
                            "input": _safe_preview(case_inputs[idx], max_chars=400),
                            "actual_kind": actual.kind,
                            "actual_value": _safe_preview(actual.value, max_chars=400) if actual.kind == "OK" else "",
                            "actual_error_type": actual.error_type or "",
                            "actual_error_msg": summarize_error(actual.error_msg or "", trace_max_chars, trace_max_lines),
                            "raw_output": summarize_error(exec_output, trace_max_chars, trace_max_lines),
                            "parsed_reason": summarize_error(exec_reason, trace_max_chars, trace_max_lines),
                            "parsed_prediction": _safe_preview(exec_prediction, max_chars=400),
                            "prompt_preview": exec_prompt_preview,
                            "format_ok": exec_format_ok,
                            "match": bool(exec_correct),
                            "correct_raw": exec_correct,
                            "format_term": exec_format_term,
                            "score_after_penalty": exec_score,
                        }
                    )

        R_soft = _mean(logic_scores) if logic_scores else 0.0
        R_soft_match_raw = _mean(logic_match_scores) if logic_match_scores else 0.0
        logic_format_ok_rate = _mean(logic_format_flags) if logic_format_flags else 0.0

        # Guard against reward hacking while preserving early learning signal:
        # ineligible nodes do not get full soft reward, but can keep a down-scaled portion.
        soft_reward_eligible = status_code != "SYNTAX_ERROR" and _has_solve_entrypoint(code)
        soft_scale = 1.0 if soft_reward_eligible else self.args.soft_reward_ineligible_scale
        R_soft_effective = _clamp01(R_soft * soft_scale)
        R_code = _compute_code_reward(
            pass_rate=pass_rate,
            r_soft=R_soft_effective,
            lambda_soft=self.args.lambda_soft,
            compile_score=compile_score,
            compile_scale=self.args.code_compile_reward_scale,
            generation_format_score=generation_format_score,
            generation_format_scale=self.args.code_format_reward_scale,
        )

        if can_reuse_reason:
            R_reason = parent.R_reason
            status_reason = parent.status_reason
            exec_format_ok_rate = float(parent.exec_summary.get("exec_format_ok_rate", 0.0))
            exec_audit_details = list(parent.exec_summary.get("exec_audit", []))
            exec_train_samples = []
        else:
            R_reason = _mean(exec_scores) if exec_scores else 0.0
            exec_format_ok_rate = _mean(exec_format_flags) if exec_format_flags else 0.0
            if not exec_scores:
                status_reason = "NOT_RUN"
            else:
                status_reason = "HONEST" if all(score == 1.0 for score in exec_raw_correct_flags) else "HALLUCINATION"
        logic_mismatch_count = int(sum(1 for item in logic_audit_details if not item.get("match", False)))
        exec_mismatch_count = int(sum(1 for item in exec_audit_details if not item.get("match", False)))
        logic_failed = next((item for item in logic_audit_details if not item.get("match", False)), None)
        exec_failed = next((item for item in exec_audit_details if not item.get("match", False)), None)
        child_frozen_code = parent.frozen_code or pass_rate == 1.0
        child_frozen_reason = (child_frozen_code and R_reason == 1.0) or (
            parent.frozen_reason and parent.frozen_code and parent.code == code
        )

        child = Node(
            node_id=node_id,
            parent_id=parent.node_id,
            round_idx=round_idx,
            code=code,
            reasoning=reasoning,
            pass_rate=pass_rate,
            R_soft=R_soft,
            R_code=R_code,
            R_reason=R_reason,
            status_code=status_code,  # type: ignore[arg-type]
            status_reason=status_reason,  # type: ignore[arg-type]
            frozen_code=child_frozen_code,
            frozen_reason=child_frozen_reason,
            exec_summary={
                "error_summary": error_summary,
                "logic_format_ok_rate": logic_format_ok_rate,
                "exec_format_ok_rate": exec_format_ok_rate,
                "generation_format_ok": generation_format_ok,
                "generation_format_score": generation_format_score,
                "compile_score": compile_score,
                "soft_reward_eligible": soft_reward_eligible,
                "R_soft_raw": R_soft,
                "R_soft_match_raw": R_soft_match_raw,
                "R_soft_effective": R_soft_effective,
                "generation_debug": {
                    "prompt_preview": summarize_error(prompt_text_raw, trace_max_chars, trace_max_lines),
                    "latest_feedback_summary": summarize_error(
                        prompt_history_debug.get("latest_feedback", ""),
                        trace_max_chars,
                        trace_max_lines,
                    ),
                    "earlier_history_summary": summarize_error(
                        prompt_history_debug.get("earlier_summary", ""),
                        trace_max_chars,
                        trace_max_lines,
                    ),
                    "raw_output": summarize_error(raw_output, trace_max_chars, trace_max_lines),
                    "parsed_reason": summarize_error(parsed_reason, trace_max_chars, trace_max_lines),
                    "parsed_logic_prediction": _safe_preview(parsed_logic_prediction, max_chars=400),
                    "parsed_exec_prediction": _safe_preview(parsed_exec_prediction, max_chars=400),
                    **(
                        {
                            "full_prompt_raw": prompt_text_raw,
                            "full_prompt_rendered": prompt_text,
                            "full_raw_output": raw_output,
                        }
                        if trace_store_full_text
                        else {}
                    ),
                },
                "logic_audit": logic_audit_details,
                "logic_train_samples": logic_train_samples,
                "exec_audit": exec_audit_details,
                "exec_train_samples": exec_train_samples,
                "history": history
                + [
                    {
                        "round": str(round_idx),
                        "status_code": status_code,
                        "status_reason": status_reason,
                        "error_summary": error_summary,
                        "code_preview": summarize_error(
                            code,
                            max_chars=self.args.error_max_chars,
                            max_lines=self.args.error_max_lines,
                        ),
                        "failed_input": failed_case_input,
                        "failed_actual": failed_case_actual,
                        "logic_mismatch_count": logic_mismatch_count,
                        "exec_mismatch_count": exec_mismatch_count,
                        "logic_format_ok_rate": logic_format_ok_rate,
                        "exec_format_ok_rate": exec_format_ok_rate,
                        "generation_format_ok": generation_format_ok,
                        "compile_score": compile_score,
                        "logic_failed_input": logic_failed.get("input") if logic_failed else None,
                        "logic_failed_prediction": logic_failed.get("parsed_prediction") if logic_failed else None,
                        "exec_failed_input": exec_failed.get("input") if exec_failed else None,
                        "exec_failed_prediction": exec_failed.get("parsed_prediction") if exec_failed else None,
                        "exec_actual_kind": exec_failed.get("actual_kind") if exec_failed else None,
                        "soft_reward_eligible": soft_reward_eligible,
                        "R_soft_raw": R_soft,
                        "R_soft_match_raw": R_soft_match_raw,
                        "R_soft_effective": R_soft_effective,
                    }
                ],
            },
            completion_text=completion_text,
            code_token_mask=code_mask,
            reason_token_mask=reason_mask,
            prompt_text=prompt_text,
        )
        if log_reward:
            self.logger.info(
                "[REWARD] node=%s pass_rate=%.4f R_code=%.4f R_reason=%.4f "
                "R_soft_raw=%.4f R_soft_match=%.4f R_soft_effective=%.4f compile=%.2f gen_fmt=%.2f soft_eligible=%s",
                child.node_id,
                child.pass_rate,
                child.R_code,
                child.R_reason,
                R_soft,
                R_soft_match_raw,
                R_soft_effective,
                compile_score,
                generation_format_score,
                soft_reward_eligible,
            )
        return child

    def _assign_group_advantages(self, siblings: list[Node], update_exec_baseline: bool = True) -> None:
        code_vals = [node.R_code for node in siblings]
        mean_code = _mean(code_vals)
        std_code = _std(code_vals)
        eps = 1e-8
        for node in siblings:
            node.A_code = (node.R_code - mean_code) / (std_code + eps)

        # Reset per-case advantages.
        for node in siblings:
            for item in node.exec_summary.get("logic_train_samples", []):
                item["advantage"] = 0.0
            for item in node.exec_summary.get("exec_train_samples", []):
                item["advantage"] = 0.0
                item["advantage_source"] = ""

        # Logic case-level advantages: same parent siblings, same case index (cross-code allowed).
        logic_groups: dict[int, list[tuple[dict[str, Any], float]]] = {}
        for node in siblings:
            for item in node.exec_summary.get("logic_train_samples", []):
                case_index = int(item.get("case_index", -1))
                base_score = float(item.get("score", 0.0))
                terminal_bonus = float(item.get("terminal_bonus", 0.0))
                confirmed = bool(item.get("confirmed", False))
                score = base_score + terminal_bonus if confirmed else 0.0
                logic_groups.setdefault(case_index, []).append((item, score))
        for rows in logic_groups.values():
            vals = [score for _, score in rows]
            mean_v = _mean(vals)
            std_v = _std(vals)
            for item, score in rows:
                item["advantage"] = (score - mean_v) / (std_v + eps) if len(rows) > 1 else 0.0

        # Exec case-level advantages:
        # 1) strict same-code + same-case z-score when group size >=2
        # 2) fallback to case baseline EMA when strict group has <2 samples
        exec_groups: dict[tuple[int, str], list[tuple[dict[str, Any], float]]] = {}
        for node in siblings:
            code_key = node.code or ""
            for item in node.exec_summary.get("exec_train_samples", []):
                case_index = int(item.get("case_index", -1))
                score = float(item.get("score", 0.0))
                exec_groups.setdefault((case_index, code_key), []).append((item, score))

        for (case_index, _code_key), rows in exec_groups.items():
            if len(rows) >= 2:
                vals = [score for _, score in rows]
                mean_v = _mean(vals)
                std_v = _std(vals)
                for item, score in rows:
                    item["advantage"] = (score - mean_v) / (std_v + eps)
                    item["advantage_source"] = "same_code_case"
            else:
                for item, score in rows:
                    qid = str(item.get("question_id", ""))
                    baseline = self._get_exec_case_baseline(qid, case_index)
                    item["advantage"] = score - baseline
                    item["advantage_source"] = "ema_case_fallback"

        # Update case baseline EMA once for the newly generated sibling group.
        if update_exec_baseline:
            for node in siblings:
                for item in node.exec_summary.get("exec_train_samples", []):
                    qid = str(item.get("question_id", ""))
                    case_index = int(item.get("case_index", -1))
                    score = float(item.get("score", 0.0))
                    self._update_exec_case_baseline(qid, case_index, score)

        for node in siblings:
            all_reason_adv = [
                float(item.get("advantage", 0.0))
                for item in node.exec_summary.get("logic_train_samples", [])
            ] + [
                float(item.get("advantage", 0.0))
                for item in node.exec_summary.get("exec_train_samples", [])
            ]
            node.A_reason = _mean(all_reason_adv) if all_reason_adv else 0.0

            if node.pass_rate == 1.0 and node.R_reason == 1.0:
                node.A_code *= self.args.gamma_shrink
                node.A_reason *= self.args.gamma_shrink
                for item in node.exec_summary.get("logic_train_samples", []):
                    item["advantage"] = float(item.get("advantage", 0.0)) * self.args.gamma_shrink
                for item in node.exec_summary.get("exec_train_samples", []):
                    item["advantage"] = float(item.get("advantage", 0.0)) * self.args.gamma_shrink

            logic_adv_map = {
                int(item.get("case_index", -1)): float(item.get("advantage", 0.0))
                for item in node.exec_summary.get("logic_train_samples", [])
            }
            for row in node.exec_summary.get("logic_audit", []):
                case_index = int(row.get("case_index", -1))
                row["advantage"] = logic_adv_map.get(case_index, 0.0)
                logic_item = next(
                    (
                        item
                        for item in node.exec_summary.get("logic_train_samples", [])
                        if int(item.get("case_index", -1)) == case_index
                    ),
                    {},
                )
                row["terminal_bonus"] = float(logic_item.get("terminal_bonus", 0.0))
                row["confirmed"] = bool(logic_item.get("confirmed", False))
                row["training_value"] = (
                    float(logic_item.get("score", 0.0)) + float(logic_item.get("terminal_bonus", 0.0))
                    if bool(logic_item.get("confirmed", False))
                    else 0.0
                )

            exec_adv_map = {
                int(item.get("case_index", -1)): (
                    float(item.get("advantage", 0.0)),
                    str(item.get("advantage_source", "")),
                )
                for item in node.exec_summary.get("exec_train_samples", [])
            }
            for row in node.exec_summary.get("exec_audit", []):
                case_index = int(row.get("case_index", -1))
                adv, source = exec_adv_map.get(case_index, (0.0, ""))
                row["advantage"] = adv
                row["advantage_source"] = source
                row["training_value"] = float(
                    next(
                        (
                            item.get("score", 0.0)
                            for item in node.exec_summary.get("exec_train_samples", [])
                            if int(item.get("case_index", -1)) == case_index
                        ),
                        0.0,
                    )
                )

    def _is_test_pass(self, expected_output: Any, actual: ExecResult) -> bool:
        if actual.kind != "OK":
            return False
        return values_equal(expected_output, actual.value)

    def _compute_eval_metrics(self, rounds: list[dict[str, Any]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        round_n = self.args.eval_round_n
        k_list = list(self.args.eval_k_list)
        search_rounds = [round_item for round_item in rounds if str(round_item.get("stage", "search")) == "search"]
        eval_rounds = search_rounds if search_rounds else rounds

        if 1 <= round_n <= len(eval_rounds):
            round_nodes = eval_rounds[round_n - 1]["nodes"]
        else:
            round_nodes = []
        for k in k_list:
            key = f"pass_at_{k}_round_{round_n}"
            metrics[key] = 1.0 if any(node["pass_rate"] == 1.0 for node in round_nodes[:k]) else 0.0
        if k_list:
            metrics["pass_at_k_round_n"] = metrics.get(f"pass_at_{k_list[0]}_round_{round_n}", 0.0)

        flat_nodes = [node for round_item in eval_rounds for node in round_item["nodes"]]
        for k in k_list:
            metrics[f"pass_at_{k}"] = 1.0 if any(node["pass_rate"] == 1.0 for node in flat_nodes[:k]) else 0.0

        metrics["best_pass_rate_overall"] = max((node["pass_rate"] for node in flat_nodes), default=0.0)
        metrics[f"best_pass_rate_round_{round_n}"] = max((node["pass_rate"] for node in round_nodes), default=0.0)
        return metrics

    def _compute_code_only_eval_metrics(self, rounds: list[dict[str, Any]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        search_rounds = [round_item for round_item in rounds if str(round_item.get("stage", "search")) == "search"]
        eval_rounds = search_rounds if search_rounds else rounds
        flat_nodes = [round_item["nodes"][0] for round_item in eval_rounds if round_item.get("nodes")]
        cumulative_solved = False
        cumulative_best = 0.0
        eval_max_rounds = int(getattr(self.args, "eval_T_max_override", 0) or self.args.T_max)
        max_rounds = max(eval_max_rounds, len(flat_nodes))

        for round_idx in range(1, max_rounds + 1):
            if round_idx <= len(flat_nodes):
                node = flat_nodes[round_idx - 1]
                cumulative_solved = cumulative_solved or (float(node.get("pass_rate", 0.0)) == 1.0)
                cumulative_best = max(cumulative_best, float(node.get("pass_rate", 0.0)))
            metrics[f"pass_at_1_round_{round_idx}"] = 1.0 if cumulative_solved else 0.0
            metrics[f"best_pass_rate_round_{round_idx}"] = cumulative_best

        metrics["pass_at_1"] = metrics.get(f"pass_at_1_round_{max_rounds}", 0.0)
        eval_round_n = min(max(1, int(self.args.eval_round_n)), max_rounds)
        metrics["pass_at_k_round_n"] = metrics.get(f"pass_at_1_round_{eval_round_n}", 0.0)
        metrics["best_pass_rate_overall"] = max((float(node["pass_rate"]) for node in flat_nodes), default=0.0)
        return metrics
