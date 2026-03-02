import math
import random
from typing import Any

from .error_utils import summarize_error
from .executor import execute
from .matcher import is_match, values_equal
from .parser import (
    build_canonical_completion,
    build_token_masks,
    parse_exec_response,
    parse_generation_output,
    parse_logic_response,
)
from .prompts import build_exec_prompt, build_generation_prompt, build_logic_prompt
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


def _compute_code_reward(pass_rate: float, r_soft: float, lambda_soft: float) -> float:
    """Scale soft reward by remaining hard-reward margin to avoid overshadowing pass_rate."""
    return pass_rate + lambda_soft * (1.0 - pass_rate) * r_soft


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _apply_format_penalty(correct_score: float, format_ok: bool, penalty: float) -> float:
    if format_ok:
        return _clamp01(correct_score)
    return _clamp01(correct_score - penalty)


def _safe_preview(value: Any, max_chars: int = 200) -> str:
    text = repr(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _is_double_zero_node(node: Node, eps: float = 1e-12) -> bool:
    """Return True when both rewards are (numerically) zero."""
    return abs(node.R_code) <= eps and abs(node.R_reason) <= eps


class CodeGRPOTreeRunner:
    def __init__(self, backend, tokenizer, args, logger):
        self.backend = backend
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger

    def run_question(self, sample: dict[str, Any], rng: random.Random) -> QuestionRollout:
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

        node_count = 0
        resample_count = 0
        node_serial = 0
        solved = False

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
                    parent=parent,
                    prompt=prompt,
                    test_cases=test_cases,
                    audit_indices=audit_indices,
                    round_idx=round_idx,
                    node_serial=node_serial,
                    budget=self.args.N_max - node_count,
                )
                node_count += produced
                resample_count += resampled

                if not siblings:
                    continue

                round_nodes.extend(siblings)
                collected_nodes.extend(siblings)
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

            rounds.append(
                {
                    "round": round_idx,
                    "nodes": [
                        {
                            "node_id": node.node_id,
                            "parent_id": node.parent_id,
                            "pass_rate": node.pass_rate,
                            "status_code": node.status_code,
                            "R_code": node.R_code,
                            "R_reason": node.R_reason,
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
                        for node in round_nodes
                    ],
                }
            )

            frontier = [node for node in next_frontier if not (node.frozen_code and node.frozen_reason)]
            if solved:
                break

        train_samples = [
            TrainSample(
                question_id=question_id,
                prompt_text=node.prompt_text,
                completion_text=node.completion_text,
                code_token_mask=node.code_token_mask,
                reason_token_mask=node.reason_token_mask,
                A_code=node.A_code,
                A_reason=node.A_reason,
                R_code=node.R_code,
                R_reason_final=node.R_reason_final,
                pass_rate=node.pass_rate,
            )
            for node in collected_nodes
            if node.completion_text
        ]

        r_code = [node.R_code for node in collected_nodes]
        r_reason = [node.R_reason_final for node in collected_nodes]
        pass_rates = [node.pass_rate for node in collected_nodes]
        eval_metrics = self._compute_eval_metrics(rounds)
        if collected_nodes:
            logic_format_ok_rate = _mean(
                [float(node.exec_summary.get("logic_format_ok_rate", 0.0)) for node in collected_nodes]
            )
            exec_format_ok_rate = _mean(
                [float(node.exec_summary.get("exec_format_ok_rate", 0.0)) for node in collected_nodes]
            )
            syntax_error_rate = _mean([1.0 if node.status_code == "SYNTAX_ERROR" else 0.0 for node in collected_nodes])
            timeout_rate = _mean([1.0 if node.status_code == "TIMEOUT" else 0.0 for node in collected_nodes])
            soft_lift = _mean([max(0.0, node.R_code - node.pass_rate) for node in collected_nodes])
            eval_metrics.update(
                {
                    "logic_format_ok_rate": logic_format_ok_rate,
                    "exec_format_ok_rate": exec_format_ok_rate,
                    "syntax_error_rate": syntax_error_rate,
                    "timeout_rate": timeout_rate,
                    "soft_lift": soft_lift,
                }
            )

        self.logger.info(
            "[TREE] question_id=%s rounds=%d node_count=%d resample_count=%d solved=%s",
            question_id,
            len(rounds),
            node_count,
            resample_count,
            solved,
        )

        return QuestionRollout(
            question_id=question_id,
            audit_indices=audit_indices,
            rounds=rounds,
            node_count=node_count,
            resample_count=resample_count,
            train_samples=train_samples,
            mean_R_code=_mean(r_code),
            mean_R_reason_final=_mean(r_reason),
            mean_pass_rate=_mean(pass_rates),
            std_R_code=_std(r_code),
            std_R_reason_final=_std(r_reason),
            eval_metrics=eval_metrics,
        )

    def _expand_parent(
        self,
        parent: Node,
        prompt: str,
        test_cases: list[dict[str, Any]],
        audit_indices: list[int],
        round_idx: int,
        node_serial: int,
        budget: int,
    ) -> tuple[list[Node], int, int, int]:
        produced_total = 0
        resample_count = 0
        accepted: list[Node] = []

        for retry_idx in range(self.args.M_retry + 1):
            local_budget = budget - produced_total
            if local_budget <= 0:
                break
            to_generate = min(self.args.K, local_budget)
            siblings: list[Node] = []
            for _ in range(to_generate):
                node_serial += 1
                child = self._generate_node(
                    node_id=f"n{node_serial}",
                    parent=parent,
                    prompt=prompt,
                    test_cases=test_cases,
                    audit_indices=audit_indices,
                    round_idx=round_idx,
                )
                siblings.append(child)
            produced_total += len(siblings)

            # Group-level retry: if all siblings are double-zero, drop the whole group and resample.
            if siblings and all(_is_double_zero_node(node) for node in siblings) and retry_idx < self.args.M_retry:
                resample_count += 1
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
            self._assign_group_advantages(accepted)

        return accepted, produced_total, resample_count, node_serial

    def _generate_node(
        self,
        node_id: str,
        parent: Node,
        prompt: str,
        test_cases: list[dict[str, Any]],
        audit_indices: list[int],
        round_idx: int,
    ) -> Node:
        history = list(parent.exec_summary.get("history", []))
        context_history = history[-self.args.context_round_window :]
        need_code = not parent.frozen_code
        need_reason = not parent.frozen_reason

        prompt_text = build_generation_prompt(
            question_prompt=prompt,
            history=context_history,
            need_code=need_code,
            need_reason=need_reason,
            parent_code=parent.code,
        )
        raw_output = self.backend.generate(prompt_text)
        parsed_code, parsed_reason, parsed_logic_prediction, parsed_exec_prediction = parse_generation_output(raw_output)
        trace_max_chars = max(self.args.error_max_chars * 4, self.args.error_max_chars)
        trace_max_lines = max(self.args.error_max_lines * 4, self.args.error_max_lines)

        code = parent.code if parent.frozen_code else parsed_code
        reasoning = parent.reasoning if parent.frozen_reason else parsed_reason
        if code is None:
            code = ""
        if reasoning is None:
            reasoning = ""

        completion_text = build_canonical_completion(
            code=code,
            reasoning=reasoning,
            logic_prediction=parsed_logic_prediction,
            exec_prediction=parsed_exec_prediction,
        )
        _, code_mask, reason_mask = build_token_masks(self.tokenizer, completion_text)

        exec_results: list[ExecResult] = []
        pass_flags: list[bool] = []
        for case in test_cases:
            actual = execute(
                code=code,
                case_input=case["input"],
                timeout_s=self.args.code_timeout_seconds,
                error_max_chars=self.args.error_max_chars,
                error_max_lines=self.args.error_max_lines,
            )
            exec_results.append(actual)
            pass_flags.append(self._is_test_pass(case["output"], actual))

        pass_rate = _mean([1.0 if ok else 0.0 for ok in pass_flags])
        status_code = _code_status(exec_results, pass_all=all(pass_flags))
        error_summary = next((res.error_msg for res in exec_results if res.kind != "OK" and res.error_msg), "")
        failed_case_input: Any | None = None
        failed_case_actual: str | None = None
        for case, actual, passed in zip(test_cases, exec_results, pass_flags, strict=True):
            if passed:
                continue
            failed_case_input = case["input"]
            if actual.kind == "OK":
                failed_case_actual = _safe_preview(actual.value)
            else:
                failed_case_actual = actual.error_type or actual.kind
            break

        if audit_indices:
            logic_scores = []
            logic_format_flags = []
            logic_audit_details: list[dict[str, Any]] = []
            for idx in audit_indices:
                case = test_cases[idx]
                logic_output = self.backend.generate(
                    build_logic_prompt(code, case["input"], question_prompt=prompt)
                )
                logic_reason, logic_prediction, logic_format_ok = parse_logic_response(
                    logic_output, require_reason_before_prediction=self.args.require_reason_before_prediction
                )
                logic_correct = 1.0 if values_equal(logic_prediction, case["output"]) else 0.0
                logic_score = _apply_format_penalty(logic_correct, logic_format_ok, penalty=self.args.format_penalty_logic)
                logic_scores.append(logic_score)
                logic_format_flags.append(1.0 if logic_format_ok else 0.0)
                logic_audit_details.append(
                    {
                        "case_index": idx,
                        "input": _safe_preview(case["input"], max_chars=400),
                        "expected_output": _safe_preview(case["output"], max_chars=400),
                        "raw_output": summarize_error(logic_output, trace_max_chars, trace_max_lines),
                        "parsed_reason": summarize_error(logic_reason, trace_max_chars, trace_max_lines),
                        "parsed_prediction": _safe_preview(logic_prediction, max_chars=400),
                        "format_ok": logic_format_ok,
                        "match": bool(logic_correct),
                        "score_after_penalty": logic_score,
                    }
                )
            R_soft = _mean(logic_scores)
            logic_format_ok_rate = _mean(logic_format_flags)
        else:
            R_soft = 0.0
            logic_format_ok_rate = 0.0
            logic_audit_details = []
        R_code = _compute_code_reward(pass_rate=pass_rate, r_soft=R_soft, lambda_soft=self.args.lambda_soft)

        exec_audit_details: list[dict[str, Any]] = []
        if parent.frozen_reason:
            R_reason = parent.R_reason
            status_reason = parent.status_reason
            exec_format_ok_rate = float(parent.exec_summary.get("exec_format_ok_rate", 0.0))
            exec_audit_details = list(parent.exec_summary.get("exec_audit", []))
        elif audit_indices:
            correctness = []
            exec_format_flags = []
            exec_raw_correct_flags = []
            for idx in audit_indices:
                case = test_cases[idx]
                actual = exec_results[idx]
                exec_output = self.backend.generate(
                    build_exec_prompt(
                        code,
                        case["input"],
                    )
                )
                exec_reason, exec_prediction, exec_format_ok = parse_exec_response(
                    exec_output, require_reason_before_prediction=self.args.require_reason_before_prediction
                )
                exec_correct = 1.0 if is_match(exec_prediction, actual) else 0.0
                exec_raw_correct_flags.append(exec_correct)
                exec_score = _apply_format_penalty(exec_correct, exec_format_ok, penalty=self.args.format_penalty_exec)
                correctness.append(exec_score)
                exec_format_flags.append(1.0 if exec_format_ok else 0.0)
                exec_audit_details.append(
                    {
                        "case_index": idx,
                        "input": _safe_preview(case["input"], max_chars=400),
                        "actual_kind": actual.kind,
                        "actual_value": _safe_preview(actual.value, max_chars=400) if actual.kind == "OK" else "",
                        "actual_error_type": actual.error_type or "",
                        "actual_error_msg": summarize_error(actual.error_msg or "", trace_max_chars, trace_max_lines),
                        "raw_output": summarize_error(exec_output, trace_max_chars, trace_max_lines),
                        "parsed_reason": summarize_error(exec_reason, trace_max_chars, trace_max_lines),
                        "parsed_prediction": _safe_preview(exec_prediction, max_chars=400),
                        "format_ok": exec_format_ok,
                        "match": bool(exec_correct),
                        "score_after_penalty": exec_score,
                    }
                )
            R_reason = _mean(correctness)
            exec_format_ok_rate = _mean(exec_format_flags)
            status_reason = "HONEST" if all(score == 1.0 for score in exec_raw_correct_flags) else "HALLUCINATION"
        else:
            R_reason = 0.0
            status_reason = "NOT_RUN"
            exec_format_ok_rate = 0.0
            exec_audit_details = []
        R_reason_final = R_reason * (0.5 + 0.5 * pass_rate)

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
            R_reason_final=R_reason_final,
            status_code=status_code,  # type: ignore[arg-type]
            status_reason=status_reason,  # type: ignore[arg-type]
            frozen_code=parent.frozen_code or pass_rate == 1.0,
            frozen_reason=parent.frozen_reason or R_reason == 1.0,
            exec_summary={
                "error_summary": error_summary,
                "logic_format_ok_rate": logic_format_ok_rate,
                "exec_format_ok_rate": exec_format_ok_rate,
                "generation_debug": {
                    "raw_output": summarize_error(raw_output, trace_max_chars, trace_max_lines),
                    "parsed_reason": summarize_error(parsed_reason, trace_max_chars, trace_max_lines),
                    "parsed_logic_prediction": _safe_preview(parsed_logic_prediction, max_chars=400),
                    "parsed_exec_prediction": _safe_preview(parsed_exec_prediction, max_chars=400),
                },
                "logic_audit": logic_audit_details,
                "exec_audit": exec_audit_details,
                "history": history
                + [
                    {
                        "round": str(round_idx),
                        "status_code": status_code,
                        "error_summary": error_summary,
                        "code_preview": summarize_error(
                            code,
                            max_chars=self.args.error_max_chars,
                            max_lines=self.args.error_max_lines,
                        ),
                        "failed_input": failed_case_input,
                        "failed_actual": failed_case_actual,
                    }
                ],
            },
            completion_text=completion_text,
            code_token_mask=code_mask,
            reason_token_mask=reason_mask,
            prompt_text=prompt_text,
        )
        self.logger.info(
            "[REWARD] node=%s pass_rate=%.4f R_code=%.4f R_reason=%.4f R_reason_final=%.4f",
            child.node_id,
            child.pass_rate,
            child.R_code,
            child.R_reason,
            child.R_reason_final,
        )
        return child

    def _assign_group_advantages(self, siblings: list[Node]) -> None:
        code_vals = [node.R_code for node in siblings]
        reason_vals = [node.R_reason_final for node in siblings]
        mean_code = _mean(code_vals)
        mean_reason = _mean(reason_vals)
        std_code = _std(code_vals)
        std_reason = _std(reason_vals)
        eps = 1e-8
        for node in siblings:
            node.A_code = (node.R_code - mean_code) / (std_code + eps)
            node.A_reason = (node.R_reason_final - mean_reason) / (std_reason + eps)
            if node.pass_rate == 1.0 and node.R_reason == 1.0:
                node.A_code *= self.args.gamma_shrink
                node.A_reason *= self.args.gamma_shrink

    def _is_test_pass(self, expected_output: Any, actual: ExecResult) -> bool:
        if actual.kind != "OK":
            return False
        return values_equal(expected_output, actual.value)

    def _compute_eval_metrics(self, rounds: list[dict[str, Any]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        round_n = self.args.eval_round_n
        k_list = list(self.args.eval_k_list)

        if 1 <= round_n <= len(rounds):
            round_nodes = rounds[round_n - 1]["nodes"]
        else:
            round_nodes = []
        for k in k_list:
            key = f"pass_at_{k}_round_{round_n}"
            metrics[key] = 1.0 if any(node["pass_rate"] == 1.0 for node in round_nodes[:k]) else 0.0
        if k_list:
            metrics["pass_at_k_round_n"] = metrics.get(f"pass_at_{k_list[0]}_round_{round_n}", 0.0)

        flat_nodes = [node for round_item in rounds for node in round_item["nodes"]]
        for k in k_list:
            metrics[f"pass_at_{k}"] = 1.0 if any(node["pass_rate"] == 1.0 for node in flat_nodes[:k]) else 0.0

        metrics["best_pass_rate_overall"] = max((node["pass_rate"] for node in flat_nodes), default=0.0)
        metrics[f"best_pass_rate_round_{round_n}"] = max((node["pass_rate"] for node in round_nodes), default=0.0)
        return metrics
