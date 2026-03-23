import ast
import math
import random
from typing import Any

from .error_utils import summarize_error
from .executor import execute_batch
from .matcher import stripped_text_equal, values_equal
from .parser import build_generation_completion, build_token_masks, parse_generation_response
from .prompts import build_code_io_training_prompt, build_generation_prompt, summarize_generation_history
from .soft_reward import (
    build_diagnostic_inputs,
    compute_soft_reward,
    compute_zero_pass_beta,
    get_oracle_outputs,
    normalize_soft_reward_to_unit_interval,
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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


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
    aux_reward_scale: float = 1.0,
) -> float:
    return _clamp01(
        pass_rate
        + _clamp01(aux_reward_scale) * _clamp01(lambda_soft) * _clamp01(r_soft)
        + _clamp01(aux_reward_scale) * _clamp01(compile_scale) * _clamp01(compile_score)
        + _clamp01(generation_format_scale) * _clamp01(generation_format_score)
    )


def _safe_preview(value: Any, max_chars: int = 200) -> str:
    text = repr(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _baseline_aux_target_text(actual: ExecResult) -> str:
    if actual.kind == "OK":
        if isinstance(actual.value, str):
            return actual.value.strip()
        try:
            import json

            return json.dumps(actual.value, ensure_ascii=False)
        except Exception:
            return str(actual.value).strip()
    return str(actual.error_type or actual.kind).strip()


def _build_code_io_aux_samples(
    *,
    question_id: str,
    code: str,
    test_cases: list[dict[str, Any]],
    case_inputs: list[Any],
    exec_results: list[ExecResult],
    pass_flags: list[bool],
    enabled: bool,
    case_count: int,
    include_correct: bool,
    include_incorrect: bool,
    include_errors: bool,
    sft_weight_correct: float,
    sft_weight_incorrect: float,
    prompt_renderer,
) -> list[dict[str, Any]]:
    if not enabled or case_count <= 0:
        return []

    samples: list[dict[str, Any]] = []
    for idx, (_case, parsed_input, actual, passed) in enumerate(
        zip(test_cases, case_inputs, exec_results, pass_flags, strict=True)
    ):
        if passed and not include_correct:
            continue
        if (not passed) and not include_incorrect:
            continue
        if actual.kind != "OK" and not include_errors:
            continue
        target_text = _baseline_aux_target_text(actual)
        if not target_text:
            continue
        prompt_text = prompt_renderer(build_code_io_training_prompt(code=code, case_input=parsed_input))
        samples.append(
            {
                "question_id": question_id,
                "case_index": idx,
                "prompt_text": prompt_text,
                "completion_text": target_text,
                "sft_weight": float(sft_weight_correct if passed else sft_weight_incorrect),
                "match": bool(passed),
                "target_kind": str(actual.kind),
                "input_preview": _safe_preview(parsed_input, max_chars=400),
                "target_preview": _safe_preview(target_text, max_chars=400),
            }
        )
        if len(samples) >= case_count:
            break
    return samples


def _is_double_zero_node(node: Node, eps: float = 1e-12) -> bool:
    return abs(node.R_code) <= eps


def _parse_case_input(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _has_solve_entrypoint(code: str) -> bool:
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        module = ast.parse(code)
    except Exception:
        return False
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'solve':
            return True
    return False


def _is_soft_reward_eligible(code: str, io_mode: str) -> bool:
    if not isinstance(code, str) or not code.strip():
        return False
    normalized_io_mode = str(io_mode or "call").strip().lower()
    if normalized_io_mode == "stdio":
        return True
    return _has_solve_entrypoint(code)


class CodeGRPOTreeRunner:
    def __init__(self, backend, tokenizer, args, logger):
        self.backend = backend
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger
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
        except Exception:
            pass
        return prompt_text

    def _code_completion_length(self) -> int:
        return int(getattr(self.args, "max_completion_length_code", None) or self.args.max_completion_length)

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

    def _use_zero_pass_baseline(self) -> bool:
        return True

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

    def _build_round_record(self, round_idx: int, nodes: list[Node], stage: str = "search") -> dict[str, Any]:
        payload_nodes: list[dict[str, Any]] = []
        for node in nodes:
            payload_nodes.append(
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "round_idx": node.round_idx,
                    "stage": stage,
                    "pass_rate": node.pass_rate,
                    "R_code": node.R_code,
                    "A_code": node.A_code,
                    "status_code": node.status_code,
                    "code": node.code,
                    "prompt_text": node.prompt_text,
                    "main_sample_active": bool(node.completion_text),
                    "hard_reward": float(node.exec_summary.get("hard_reward", node.pass_rate)),
                    "raw_soft_reward": float(node.exec_summary.get("raw_soft_reward", 0.0)),
                    "normalized_soft_reward": float(node.exec_summary.get("normalized_soft_reward", 0.0)),
                    "soft_reward_beta": float(node.exec_summary.get("soft_reward_beta", 0.0)),
                    "final_reward": float(node.exec_summary.get("final_reward", node.R_code)),
                    "compile_score": float(node.exec_summary.get("compile_score", 0.0)),
                    "generation_format_ok": bool(node.exec_summary.get("generation_format_ok", False)),
                    "code_io_aux_sample_count": len(node.exec_summary.get("code_io_train_samples", [])),
                    "error_summary": str(node.exec_summary.get("error_summary", "")),
                    "history": list(node.exec_summary.get("history", [])),
                    "generation_debug": dict(node.exec_summary.get("generation_debug", {})),
                }
            )
        return {"round_idx": round_idx, "stage": stage, "nodes": payload_nodes}

    def run_question(self, sample: dict[str, Any], rng: random.Random) -> QuestionRollout:
        del rng
        question_id = str(sample["question_id"])
        prompt = str(sample["prompt"])
        test_cases = list(sample["test_cases"])
        diagnostic_inputs = list(sample.get("diagnostic_inputs", []) or [])
        diagnostic_outputs = list(sample.get("diagnostic_outputs", []) or [])
        io_mode = str(sample.get("io_mode", "call") or "call").strip().lower()

        root = Node(node_id="root", parent_id=None, round_idx=0, code=None, exec_summary={"history": []})
        siblings, produced, resampled, _node_serial, undiff_triggered, undiff_succeeded = self._expand_parent(
            question_id=question_id,
            parent=root,
            prompt=prompt,
            test_cases=test_cases,
            diagnostic_inputs=diagnostic_inputs,
            diagnostic_outputs=diagnostic_outputs,
            io_mode=io_mode,
            round_idx=1,
            node_serial=0,
            budget=min(int(self.args.N_max), int(self.args.K)),
        )
        rounds = [self._build_round_record(1, siblings, stage="search")] if siblings else []

        train_samples: list[TrainSample] = []
        for node in siblings:
            if node.completion_text:
                train_samples.append(
                    TrainSample(
                        question_id=question_id,
                        prompt_text=node.prompt_text,
                        completion_text=node.completion_text,
                        code_token_mask=node.code_token_mask,
                        A_code=node.A_code,
                        sft_token_mask=[0] * len(node.code_token_mask),
                        sft_weight=0.0,
                        R_code=node.R_code,
                        pass_rate=node.pass_rate,
                    )
                )
            for item in node.exec_summary.get("code_io_train_samples", []):
                aux_prompt_text = str(item.get("prompt_text", ""))
                aux_completion_text = str(item.get("completion_text", ""))
                if not aux_prompt_text or not aux_completion_text:
                    continue
                token_ids = self.tokenizer(aux_completion_text, add_special_tokens=False)["input_ids"]
                if not token_ids:
                    continue
                train_samples.append(
                    TrainSample(
                        question_id=question_id,
                        prompt_text=aux_prompt_text,
                        completion_text=aux_completion_text,
                        code_token_mask=[0] * len(token_ids),
                        A_code=0.0,
                        sft_token_mask=[1] * len(token_ids),
                        sft_weight=float(item.get("sft_weight", 0.0)),
                        R_code=node.R_code,
                        pass_rate=node.pass_rate,
                    )
                )

        r_code = [node.R_code for node in siblings]
        pass_rates = [node.pass_rate for node in siblings]
        eval_metrics = self._compute_eval_metrics(rounds)
        if siblings:
            eval_metrics.update(
                {
                    "search_node_count": float(len(siblings)),
                    "generation_format_ok_rate": _mean([float(node.exec_summary.get("generation_format_ok", 0.0)) for node in siblings if node.completion_text]),
                    "compile_ok_rate": _mean([float(node.exec_summary.get("compile_score", 0.0)) for node in siblings]),
                    "syntax_error_rate": _mean([1.0 if node.status_code == "SYNTAX_ERROR" else 0.0 for node in siblings]),
                    "timeout_rate": _mean([1.0 if node.status_code == "TIMEOUT" else 0.0 for node in siblings]),
                    "soft_lift": _mean([max(0.0, node.R_code - node.pass_rate) for node in siblings]),
                    "mean_R_soft_raw": _mean([float(node.exec_summary.get("raw_soft_reward", 0.0)) for node in siblings]),
                    "mean_R_soft_match_raw": _mean([float(node.exec_summary.get("normalized_soft_reward", 0.0)) for node in siblings]),
                    "mean_R_soft_effective": _mean([float(node.exec_summary.get("normalized_soft_reward", 0.0)) for node in siblings]),
                    "soft_reward_eligible_rate": _mean([1.0 if bool(node.exec_summary.get("soft_reward_eligible", False)) else 0.0 for node in siblings]),
                    "main_sample_count": float(sum(1 for node in siblings if node.completion_text)),
                    "code_io_aux_sample_count": float(sum(len(node.exec_summary.get("code_io_train_samples", [])) for node in siblings)),
                    "zero_pass_soft_trigger_rate": _mean([1.0 if bool(node.exec_summary.get("zero_pass_soft_reward_triggered", False)) else 0.0 for node in siblings]),
                    "mean_hard_reward": _mean([float(node.exec_summary.get("hard_reward", node.pass_rate)) for node in siblings]),
                    "mean_raw_soft_reward": _mean([float(node.exec_summary.get("raw_soft_reward", 0.0)) for node in siblings]),
                    "mean_normalized_soft_reward": _mean([float(node.exec_summary.get("normalized_soft_reward", 0.0)) for node in siblings]),
                    "mean_soft_reward_beta": _mean([float(node.exec_summary.get("soft_reward_beta", 0.0)) for node in siblings]),
                    "sibling_group_zero_std_R_code_rate": 1.0 if _std(r_code) <= 1e-8 else 0.0,
                    "pair_same_R_code_rate": (1.0 if len(siblings) == 2 and abs(float(siblings[0].R_code) - float(siblings[1].R_code)) <= 1e-8 else 0.0),
                }
            )

        return QuestionRollout(
            question_id=question_id,
            rounds=rounds,
            node_count=produced,
            resample_count=resampled,
            train_samples=train_samples,
            mean_R_code=_mean(r_code),
            mean_pass_rate=_mean(pass_rates),
            std_R_code=_std(r_code),
            eval_metrics=eval_metrics,
        )

    def run_question_eval_code_only(self, sample: dict[str, Any], rng: random.Random) -> QuestionRollout:
        return self.run_question(sample, rng)

    def _expand_parent(
        self,
        question_id: str,
        parent: Node,
        prompt: str,
        test_cases: list[dict[str, Any]],
        diagnostic_inputs: list[Any],
        diagnostic_outputs: list[Any],
        io_mode: str,
        round_idx: int,
        node_serial: int,
        budget: int,
    ) -> tuple[list[Node], int, int, int, int, int]:
        produced_total = 0
        resample_count = 0
        accepted: list[Node] = []

        branch_k = min(max(0, int(self.args.K)), max(0, int(budget)))
        if branch_k <= 0:
            return accepted, produced_total, resample_count, node_serial, 0, 0

        history = list(parent.exec_summary.get("history", []))
        context_history = history[-self.args.context_round_window :]
        prompt_text = build_generation_prompt(
            question_prompt=prompt,
            history=context_history,
            parent_code=parent.code,
        )
        rendered_prompt_text = self._render_prompt_with_chat_template(prompt_text)
        generation_kwargs = self._code_generation_kwargs()
        soft_reward_problem_logprob_cache: dict[tuple[str, str], float] = {}

        for retry_idx in range(int(self.args.M_retry) + 1):
            raw_outputs = self.backend.generate_many(
                [rendered_prompt_text],
                num_generations=branch_k,
                **generation_kwargs,
            )
            raw_outputs = self._retry_empty_generation_outputs(
                rendered_prompt_text,
                raw_outputs,
                generation_kwargs=generation_kwargs,
            )
            accepted = []
            for raw_output in raw_outputs[:branch_k]:
                node_serial += 1
                accepted.append(
                    self._generate_node(
                        question_id=question_id,
                        node_id=f"n{node_serial}",
                        parent=parent,
                        prompt=prompt,
                        test_cases=test_cases,
                        diagnostic_inputs=diagnostic_inputs,
                        diagnostic_outputs=diagnostic_outputs,
                        io_mode=io_mode,
                        round_idx=round_idx,
                        prompt_text_override=rendered_prompt_text,
                        prompt_text_raw_override=prompt_text,
                        raw_output_override=raw_output,
                        problem_logprob_cache=soft_reward_problem_logprob_cache,
                    )
                )
            produced_total += len(accepted)
            if accepted and all(_is_double_zero_node(node) for node in accepted) and retry_idx < int(self.args.M_retry):
                resample_count += 1
                continue
            break

        undiff_retry_triggered = 0
        undiff_retry_succeeded = 0
        undiff_retry_enabled = bool(getattr(self.args, "undiff_retry_enabled", False))
        undiff_retry_max = int(getattr(self.args, "undiff_retry_max", 1))
        if undiff_retry_enabled and len(accepted) == 2:
            eps = 1e-8
            for _ in range(undiff_retry_max):
                a, b = accepted[0], accepted[1]
                all_pass = (a.pass_rate == 1.0) and (b.pass_rate == 1.0)
                r_code_same = abs(a.R_code - b.R_code) < eps
                pass_rate_same = abs(a.pass_rate - b.pass_rate) < eps
                if all_pass or not r_code_same or not pass_rate_same:
                    break
                undiff_retry_triggered += 1
                node_serial += 1
                produced_total += 1
                accepted[1] = self._generate_node(
                    question_id=question_id,
                    node_id=f"n{node_serial}",
                    parent=parent,
                    prompt=prompt,
                    test_cases=test_cases,
                    diagnostic_inputs=diagnostic_inputs,
                    diagnostic_outputs=diagnostic_outputs,
                    io_mode=io_mode,
                    round_idx=round_idx,
                    prompt_text_override=rendered_prompt_text,
                    prompt_text_raw_override=prompt_text,
                    problem_logprob_cache=soft_reward_problem_logprob_cache,
                )
                if abs(accepted[0].R_code - accepted[1].R_code) >= eps or abs(accepted[0].pass_rate - accepted[1].pass_rate) >= eps:
                    undiff_retry_succeeded += 1
                    break

        if accepted:
            self._assign_group_advantages(accepted)
        return accepted, produced_total, resample_count, node_serial, undiff_retry_triggered, undiff_retry_succeeded

    def _generate_node(
        self,
        question_id: str,
        node_id: str,
        parent: Node,
        prompt: str,
        test_cases: list[dict[str, Any]],
        diagnostic_inputs: list[Any],
        diagnostic_outputs: list[Any],
        io_mode: str,
        round_idx: int,
        prompt_text_override: str | None = None,
        prompt_text_raw_override: str | None = None,
        raw_output_override: str | None = None,
        generation_kwargs_override: dict[str, Any] | None = None,
        problem_logprob_cache: dict[tuple[str, str], float] | None = None,
        log_reward: bool = False,
    ) -> Node:
        history = list(parent.exec_summary.get("history", []))
        context_history = history[-self.args.context_round_window :]
        prompt_history_debug = summarize_generation_history(context_history)

        if prompt_text_override is None:
            prompt_text_raw = build_generation_prompt(
                question_prompt=prompt,
                history=context_history,
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
        parsed_code, _, _, _, generation_format_ok = (
            parse_generation_response(
                raw_output,
                allow_outside_noise_chars=self.args.generation_outside_noise_chars,
                prefilled_code=False,
            )
        )
        trace_max_chars = self.args.error_max_chars * 4
        trace_max_lines = self.args.error_max_lines * 4
        trace_store_full_text = bool(getattr(self.args, "trace_store_full_text", False))

        code = parsed_code or ""
        completion_text = build_generation_completion(code)
        token_ids, _, _ = build_token_masks(self.tokenizer, completion_text)
        code_mask = [1] * len(token_ids)

        case_inputs = [str(case["input"]) if io_mode == "stdio" else _parse_case_input(case["input"]) for case in test_cases]
        exec_results = execute_batch(
            code=code,
            case_inputs=case_inputs,
            timeout_s=self.args.code_timeout_seconds,
            error_max_chars=self.args.error_max_chars,
            error_max_lines=self.args.error_max_lines,
            io_mode=io_mode,
        )
        pass_flags = [
            self._is_test_pass(case["output"], actual, io_mode=io_mode)
            for case, actual in zip(test_cases, exec_results, strict=True)
        ]
        pass_cnt = int(sum(1 for ok in pass_flags if ok))
        test_count = len(test_cases)
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
            failed_case_actual = _safe_preview(actual.value) if actual.kind == "OK" else (actual.error_type or actual.kind)
            break

        hard_reward = float(pass_cnt / test_count) if test_count > 0 else 0.0
        zero_pass_triggered = bool(pass_cnt == 0 and test_count > 0 and bool(getattr(self.args, "zero_pass_soft_reward_enabled", True)))
        raw_soft_reward = 0.0
        normalized_soft_reward = 0.0
        soft_reward_beta = 0.0
        soft_reward_details: list[dict[str, Any]] = []
        if zero_pass_triggered:
            problem_payload = {
                "prompt": prompt,
                "test_cases": test_cases,
                "diagnostic_inputs": list(diagnostic_inputs),
                "diagnostic_outputs": list(diagnostic_outputs),
                "io_mode": io_mode,
            }
            diag_count = int(getattr(self.args, "zero_pass_soft_reward_diag_count", 0) or 0)
            selected_diagnostic_inputs = build_diagnostic_inputs(problem_payload, max_count=diag_count)
            oracle_outputs = get_oracle_outputs(problem_payload, selected_diagnostic_inputs)
            raw_soft_reward, soft_reward_details = compute_soft_reward(
                problem=problem_payload,
                code=code,
                diagnostic_inputs=selected_diagnostic_inputs,
                oracle_outputs=oracle_outputs,
                evaluator=self.backend,
                problem_logprob_cache=problem_logprob_cache,
            )
            normalized_soft_reward = normalize_soft_reward_to_unit_interval(
                raw_value=raw_soft_reward,
                clip_low=float(getattr(self.args, "zero_pass_soft_reward_clip_low", -2.0)),
                clip_high=float(getattr(self.args, "zero_pass_soft_reward_clip_high", 2.0)),
            )
            soft_reward_beta = compute_zero_pass_beta(
                test_count=test_count,
                beta_scale=float(getattr(self.args, "zero_pass_soft_reward_beta_scale", 0.5)),
            )
        soft_reward_eligible = zero_pass_triggered and _is_soft_reward_eligible(code, io_mode)
        if zero_pass_triggered and not soft_reward_eligible:
            normalized_soft_reward *= float(getattr(self.args, "soft_reward_ineligible_scale", 0.3))

        if pass_cnt > 0:
            final_reward = _compute_code_reward(
                pass_rate=hard_reward,
                r_soft=0.0,
                lambda_soft=0.0,
                compile_score=compile_score,
                compile_scale=self.args.code_compile_reward_scale,
                generation_format_score=generation_format_score,
                generation_format_scale=self.args.code_format_reward_scale,
                aux_reward_scale=(1.0 if generation_format_ok else self.args.code_aux_reward_without_format_scale),
            )
        else:
            final_reward = hard_reward + soft_reward_beta * normalized_soft_reward

        code_io_train_samples = _build_code_io_aux_samples(
            question_id=question_id,
            code=code,
            test_cases=test_cases,
            case_inputs=case_inputs,
            exec_results=exec_results,
            pass_flags=pass_flags,
            enabled=bool(getattr(self.args, "code_io_aux_training_enabled", False)),
            case_count=int(getattr(self.args, "code_io_aux_case_count", 0) or 0),
            include_correct=bool(getattr(self.args, "code_io_aux_include_correct", True)),
            include_incorrect=bool(getattr(self.args, "code_io_aux_include_incorrect", True)),
            include_errors=bool(getattr(self.args, "code_io_aux_include_errors", True)),
            sft_weight_correct=float(getattr(self.args, "code_io_aux_sft_weight_correct", 1.0)),
            sft_weight_incorrect=float(getattr(self.args, "code_io_aux_sft_weight_incorrect", 1.0)),
            prompt_renderer=self._render_prompt_with_chat_template,
        )

        child = Node(
            node_id=node_id,
            parent_id=parent.node_id,
            round_idx=round_idx,
            code=code,
            pass_rate=pass_rate,
            R_soft=normalized_soft_reward,
            R_code=final_reward,
            status_code=status_code,
            frozen_code=pass_rate == 1.0,
            exec_summary={
                "error_summary": error_summary,
                "generation_format_ok": generation_format_ok,
                "generation_format_score": generation_format_score,
                "compile_score": compile_score,
                "soft_reward_eligible": bool(soft_reward_eligible),
                "pass_cnt": pass_cnt,
                "test_count": test_count,
                "hard_reward": hard_reward,
                "zero_pass_soft_reward_triggered": zero_pass_triggered,
                "raw_soft_reward": raw_soft_reward,
                "normalized_soft_reward": normalized_soft_reward,
                "soft_reward_beta": soft_reward_beta,
                "final_reward": final_reward,
                "soft_reward_details": soft_reward_details,
                "code_io_train_samples": code_io_train_samples,
                "generation_debug": {
                    "prompt_preview": summarize_error(prompt_text_raw, trace_max_chars, trace_max_lines),
                    "latest_feedback_summary": summarize_error(prompt_history_debug.get("latest_feedback", ""), trace_max_chars, trace_max_lines),
                    "earlier_history_summary": summarize_error(prompt_history_debug.get("earlier_summary", ""), trace_max_chars, trace_max_lines),
                    "raw_output": summarize_error(raw_output, trace_max_chars, trace_max_lines),
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
                "history": history
                + [
                    {
                        "round": str(round_idx),
                        "status_code": status_code,
                        "error_summary": error_summary,
                        "code_preview": summarize_error(code, max_chars=self.args.error_max_chars, max_lines=self.args.error_max_lines),
                        "failed_input": failed_case_input,
                        "failed_actual": failed_case_actual,
                        "generation_format_ok": generation_format_ok,
                        "compile_score": compile_score,
                        "pass_cnt": pass_cnt,
                        "test_count": test_count,
                        "hard_reward": hard_reward,
                        "zero_pass_soft_reward_triggered": zero_pass_triggered,
                        "raw_soft_reward": raw_soft_reward,
                        "normalized_soft_reward": normalized_soft_reward,
                        "soft_reward_beta": soft_reward_beta,
                        "final_reward": final_reward,
                        "code_io_aux_sample_count": len(code_io_train_samples),
                    }
                ],
            },
            completion_text=completion_text,
            code_token_mask=code_mask,
            prompt_text=prompt_text,
        )
        if log_reward or bool(getattr(self.args, "log_train_rollout_details", False)):
            self.logger.info(
                "[BASELINE] qid=%s node=%s pass=%d/%d hard=%.4f soft_on=%s raw_soft=%.4f norm_soft=%.4f beta=%.6f final=%.4f",
                question_id,
                node_id,
                pass_cnt,
                test_count,
                hard_reward,
                zero_pass_triggered,
                raw_soft_reward,
                normalized_soft_reward,
                soft_reward_beta,
                final_reward,
            )
        return child

    def _assign_group_advantages(self, siblings: list[Node]) -> None:
        advantage_base = getattr(self.args, "advantage_base", "R_code")
        advantage_mode = getattr(self.args, "advantage_mode", "zscore")
        code_vals = [node.pass_rate if advantage_base == "pass_rate" else node.R_code for node in siblings]
        mean_code = _mean(code_vals)
        std_code = _std(code_vals)
        eps = 1e-8
        for node, val in zip(siblings, code_vals, strict=True):
            if advantage_mode == "sign":
                if len(siblings) != 2:
                    node.A_code = val - mean_code
                elif std_code <= eps:
                    node.A_code = 0.0
                elif val > mean_code:
                    node.A_code = 1.0
                else:
                    node.A_code = -1.0
            elif advantage_mode == "mean_only":
                node.A_code = val - mean_code
            else:
                node.A_code = (val - mean_code) / (std_code + eps)
    def _is_test_pass(self, expected_output: Any, actual: ExecResult, io_mode: str = "call") -> bool:
        if actual.kind != "OK":
            return False
        if io_mode == "stdio":
            return stripped_text_equal(expected_output, actual.value)
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
            metrics[key] = 1.0 if any(float(node["pass_rate"]) == 1.0 for node in round_nodes[:k]) else 0.0
        if k_list:
            metrics["pass_at_k_round_n"] = metrics.get(f"pass_at_{k_list[0]}_round_{round_n}", 0.0)
        flat_nodes = [node for round_item in eval_rounds for node in round_item["nodes"]]
        for k in k_list:
            metrics[f"pass_at_{k}"] = 1.0 if any(float(node["pass_rate"]) == 1.0 for node in flat_nodes[:k]) else 0.0
        metrics["best_pass_rate_overall"] = max((float(node["pass_rate"]) for node in flat_nodes), default=0.0)
        metrics[f"best_pass_rate_round_{round_n}"] = max((float(node["pass_rate"]) for node in round_nodes), default=0.0)
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
