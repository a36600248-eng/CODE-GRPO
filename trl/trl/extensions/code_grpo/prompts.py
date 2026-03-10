import json
from collections import Counter
from typing import Any


def serialize_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return repr(value)


GENERATION_FEWSHOT = """Format example (follow exactly this structure):
<CODE>
def solve(x):
    return x * 2
</CODE>"""

GENERATION_PREFILL = "<CODE>\n"


LOGIC_FEWSHOT = """Example:
Code:
def solve(x)
    return x * 2
Input:
3
Answer:
<REASON>The intended logic is to multiply input by 2, ignoring syntax issues.</REASON>
<LOGIC_PREDICTION>6</LOGIC_PREDICTION>"""


EXEC_FEWSHOT = """Example:
Code:
def solve(x)
    return x * 2
Input:
3
Answer:
<REASON>This code has a function-definition syntax problem, so execution raises an error.</REASON>
<EXEC_PREDICTION>SyntaxError</EXEC_PREDICTION>"""


def summarize_generation_history(history: list[dict[str, Any]]) -> dict[str, str]:
    latest_feedback = ""
    earlier_summary = ""
    if not history:
        return {"latest_feedback": latest_feedback, "earlier_summary": earlier_summary}

    latest = history[-1]
    latest_lines: list[str] = []
    status = str(latest.get("status_code", "") or "").strip()
    if status:
        latest_lines.append(f"- latest_status={status}")
    status_reason = str(latest.get("status_reason", "") or "").strip()
    if status_reason:
        latest_lines.append(f"- latest_reason_status={status_reason}")
    failed_input = latest.get("failed_input")
    if failed_input is not None:
        latest_lines.append(f"- failed_input={serialize_value(failed_input)}")
    failed_actual = latest.get("failed_actual")
    if failed_actual is not None:
        latest_lines.append(f"- actual_result_or_error={serialize_value(failed_actual)}")
    error_summary = str(latest.get("error_summary", "") or "").strip()
    if error_summary:
        latest_lines.append(f"- error_summary={error_summary}")
    logic_mismatch_count = latest.get("logic_mismatch_count")
    if logic_mismatch_count not in (None, ""):
        latest_lines.append(f"- logic_mismatch_count={logic_mismatch_count}")
    exec_mismatch_count = latest.get("exec_mismatch_count")
    if exec_mismatch_count not in (None, ""):
        latest_lines.append(f"- exec_mismatch_count={exec_mismatch_count}")
    latest_feedback = "\n".join(latest_lines)

    earlier = history[:-1]
    if earlier:
        earlier_lines: list[str] = [f"- earlier_rounds={len(earlier)}"]
        status_counts = Counter(str(item.get("status_code", "") or "").strip() for item in earlier if item.get("status_code"))
        if status_counts:
            packed = ", ".join(f"{key}:{status_counts[key]}" for key in sorted(status_counts))
            earlier_lines.append(f"- status_counts={packed}")
        repeated_failed_inputs = Counter(
            serialize_value(item.get("failed_input"))
            for item in earlier
            if item.get("failed_input") is not None
        )
        if repeated_failed_inputs:
            top_input, top_count = repeated_failed_inputs.most_common(1)[0]
            if top_count >= 2:
                earlier_lines.append(f"- repeated_failure_input={top_input}")
        if any(float(item.get("compile_score", 0.0) or 0.0) <= 0.0 for item in earlier):
            earlier_lines.append("- some_earlier_attempts_failed_to_compile")
        if any(int(item.get("logic_mismatch_count", 0) or 0) > 0 for item in earlier):
            earlier_lines.append("- earlier_attempts_had_logic_mismatches")
        if any(int(item.get("exec_mismatch_count", 0) or 0) > 0 for item in earlier):
            earlier_lines.append("- earlier_attempts_had_execution_mismatches")
        earlier_summary = "\n".join(earlier_lines)

    return {"latest_feedback": latest_feedback, "earlier_summary": earlier_summary}


def build_generation_prompt(
    question_prompt: str,
    history: list[dict[str, Any]],
    need_code: bool,
    need_reason: bool,
    parent_code: str | None = None,
) -> str:
    del need_reason
    history_summary = summarize_generation_history(history)
    parts = [
        "You solve one Python programming task.",
        "The assistant reply is prefilled with the opening tag below.",
        "Write only Python code, then close the block with </CODE> exactly once.",
        "Rules:",
        "1) No markdown fences.",
        "2) Do not repeat <CODE>.",
        "3) No text after </CODE>.",
        "4) Do NOT output any reasoning or prediction tags in this stage.",
        GENERATION_FEWSHOT,
        "Question:",
        question_prompt.strip(),
    ]
    if parent_code:
        parts.extend(
            [
                "Parent code from previous round (revise this code instead of rewriting from scratch):",
                "<PARENT_CODE>",
                parent_code.strip(),
                "</PARENT_CODE>",
            ]
        )
    if history_summary["latest_feedback"]:
        parts.extend(["Latest feedback:", history_summary["latest_feedback"]])
    if history_summary["earlier_summary"]:
        parts.extend(["Earlier history summary:", history_summary["earlier_summary"]])
    if not need_code:
        parts.append("Code is frozen; keep prior code behavior unchanged.")
    parts.extend(
        [
            "Continue immediately after this opening tag and output code only:",
            "<CODE>",
        ]
    )
    return "\n".join(parts)


def build_logic_prompt(code: str, case_input: Any, question_prompt: str = "") -> str:
    return (
        "You evaluate intended code logic; do not rewrite code.\n"
        "Infer the intended output for this input.\n"
        "Ignore implementation-level issues such as syntax/runtime errors.\n"
        "Output exactly these tags, once, in this order:\n"
        "<REASON>...</REASON>\n"
        "<LOGIC_PREDICTION>...</LOGIC_PREDICTION>\n"
        "No markdown, no extra text outside tags.\n"
        "<REASON> must be one short sentence.\n"
        "<LOGIC_PREDICTION> must be one-line final value only.\n"
        f"{LOGIC_FEWSHOT}\n"
        f"Question:\n{question_prompt.strip()}\n"
        f"Code:\n{code}\n"
        f"Input:\n{serialize_value(case_input)}\n"
        "Now respond with tags only."
    )


def build_exec_prompt(
    code: str,
    case_input: Any,
) -> str:
    return (
        "You simulate actual execution of the given code.\n"
        "Predict actual result for the input (value or error type).\n"
        "Output exactly these tags, once, in this order:\n"
        "<REASON>...</REASON>\n"
        "<EXEC_PREDICTION>...</EXEC_PREDICTION>\n"
        "No markdown, no extra text outside tags.\n"
        "<REASON> must be one short sentence.\n"
        "<EXEC_PREDICTION> must be one-line final value or error type (e.g., SyntaxError).\n"
        f"{EXEC_FEWSHOT}\n"
        f"Code:\n{code}\n"
        f"Input:\n{serialize_value(case_input)}\n"
        "Now respond with tags only."
    )


def build_frozen_reason_prompt(
    code: str,
    case_input: Any,
) -> str:
    return (
        "You are given a fixed Python function. Do not modify code.\n"
        "Simulate execution for the input and output exactly these tags once:\n"
        "<REASON>...</REASON>\n"
        "<EXEC_PREDICTION>...</EXEC_PREDICTION>\n"
        "No markdown, no extra text outside tags.\n"
        "<REASON> must be one short sentence.\n"
        "<EXEC_PREDICTION> must be one-line final value or error type.\n"
        f"{EXEC_FEWSHOT}\n"
        f"Code:\n{code}\n"
        f"Input:\n{serialize_value(case_input)}\n"
        "Now respond with tags only."
    )


def build_soft_prompt(code: str, case_input: Any, question_prompt: str = "") -> str:
    """Backward-compatible alias for logic prediction prompt."""
    return build_logic_prompt(code=code, case_input=case_input, question_prompt=question_prompt)


def build_reason_prompt(code: str, case_input: Any) -> str:
    """Backward-compatible alias for execution prediction prompt."""
    return build_exec_prompt(
        code=code,
        case_input=case_input,
    )
