import json
from typing import Any


def serialize_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return repr(value)


GENERATION_FEWSHOT = """Format example (follow exactly this structure):
<REASON>
Use multiplication by 2.
</REASON>
<CODE>
def solve(x):
    return x * 2
</CODE>"""


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


def build_generation_prompt(
    question_prompt: str,
    history: list[dict[str, Any]],
    need_code: bool,
    need_reason: bool,
    parent_code: str | None = None,
) -> str:
    parts = [
        "You solve one Python programming task.",
        "Return exactly two tags in this order, each exactly once:",
        "<REASON>...</REASON>",
        "<CODE>...</CODE>",
        "Rules:",
        "1) No markdown fences.",
        "2) No text before <REASON> or after </CODE>.",
        "3) Do NOT output prediction tags in this stage.",
        "4) <REASON> must be one short sentence describing algorithm intent.",
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
    if history:
        parts.append("Recent execution and reasoning feedback summaries:")
        for item in history:
            entry = (
                f"- round={item.get('round', '')}, status={item.get('status_code', '')}, "
                f"error={item.get('error_summary', '')}"
            )
            status_reason = item.get("status_reason")
            if status_reason:
                entry += f", reason_status={status_reason}"
            code_preview = (item.get("code_preview") or "").strip()
            if code_preview:
                entry += f", code_snippet={code_preview}"
            logic_mismatch_count = item.get("logic_mismatch_count")
            if logic_mismatch_count is not None:
                entry += f", logic_mismatch_count={logic_mismatch_count}"
            exec_mismatch_count = item.get("exec_mismatch_count")
            if exec_mismatch_count is not None:
                entry += f", exec_mismatch_count={exec_mismatch_count}"
            logic_fmt = item.get("logic_format_ok_rate")
            if logic_fmt is not None:
                entry += f", logic_format_ok_rate={logic_fmt}"
            exec_fmt = item.get("exec_format_ok_rate")
            if exec_fmt is not None:
                entry += f", exec_format_ok_rate={exec_fmt}"
            failed_input = item.get("failed_input")
            failed_actual = item.get("failed_actual")
            if failed_input is not None:
                entry += (
                    f", failed_input={serialize_value(failed_input)}, "
                    f"failed_actual={serialize_value(failed_actual)}"
                )
            logic_failed_input = item.get("logic_failed_input")
            if logic_failed_input is not None:
                entry += (
                    f", logic_failed_input={serialize_value(logic_failed_input)}, "
                    f"logic_failed_prediction={serialize_value(item.get('logic_failed_prediction'))}"
                )
            exec_failed_input = item.get("exec_failed_input")
            if exec_failed_input is not None:
                entry += (
                    f", exec_failed_input={serialize_value(exec_failed_input)}, "
                    f"exec_failed_prediction={serialize_value(item.get('exec_failed_prediction'))}, "
                    f"exec_actual_kind={serialize_value(item.get('exec_actual_kind'))}"
                )
            parts.append(entry)
    if not need_code:
        parts.append("Code is frozen; keep prior code behavior unchanged.")
    if not need_reason:
        parts.append("Reasoning is frozen; keep prior reasoning behavior unchanged.")
    parts.append("Now answer the current question using the exact required tags only.")
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
