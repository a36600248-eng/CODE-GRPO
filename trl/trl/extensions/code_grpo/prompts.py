import json
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
</CODE>
<REASON>
The task asks for double of the input.
<LOGIC_PREDICTION>
6
</LOGIC_PREDICTION>
<EXEC_PREDICTION>
6
</EXEC_PREDICTION>
</REASON>"""


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
        "You solve programming tasks.",
        "Return output in this exact tag order and include all tags exactly once:",
        "<CODE>...</CODE>",
        "<REASON>...<LOGIC_PREDICTION>...</LOGIC_PREDICTION><EXEC_PREDICTION>...</EXEC_PREDICTION></REASON>",
        "Hard constraints:",
        "1) Output only these tags and their contents. No markdown, no extra prose outside tags.",
        "2) <REASON> must appear before both prediction tags.",
        "3) <LOGIC_PREDICTION> is the ideal output by intended logic (ignore minor implementation issues).",
        "4) <EXEC_PREDICTION> is the actual execution outcome for the code (value or error type).",
        "5) Always provide concise final predictions (single value or error name).",
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
        parts.append("Recent execution feedback and failed-case summaries:")
        for item in history:
            entry = (
                f"- round={item.get('round', '')}, status={item.get('status_code', '')}, "
                f"error={item.get('error_summary', '')}"
            )
            code_preview = (item.get("code_preview") or "").strip()
            if code_preview:
                entry += f", code_snippet={code_preview}"
            failed_input = item.get("failed_input")
            failed_actual = item.get("failed_actual")
            if failed_input is not None:
                entry += (
                    f", failed_input={serialize_value(failed_input)}, "
                    f"failed_actual={serialize_value(failed_actual)}"
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
        "Given task statement, Python code, and an input, infer intended logic and predict the ideal output.\n"
        "Ignore implementation-level issues such as syntax errors.\n"
        "Output format is mandatory and must contain only these tags:\n"
        "<REASON>...</REASON>\n"
        "<LOGIC_PREDICTION>...</LOGIC_PREDICTION>\n"
        "Hard constraints:\n"
        "1) <REASON> must appear before <LOGIC_PREDICTION>.\n"
        "2) No extra text outside tags.\n"
        "3) Prediction must be a concise final value.\n"
        f"{LOGIC_FEWSHOT}\n"
        f"Question:\n{question_prompt.strip()}\n"
        f"Code:\n{code}\n"
        f"Input:\n{serialize_value(case_input)}\n"
        "Now answer for the current code/input with the required tags only."
    )


def build_exec_prompt(
    code: str,
    case_input: Any,
) -> str:
    return (
        "Analyze Python code and predict actual execution result for the input.\n"
        "Output format is mandatory and must contain only these tags:\n"
        "<REASON>...</REASON>\n"
        "<EXEC_PREDICTION>...</EXEC_PREDICTION>\n"
        "Hard constraints:\n"
        "1) <REASON> must appear before <EXEC_PREDICTION>.\n"
        "2) No extra text outside tags.\n"
        "3) Prediction must be either the concrete output value or an error type (e.g., SyntaxError, Timeout).\n"
        f"{EXEC_FEWSHOT}\n"
        f"Code:\n{code}\n"
        f"Input:\n{serialize_value(case_input)}\n"
        "Now answer for the current code/input with the required tags only."
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
