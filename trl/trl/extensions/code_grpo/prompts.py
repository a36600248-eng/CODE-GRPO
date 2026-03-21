import json
from collections import Counter
from typing import Any


def serialize_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return repr(value)


GENERATION_FEWSHOT = """Format example (follow exactly this structure):
```python
def solve(x):
    return x * 2
```"""


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
    failed_input = latest.get("failed_input")
    if failed_input is not None:
        latest_lines.append(f"- failed_input={serialize_value(failed_input)}")
    failed_actual = latest.get("failed_actual")
    if failed_actual is not None:
        latest_lines.append(f"- actual_result_or_error={serialize_value(failed_actual)}")
    error_summary = str(latest.get("error_summary", "") or "").strip()
    if error_summary:
        latest_lines.append(f"- error_summary={error_summary}")
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
        earlier_summary = "\n".join(earlier_lines)

    return {"latest_feedback": latest_feedback, "earlier_summary": earlier_summary}


def build_generation_prompt(
    question_prompt: str,
    history: list[dict[str, Any]],
    parent_code: str | None = None,
) -> str:
    history_summary = summarize_generation_history(history)
    parts = [
        "You solve one Python programming task.",
        "Return exactly one fenced Python code block.",
        "Rules:",
        "1) Use ```python as the opening fence and ``` as the closing fence.",
        "2) No text before the opening fence or after the closing fence.",
        "3) Do NOT output any reasoning or prediction tags in this stage.",
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
    parts.append("Now answer with one fenced Python code block only.")
    return "\n".join(parts)


def build_zero_pass_problem_view_prompt(question_prompt: str, case_input: Any) -> str:
    return (
        "You are given a Python programming task and one concrete input.\n"
        "Predict the exact output for the input.\n"
        "Output the final answer only.\n"
        "No reasoning. No explanation. No markdown. No extra text.\n"
        f"Problem:\n{question_prompt.strip()}\n"
        f"Input:\n{serialize_value(case_input)}\n"
        "Output:\n"
    )


def build_zero_pass_code_view_prompt(code: str, case_input: Any) -> str:
    return (
        "You are given Python code and one concrete input.\n"
        "Predict the exact output produced by the code for the input.\n"
        "Output the final answer only.\n"
        "No reasoning. No explanation. No markdown. No extra text.\n"
        f"Code:\n{code}\n"
        f"Input:\n{serialize_value(case_input)}\n"
        "Output:\n"
    )


def build_code_io_training_prompt(code: str, case_input: Any) -> str:
    return (
        "You are given Python code and one concrete input.\n"
        "Predict the exact program behavior for this input.\n"
        "If execution returns a value, output that value only.\n"
        "If execution fails, output the error type only (for example: SyntaxError, RuntimeError, Timeout).\n"
        "Do not explain. Do not reason. Do not use markdown. Output one final answer only.\n"
        f"Code:\n{code}\n"
        f"Input:\n{serialize_value(case_input)}\n"
        "Output:\n"
    )
