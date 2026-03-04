import ast
from typing import Any

from .base import DatasetAdapter


def _to_test_cases(value: Any, example: dict[str, Any]) -> list[dict[str, Any]]:
    def _decode_case_value(v: Any) -> Any:
        if not isinstance(v, str):
            return v
        text = v.strip()
        if not text:
            return v
        try:
            return ast.literal_eval(text)
        except Exception:  # noqa: BLE001
            return v

    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, dict):
                if "input" in item and "output" in item:
                    normalized.append({"input": _decode_case_value(item["input"]), "output": _decode_case_value(item["output"])})
                elif "stdin" in item and "stdout" in item:
                    normalized.append(
                        {"input": _decode_case_value(item["stdin"]), "output": _decode_case_value(item["stdout"])}
                    )
        if normalized:
            return normalized

    if isinstance(example.get("input"), list) and isinstance(example.get("output"), list):
        return [
            {"input": _decode_case_value(i), "output": _decode_case_value(o)}
            for i, o in zip(example["input"], example["output"], strict=True)
        ]

    raise ValueError("Could not normalize test cases. Expected list of {input, output} or parallel input/output lists.")


class DefaultCodeDatasetAdapter(DatasetAdapter):
    """Heuristic adapter to normalize common code dataset schemas."""

    def adapt_example(self, example: dict[str, Any], index: int) -> dict[str, Any]:
        question_id = str(example.get("question_id") or example.get("id") or f"q_{index}")
        prompt = (
            example.get("prompt")
            or example.get("question")
            or example.get("instruction")
            or example.get("problem")
            or ""
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Example {question_id} has empty prompt.")

        test_cases_source = example.get("test_cases", example.get("tests"))
        test_cases = _to_test_cases(test_cases_source, example)

        return {
            **example,
            "question_id": question_id,
            "prompt": prompt,
            "test_cases": test_cases,
        }
