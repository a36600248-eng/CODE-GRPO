from typing import Any

from .base import DatasetAdapter


def _to_test_cases(value: Any, example: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, dict):
                if "input" in item and "output" in item:
                    normalized.append({"input": item["input"], "output": item["output"]})
                elif "stdin" in item and "stdout" in item:
                    normalized.append({"input": item["stdin"], "output": item["stdout"]})
        if normalized:
            return normalized

    if isinstance(example.get("input"), list) and isinstance(example.get("output"), list):
        return [{"input": i, "output": o} for i, o in zip(example["input"], example["output"], strict=True)]

    raise ValueError("Could not normalize test cases. Expected list of {input, output} or parallel input/output lists.")


def _infer_io_mode(example: dict[str, Any], test_cases_source: Any) -> str:
    explicit = example.get("io_mode")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip().lower()
    if isinstance(test_cases_source, list):
        for item in test_cases_source:
            if isinstance(item, dict) and "stdin" in item and "stdout" in item:
                return "stdio"
    return "call"


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
        io_mode = _infer_io_mode(example, test_cases_source)
        if io_mode not in {"call", "stdio"}:
            raise ValueError(f"Example {question_id} has unsupported io_mode: {io_mode}")

        return {
            **example,
            "question_id": question_id,
            "prompt": prompt,
            "test_cases": test_cases,
            "io_mode": io_mode,
        }
