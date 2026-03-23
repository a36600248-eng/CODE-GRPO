import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .executor import execute_batch
from .matcher import stripped_text_equal


PYTHON_LANGUAGE_ALIASES = {"py", "python", "python3", "cpython"}
PYTHON_LANGUAGE_IDS = {1, 3}
DISALLOWED_IMPORT_PATTERNS = (
    r"(^|\n)\s*import\s+(numpy|pandas|scipy|torch|tensorflow|sklearn|requests)\b",
    r"(^|\n)\s*from\s+(numpy|pandas|scipy|torch|tensorflow|sklearn|requests)\b",
)
INTERACTIVE_PATTERNS = (
    "interactive",
    "flush",
    "output flushing",
    "hacked interaction",
)
FILE_IO_PATTERNS = (
    "input.txt",
    "output.txt",
    "read from file",
    "write to file",
    "file input",
    "file output",
)
BASELINE_BANK = {
    "print_0": "print(0)",
    "print_1": "print(1)",
    "empty_output": "pass",
    "echo_input": "import sys\nprint(sys.stdin.read(), end='')",
    "placeholder": "def solve(x):\n    return None\n",
}
DEFAULT_SOURCE_TARGETS = {
    "apps": 100,
    "codecontests": 200,
    "taco": 200,
    "codeforces": 0,
}
DEFAULT_MAX_CASE_INPUT_CHARS = 5000
DEFAULT_MAX_CASE_OUTPUT_CHARS = 5000
DEFAULT_MAX_DIAGNOSTIC_CASES = 4
STDIO_PROMPT_PREFIX = (
    "Solve the following programming problem in Python.\n"
    "Read from standard input and write to standard output.\n"
    "Do not add explanations.\n\n"
    "Problem:\n"
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _load_records_from_path(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return _read_jsonl(path, limit=limit)
        if suffix == ".json":
            payload = _read_json(path)
            if isinstance(payload, list):
                rows = [row for row in payload if isinstance(row, dict)]
                return rows[:limit] if limit is not None else rows
            if isinstance(payload, dict):
                for candidate_key in ("train", "test", "validation", "data", "examples", "problems"):
                    candidate = payload.get(candidate_key)
                    if isinstance(candidate, list):
                        rows = [row for row in candidate if isinstance(row, dict)]
                        return rows[:limit] if limit is not None else rows
                return [payload]
            return []
        if suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(path), split="train")
            rows: list[dict[str, Any]] = []
            for row in dataset:
                rows.append(dict(row))
                if limit is not None and len(rows) >= limit:
                    break
            return rows
        raise ValueError(f"Unsupported source file: {path}")

    if path.is_dir():
        files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".jsonl", ".json", ".parquet"}
        )
        rows: list[dict[str, Any]] = []
        for file_path in files:
            remaining = None if limit is None else max(0, limit - len(rows))
            if remaining == 0:
                break
            rows.extend(_load_records_from_path(file_path, limit=remaining))
            if limit is not None and len(rows) >= limit:
                break
        return rows

    raise FileNotFoundError(path)


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except Exception:
            return value
    return value


def _clean_prompt(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _approx_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _normalize_text_signature(text: str) -> str:
    lowered = _clean_prompt(text).lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_test_case_io(value: Any) -> str | None:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _normalize_optional_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _extract_tests_from_mapping(mapping: dict[str, Any]) -> list[dict[str, str]]:
    inputs = mapping.get("inputs", mapping.get("input", mapping.get("stdin")))
    outputs = mapping.get("outputs", mapping.get("output", mapping.get("stdout")))
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(outputs, str):
        outputs = [outputs]
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return []
    cases: list[dict[str, str]] = []
    for raw_input, raw_output in zip(inputs, outputs, strict=False):
        normalized_input = _normalize_test_case_io(raw_input)
        normalized_output = _normalize_test_case_io(raw_output)
        if normalized_input is None or normalized_output is None:
            return []
        cases.append({"input": normalized_input, "output": normalized_output})
    return cases


def _collect_codecontests_tests(example: dict[str, Any]) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for key in ("public_tests", "private_tests", "generated_tests", "tests"):
        payload = _parse_jsonish(example.get(key))
        if isinstance(payload, dict):
            cases.extend(_extract_tests_from_mapping(payload))
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    if "input" in item and "output" in item:
                        normalized_input = _normalize_test_case_io(item["input"])
                        normalized_output = _normalize_test_case_io(item["output"])
                        if normalized_input is None or normalized_output is None:
                            return []
                        cases.append({"input": normalized_input, "output": normalized_output})
                    else:
                        extracted = _extract_tests_from_mapping(item)
                        if not extracted:
                            return []
                        cases.extend(extracted)
    return cases


def _extract_python_solutions(value: Any) -> list[str]:
    payload = _parse_jsonish(value)
    if isinstance(payload, str):
        return [payload] if payload.strip() else []
    if isinstance(payload, list):
        if all(isinstance(item, str) for item in payload):
            return [item for item in payload if item.strip()]
        solutions: list[str] = []
        for item in payload:
            if isinstance(item, dict):
                raw_language = item.get("language", item.get("lang", ""))
                language = str(raw_language).strip().lower()
                code = item.get("solution", item.get("code", item.get("text", "")))
                is_python_language = language in PYTHON_LANGUAGE_ALIASES
                if isinstance(raw_language, int):
                    is_python_language = is_python_language or raw_language in PYTHON_LANGUAGE_IDS
                if is_python_language and isinstance(code, str) and code.strip():
                    solutions.append(code)
        return solutions
    if isinstance(payload, dict):
        languages = payload.get("language", payload.get("languages", payload.get("lang")))
        solutions = payload.get("solution", payload.get("solutions", payload.get("code")))
        if isinstance(languages, list) and isinstance(solutions, list):
            collected = []
            for language, solution in zip(languages, solutions, strict=False):
                normalized_language = str(language).strip().lower()
                is_python_language = normalized_language in PYTHON_LANGUAGE_ALIASES
                if isinstance(language, int):
                    is_python_language = is_python_language or language in PYTHON_LANGUAGE_IDS
                if is_python_language and isinstance(solution, str) and solution.strip():
                    collected.append(solution)
            return collected
    return []


def _looks_interactive(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(pattern in lowered for pattern in INTERACTIVE_PATTERNS)


def _looks_like_file_io(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(pattern in lowered for pattern in FILE_IO_PATTERNS)


def _uses_disallowed_imports(code: str) -> bool:
    return any(re.search(pattern, code) for pattern in DISALLOWED_IMPORT_PATTERNS)


def _dedupe_cases(cases: list[dict[str, str]], max_cases: int | None = None) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique_cases: list[dict[str, str]] = []
    for case in cases:
        key = (case["input"], case["output"])
        if key in seen:
            continue
        seen.add(key)
        unique_cases.append(case)
        if max_cases is not None and max_cases > 0 and len(unique_cases) >= max_cases:
            break
    return unique_cases


def _case_input_len(case: dict[str, str]) -> int:
    return len(str(case.get("input", "")))


def _case_output_len(case: dict[str, str]) -> int:
    return len(str(case.get("output", "")))


def _case_total_len(case: dict[str, str]) -> int:
    return _case_input_len(case) + _case_output_len(case)


def _select_representative_cases(indexed_cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(indexed_cases) <= limit:
        return list(indexed_cases)
    ranked = sorted(indexed_cases, key=lambda case: (_case_total_len(case), int(case["_idx"])))
    if limit == 1:
        return [ranked[0]]

    selected_positions: list[int] = []
    for slot in range(limit):
        position = round(slot * (len(ranked) - 1) / (limit - 1))
        selected_positions.append(position)

    selected_indices: set[int] = set()
    selected: list[dict[str, Any]] = []
    for position in selected_positions:
        candidate_index = None
        for radius in range(len(ranked)):
            left = position - radius
            if left >= 0 and left not in selected_indices:
                candidate_index = left
                break
            right = position + radius
            if right < len(ranked) and right not in selected_indices:
                candidate_index = right
                break
        if candidate_index is None:
            continue
        selected_indices.add(candidate_index)
        selected.append(ranked[candidate_index])

    if len(selected) < limit:
        for idx, case in enumerate(ranked):
            if idx in selected_indices:
                continue
            selected.append(case)
            if len(selected) >= limit:
                break

    return sorted(selected, key=lambda case: int(case["_idx"]))


def _prepare_problem_cases(
    cases: list[dict[str, str]],
    *,
    max_train_cases: int,
    max_diagnostic_cases: int,
    max_case_input_chars: int,
    max_case_output_chars: int,
) -> dict[str, Any]:
    deduped_cases = _dedupe_cases(cases, max_cases=None)
    indexed_cases = [{**case, "_idx": idx} for idx, case in enumerate(deduped_cases)]
    filtered_cases = [
        case
        for case in indexed_cases
        if _case_input_len(case) <= max_case_input_chars and _case_output_len(case) <= max_case_output_chars
    ]
    selected_cases = _select_representative_cases(filtered_cases, max_train_cases)
    diagnostic_cases = sorted(selected_cases, key=lambda case: (_case_total_len(case), int(case["_idx"])))[:max_diagnostic_cases]

    return {
        "test_cases": [{"input": str(case["input"]), "output": str(case["output"])} for case in selected_cases],
        "diagnostic_inputs": [str(case["input"]) for case in diagnostic_cases],
        "diagnostic_outputs": [str(case["output"]) for case in diagnostic_cases],
        "source_test_count": len(deduped_cases),
        "selected_test_count": len(selected_cases),
        "oversized_test_count": max(0, len(deduped_cases) - len(filtered_cases)),
    }


def _extract_apps_like_problem(
    source: str,
    example: dict[str, Any],
    index: int,
    max_tests: int,
    max_case_input_chars: int,
    max_case_output_chars: int,
    max_diagnostic_cases: int,
) -> tuple[dict[str, Any] | None, str | None]:
    input_output = _parse_jsonish(example.get("input_output"))
    if not isinstance(input_output, dict):
        return None, "missing_input_output"
    if input_output.get("fn_name"):
        return None, "non_stdio_task"

    test_cases = _extract_tests_from_mapping(input_output)
    if not test_cases:
        return None, "missing_tests"

    prompt = str(example.get("question") or example.get("prompt") or example.get("problem") or "").strip()
    if not prompt:
        return None, "missing_prompt"

    solutions = _extract_python_solutions(example.get("solutions"))
    if not solutions:
        solutions = _extract_python_solutions(example.get("reference_solution"))
    if not solutions:
        return None, "missing_python_reference_solution"

    source_problem_id = example.get("problem_id", example.get("id", f"{source}_{index}"))
    prepared_cases = _prepare_problem_cases(
        test_cases,
        max_train_cases=max_tests,
        max_diagnostic_cases=max_diagnostic_cases,
        max_case_input_chars=max_case_input_chars,
        max_case_output_chars=max_case_output_chars,
    )

    row = {
        "question_id": f"{source}_{source_problem_id}",
        "prompt": f"{STDIO_PROMPT_PREFIX}{_clean_prompt(prompt)}",
        "test_cases": prepared_cases["test_cases"],
        "diagnostic_inputs": prepared_cases["diagnostic_inputs"],
        "diagnostic_outputs": prepared_cases["diagnostic_outputs"],
        "reference_solution": solutions[0],
        "source": source,
        "io_mode": "stdio",
        "source_problem_id": str(source_problem_id),
        "difficulty": _normalize_optional_string(example.get("difficulty")),
        "source_url": _normalize_optional_string(example.get("url")),
        "source_prompt": prompt.strip(),
        "source_test_count": int(prepared_cases["source_test_count"]),
        "selected_test_count": int(prepared_cases["selected_test_count"]),
        "oversized_test_count": int(prepared_cases["oversized_test_count"]),
    }
    return row, None


def _extract_codecontests_problem(
    example: dict[str, Any],
    index: int,
    max_tests: int,
    max_case_input_chars: int,
    max_case_output_chars: int,
    max_diagnostic_cases: int,
) -> tuple[dict[str, Any] | None, str | None]:
    prompt = str(example.get("description") or example.get("prompt") or example.get("problem") or "").strip()
    if not prompt:
        return None, "missing_prompt"

    test_cases = _collect_codecontests_tests(example)
    if not test_cases:
        return None, "missing_tests"

    solutions = _extract_python_solutions(example.get("solutions"))
    if not solutions:
        solutions = _extract_python_solutions(example.get("reference_solution"))
    if not solutions:
        return None, "missing_python_reference_solution"

    source_problem_id = example.get("name", example.get("problem_id", example.get("id", f"codecontests_{index}")))
    prepared_cases = _prepare_problem_cases(
        test_cases,
        max_train_cases=max_tests,
        max_diagnostic_cases=max_diagnostic_cases,
        max_case_input_chars=max_case_input_chars,
        max_case_output_chars=max_case_output_chars,
    )

    row = {
        "question_id": f"codecontests_{source_problem_id}",
        "prompt": f"{STDIO_PROMPT_PREFIX}{_clean_prompt(prompt)}",
        "test_cases": prepared_cases["test_cases"],
        "diagnostic_inputs": prepared_cases["diagnostic_inputs"],
        "diagnostic_outputs": prepared_cases["diagnostic_outputs"],
        "reference_solution": solutions[0],
        "source": "codecontests",
        "io_mode": "stdio",
        "source_problem_id": str(source_problem_id),
        "difficulty": _normalize_optional_string(example.get("difficulty")),
        "source_url": _normalize_optional_string(example.get("url")),
        "source_prompt": prompt.strip(),
        "source_test_count": int(prepared_cases["source_test_count"]),
        "selected_test_count": int(prepared_cases["selected_test_count"]),
        "oversized_test_count": int(prepared_cases["oversized_test_count"]),
    }
    return row, None


def normalize_source_record(
    source: str,
    example: dict[str, Any],
    index: int,
    max_tests: int,
    max_case_input_chars: int = DEFAULT_MAX_CASE_INPUT_CHARS,
    max_case_output_chars: int = DEFAULT_MAX_CASE_OUTPUT_CHARS,
    max_diagnostic_cases: int = DEFAULT_MAX_DIAGNOSTIC_CASES,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized_source = str(source).strip().lower()
    if normalized_source in {"apps", "taco", "codeforces"}:
        return _extract_apps_like_problem(
            normalized_source,
            example,
            index,
            max_tests=max_tests,
            max_case_input_chars=max_case_input_chars,
            max_case_output_chars=max_case_output_chars,
            max_diagnostic_cases=max_diagnostic_cases,
        )
    if normalized_source == "codecontests":
        return _extract_codecontests_problem(
            example,
            index,
            max_tests=max_tests,
            max_case_input_chars=max_case_input_chars,
            max_case_output_chars=max_case_output_chars,
            max_diagnostic_cases=max_diagnostic_cases,
        )
    return None, "unsupported_source"


def _reference_pass_rate(reference_solution: str, test_cases: list[dict[str, str]], timeout_seconds: float) -> tuple[float, list[Any]]:
    inputs = [case["input"] for case in test_cases]
    results = execute_batch(
        code=reference_solution,
        case_inputs=inputs,
        timeout_s=timeout_seconds,
        error_max_chars=800,
        error_max_lines=16,
        io_mode="stdio",
    )
    passes = [
        1.0 if result.kind == "OK" and stripped_text_equal(case["output"], result.value) else 0.0
        for case, result in zip(test_cases, results, strict=True)
    ]
    pass_rate = sum(passes) / len(passes) if passes else 0.0
    return pass_rate, results


def _run_trivial_baselines(test_cases: list[dict[str, str]], timeout_seconds: float) -> dict[str, float]:
    inputs = [case["input"] for case in test_cases]
    results: dict[str, float] = {}
    for baseline_name, baseline_code in BASELINE_BANK.items():
        exec_results = execute_batch(
            code=baseline_code,
            case_inputs=inputs,
            timeout_s=timeout_seconds,
            error_max_chars=400,
            error_max_lines=8,
            io_mode="stdio",
        )
        pass_count = sum(
            1
            for case, result in zip(test_cases, exec_results, strict=True)
            if result.kind == "OK" and stripped_text_equal(case["output"], result.value)
        )
        results[baseline_name] = pass_count / len(test_cases) if test_cases else 0.0
    return results


def _candidate_score(row: dict[str, Any]) -> tuple[int, int, int]:
    test_count = len(row.get("test_cases", []))
    prompt_tokens = _approx_token_count(str(row.get("prompt", "")))
    return (
        test_count,
        -prompt_tokens,
        1 if str(row.get("source", "")) == "codecontests" else 0,
    )


def _build_exact_key(row: dict[str, Any]) -> str:
    payload = {
        "prompt": _normalize_text_signature(str(row.get("prompt", ""))),
        "tests": row.get("test_cases", []),
        "io_mode": row.get("io_mode", "stdio"),
    }
    return _hash_payload(payload)


def _build_near_key(row: dict[str, Any]) -> str:
    prompt_signature = _normalize_text_signature(str(row.get("prompt", "")))
    reduced_prompt = " ".join(prompt_signature.split()[:80])
    io_pairs = [
        {
            "input": case["input"].strip(),
            "output": case["output"].strip(),
        }
        for case in list(row.get("test_cases", []))[:3]
    ]
    return _hash_payload({"prompt": reduced_prompt, "sample_io": io_pairs})


def _split_rows(rows: list[dict[str, Any]], split_seed: int, train_size: int, validation_size: int, test_size: int) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(rows)
    rng = random.Random(split_seed)
    rng.shuffle(shuffled)
    return {
        "train": shuffled[:train_size],
        "validation": shuffled[train_size : train_size + validation_size],
        "test": shuffled[train_size + validation_size : train_size + validation_size + test_size],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_mixed_dataset_bundle(
    *,
    source_paths: dict[str, Path | None],
    output_dir: Path,
    split_seed: int = 42,
    max_prompt_tokens: int = 2000,
    min_tests: int = 3,
    max_tests_per_problem: int = 10,
    max_diagnostic_cases_per_problem: int = DEFAULT_MAX_DIAGNOSTIC_CASES,
    max_case_input_chars: int = DEFAULT_MAX_CASE_INPUT_CHARS,
    max_case_output_chars: int = DEFAULT_MAX_CASE_OUTPUT_CHARS,
    timeout_seconds: float = 2.0,
    target_total: int = 500,
    source_targets: dict[str, int] | None = None,
    source_input_limits: dict[str, int] | None = None,
    split_sizes: tuple[int, int, int] = (400, 50, 50),
) -> dict[str, Any]:
    targets = dict(DEFAULT_SOURCE_TARGETS)
    if source_targets:
        targets.update({key: int(value) for key, value in source_targets.items()})

    per_source_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejects: list[dict[str, Any]] = []
    source_counts_before = Counter()
    source_counts_after = Counter()
    reject_counts = Counter()
    trivial_reject_counts = Counter()
    duplicate_counts = Counter()

    def reject(source: str, example_id: Any, reason: str, extra: dict[str, Any] | None = None) -> None:
        reject_counts[reason] += 1
        payload = {
            "source": source,
            "example_id": str(example_id),
            "reason": reason,
        }
        if extra:
            payload.update(extra)
        rejects.append(payload)

    for source, maybe_path in source_paths.items():
        if maybe_path is None:
            continue
        input_limit = None
        if source_input_limits is not None and source in source_input_limits:
            raw_limit_value = int(source_input_limits[source])
            input_limit = raw_limit_value if raw_limit_value > 0 else None
        print(f"[builder] loading source={source} path={maybe_path} input_limit={input_limit}")
        raw_rows = _load_records_from_path(maybe_path, limit=input_limit)
        print(f"[builder] loaded source={source} raw_rows={len(raw_rows)}")
        source_counts_before[source] = len(raw_rows)
        for index, example in enumerate(raw_rows):
            if index > 0 and index % 25 == 0:
                print(
                    f"[builder] source={source} normalized={len(per_source_candidates[source])} "
                    f"processed={index}/{len(raw_rows)} rejects={sum(1 for item in rejects if item['source'] == source)}"
                )
            example_id = example.get("id", example.get("problem_id", example.get("name", index)))
            normalized, reason = normalize_source_record(
                source,
                example,
                index=index,
                max_tests=max_tests_per_problem,
                max_case_input_chars=max_case_input_chars,
                max_case_output_chars=max_case_output_chars,
                max_diagnostic_cases=max_diagnostic_cases_per_problem,
            )
            if normalized is None:
                reject(source, example_id, reason or "normalize_failed")
                continue

            prompt = str(normalized["prompt"])
            if _approx_token_count(prompt) > max_prompt_tokens:
                reject(source, example_id, "prompt_too_long")
                continue
            if _looks_interactive(prompt):
                reject(source, example_id, "interactive_task")
                continue
            if _looks_like_file_io(prompt):
                reject(source, example_id, "file_io_task")
                continue

            test_cases = list(normalized["test_cases"])
            if len(test_cases) < min_tests:
                reject(source, example_id, "too_few_tests")
                continue

            reference_solution = str(normalized["reference_solution"])
            if _uses_disallowed_imports(reference_solution):
                reject(source, example_id, "disallowed_imports")
                continue

            reference_pass_rate, exec_results = _reference_pass_rate(
                reference_solution=reference_solution,
                test_cases=test_cases,
                timeout_seconds=timeout_seconds,
            )
            if reference_pass_rate < 1.0:
                reject(
                    source,
                    example_id,
                    "reference_solution_failed",
                    extra={"reference_pass_rate": reference_pass_rate},
                )
                continue

            trivial_rates = _run_trivial_baselines(test_cases, timeout_seconds=timeout_seconds)
            triggered = [name for name, rate in trivial_rates.items() if rate > 0.0]
            if triggered:
                for name in triggered:
                    trivial_reject_counts[name] += 1
                reject(
                    source,
                    example_id,
                    "trivial_baseline_passed",
                    extra={"baseline_hits": triggered, "baseline_rates": trivial_rates},
                )
                continue

            normalized["reference_exec_ok_count"] = sum(1 for result in exec_results if result.kind == "OK")
            normalized["prompt_token_estimate"] = _approx_token_count(prompt)
            per_source_candidates[source].append(normalized)
        print(
            f"[builder] source={source} kept_candidates={len(per_source_candidates[source])} "
            f"rejects={sum(1 for item in rejects if item['source'] == source)}"
        )

    print("[builder] starting dedupe")
    exact_seen: dict[str, dict[str, Any]] = {}
    near_seen: dict[str, dict[str, Any]] = {}
    deduped_rows: list[dict[str, Any]] = []
    for source in ("apps", "codecontests", "taco", "codeforces"):
        for row in per_source_candidates.get(source, []):
            exact_key = _build_exact_key(row)
            current_best = exact_seen.get(exact_key)
            if current_best is not None:
                duplicate_counts["exact"] += 1
                if _candidate_score(row) > _candidate_score(current_best):
                    exact_seen[exact_key] = row
                continue
            exact_seen[exact_key] = row

    for row in exact_seen.values():
        near_key = _build_near_key(row)
        current_best = near_seen.get(near_key)
        if current_best is not None:
            duplicate_counts["near"] += 1
            if _candidate_score(row) > _candidate_score(current_best):
                near_seen[near_key] = row
            continue
        near_seen[near_key] = row
    deduped_rows = list(near_seen.values())
    print(f"[builder] dedupe complete rows={len(deduped_rows)}")

    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deduped_rows:
        rows_by_source[str(row["source"])].append(row)
    for rows in rows_by_source.values():
        rows.sort(key=_candidate_score, reverse=True)

    rng = random.Random(split_seed)
    selected: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    for source in ("taco", "codecontests", "apps", "codeforces"):
        rows = rows_by_source.get(source, [])
        quota = max(0, int(targets.get(source, 0)))
        selected.extend(rows[:quota])
        extras.extend(rows[quota:])

    if len(selected) < target_total:
        rng.shuffle(extras)
        selected.extend(extras[: max(0, target_total - len(selected))])
    elif len(selected) > target_total:
        rng.shuffle(selected)
        selected = selected[:target_total]

    selected.sort(key=lambda row: (str(row["source"]), str(row["question_id"])))
    print(f"[builder] selection complete selected={len(selected)} extras={len(extras)}")
    for row in selected:
        source_counts_after[str(row["source"])] += 1

    train_size, validation_size, test_size = split_sizes
    split_payload = _split_rows(
        selected,
        split_seed=split_seed,
        train_size=min(train_size, len(selected)),
        validation_size=min(validation_size, max(0, len(selected) - train_size)),
        test_size=min(test_size, max(0, len(selected) - train_size - validation_size)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "code_rl_mixed_small_merged.jsonl"
    train_path = output_dir / "code_rl_mixed_small_train.jsonl"
    validation_path = output_dir / "code_rl_mixed_small_validation.jsonl"
    test_path = output_dir / "code_rl_mixed_small_test.jsonl"
    summary_path = output_dir / "code_rl_mixed_small_summary.json"
    reject_path = output_dir / "code_rl_mixed_small_rejected.jsonl"

    _write_jsonl(merged_path, selected)
    _write_jsonl(train_path, split_payload["train"])
    _write_jsonl(validation_path, split_payload["validation"])
    _write_jsonl(test_path, split_payload["test"])
    _write_jsonl(reject_path, rejects)
    print(f"[builder] wrote merged={len(selected)} train={len(split_payload['train'])} validation={len(split_payload['validation'])} test={len(split_payload['test'])}")

    prompt_lengths = [int(row.get("prompt_token_estimate", _approx_token_count(str(row.get("prompt", ""))))) for row in selected]
    test_counts = [len(row.get("test_cases", [])) for row in selected]
    summary = {
        "target_total": target_total,
        "source_targets": targets,
        "source_counts_before_filtering": dict(source_counts_before),
        "source_counts_after_filtering": dict(source_counts_after),
        "reject_counts": dict(reject_counts),
        "trivial_baseline_reject_counts": dict(trivial_reject_counts),
        "duplicate_removal_counts": dict(duplicate_counts),
        "final_counts": {
            "merged": len(selected),
            "train": len(split_payload["train"]),
            "validation": len(split_payload["validation"]),
            "test": len(split_payload["test"]),
        },
        "prompt_length_stats": {
            "min": min(prompt_lengths) if prompt_lengths else 0,
            "max": max(prompt_lengths) if prompt_lengths else 0,
            "mean": (sum(prompt_lengths) / len(prompt_lengths)) if prompt_lengths else 0.0,
        },
        "test_count_stats": {
            "min": min(test_counts) if test_counts else 0,
            "max": max(test_counts) if test_counts else 0,
            "mean": (sum(test_counts) / len(test_counts)) if test_counts else 0.0,
        },
        "train_test_filtering": {
            "max_tests_per_problem": max_tests_per_problem,
            "max_diagnostic_cases_per_problem": max_diagnostic_cases_per_problem,
            "max_case_input_chars": max_case_input_chars,
            "max_case_output_chars": max_case_output_chars,
        },
        "split_seed": split_seed,
        "output_files": {
            "merged": str(merged_path),
            "train": str(train_path),
            "validation": str(validation_path),
            "test": str(test_path),
            "summary": str(summary_path),
            "rejected": str(reject_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a small high-quality mixed Code RL dataset.")
    parser.add_argument("--apps-path", type=Path, default=None)
    parser.add_argument("--codecontests-path", type=Path, default=None)
    parser.add_argument("--taco-path", type=Path, default=None)
    parser.add_argument("--codeforces-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-prompt-tokens", type=int, default=2000)
    parser.add_argument("--min-tests", type=int, default=3)
    parser.add_argument("--max-tests-per-problem", type=int, default=10)
    parser.add_argument("--max-diagnostic-cases-per-problem", type=int, default=DEFAULT_MAX_DIAGNOSTIC_CASES)
    parser.add_argument("--max-case-input-chars", type=int, default=DEFAULT_MAX_CASE_INPUT_CHARS)
    parser.add_argument("--max-case-output-chars", type=int, default=DEFAULT_MAX_CASE_OUTPUT_CHARS)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--target-total", type=int, default=500)
    parser.add_argument("--apps-target", type=int, default=100)
    parser.add_argument("--codecontests-target", type=int, default=200)
    parser.add_argument("--taco-target", type=int, default=200)
    parser.add_argument("--codeforces-target", type=int, default=0)
    parser.add_argument("--apps-input-limit", type=int, default=0)
    parser.add_argument("--codecontests-input-limit", type=int, default=0)
    parser.add_argument("--taco-input-limit", type=int, default=0)
    parser.add_argument("--codeforces-input-limit", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--validation-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = build_mixed_dataset_bundle(
        source_paths={
            "apps": args.apps_path,
            "codecontests": args.codecontests_path,
            "taco": args.taco_path,
            "codeforces": args.codeforces_path,
        },
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        max_prompt_tokens=args.max_prompt_tokens,
        min_tests=args.min_tests,
        max_tests_per_problem=args.max_tests_per_problem,
        max_diagnostic_cases_per_problem=args.max_diagnostic_cases_per_problem,
        max_case_input_chars=args.max_case_input_chars,
        max_case_output_chars=args.max_case_output_chars,
        timeout_seconds=args.timeout_seconds,
        target_total=args.target_total,
        source_targets={
            "apps": args.apps_target,
            "codecontests": args.codecontests_target,
            "taco": args.taco_target,
            "codeforces": args.codeforces_target,
        },
        source_input_limits={
            "apps": args.apps_input_limit,
            "codecontests": args.codecontests_input_limit,
            "taco": args.taco_input_limit,
            "codeforces": args.codeforces_input_limit,
        },
        split_sizes=(args.train_size, args.validation_size, args.test_size),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
