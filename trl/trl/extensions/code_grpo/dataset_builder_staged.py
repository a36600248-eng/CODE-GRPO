import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .dataset_builder import (
    DEFAULT_SOURCE_TARGETS,
    _approx_token_count,
    _build_exact_key,
    _build_near_key,
    _candidate_score,
    _load_records_from_path,
    _looks_interactive,
    _looks_like_file_io,
    _read_json,
    _reference_pass_rate,
    _run_trivial_baselines,
    _split_rows,
    _uses_disallowed_imports,
    _write_json,
    _write_jsonl,
    normalize_source_record,
)

SOURCE_ORDER = ("apps", "codecontests", "taco", "codeforces")

EXPECTED_STRING_FIELDS = (
    "question_id",
    "prompt",
    "reference_solution",
    "source",
    "io_mode",
    "source_problem_id",
    "difficulty",
    "source_url",
    "source_prompt",
)
EXPECTED_INT_FIELDS = (
    "source_test_count",
    "selected_test_count",
    "oversized_test_count",
    "prompt_token_estimate",
    "reference_exec_ok_count",
)
EXPECTED_FLOATLIKE_FIELDS = ("reference_pass_rate",)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl_safe(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                break
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _validate_dataset_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        question_id = str(row.get("question_id", "unknown"))
        for field_name in EXPECTED_STRING_FIELDS:
            value = row.get(field_name)
            if not isinstance(value, str):
                raise ValueError(f"{question_id}: field '{field_name}' must be str, got {type(value).__name__}")
        for field_name in EXPECTED_INT_FIELDS:
            value = row.get(field_name)
            if not isinstance(value, int):
                raise ValueError(f"{question_id}: field '{field_name}' must be int, got {type(value).__name__}")
        for field_name in EXPECTED_FLOATLIKE_FIELDS:
            value = row.get(field_name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{question_id}: field '{field_name}' must be numeric, got {type(value).__name__}")
        test_cases = row.get("test_cases")
        if not isinstance(test_cases, list) or len(test_cases) < 1:
            raise ValueError(f"{question_id}: field 'test_cases' must be a non-empty list")
        for case in test_cases:
            if not isinstance(case, dict):
                raise ValueError(f"{question_id}: each test case must be an object")
            if not isinstance(case.get("input"), str) or not isinstance(case.get("output"), str):
                raise ValueError(f"{question_id}: test case input/output must be str")
        for field_name in ("diagnostic_inputs", "diagnostic_outputs"):
            value = row.get(field_name, [])
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"{question_id}: field '{field_name}' must be list[str]")


def _normalize_static_filter(
    *,
    source: str,
    example: dict[str, Any],
    index: int,
    max_tests: int,
    max_diagnostic_cases: int,
    max_case_input_chars: int,
    max_case_output_chars: int,
    max_prompt_tokens: int,
    min_tests: int,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    example_id = example.get("id", example.get("problem_id", example.get("name", index)))
    normalized, reason = normalize_source_record(
        source,
        example,
        index=index,
        max_tests=max_tests,
        max_case_input_chars=max_case_input_chars,
        max_case_output_chars=max_case_output_chars,
        max_diagnostic_cases=max_diagnostic_cases,
    )
    if normalized is None:
        return None, {"source": source, "example_id": str(example_id), "reason": reason or "normalize_failed"}
    prompt = str(normalized["prompt"])
    if _approx_token_count(prompt) > max_prompt_tokens:
        return None, {"source": source, "example_id": str(example_id), "reason": "prompt_too_long"}
    if _looks_interactive(prompt):
        return None, {"source": source, "example_id": str(example_id), "reason": "interactive_task"}
    if _looks_like_file_io(prompt):
        return None, {"source": source, "example_id": str(example_id), "reason": "file_io_task"}
    if len(normalized["test_cases"]) < min_tests:
        return None, {"source": source, "example_id": str(example_id), "reason": "too_few_tests"}
    if _uses_disallowed_imports(str(normalized["reference_solution"])):
        return None, {"source": source, "example_id": str(example_id), "reason": "disallowed_imports"}
    normalized["prompt_token_estimate"] = _approx_token_count(prompt)
    return normalized, None


def normalize_stage(
    *,
    source_paths: dict[str, Optional[Path]],
    output_dir: Path,
    source_input_limits: dict[str, int],
    source_index_offsets: Optional[dict[str, int]],
    split_seed: int,
    max_prompt_tokens: int,
    min_tests: int,
    max_tests_per_problem: int,
    max_diagnostic_cases_per_problem: int,
    max_case_input_chars: int,
    max_case_output_chars: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rejects: list[dict[str, Any]] = []
    normalized_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts_before = Counter()
    source_counts_after = Counter()
    reject_counts = Counter()
    duplicate_counts = Counter()
    cross_source_near_hits: list[dict[str, Any]] = []

    for source in SOURCE_ORDER:
        maybe_path = source_paths.get(source)
        if maybe_path is None:
            continue
        raw_limit = int(source_input_limits.get(source, 0))
        index_offset = int((source_index_offsets or {}).get(source, 0))
        input_limit = raw_limit if raw_limit > 0 else None
        print(f"[normalize] loading source={source} path={maybe_path} input_limit={input_limit} index_offset={index_offset}")
        raw_rows = _load_records_from_path(maybe_path, limit=input_limit)
        source_counts_before[source] = len(raw_rows)
        print(f"[normalize] loaded source={source} raw_rows={len(raw_rows)}")
        for index, example in enumerate(raw_rows, start=index_offset):
            if index > 0 and index % 25 == 0:
                local_processed = index - index_offset
                print(f"[normalize] source={source} processed={local_processed}/{len(raw_rows)} kept={len(normalized_by_source[source])}")
            normalized, reject_payload = _normalize_static_filter(
                source=source,
                example=example,
                index=index,
                max_tests=max_tests_per_problem,
                max_diagnostic_cases=max_diagnostic_cases_per_problem,
                max_case_input_chars=max_case_input_chars,
                max_case_output_chars=max_case_output_chars,
                max_prompt_tokens=max_prompt_tokens,
                min_tests=min_tests,
            )
            if normalized is None:
                reject_counts[reject_payload["reason"]] += 1
                rejects.append(reject_payload)
                continue
            normalized_by_source[source].append(normalized)
        source_counts_after[source] = len(normalized_by_source[source])
        _write_jsonl(output_dir / f"normalized_{source}.jsonl", normalized_by_source[source])

    exact_seen: dict[str, dict[str, Any]] = {}
    for source in SOURCE_ORDER:
        for row in normalized_by_source.get(source, []):
            exact_key = _build_exact_key(row)
            current_best = exact_seen.get(exact_key)
            if current_best is not None:
                duplicate_counts["exact"] += 1
                if _candidate_score(row) > _candidate_score(current_best):
                    exact_seen[exact_key] = row
                continue
            exact_seen[exact_key] = row

    near_seen: dict[str, dict[str, Any]] = {}
    for row in exact_seen.values():
        near_key = _build_near_key(row)
        current_best = near_seen.get(near_key)
        if current_best is not None:
            duplicate_counts["near"] += 1
            better_row = row if _candidate_score(row) > _candidate_score(current_best) else current_best
            removed_row = current_best if better_row is row else row
            if str(removed_row.get("source")) != str(better_row.get("source")):
                cross_source_near_hits.append({"kept": str(better_row.get("question_id")), "removed": str(removed_row.get("question_id")), "reason": "near_dup"})
            near_seen[near_key] = better_row
            continue
        near_seen[near_key] = row

    deduped_rows = list(near_seen.values())
    deduped_rows.sort(key=lambda row: (str(row["source"]), str(row["question_id"])))
    _write_jsonl(output_dir / "deduped_pool.jsonl", deduped_rows)
    _write_jsonl(output_dir / "rejected_normalize.jsonl", rejects)
    summary = {
        "status": "completed",
        "split_seed": split_seed,
        "source_counts_before": dict(source_counts_before),
        "source_counts_after_normalize": dict(source_counts_after),
        "deduped_pool_count": len(deduped_rows),
        "reject_counts": dict(reject_counts),
        "duplicate_removal_counts": dict(duplicate_counts),
        "cross_source_near_hits": cross_source_near_hits[:200],
        "train_test_filtering": {
            "max_tests_per_problem": max_tests_per_problem,
            "max_diagnostic_cases_per_problem": max_diagnostic_cases_per_problem,
            "max_case_input_chars": max_case_input_chars,
            "max_case_output_chars": max_case_output_chars,
        },
    }
    _write_json(output_dir / "normalize_summary.json", summary)
    print(f"[normalize] dedupe complete rows={len(deduped_rows)}")
    return summary


def _validate_single_row(row: dict[str, Any], timeout_seconds: float) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    source = str(row.get("source", "unknown"))
    question_id = str(row.get("question_id", "unknown"))
    test_cases = list(row.get("test_cases", []))
    reference_pass_rate, exec_results = _reference_pass_rate(
        reference_solution=str(row.get("reference_solution", "")),
        test_cases=test_cases,
        timeout_seconds=timeout_seconds,
    )
    if reference_pass_rate < 1.0:
        return None, {"source": source, "question_id": question_id, "reason": "reference_solution_failed", "reference_pass_rate": reference_pass_rate}
    trivial_rates = _run_trivial_baselines(test_cases, timeout_seconds=timeout_seconds)
    triggered = [name for name, rate in trivial_rates.items() if rate > 0.0]
    if triggered:
        return None, {"source": source, "question_id": question_id, "reason": "trivial_baseline_passed", "baseline_hits": triggered, "baseline_rates": trivial_rates}
    accepted = dict(row)
    accepted["reference_exec_ok_count"] = sum(1 for result in exec_results if result.kind == "OK")
    accepted["reference_pass_rate"] = reference_pass_rate
    return accepted, None


def _rebuild_processed_ids(validate_dir: Path) -> set[str]:
    processed_ids: set[str] = set()
    for file_path in validate_dir.glob("validated_*.jsonl"):
        for row in _read_jsonl_safe(file_path):
            if row.get("question_id"):
                processed_ids.add(str(row["question_id"]))
    for row in _read_jsonl_safe(validate_dir / "rejected_validate.jsonl"):
        question_id = row.get("question_id", row.get("example_id"))
        if question_id:
            processed_ids.add(str(question_id))
    return processed_ids


def _build_validate_counts(pool_rows: list[dict[str, Any]], validate_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "processed": 0, "accepted": 0, "rejected": 0})
    for row in pool_rows:
        counts[str(row["source"])]["total"] += 1
    for file_path in validate_dir.glob("validated_*.jsonl"):
        source = file_path.stem.replace("validated_", "")
        counts[source]["accepted"] = len(_read_jsonl_safe(file_path))
    for row in _read_jsonl_safe(validate_dir / "rejected_validate.jsonl"):
        counts[str(row.get("source", "unknown"))]["rejected"] += 1
    for source in counts:
        counts[source]["processed"] = counts[source]["accepted"] + counts[source]["rejected"]
    return {source: dict(payload) for source, payload in counts.items()}


def _write_validate_summary(
    *,
    validate_dir: Path,
    pool_rows: list[dict[str, Any]],
    processed_ids: set[str],
    counts: dict[str, dict[str, int]],
    last_batch: Optional[dict[str, Any]],
    started_at: str,
    batch_size: int,
) -> dict[str, Any]:
    rejected_rows = _read_jsonl_safe(validate_dir / "rejected_validate.jsonl")
    reject_reasons_by_source: dict[str, Counter] = defaultdict(Counter)
    for row in rejected_rows:
        reject_reasons_by_source[str(row.get("source", "unknown"))][str(row.get("reason", "unknown"))] += 1
    elapsed_seconds = max(0.0, time.time() - datetime.fromisoformat(started_at).timestamp())
    source_payload: dict[str, Any] = {}
    total_accepted = 0
    total_rejected = 0
    for source, payload in counts.items():
        total_accepted += int(payload["accepted"])
        total_rejected += int(payload["rejected"])
        progress = (payload["processed"] / payload["total"] * 100.0) if payload["total"] else 0.0
        source_payload[source] = {
            "pool_size": payload["total"],
            "processed": payload["processed"],
            "accepted": payload["accepted"],
            "rejected": payload["rejected"],
            "progress_pct": f"{progress:.1f}%",
            "reject_reasons": dict(reject_reasons_by_source.get(source, Counter())),
        }
    overall_progress = (len(processed_ids) / len(pool_rows) * 100.0) if pool_rows else 0.0
    summary = {
        "status": "in_progress" if len(processed_ids) < len(pool_rows) else "completed",
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "batch_size": batch_size,
        "sources": source_payload,
        "overall": {
            "total_pool": len(pool_rows),
            "total_processed": len(processed_ids),
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
            "overall_progress_pct": f"{overall_progress:.1f}%",
            "estimated_accept_rate": f"{(total_accepted / len(processed_ids) * 100.0):.1f}%" if processed_ids else "0.0%",
        },
        "last_batch": last_batch,
    }
    _atomic_write_json(validate_dir / "validate_summary.json", summary)
    return summary


def validate_stage(
    *,
    pool_path: Path,
    output_dir: Path,
    timeout_seconds: float,
    batch_size: int,
    resume: bool,
    max_batches: int = 0,
    source_targets: Optional[dict[str, int]] = None,
    stop_when_targets_met: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_rows = _read_jsonl_safe(pool_path)
    progress_path = output_dir / "progress.json"
    existing_progress = _read_json(progress_path) if resume and progress_path.exists() else None
    started_at = str(existing_progress.get("started_at")) if isinstance(existing_progress, dict) and existing_progress.get("started_at") else datetime.now().isoformat(timespec="seconds")
    processed_ids = _rebuild_processed_ids(output_dir) if resume else set()
    if isinstance(existing_progress, dict):
        processed_ids.update(str(item) for item in existing_progress.get("processed_ids", []))
    counts = _build_validate_counts(pool_rows, output_dir)
    pending = [row for row in pool_rows if str(row["question_id"]) not in processed_ids]
    print(f"[validate] pool={len(pool_rows)} processed={len(processed_ids)} pending={len(pending)} batch_size={batch_size}")
    total_batches = (len(pending) + batch_size - 1) // batch_size if batch_size > 0 else 0
    last_batch_summary: Optional[dict[str, Any]] = None
    for batch_index, start in enumerate(range(0, len(pending), batch_size), start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        batch = pending[start : start + batch_size]
        batch_started_at = time.time()
        accepted_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        for row in batch:
            accepted_row, rejected_row = _validate_single_row(row, timeout_seconds=timeout_seconds)
            if accepted_row is not None:
                accepted_rows.append(accepted_row)
            elif rejected_row is not None:
                rejected_rows.append(rejected_row)
        accepted_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in accepted_rows:
            accepted_by_source[str(row["source"])].append(row)
        for source, rows in accepted_by_source.items():
            _append_jsonl(output_dir / f"validated_{source}.jsonl", rows)
        _append_jsonl(output_dir / "rejected_validate.jsonl", rejected_rows)
        for row in batch:
            processed_ids.add(str(row["question_id"]))
        for row in accepted_rows:
            counts[str(row["source"])]["accepted"] += 1
            counts[str(row["source"])]["processed"] += 1
        for row in rejected_rows:
            counts[str(row["source"])]["rejected"] += 1
            counts[str(row["source"])]["processed"] += 1
        last_batch_summary = {
            "batch_idx": batch_index,
            "batch_count": total_batches,
            "batch_size": len(batch),
            "accepted": len(accepted_rows),
            "rejected": len(rejected_rows),
            "wall_time_seconds": round(time.time() - batch_started_at, 2),
        }
        _atomic_write_json(progress_path, {"schema_version": 1, "started_at": started_at, "updated_at": datetime.now().isoformat(timespec="seconds"), "batch_size": batch_size, "processed_ids": sorted(processed_ids), "counts": counts})
        summary = _write_validate_summary(validate_dir=output_dir, pool_rows=pool_rows, processed_ids=processed_ids, counts=counts, last_batch=last_batch_summary, started_at=started_at, batch_size=batch_size)
        print(f"[validate] batch {batch_index}/{total_batches} | accepted={len(accepted_rows)} rejected={len(rejected_rows)} | total {summary['overall']['total_processed']}/{summary['overall']['total_pool']} acc={summary['overall']['estimated_accept_rate']} | {last_batch_summary['wall_time_seconds']}s")
        if stop_when_targets_met and source_targets:
            if all(counts.get(source, {}).get("accepted", 0) >= int(target) for source, target in source_targets.items() if int(target) > 0):
                print("[validate] stopping early because source targets were met")
                break
    return _write_validate_summary(validate_dir=output_dir, pool_rows=pool_rows, processed_ids=processed_ids, counts=counts, last_batch=last_batch_summary, started_at=started_at, batch_size=batch_size)


def finalize_stage(*, validated_dir: Path, output_dir: Path, split_seed: int, target_total: int, source_targets: dict[str, int], split_sizes: tuple[int, int, int]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = dict(DEFAULT_SOURCE_TARGETS)
    targets.update({key: int(value) for key, value in source_targets.items()})
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in SOURCE_ORDER:
        rows = _read_jsonl_safe(validated_dir / f"validated_{source}.jsonl")
        rows.sort(key=_candidate_score, reverse=True)
        rows_by_source[source] = rows
        print(f"[finalize] source={source} validated_rows={len(rows)}")
    selected: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    for source in SOURCE_ORDER:
        rows = rows_by_source.get(source, [])
        quota = max(0, int(targets.get(source, 0)))
        selected.extend(rows[:quota])
        extras.extend(rows[quota:])
    rng = __import__("random").Random(split_seed)
    if len(selected) < target_total:
        rng.shuffle(extras)
        selected.extend(extras[: max(0, target_total - len(selected))])
    elif len(selected) > target_total:
        rng.shuffle(selected)
        selected = selected[:target_total]
    selected.sort(key=lambda row: (str(row["source"]), str(row["question_id"])))
    _validate_dataset_rows(selected)
    split_payload = _split_rows(selected, split_seed=split_seed, train_size=min(split_sizes[0], len(selected)), validation_size=min(split_sizes[1], max(0, len(selected) - split_sizes[0])), test_size=min(split_sizes[2], max(0, len(selected) - split_sizes[0] - split_sizes[1])))
    _write_jsonl(output_dir / "merged.jsonl", selected)
    _write_jsonl(output_dir / "train.jsonl", split_payload["train"])
    _write_jsonl(output_dir / "validation.jsonl", split_payload["validation"])
    _write_jsonl(output_dir / "test.jsonl", split_payload["test"])
    summary = {"status": "completed", "target_total": target_total, "source_targets": targets, "source_counts": {source: len(rows_by_source.get(source, [])) for source in SOURCE_ORDER}, "final_counts": {"merged": len(selected), "train": len(split_payload["train"]), "validation": len(split_payload["validation"]), "test": len(split_payload["test"])}}
    _write_json(output_dir / "bundle_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-based mixed Code RL dataset builder.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("--apps-path", type=Path, default=None)
    normalize_parser.add_argument("--codecontests-path", type=Path, default=None)
    normalize_parser.add_argument("--taco-path", type=Path, default=None)
    normalize_parser.add_argument("--codeforces-path", type=Path, default=None)
    normalize_parser.add_argument("--output-dir", type=Path, required=True)
    normalize_parser.add_argument("--split-seed", type=int, default=42)
    normalize_parser.add_argument("--max-prompt-tokens", type=int, default=2000)
    normalize_parser.add_argument("--min-tests", type=int, default=3)
    normalize_parser.add_argument("--max-tests-per-problem", type=int, default=10)
    normalize_parser.add_argument("--max-diagnostic-cases-per-problem", type=int, default=4)
    normalize_parser.add_argument("--max-case-input-chars", type=int, default=5000)
    normalize_parser.add_argument("--max-case-output-chars", type=int, default=5000)
    normalize_parser.add_argument("--apps-input-limit", type=int, default=0)
    normalize_parser.add_argument("--codecontests-input-limit", type=int, default=0)
    normalize_parser.add_argument("--taco-input-limit", type=int, default=0)
    normalize_parser.add_argument("--codeforces-input-limit", type=int, default=0)
    normalize_parser.add_argument("--apps-index-offset", type=int, default=0)
    normalize_parser.add_argument("--codecontests-index-offset", type=int, default=0)
    normalize_parser.add_argument("--taco-index-offset", type=int, default=0)
    normalize_parser.add_argument("--codeforces-index-offset", type=int, default=0)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--pool", type=Path, required=True)
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.add_argument("--timeout-seconds", type=float, default=2.0)
    validate_parser.add_argument("--batch-size", type=int, default=10)
    validate_parser.add_argument("--resume", action="store_true")
    validate_parser.add_argument("--max-batches", type=int, default=0)
    validate_parser.add_argument("--apps-target", type=int, default=0)
    validate_parser.add_argument("--codecontests-target", type=int, default=0)
    validate_parser.add_argument("--taco-target", type=int, default=0)
    validate_parser.add_argument("--codeforces-target", type=int, default=0)
    validate_parser.add_argument("--stop-when-targets-met", action="store_true")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--validated-dir", type=Path, required=True)
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser.add_argument("--split-seed", type=int, default=42)
    finalize_parser.add_argument("--target-total", type=int, default=500)
    finalize_parser.add_argument("--apps-target", type=int, default=100)
    finalize_parser.add_argument("--codecontests-target", type=int, default=200)
    finalize_parser.add_argument("--taco-target", type=int, default=200)
    finalize_parser.add_argument("--codeforces-target", type=int, default=0)
    finalize_parser.add_argument("--train-size", type=int, default=400)
    finalize_parser.add_argument("--validation-size", type=int, default=50)
    finalize_parser.add_argument("--test-size", type=int, default=50)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "normalize":
        summary = normalize_stage(
            source_paths={"apps": args.apps_path, "codecontests": args.codecontests_path, "taco": args.taco_path, "codeforces": args.codeforces_path},
            output_dir=args.output_dir,
            source_input_limits={"apps": args.apps_input_limit, "codecontests": args.codecontests_input_limit, "taco": args.taco_input_limit, "codeforces": args.codeforces_input_limit},
            source_index_offsets={"apps": args.apps_index_offset, "codecontests": args.codecontests_index_offset, "taco": args.taco_index_offset, "codeforces": args.codeforces_index_offset},
            split_seed=args.split_seed,
            max_prompt_tokens=args.max_prompt_tokens,
            min_tests=args.min_tests,
            max_tests_per_problem=args.max_tests_per_problem,
            max_diagnostic_cases_per_problem=args.max_diagnostic_cases_per_problem,
            max_case_input_chars=args.max_case_input_chars,
            max_case_output_chars=args.max_case_output_chars,
        )
    elif args.command == "validate":
            summary = validate_stage(
                pool_path=args.pool,
                output_dir=args.output_dir,
                timeout_seconds=args.timeout_seconds,
                batch_size=args.batch_size,
                resume=args.resume,
                max_batches=args.max_batches,
                source_targets={
                    "apps": args.apps_target,
                    "codecontests": args.codecontests_target,
                    "taco": args.taco_target,
                    "codeforces": args.codeforces_target,
                },
                stop_when_targets_met=args.stop_when_targets_met,
            )
    else:
        summary = finalize_stage(
            validated_dir=args.validated_dir,
            output_dir=args.output_dir,
            split_seed=args.split_seed,
            target_total=args.target_total,
            source_targets={"apps": args.apps_target, "codecontests": args.codecontests_target, "taco": args.taco_target, "codeforces": args.codeforces_target},
            split_sizes=(args.train_size, args.validation_size, args.test_size),
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
