import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional


SOURCE_DEFAULTS = {
    "apps": {
        "dataset_names": ["Zuijkkk/apps", "likaixin/APPS-verified", "codeparrot/apps"],
        "splits": ["train"],
        "default_count": 300,
    },
    "codecontests": {
        "dataset_names": ["Imandra/code_contests", "deepmind/code_contests"],
        "splits": ["train"],
        "default_count": 500,
    },
    "taco": {
        "dataset_names": ["felixZzz/TACO", "BEE-spoke-data/TACO-hf", "BAAI/TACO"],
        "splits": ["train"],
        "default_count": 500,
    },
}


def _load_source_rows(
    *,
    source_name: str,
    dataset_names: list[str],
    splits: list[str],
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    last_error: Exception | None = None
    for candidate_name in dataset_names:
        try:
            print(f"[download] source={source_name} candidate={candidate_name} limit={limit}")
            rows: list[dict[str, Any]] = []
            for split_name in splits:
                print(f"[download] source={source_name} split={split_name} starting stream")
                dataset = load_dataset(candidate_name, split=split_name, streaming=True)
                if limit > 0:
                    buffer_size = max(1000, min(limit * 20, 20000))
                    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
                for row in dataset:
                    rows.append(dict(row))
                    if len(rows) % 25 == 0:
                        print(f"[download] source={source_name} collected={len(rows)}")
                    if limit > 0 and len(rows) >= limit:
                        print(f"[download] source={source_name} reached limit={limit}")
                        return rows
            print(f"[download] source={source_name} completed candidate={candidate_name} rows={len(rows)}")
            return rows
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[download] source={source_name} candidate={candidate_name} failed: {exc}")
            continue
    if last_error is not None:
        raise RuntimeError(f"Failed to download dataset from candidates {dataset_names}") from last_error
    return []


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download raw code RL source datasets from Hugging Face.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", type=str, default="")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--apps-count", type=int, default=SOURCE_DEFAULTS["apps"]["default_count"])
    parser.add_argument(
        "--codecontests-count",
        type=int,
        default=SOURCE_DEFAULTS["codecontests"]["default_count"],
    )
    parser.add_argument("--taco-count", type=int, default=SOURCE_DEFAULTS["taco"]["default_count"])
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (output_dir / "hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        os.environ["HUGGINGFACE_HUB_BASE_URL"] = args.hf_endpoint

    summary: dict[str, Any] = {"seed": args.seed, "cache_dir": str(cache_dir), "sources": {}}
    count_by_source = {
        "apps": int(args.apps_count),
        "codecontests": int(args.codecontests_count),
        "taco": int(args.taco_count),
    }

    for source_name, config in SOURCE_DEFAULTS.items():
        requested_count = int(count_by_source[source_name])
        if requested_count <= 0:
            print(f"[download] source={source_name} skipped requested_count=0")
            summary["sources"][source_name] = {
                "dataset_name_candidates": config["dataset_names"],
                "splits": config["splits"],
                "downloaded_rows": 0,
                "skipped": True,
            }
            continue
        print(f"[download] source={source_name} requested_count={requested_count}")
        rows = _load_source_rows(
            source_name=source_name,
            dataset_names=list(config["dataset_names"]),
            splits=list(config["splits"]),
            limit=requested_count,
            seed=int(args.seed),
        )
        target_path = output_dir / f"{source_name}_raw.jsonl"
        _write_jsonl(target_path, rows)
        print(f"[download] source={source_name} wrote rows={len(rows)} path={target_path}")
        summary["sources"][source_name] = {
            "dataset_name_candidates": config["dataset_names"],
            "splits": config["splits"],
            "downloaded_rows": len(rows),
            "output_path": str(target_path),
        }

    summary_path = output_dir / "download_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
