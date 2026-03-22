import argparse
from pathlib import Path
from typing import Optional

from download_code_rl_sources import main as download_main
from prepare_code_rl_mixed_small import main as build_main


def _resolve_download_count(explicit_count: int, target_count: int, buffer_factor: float, minimum_count: int) -> int:
    if explicit_count > 0:
        return explicit_count
    derived_count = int(round(target_count * buffer_factor))
    return max(minimum_count, derived_count)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download raw source subsets and build the mixed Code RL dataset.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", type=str, default="")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--buffer-factor", type=float, default=2.0)
    parser.add_argument("--apps-count", type=int, default=0)
    parser.add_argument("--codecontests-count", type=int, default=0)
    parser.add_argument("--taco-count", type=int, default=0)
    parser.add_argument("--target-total", type=int, default=500)
    parser.add_argument("--apps-target", type=int, default=100)
    parser.add_argument("--codecontests-target", type=int, default=200)
    parser.add_argument("--taco-target", type=int, default=200)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--validation-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=50)
    args = parser.parse_args(argv)

    work_dir = args.work_dir
    raw_dir = work_dir / "raw_sources"
    output_dir = work_dir / "code_rl_mixed_small"
    cache_dir = args.cache_dir or (work_dir / "hf_cache")
    apps_count = _resolve_download_count(args.apps_count, args.apps_target, args.buffer_factor, minimum_count=120)
    codecontests_count = _resolve_download_count(
        args.codecontests_count,
        args.codecontests_target,
        args.buffer_factor,
        minimum_count=220,
    )
    taco_count = _resolve_download_count(args.taco_count, args.taco_target, args.buffer_factor, minimum_count=220)
    apps_existing = _count_jsonl_rows(raw_dir / "apps_raw.jsonl")
    codecontests_existing = _count_jsonl_rows(raw_dir / "codecontests_raw.jsonl")
    taco_existing = _count_jsonl_rows(raw_dir / "taco_raw.jsonl")
    apps_count = 0 if apps_existing >= apps_count else apps_count
    codecontests_count = 0 if codecontests_existing >= codecontests_count else codecontests_count
    taco_count = 0 if taco_existing >= taco_count else taco_count
    print(
        "[build] existing_rows "
        f"apps={apps_existing} codecontests={codecontests_existing} taco={taco_existing}"
    )
    print(
        "[build] requested_downloads "
        f"apps={apps_count} codecontests={codecontests_count} taco={taco_count}"
    )
    print(f"[build] cache_dir={cache_dir}")

    download_main(
        [
            "--output-dir",
            str(raw_dir),
            "--cache-dir",
            str(cache_dir),
            "--seed",
            str(args.seed),
            "--hf-endpoint",
            str(args.hf_endpoint),
            "--apps-count",
            str(apps_count),
            "--codecontests-count",
            str(codecontests_count),
            "--taco-count",
            str(taco_count),
        ]
    )
    print("[build] download stage complete; starting normalization/build")

    build_main(
        [
            "--apps-path",
            str(raw_dir / "apps_raw.jsonl"),
            "--codecontests-path",
            str(raw_dir / "codecontests_raw.jsonl"),
            "--taco-path",
            str(raw_dir / "taco_raw.jsonl"),
            "--output-dir",
            str(output_dir),
            "--split-seed",
            str(args.seed),
            "--target-total",
            str(args.target_total),
            "--apps-target",
            str(args.apps_target),
            "--codecontests-target",
            str(args.codecontests_target),
            "--taco-target",
            str(args.taco_target),
            "--train-size",
            str(args.train_size),
            "--validation-size",
            str(args.validation_size),
            "--test-size",
            str(args.test_size),
        ]
    )
    print(f"[build] dataset bundle written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
