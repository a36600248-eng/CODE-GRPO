import json
import shutil
from pathlib import Path

from trl.extensions.code_grpo import dataset_builder
from trl.extensions.code_grpo.adapters.default_adapter import DefaultCodeDatasetAdapter
from trl.extensions.code_grpo.dataset_builder import build_mixed_dataset_bundle, normalize_source_record
from trl.extensions.code_grpo.executor import _exec_stdio_mode
from trl.extensions.code_grpo.types import ExecResult


def test_exec_stdio_mode_captures_stdout():
    code = "import sys\ntext = sys.stdin.read().strip()\nprint(text[::-1])\n"
    compiled = compile(code, "<candidate>", "exec")
    assert _exec_stdio_mode(compiled, "abc\n").strip() == "cba"
    assert _exec_stdio_mode(compiled, "xy\n").strip() == "yx"


def test_default_adapter_defaults_io_mode_to_call():
    adapter = DefaultCodeDatasetAdapter()
    example = adapter.adapt_example(
        {
            "question_id": "q1",
            "prompt": "demo",
            "test_cases": [{"input": "1", "output": "2"}],
        },
        0,
    )
    assert example["io_mode"] == "call"


def test_default_adapter_accepts_stdio_rows():
    adapter = DefaultCodeDatasetAdapter()
    example = adapter.adapt_example(
        {
            "id": "q2",
            "question": "demo",
            "tests": [{"stdin": "1\n", "stdout": "2\n"}],
            "io_mode": "stdio",
        },
        0,
    )
    assert example["question_id"] == "q2"
    assert example["io_mode"] == "stdio"
    assert example["test_cases"] == [{"input": "1\n", "output": "2\n"}]


def test_normalize_apps_like_record():
    example = {
        "problem_id": "apps_1",
        "question": "Add two integers.",
        "input_output": json.dumps(
            {
                "fn_name": None,
                "inputs": ["1 2\n", "10 5\n", "7 8\n"],
                "outputs": ["3\n", "15\n", "15\n"],
            }
        ),
        "solutions": json.dumps(
            [
                "a, b = map(int, input().split())\nprint(a + b)\n",
            ]
        ),
    }
    row, reason = normalize_source_record("apps", example, index=0, max_tests=10)
    assert reason is None
    assert row is not None
    assert row["io_mode"] == "stdio"
    assert row["source"] == "apps"
    assert len(row["test_cases"]) == 3


def test_normalize_apps_like_rejects_fn_name():
    example = {
        "problem_id": "apps_2",
        "question": "Function task.",
        "input_output": json.dumps(
            {
                "fn_name": "solve",
                "inputs": ["1\n", "2\n", "3\n"],
                "outputs": ["1\n", "2\n", "3\n"],
            }
        ),
        "solutions": json.dumps(["def solve(x):\n    return x\n"]),
    }
    row, reason = normalize_source_record("apps", example, index=0, max_tests=10)
    assert row is None
    assert reason == "non_stdio_task"


def test_normalize_codecontests_record():
    example = {
        "name": "cc_1",
        "description": "Multiply two integers.",
        "public_tests": {"input": ["2 3\n"], "output": ["6\n"]},
        "private_tests": {"input": ["5 5\n"], "output": ["25\n"]},
        "generated_tests": {"input": ["7 8\n"], "output": ["56\n"]},
        "solutions": {
            "language": [3, "python3", "cpp"],
            "solution": ["ignored", "a, b = map(int, input().split())\nprint(a * b)\n", "ignored"],
        },
    }
    row, reason = normalize_source_record("codecontests", example, index=0, max_tests=10)
    assert reason is None
    assert row is not None
    assert row["source"] == "codecontests"
    assert len(row["test_cases"]) == 3


def test_build_mixed_dataset_bundle_filters_trivial_and_dedupes(monkeypatch):
    def fake_execute_batch(code, case_inputs, timeout_s, error_max_chars, error_max_lines, io_mode="stdio"):
        assert io_mode == "stdio"
        try:
            compiled = compile(code, "<candidate>", "exec")
        except SyntaxError as exc:
            return [
                ExecResult(kind="SYNTAX_ERROR", value=None, error_type="SyntaxError", error_msg=str(exc))
                for _ in case_inputs
            ]

        results = []
        for case_input in case_inputs:
            try:
                value = _exec_stdio_mode(compiled, case_input)
                results.append(ExecResult(kind="OK", value=value, error_type=None, error_msg=None))
            except Exception as exc:  # noqa: BLE001
                results.append(
                    ExecResult(
                        kind="RUNTIME_ERROR",
                        value=None,
                        error_type=type(exc).__name__,
                        error_msg=str(exc),
                    )
                )
        return results

    monkeypatch.setattr(dataset_builder, "execute_batch", fake_execute_batch)

    tmp_root = Path("trl") / "test-code-rl-mixed-artifacts"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    apps_path = tmp_root / "apps.jsonl"
    codecontests_path = tmp_root / "codecontests.jsonl"
    taco_path = tmp_root / "taco.jsonl"

    apps_rows = [
        {
            "problem_id": "apps_good",
            "question": "Add two integers.",
            "input_output": json.dumps(
                {
                    "fn_name": None,
                    "inputs": ["1 2\n", "3 4\n", "5 6\n"],
                    "outputs": ["3\n", "7\n", "11\n"],
                }
            ),
            "solutions": json.dumps(["a, b = map(int, input().split())\nprint(a + b)\n"]),
        },
        {
            "problem_id": "apps_bad",
            "question": "Always output zero.",
            "input_output": json.dumps(
                {
                    "fn_name": None,
                    "inputs": ["1\n", "2\n", "3\n"],
                    "outputs": ["0\n", "0\n", "0\n"],
                }
            ),
            "solutions": json.dumps(["_ = input()\nprint(0)\n"]),
        },
    ]
    codecontests_rows = [
        {
            "name": "cc_dup",
            "description": "Multiply two integers.",
            "public_tests": {"input": ["2 3\n"], "output": ["6\n"]},
            "private_tests": {"input": ["5 5\n"], "output": ["25\n"]},
            "generated_tests": {"input": ["7 8\n"], "output": ["56\n"]},
            "solutions": {
                "language": ["python3"],
                "solution": ["a, b = map(int, input().split())\nprint(a * b)\n"],
            },
        }
    ]
    taco_rows = [
        {
            "problem_id": "taco_dup",
            "question": "Multiply two integers.",
            "input_output": json.dumps(
                {
                    "fn_name": None,
                    "inputs": ["2 3\n", "5 5\n", "7 8\n", "9 9\n"],
                    "outputs": ["6\n", "25\n", "56\n", "81\n"],
                }
            ),
            "solutions": json.dumps(["a, b = map(int, input().split())\nprint(a * b)\n"]),
        }
    ]

    for path, rows in ((apps_path, apps_rows), (codecontests_path, codecontests_rows), (taco_path, taco_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    try:
        summary = build_mixed_dataset_bundle(
            source_paths={
                "apps": apps_path,
                "codecontests": codecontests_path,
                "taco": taco_path,
                "codeforces": None,
            },
            output_dir=tmp_root / "out",
            split_seed=7,
            target_total=3,
            source_targets={"apps": 1, "codecontests": 1, "taco": 1, "codeforces": 0},
            split_sizes=(2, 1, 0),
        )

        assert summary["final_counts"]["merged"] == 2
        assert summary["final_counts"]["train"] == 2
        assert summary["final_counts"]["validation"] == 0
        assert summary["reject_counts"]["trivial_baseline_passed"] == 1
        assert summary["duplicate_removal_counts"]["near"] == 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
