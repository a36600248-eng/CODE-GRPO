import argparse
import ast
import json
import operator
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_SAFE_CALLS = {
    "set": set,
    "list": list,
    "tuple": tuple,
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
}


def _safe_eval_expr(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_eval_expr(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_expr(elt) for elt in node.elts)
    if isinstance(node, ast.Set):
        return {_safe_eval_expr(elt) for elt in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _safe_eval_expr(key): _safe_eval_expr(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval_expr(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval_expr(node.left), _safe_eval_expr(node.right))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_CALLS:
            fn = _SAFE_CALLS[node.func.id]
            args = [_safe_eval_expr(arg) for arg in node.args]
            kwargs = {kw.arg: _safe_eval_expr(kw.value) for kw in node.keywords if kw.arg is not None}
            return fn(*args, **kwargs)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sys"
            and node.func.attr == "getsizeof"
        ):
            args = [_safe_eval_expr(arg) for arg in node.args]
            kwargs = {kw.arg: _safe_eval_expr(kw.value) for kw in node.keywords if kw.arg is not None}
            return sys.getsizeof(*args, **kwargs)
    raise ValueError(f"Unsupported literal: {ast.unparse(node)}")


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception as exc:  # noqa: BLE001
        try:
            return _safe_eval_expr(node)
        except Exception as safe_exc:  # noqa: BLE001
            raise ValueError(f"Unsupported literal: {ast.unparse(node)}") from safe_exc


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    raise ValueError(f"Unsupported callable: {ast.dump(node)}")


def _parse_assert_case(assert_src: str) -> dict[str, Any]:
    tree = ast.parse(assert_src)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        raise ValueError("Expected one assert statement.")

    test = tree.body[0].test
    call_node: ast.Call | None = None
    expected: Any

    if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
        left = test.left
        right = test.comparators[0]
        op = test.ops[0]
        if isinstance(op, ast.Eq):
            if isinstance(left, ast.Call):
                call_node = left
                expected = _literal(right)
            elif isinstance(right, ast.Call):
                call_node = right
                expected = _literal(left)
            else:
                raise ValueError("Equality assert does not compare a function call.")
        else:
            raise ValueError(f"Unsupported compare operator: {type(op).__name__}")
    elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and isinstance(test.operand, ast.Call):
        call_node = test.operand
        expected = False
    elif isinstance(test, ast.Call):
        call_node = test
        expected = True
    else:
        raise ValueError(f"Unsupported assert shape: {ast.dump(test)}")

    if call_node is None:
        raise ValueError("Could not locate target function call.")

    args = [_literal(arg) for arg in call_node.args]
    kwargs = {}
    for kw in call_node.keywords:
        if kw.arg is None:
            raise ValueError("Star kwargs are not supported.")
        kwargs[kw.arg] = _literal(kw.value)

    return {
        "function_name": _call_name(call_node.func),
        "args": args,
        "kwargs": kwargs,
        "expected": expected,
        "raw_assert": assert_src,
    }


def _infer_input_mode(cases: list[dict[str, Any]]) -> str:
    if not cases:
        raise ValueError("No parsed cases.")

    has_kwargs = any(case["kwargs"] for case in cases)
    if has_kwargs:
        return "args_kwargs"

    arg_counts = {len(case["args"]) for case in cases}
    if len(arg_counts) != 1:
        return "args_list"

    only_count = next(iter(arg_counts))
    if only_count == 0:
        return "no_args"
    if only_count == 1:
        return "single_arg"
    return "args_list"


def _serialize_case_input(case: dict[str, Any], input_mode: str) -> Any:
    if input_mode == "single_arg":
        return case["args"][0]
    if input_mode == "args_list":
        return list(case["args"])
    if input_mode == "args_kwargs":
        return {"args": list(case["args"]), "kwargs": dict(case["kwargs"])}
    if input_mode == "no_args":
        return None
    raise ValueError(f"Unsupported input mode: {input_mode}")


def _runtime_contract(function_name: str, input_mode: str) -> str:
    lines = [
        "Write Python function solve(x).",
        "Return the answer directly. Do not print.",
        f"The original MBPP tests called function `{function_name}`.",
        "Runtime contract for this dataset:",
    ]
    if input_mode == "single_arg":
        lines.append(f"- `solve(x)` receives the original single argument from `{function_name}(arg0)` directly.")
    elif input_mode == "args_list":
        lines.append(
            f"- `solve(x)` receives a Python list of positional arguments in order. "
            f"The original call `{function_name}(a, b, ...)` becomes `solve([a, b, ...])`."
        )
    elif input_mode == "args_kwargs":
        lines.append(
            "- `solve(x)` receives a Python dict with two keys: "
            "`{'args': [...], 'kwargs': {...}}`."
        )
        lines.append(
            f"- The original call `{function_name}(*args, **kwargs)` becomes "
            "`solve({'args': args, 'kwargs': kwargs})`."
        )
    elif input_mode == "no_args":
        lines.append(f"- `solve(x)` receives `None` because `{function_name}()` takes no arguments.")
    else:
        raise ValueError(f"Unsupported input mode: {input_mode}")
    return "\n".join(lines)


def _build_prompt(task_prompt: str, function_name: str, input_mode: str) -> str:
    return (
        f"{_runtime_contract(function_name, input_mode)}\n\n"
        f"Original MBPP task:\n{task_prompt.strip()}"
    )


def _convert_example(example: dict[str, Any], split: str) -> dict[str, Any]:
    test_list = list(example.get("test_list") or [])
    if not test_list:
        raise ValueError("Missing test_list.")

    parsed_cases = [_parse_assert_case(test_src) for test_src in test_list]
    function_names = {case["function_name"] for case in parsed_cases}
    if len(function_names) != 1:
        raise ValueError(f"Inconsistent function names: {sorted(function_names)}")

    function_name = next(iter(function_names))
    input_mode = _infer_input_mode(parsed_cases)
    prompt = _build_prompt(str(example["prompt"]), function_name, input_mode)

    test_cases = []
    for case in parsed_cases:
        case_input = _serialize_case_input(case, input_mode)
        test_cases.append(
            {
                "input": repr(case_input),
                "output": repr(case["expected"]),
            }
        )

    task_id = int(example["task_id"])
    return {
        "question_id": f"mbpp_{split}_{task_id}",
        "prompt": prompt,
        "test_cases": test_cases,
        "source_dataset": "mbpp",
        "source_config": "sanitized",
        "source_split": split,
        "source_task_id": task_id,
        "source_function_name": function_name,
        "source_test_count": len(test_cases),
        "source_prompt": str(example["prompt"]).strip(),
        "source_test_list": test_list,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_mbpp(output_dir: Path, small_train_size: int) -> dict[str, Any]:
    dataset = load_dataset("mbpp", "sanitized")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "dataset": "mbpp",
        "config": "sanitized",
        "splits": {},
        "files": {},
        "skipped_examples": defaultdict(list),
    }

    for split in ("train", "validation", "test"):
        converted: list[dict[str, Any]] = []
        skipped = 0
        reasons = Counter()
        for example in dataset[split]:
            try:
                converted.append(_convert_example(example, split))
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                reasons[type(exc).__name__] += 1
                summary["skipped_examples"][split].append(
                    {
                        "task_id": int(example.get("task_id", -1)),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

        split_rows[split] = converted
        summary["splits"][split] = {
            "raw_count": len(dataset[split]),
            "converted_count": len(converted),
            "skipped_count": skipped,
            "skip_reasons": dict(reasons),
        }

    train_path = output_dir / "mbpp_sanitized_codegrpo_train.jsonl"
    val_path = output_dir / "mbpp_sanitized_codegrpo_validation.jsonl"
    test_path = output_dir / "mbpp_sanitized_codegrpo_test.jsonl"
    trainval_path = output_dir / "mbpp_sanitized_codegrpo_trainval.jsonl"
    small_path = output_dir / f"mbpp_sanitized_codegrpo_small{small_train_size}.jsonl"

    _write_jsonl(train_path, split_rows["train"])
    _write_jsonl(val_path, split_rows["validation"])
    _write_jsonl(test_path, split_rows["test"])
    _write_jsonl(trainval_path, split_rows["train"] + split_rows["validation"])
    _write_jsonl(small_path, split_rows["train"][:small_train_size])

    summary["files"] = {
        "train": str(train_path.name),
        "validation": str(val_path.name),
        "test": str(test_path.name),
        "trainval": str(trainval_path.name),
        "small_train": str(small_path.name),
        "small_train_size": small_train_size,
    }
    summary["skipped_examples"] = dict(summary["skipped_examples"])

    summary_path = output_dir / "mbpp_sanitized_codegrpo_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MBPP and convert it to Code-GRPO JSONL format.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--small-train-size", type=int, default=64)
    args = parser.parse_args()

    summary = convert_mbpp(args.output_dir, args.small_train_size)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
