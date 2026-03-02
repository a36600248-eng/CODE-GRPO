import ast
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .types import ExecResult


def _normalize_ws(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_error_type(text: str | None) -> str:
    if not text:
        return ""
    candidate = text.strip().split(":")[0].split(".")[-1]
    return candidate.upper()


def _try_parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def values_equal(lhs: Any, rhs: Any, float_tol: float = 1e-6) -> bool:
    lhs = _try_parse_literal(lhs)
    rhs = _try_parse_literal(rhs)

    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=float_tol)

    if isinstance(lhs, str) and isinstance(rhs, str):
        return _normalize_ws(lhs) == _normalize_ws(rhs)

    if isinstance(lhs, Mapping) and isinstance(rhs, Mapping):
        if set(lhs.keys()) != set(rhs.keys()):
            return False
        return all(values_equal(lhs[k], rhs[k], float_tol=float_tol) for k in lhs)

    if isinstance(lhs, Sequence) and isinstance(rhs, Sequence) and not isinstance(lhs, str) and not isinstance(rhs, str):
        if len(lhs) != len(rhs):
            return False
        return all(values_equal(a, b, float_tol=float_tol) for a, b in zip(lhs, rhs, strict=True))

    return lhs == rhs


def is_match(pred: str, actual: ExecResult) -> bool:
    """Match prediction string against execution result with structured-first fallback."""
    pred = pred or ""
    if actual.kind != "OK":
        pred_error = _normalize_error_type(pred)
        actual_error = _normalize_error_type(actual.error_type or actual.kind)
        return bool(pred_error) and pred_error == actual_error

    parsed_pred = _try_parse_literal(pred)
    return values_equal(parsed_pred, actual.value)

