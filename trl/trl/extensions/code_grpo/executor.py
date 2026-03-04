import multiprocessing as mp
import traceback
from typing import Any

from .error_utils import summarize_error
from .types import ExecResult


def _exec_worker(code: str, case_input: Any, queue: "mp.Queue[tuple[str, Any, str | None, str | None]]") -> None:
    try:
        compiled = compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        queue.put(("SYNTAX_ERROR", None, "SyntaxError", "".join(traceback.format_exception(exc))))
        return

    scope: dict[str, Any] = {}
    try:
        exec(compiled, scope, scope)
        if callable(scope.get("solve")):
            value = scope["solve"](case_input)
        elif callable(scope.get("main")):
            value = scope["main"](case_input)
        elif "output" in scope:
            value = scope["output"]
        else:
            raise RuntimeError("MissingEntrypointError: expected solve(case_input), main(case_input), or global output")
        queue.put(("OK", value, None, None))
    except Exception as exc:  # noqa: BLE001
        queue.put(("RUNTIME_ERROR", None, type(exc).__name__, "".join(traceback.format_exception(exc))))


def execute(
    code: str,
    case_input: Any,
    timeout_s: float,
    error_max_chars: int,
    error_max_lines: int,
) -> ExecResult:
    """Execute candidate code against one input with timeout and normalized result."""
    queue: "mp.Queue[tuple[str, Any, str | None, str | None]]" = mp.Queue()
    process = mp.Process(target=_exec_worker, args=(code, case_input, queue))
    process.start()
    process.join(timeout=timeout_s)

    if process.is_alive():
        process.terminate()
        process.join()
        return ExecResult(
            kind="TIMEOUT",
            value=None,
            error_type="Timeout",
            error_msg=summarize_error("Execution timed out.", error_max_chars, error_max_lines),
        )

    if queue.empty():
        return ExecResult(
            kind="RUNTIME_ERROR",
            value=None,
            error_type="UnknownRuntimeError",
            error_msg=summarize_error("Execution failed without traceback.", error_max_chars, error_max_lines),
        )

    kind, value, error_type, raw_error = queue.get()
    if kind == "OK":
        return ExecResult(kind="OK", value=value, error_type=None, error_msg=None)
    return ExecResult(
        kind=kind,  # type: ignore[arg-type]
        value=None,
        error_type=error_type,
        error_msg=summarize_error(raw_error or "", error_max_chars, error_max_lines),
    )


def _exec_batch_worker(
    code: str,
    case_inputs: list[Any],
    queue: "mp.Queue[list[tuple[str, Any, str | None, str | None]]]",
) -> None:
    try:
        compiled = compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        err = "".join(traceback.format_exception(exc))
        queue.put([("SYNTAX_ERROR", None, "SyntaxError", err) for _ in case_inputs])
        return

    results: list[tuple[str, Any, str | None, str | None]] = []
    for case_input in case_inputs:
        scope: dict[str, Any] = {}
        try:
            exec(compiled, scope, scope)
            if callable(scope.get("solve")):
                value = scope["solve"](case_input)
            elif callable(scope.get("main")):
                value = scope["main"](case_input)
            elif "output" in scope:
                value = scope["output"]
            else:
                raise RuntimeError(
                    "MissingEntrypointError: expected solve(case_input), main(case_input), or global output"
                )
            results.append(("OK", value, None, None))
        except Exception as exc:  # noqa: BLE001
            results.append(("RUNTIME_ERROR", None, type(exc).__name__, "".join(traceback.format_exception(exc))))
    queue.put(results)


def execute_batch(
    code: str,
    case_inputs: list[Any],
    timeout_s: float,
    error_max_chars: int,
    error_max_lines: int,
) -> list[ExecResult]:
    """Execute candidate code against multiple inputs using one worker process."""
    if not case_inputs:
        return []

    queue: "mp.Queue[list[tuple[str, Any, str | None, str | None]]]" = mp.Queue()
    process = mp.Process(target=_exec_batch_worker, args=(code, case_inputs, queue))
    process.start()
    # Preserve per-case timeout semantics approximately via total timeout budget.
    process.join(timeout=max(timeout_s * len(case_inputs), timeout_s))

    if process.is_alive():
        process.terminate()
        process.join()
        timeout_error = summarize_error("Execution timed out.", error_max_chars, error_max_lines)
        return [
            ExecResult(kind="TIMEOUT", value=None, error_type="Timeout", error_msg=timeout_error) for _ in case_inputs
        ]

    if queue.empty():
        unknown_error = summarize_error("Execution failed without traceback.", error_max_chars, error_max_lines)
        return [
            ExecResult(
                kind="RUNTIME_ERROR",
                value=None,
                error_type="UnknownRuntimeError",
                error_msg=unknown_error,
            )
            for _ in case_inputs
        ]

    raw_results = queue.get()
    if len(raw_results) < len(case_inputs):
        raw_results.extend(
            [("RUNTIME_ERROR", None, "UnknownRuntimeError", "Missing execution result.")] * (len(case_inputs) - len(raw_results))
        )
    raw_results = raw_results[: len(case_inputs)]

    normalized: list[ExecResult] = []
    for kind, value, error_type, raw_error in raw_results:
        if kind == "OK":
            normalized.append(ExecResult(kind="OK", value=value, error_type=None, error_msg=None))
        else:
            normalized.append(
                ExecResult(
                    kind=kind,  # type: ignore[arg-type]
                    value=None,
                    error_type=error_type,
                    error_msg=summarize_error(raw_error or "", error_max_chars, error_max_lines),
                )
            )
    return normalized
