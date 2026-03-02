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
            value = None
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

