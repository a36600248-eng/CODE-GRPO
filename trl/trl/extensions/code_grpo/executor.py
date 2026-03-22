import contextlib
import io
import json
import multiprocessing as mp
import subprocess
import sys
import traceback
from typing import Any

from .error_utils import summarize_error
from .types import ExecResult


_SUBPROCESS_BATCH_RUNNER = r"""
import contextlib
import io
import json
import sys
import traceback


def _exec_call_mode(compiled, case_input):
    scope = {}
    exec(compiled, scope, scope)
    if callable(scope.get("solve")):
        return scope["solve"](case_input)
    if callable(scope.get("main")):
        return scope["main"](case_input)
    if "output" in scope:
        return scope["output"]
    raise RuntimeError("MissingEntrypointError: expected solve(case_input), main(case_input), or global output")


def _exec_stdio_mode(compiled, case_input):
    stdin_text = case_input if isinstance(case_input, str) else ("" if case_input is None else str(case_input))
    scope = {"__name__": "__main__"}
    stdin_buffer = io.StringIO(stdin_text)
    stdout_buffer = io.StringIO()
    old_stdin, old_stdout = sys.stdin, sys.stdout
    try:
        sys.stdin = stdin_buffer
        sys.stdout = stdout_buffer
        with contextlib.redirect_stdout(stdout_buffer):
            exec(compiled, scope, scope)
        return stdout_buffer.getvalue()
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout


payload = json.load(sys.stdin)
code = payload["code"]
case_inputs = payload["case_inputs"]
io_mode = payload["io_mode"]

try:
    compiled = compile(code, "<candidate>", "exec")
except SyntaxError as exc:
    err = "".join(traceback.format_exception(exc))
    json.dump([["SYNTAX_ERROR", None, "SyntaxError", err] for _ in case_inputs], sys.stdout)
    raise SystemExit(0)

results = []
for case_input in case_inputs:
    try:
        value = _exec_stdio_mode(compiled, case_input) if io_mode == "stdio" else _exec_call_mode(compiled, case_input)
        try:
            json.dumps(value)
        except TypeError:
            value = repr(value)
        results.append(["OK", value, None, None])
    except Exception as exc:
        results.append(["RUNTIME_ERROR", None, type(exc).__name__, "".join(traceback.format_exception(exc))])

json.dump(results, sys.stdout)
"""


def _validate_io_mode(io_mode: str) -> str:
    normalized = str(io_mode or "call").strip().lower()
    if normalized not in {"call", "stdio"}:
        raise ValueError(f"Unsupported io_mode: {io_mode}")
    return normalized


def _exec_call_mode(compiled, case_input: Any) -> Any:
    scope: dict[str, Any] = {}
    exec(compiled, scope, scope)
    if callable(scope.get("solve")):
        return scope["solve"](case_input)
    if callable(scope.get("main")):
        return scope["main"](case_input)
    if "output" in scope:
        return scope["output"]
    raise RuntimeError("MissingEntrypointError: expected solve(case_input), main(case_input), or global output")


def _exec_stdio_mode(compiled, case_input: Any) -> str:
    stdin_text = case_input if isinstance(case_input, str) else ("" if case_input is None else str(case_input))
    scope: dict[str, Any] = {"__name__": "__main__"}
    stdin_buffer = io.StringIO(stdin_text)
    stdout_buffer = io.StringIO()
    old_stdin, old_stdout = sys.stdin, sys.stdout
    try:
        sys.stdin = stdin_buffer
        sys.stdout = stdout_buffer
        with contextlib.redirect_stdout(stdout_buffer):
            exec(compiled, scope, scope)
        return stdout_buffer.getvalue()
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout


def _exec_worker(
    code: str,
    case_input: Any,
    queue: "mp.Queue[tuple[str, Any, str | None, str | None]]",
    io_mode: str,
) -> None:
    try:
        compiled = compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        queue.put(("SYNTAX_ERROR", None, "SyntaxError", "".join(traceback.format_exception(exc))))
        return

    try:
        if io_mode == "stdio":
            value = _exec_stdio_mode(compiled, case_input)
        else:
            value = _exec_call_mode(compiled, case_input)
        queue.put(("OK", value, None, None))
    except Exception as exc:  # noqa: BLE001
        queue.put(("RUNTIME_ERROR", None, type(exc).__name__, "".join(traceback.format_exception(exc))))


def execute(
    code: str,
    case_input: Any,
    timeout_s: float,
    error_max_chars: int,
    error_max_lines: int,
    io_mode: str = "call",
) -> ExecResult:
    """Execute candidate code against one input with timeout and normalized result."""
    validated_io_mode = _validate_io_mode(io_mode)
    try:
        queue: "mp.Queue[tuple[str, Any, str | None, str | None]]" = mp.Queue()
        process = mp.Process(target=_exec_worker, args=(code, case_input, queue, validated_io_mode))
        process.start()
    except PermissionError:
        return _execute_batch_via_subprocess(
            code,
            [case_input],
            timeout_s,
            error_max_chars,
            error_max_lines,
            validated_io_mode,
        )[0]

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
    io_mode: str,
) -> None:
    try:
        compiled = compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        err = "".join(traceback.format_exception(exc))
        queue.put([("SYNTAX_ERROR", None, "SyntaxError", err) for _ in case_inputs])
        return

    results: list[tuple[str, Any, str | None, str | None]] = []
    for case_input in case_inputs:
        try:
            if io_mode == "stdio":
                value = _exec_stdio_mode(compiled, case_input)
            else:
                value = _exec_call_mode(compiled, case_input)
            results.append(("OK", value, None, None))
        except Exception as exc:  # noqa: BLE001
            results.append(("RUNTIME_ERROR", None, type(exc).__name__, "".join(traceback.format_exception(exc))))
    queue.put(results)


def _execute_batch_via_subprocess(
    code: str,
    case_inputs: list[Any],
    timeout_s: float,
    error_max_chars: int,
    error_max_lines: int,
    io_mode: str,
) -> list[ExecResult]:
    payload = {"code": code, "case_inputs": case_inputs, "io_mode": _validate_io_mode(io_mode)}
    timeout_budget = max(timeout_s * len(case_inputs), timeout_s)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_BATCH_RUNNER],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_budget,
            check=False,
        )
    except subprocess.TimeoutExpired:
        timeout_error = summarize_error("Execution timed out.", error_max_chars, error_max_lines)
        return [ExecResult(kind="TIMEOUT", value=None, error_type="Timeout", error_msg=timeout_error) for _ in case_inputs]

    if completed.returncode != 0:
        runtime_error = summarize_error(completed.stderr or "Execution failed without traceback.", error_max_chars, error_max_lines)
        return [
            ExecResult(
                kind="RUNTIME_ERROR",
                value=None,
                error_type="SubprocessError",
                error_msg=runtime_error,
            )
            for _ in case_inputs
        ]

    try:
        raw_results = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
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

    if len(raw_results) < len(case_inputs):
        raw_results.extend(
            [["RUNTIME_ERROR", None, "UnknownRuntimeError", "Missing execution result."]]
            * (len(case_inputs) - len(raw_results))
        )
    raw_results = raw_results[: len(case_inputs)]

    normalized: list[ExecResult] = []
    for kind, value, error_type, raw_error in raw_results:
        if kind == "OK":
            normalized.append(ExecResult(kind="OK", value=value, error_type=None, error_msg=None))
        else:
            normalized.append(
                ExecResult(
                    kind=kind,
                    value=None,
                    error_type=error_type,
                    error_msg=summarize_error(raw_error or "", error_max_chars, error_max_lines),
                )
            )
    return normalized


def execute_batch(
    code: str,
    case_inputs: list[Any],
    timeout_s: float,
    error_max_chars: int,
    error_max_lines: int,
    io_mode: str = "call",
) -> list[ExecResult]:
    """Execute candidate code against multiple inputs using one worker process."""
    if not case_inputs:
        return []

    validated_io_mode = _validate_io_mode(io_mode)
    try:
        queue: "mp.Queue[list[tuple[str, Any, str | None, str | None]]]" = mp.Queue()
        process = mp.Process(target=_exec_batch_worker, args=(code, case_inputs, queue, validated_io_mode))
        process.start()
    except PermissionError:
        return _execute_batch_via_subprocess(code, case_inputs, timeout_s, error_max_chars, error_max_lines, validated_io_mode)

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
