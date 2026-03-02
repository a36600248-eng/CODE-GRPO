def summarize_error(error_text: str, max_chars: int, max_lines: int) -> str:
    """Return a compact error summary bounded by char and line limits."""
    text = (error_text or "").strip()
    if not text:
        return ""
    lines = text.splitlines()[:max_lines]
    compact = "\n".join(lines)
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip() + "..."
    return compact

