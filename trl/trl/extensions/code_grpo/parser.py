import re


CODE_PATTERN = re.compile(r"<CODE>\s*(.*?)\s*</CODE>", re.DOTALL | re.IGNORECASE)
FENCED_CODE_BLOCK_PATTERN = re.compile(r"^\s*```(?:python)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
LEADING_FENCE_PATTERN = re.compile(r"^\s*```(?:python)?[ \t]*\r?\n?", re.IGNORECASE)
REASON_PATTERN = re.compile(r"<REASON>\s*(.*?)\s*</REASON>", re.DOTALL | re.IGNORECASE)
LOGIC_PREDICTION_PATTERN = re.compile(r"<LOGIC_PREDICTION>\s*(.*?)\s*</LOGIC_PREDICTION>", re.DOTALL | re.IGNORECASE)
EXEC_PREDICTION_PATTERN = re.compile(r"<EXEC_PREDICTION>\s*(.*?)\s*</EXEC_PREDICTION>", re.DOTALL | re.IGNORECASE)
LEGACY_PREDICTION_PATTERN = re.compile(r"<PREDICTION>\s*(.*?)\s*</PREDICTION>", re.DOTALL | re.IGNORECASE)
CODE_LIKE_PATTERN = re.compile(r"```|<CODE>|def\s+\w+\s*\(|class\s+\w+\s*[:(]", re.IGNORECASE)
FENCED_BLOCK_PATTERN = re.compile(r"^\s*```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```\s*$", re.DOTALL)
CODE_CLOSE_TAG = "</CODE>"


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    match = None
    for candidate in pattern.finditer(text):
        match = candidate
    return match


def _normalize_response_text(text: str, *, strip_fenced_block: bool = True) -> str:
    text = (text or "").replace("\ufeff", "").strip()
    if not text:
        return ""
    fenced_match = FENCED_BLOCK_PATTERN.match(text) if strip_fenced_block else None
    if fenced_match:
        return fenced_match.group(1).strip()
    return text


def _extract_leading_fenced_code_block(text: str) -> tuple[str | None, list[tuple[int, int]]]:
    normalized = _normalize_response_text(text, strip_fenced_block=False)
    if not normalized:
        return None, []
    opening = LEADING_FENCE_PATTERN.match(normalized)
    if opening is None:
        return None, []
    body_start = opening.end()
    close_idx = normalized.find("```", body_start)
    if close_idx < 0:
        code = normalized[body_start:].strip()
        return code, [(opening.start(), len(normalized))]
    code = normalized[body_start:close_idx].strip()
    close_end = close_idx + 3
    return code, [(opening.start(), close_end)]


def parse_generation_output(text: str) -> tuple[str, str, str, str]:
    """Parse <CODE>, <REASON>, <LOGIC_PREDICTION>, <EXEC_PREDICTION> sections from generation output."""
    text = _normalize_response_text(text, strip_fenced_block=False)
    code_match = _last_match(CODE_PATTERN, text)
    fenced_code, _ = _extract_leading_fenced_code_block(text)
    reason_match = _last_match(REASON_PATTERN, text)
    if code_match:
        code = code_match.group(1).strip()
    elif fenced_code is not None:
        code = fenced_code
    else:
        code = text.strip()
    reason_block = reason_match.group(1).strip() if reason_match else ""

    logic_match = _last_match(LOGIC_PREDICTION_PATTERN, reason_block) or _last_match(LOGIC_PREDICTION_PATTERN, text)
    exec_match = _last_match(EXEC_PREDICTION_PATTERN, reason_block) or _last_match(EXEC_PREDICTION_PATTERN, text)
    legacy_match = _last_match(LEGACY_PREDICTION_PATTERN, reason_block) or _last_match(LEGACY_PREDICTION_PATTERN, text)

    logic_prediction = logic_match.group(1).strip() if logic_match else ""
    exec_prediction = exec_match.group(1).strip() if exec_match else ""
    if legacy_match and not logic_prediction:
        logic_prediction = legacy_match.group(1).strip()
    if legacy_match and not exec_prediction:
        exec_prediction = legacy_match.group(1).strip()

    reason = LOGIC_PREDICTION_PATTERN.sub("", reason_block)
    reason = EXEC_PREDICTION_PATTERN.sub("", reason)
    reason = LEGACY_PREDICTION_PATTERN.sub("", reason).strip()
    return code, reason, logic_prediction, exec_prediction


def _canonicalize_generation_text(text: str, *, prefilled_code: bool) -> str:
    normalized = _normalize_response_text(text, strip_fenced_block=False)
    if not normalized:
        return ""

    fenced_code, _ = _extract_leading_fenced_code_block(normalized)
    if fenced_code is not None:
        code = fenced_code
        return f"```python\n{code}\n```"

    code_matches = list(CODE_PATTERN.finditer(normalized))
    if code_matches:
        code = code_matches[0].group(1).strip()
        return f"<CODE>\n{code}\n</CODE>"

    if not prefilled_code:
        return normalized

    lowered = normalized.lower()
    close_idx = lowered.find(CODE_CLOSE_TAG.lower())
    body = normalized[:close_idx] if close_idx >= 0 else normalized
    body = body.strip()
    return f"<CODE>\n{body}\n</CODE>"


def parse_generation_response(
    text: str,
    *,
    allow_outside_noise_chars: int = 0,
    prefilled_code: bool = False,
) -> tuple[str, str, str, str, bool]:
    """
    Parse generation response and return:
    (code, reason, logic_prediction, exec_prediction, format_ok)

    Main-generation format is valid when either:
    - one fenced Python code block exists and outside noise is within threshold, or
    - one legacy <CODE> block exists and outside noise is within threshold, or
    - prefilled-code mode is enabled and the completion closes the block without trailing noise.
    """
    normalized = _normalize_response_text(text, strip_fenced_block=False)
    canonical = _canonicalize_generation_text(normalized, prefilled_code=prefilled_code)
    code_matches = list(CODE_PATTERN.finditer(normalized))
    code_match = code_matches[0] if len(code_matches) == 1 else None
    fenced_code, fenced_spans = _extract_leading_fenced_code_block(normalized)
    code, reason, logic_prediction, exec_prediction = parse_generation_output(canonical)

    format_ok = False
    if fenced_code is not None:
        outside_noise = _outside_non_ws_length(normalized, fenced_spans)
        format_ok = outside_noise <= max(0, int(allow_outside_noise_chars))
    elif prefilled_code and not code_matches:
        lowered = normalized.lower()
        close_idx = lowered.find(CODE_CLOSE_TAG.lower())
        if close_idx >= 0:
            tail_start = close_idx + len(CODE_CLOSE_TAG)
            outside_noise = len(re.sub(r"\s+", "", normalized[tail_start:]))
            format_ok = outside_noise <= max(0, int(allow_outside_noise_chars))
    elif code_match is not None:
        outside_noise = _outside_non_ws_length(normalized, [code_match.span()])
        outside_ok = outside_noise <= max(0, int(allow_outside_noise_chars))
        format_ok = bool(outside_ok)
    return code, reason, logic_prediction, exec_prediction, format_ok


def parse_logic_prediction_only(text: str) -> str:
    _, _, logic_prediction, _ = parse_generation_output(text)
    return logic_prediction or (text or "").strip()


def parse_exec_prediction_only(text: str) -> str:
    _, _, _, exec_prediction = parse_generation_output(text)
    return exec_prediction or (text or "").strip()


def parse_prediction_only(text: str) -> str:
    """Backward-compatible alias that returns exec prediction."""
    return parse_exec_prediction_only(text)


def _is_reason_before_prediction(reason_match: re.Match | None, prediction_match: re.Match | None) -> bool:
    if reason_match is None or prediction_match is None:
        return False
    return reason_match.start() < prediction_match.start() and reason_match.end() <= prediction_match.start()


def _has_non_whitespace_outside_spans(text: str, spans: list[tuple[int, int]]) -> bool:
    if not spans:
        return bool(text.strip())
    spans = sorted(spans, key=lambda x: x[0])
    cursor = 0
    for start, end in spans:
        if text[cursor:start].strip():
            return True
        cursor = max(cursor, end)
    return bool(text[cursor:].strip())


def _outside_non_ws_length(text: str, spans: list[tuple[int, int]]) -> int:
    if not spans:
        return len(re.sub(r"\s+", "", text or ""))
    spans = sorted(spans, key=lambda x: x[0])
    chunks: list[str] = []
    cursor = 0
    for start, end in spans:
        chunks.append(text[cursor:start])
        cursor = max(cursor, end)
    chunks.append(text[cursor:])
    outside = "".join(chunks)
    return len(re.sub(r"\s+", "", outside))


def _is_single_line_value(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return "\n" not in stripped and "\r" not in stripped


def _format_ok_strict(
    text: str,
    reason_match: re.Match | None,
    prediction_match: re.Match | None,
    *,
    require_reason_before_prediction: bool,
    prediction_max_chars: int,
    reason_max_chars: int,
    disallow_code_in_reasoning: bool,
    allow_outside_noise_chars: int = 0,
) -> bool:
    if reason_match is None or prediction_match is None:
        return False
    if require_reason_before_prediction and not _is_reason_before_prediction(reason_match, prediction_match):
        return False
    reason = reason_match.group(1).strip()
    prediction = prediction_match.group(1).strip()
    if len(reason) > reason_max_chars or len(prediction) > prediction_max_chars:
        return False
    if not _is_single_line_value(prediction):
        return False
    if disallow_code_in_reasoning and CODE_LIKE_PATTERN.search(reason):
        return False
    # Allow small prompt-echo noise outside tags to avoid over-sparse rewards.
    # Keep strict behavior when `allow_outside_noise_chars` is 0.
    outside_noise = _outside_non_ws_length(text, [reason_match.span(), prediction_match.span()])
    if outside_noise > max(0, int(allow_outside_noise_chars)):
        return False
    return True


def parse_logic_response(
    text: str,
    require_reason_before_prediction: bool = True,
    prediction_max_chars: int = 200,
    reason_max_chars: int = 400,
    disallow_code_in_reasoning: bool = True,
    allow_outside_noise_chars: int = 0,
) -> tuple[str, str, bool]:
    """
    Parse logic response and return:
    (reason_text, logic_prediction, format_ok)
    """
    text = _normalize_response_text(text)
    reason_match = _last_match(REASON_PATTERN, text)
    logic_match = _last_match(LOGIC_PREDICTION_PATTERN, text)

    reason = reason_match.group(1).strip() if reason_match else ""
    prediction = logic_match.group(1).strip() if logic_match else parse_logic_prediction_only(text)
    format_ok = _format_ok_strict(
        text=text,
        reason_match=reason_match,
        prediction_match=logic_match,
        require_reason_before_prediction=require_reason_before_prediction,
        prediction_max_chars=prediction_max_chars,
        reason_max_chars=reason_max_chars,
        disallow_code_in_reasoning=disallow_code_in_reasoning,
        allow_outside_noise_chars=allow_outside_noise_chars,
    )
    return reason, prediction, format_ok


def parse_exec_response(
    text: str,
    require_reason_before_prediction: bool = True,
    prediction_max_chars: int = 200,
    reason_max_chars: int = 400,
    disallow_code_in_reasoning: bool = True,
    allow_outside_noise_chars: int = 0,
) -> tuple[str, str, bool]:
    """
    Parse execution response and return:
    (reason_text, exec_prediction, format_ok)
    """
    text = _normalize_response_text(text)
    reason_match = _last_match(REASON_PATTERN, text)
    exec_match = _last_match(EXEC_PREDICTION_PATTERN, text)

    reason = reason_match.group(1).strip() if reason_match else ""
    prediction = exec_match.group(1).strip() if exec_match else parse_exec_prediction_only(text)
    format_ok = _format_ok_strict(
        text=text,
        reason_match=reason_match,
        prediction_match=exec_match,
        require_reason_before_prediction=require_reason_before_prediction,
        prediction_max_chars=prediction_max_chars,
        reason_max_chars=reason_max_chars,
        disallow_code_in_reasoning=disallow_code_in_reasoning,
        allow_outside_noise_chars=allow_outside_noise_chars,
    )
    return reason, prediction, format_ok


def build_generation_completion(code: str) -> str:
    return (
        "```python\n"
        f"{code.strip()}\n"
        "```"
    )


def build_canonical_completion(
    code: str,
    reasoning: str,
    logic_prediction: str,
    exec_prediction: str,
    include_predictions: bool = True,
    include_reason: bool = True,
) -> str:
    reason_body = reasoning.strip()
    if include_predictions:
        reason_body = (
            f"{reason_body}\n"
            "<LOGIC_PREDICTION>\n"
            f"{logic_prediction.strip()}\n"
            "</LOGIC_PREDICTION>\n"
            "<EXEC_PREDICTION>\n"
            f"{exec_prediction.strip()}\n"
            "</EXEC_PREDICTION>"
        ).strip()
    if include_reason:
        return (
            "<REASON>\n"
            f"{reason_body}\n"
            "</REASON>\n"
            "<CODE>\n"
            f"{code.strip()}\n"
            "</CODE>"
        )
    return (
        "<CODE>\n"
        f"{code.strip()}\n"
        "</CODE>"
    )


def _span_overlap(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    return max(span_a[0], span_b[0]) < min(span_a[1], span_b[1])


def build_token_masks(tokenizer, completion_text: str) -> tuple[list[int], list[int], list[int]]:
    """
    Build completion token ids and binary masks for code/reason regions.

    Returns:
        input_ids, code_mask, reason_mask
    """
    code_start_tag = completion_text.find("<CODE>")
    code_end_tag = completion_text.find("</CODE>")
    reason_start_tag = completion_text.find("<REASON>")
    reason_end_tag = completion_text.find("</REASON>")
    fenced_match = FENCED_CODE_BLOCK_PATTERN.match(completion_text)

    if code_start_tag < 0 or code_end_tag < 0:
        if fenced_match:
            code_span = fenced_match.span(1)
            encoded = tokenizer(
                completion_text,
                add_special_tokens=False,
                return_offsets_mapping=bool(getattr(tokenizer, "is_fast", False)),
            )
            ids = encoded["input_ids"]
            if "offset_mapping" not in encoded:
                return ids, [1] * len(ids), [0] * len(ids)
            code_mask: list[int] = []
            reason_mask: list[int] = []
            for start, end in encoded["offset_mapping"]:
                token_span = (start, end)
                code_mask.append(1 if _span_overlap(token_span, code_span) else 0)
                reason_mask.append(0)
            return ids, code_mask, reason_mask
        encoded = tokenizer(completion_text, add_special_tokens=False)
        ids = encoded["input_ids"]
        return ids, [1] * len(ids), [0] * len(ids)

    code_span = (code_start_tag + len("<CODE>"), code_end_tag)
    if reason_start_tag >= 0 and reason_end_tag >= 0:
        reason_span = (reason_start_tag + len("<REASON>"), reason_end_tag)
    else:
        reason_span = (0, 0)

    encoded = tokenizer(
        completion_text,
        add_special_tokens=False,
        return_offsets_mapping=bool(getattr(tokenizer, "is_fast", False)),
    )
    ids = encoded["input_ids"]

    if "offset_mapping" not in encoded:
        return ids, [0] * len(ids), [0] * len(ids)

    code_mask: list[int] = []
    reason_mask: list[int] = []
    for start, end in encoded["offset_mapping"]:
        token_span = (start, end)
        code_mask.append(1 if _span_overlap(token_span, code_span) else 0)
        reason_mask.append(1 if _span_overlap(token_span, reason_span) else 0)
    return ids, code_mask, reason_mask
