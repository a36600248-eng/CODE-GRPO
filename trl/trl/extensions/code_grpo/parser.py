import re


CODE_PATTERN = re.compile(r"<CODE>\s*(.*?)\s*</CODE>", re.DOTALL | re.IGNORECASE)
FENCED_CODE_BLOCK_PATTERN = re.compile(r"^\s*```(?:python)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
FENCED_CODE_FINDALL_PATTERN = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
REASON_PATTERN = re.compile(r"<REASON>\s*(.*?)\s*</REASON>", re.DOTALL | re.IGNORECASE)


def _normalize_response_text(text: str) -> str:
    return (text or "").replace("\ufeff", "").strip()


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
    return len(re.sub(r"\s+", "", "".join(chunks)))


def _extract_last_code_block(text: str) -> tuple[str, list[tuple[int, int]], bool]:
    fenced_matches = list(FENCED_CODE_FINDALL_PATTERN.finditer(text))
    if fenced_matches:
        match = fenced_matches[-1]
        return match.group(1).strip(), [match.span()], len(fenced_matches) == 1
    tag_matches = list(CODE_PATTERN.finditer(text))
    if tag_matches:
        match = tag_matches[-1]
        return match.group(1).strip(), [match.span()], len(tag_matches) == 1
    return text.strip(), [], False


def parse_generation_output(text: str) -> tuple[str, str, str, str]:
    normalized = _normalize_response_text(text)
    code, _, _ = _extract_last_code_block(normalized)
    reason_match = REASON_PATTERN.search(normalized)
    reason = reason_match.group(1).strip() if reason_match else ""
    return code, reason, "", ""


def parse_generation_response(
    text: str,
    *,
    allow_outside_noise_chars: int = 0,
    prefilled_code: bool = False,
) -> tuple[str, str, str, str, bool]:
    del prefilled_code
    normalized = _normalize_response_text(text)
    code, spans, single_block = _extract_last_code_block(normalized)
    outside_noise = _outside_non_ws_length(normalized, spans)
    format_ok = bool(single_block and outside_noise <= max(0, int(allow_outside_noise_chars)))
    return code, "", "", "", format_ok


def build_generation_completion(code: str) -> str:
    return "```python\n" + code.strip() + "\n```"


def _span_overlap(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    return max(span_a[0], span_b[0]) < min(span_a[1], span_b[1])


def build_token_masks(tokenizer, completion_text: str) -> tuple[list[int], list[int], list[int]]:
    completion_text = completion_text or ""
    fenced_match = FENCED_CODE_BLOCK_PATTERN.match(completion_text)
    code_start_tag = completion_text.find("<CODE>")
    code_end_tag = completion_text.find("</CODE>")
    reason_start_tag = completion_text.find("<REASON>")
    reason_end_tag = completion_text.find("</REASON>")

    encoded = tokenizer(
        completion_text,
        add_special_tokens=False,
        return_offsets_mapping=bool(getattr(tokenizer, "is_fast", False)),
    )
    ids = encoded["input_ids"]
    if "offset_mapping" not in encoded:
        return ids, [1] * len(ids), [0] * len(ids)

    if fenced_match:
        code_span = fenced_match.span(1)
        reason_span = (0, 0)
    elif code_start_tag >= 0 and code_end_tag >= 0:
        code_span = (code_start_tag + len("<CODE>"), code_end_tag)
        if reason_start_tag >= 0 and reason_end_tag >= 0:
            reason_span = (reason_start_tag + len("<REASON>"), reason_end_tag)
        else:
            reason_span = (0, 0)
    else:
        code_span = (0, len(completion_text))
        reason_span = (0, 0)

    code_mask: list[int] = []
    reason_mask: list[int] = []
    for start, end in encoded["offset_mapping"]:
        token_span = (start, end)
        code_mask.append(1 if _span_overlap(token_span, code_span) else 0)
        reason_mask.append(1 if _span_overlap(token_span, reason_span) else 0)
    return ids, code_mask, reason_mask
