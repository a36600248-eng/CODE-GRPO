import re


CODE_PATTERN = re.compile(r"<CODE>\s*(.*?)\s*</CODE>", re.DOTALL | re.IGNORECASE)
REASON_PATTERN = re.compile(r"<REASON>\s*(.*?)\s*</REASON>", re.DOTALL | re.IGNORECASE)
LOGIC_PREDICTION_PATTERN = re.compile(r"<LOGIC_PREDICTION>\s*(.*?)\s*</LOGIC_PREDICTION>", re.DOTALL | re.IGNORECASE)
EXEC_PREDICTION_PATTERN = re.compile(r"<EXEC_PREDICTION>\s*(.*?)\s*</EXEC_PREDICTION>", re.DOTALL | re.IGNORECASE)
LEGACY_PREDICTION_PATTERN = re.compile(r"<PREDICTION>\s*(.*?)\s*</PREDICTION>", re.DOTALL | re.IGNORECASE)


def parse_generation_output(text: str) -> tuple[str, str, str, str]:
    """Parse <CODE>, <REASON>, <LOGIC_PREDICTION>, <EXEC_PREDICTION> sections from generation output."""
    text = text or ""
    code_match = CODE_PATTERN.search(text)
    reason_match = REASON_PATTERN.search(text)
    code = code_match.group(1).strip() if code_match else text.strip()
    reason_block = reason_match.group(1).strip() if reason_match else ""

    logic_match = LOGIC_PREDICTION_PATTERN.search(reason_block) or LOGIC_PREDICTION_PATTERN.search(text)
    exec_match = EXEC_PREDICTION_PATTERN.search(reason_block) or EXEC_PREDICTION_PATTERN.search(text)
    legacy_match = LEGACY_PREDICTION_PATTERN.search(reason_block)

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


def parse_logic_response(text: str, require_reason_before_prediction: bool = True) -> tuple[str, str, bool]:
    """
    Parse logic response and return:
    (reason_text, logic_prediction, format_ok)
    """
    text = text or ""
    reason_match = REASON_PATTERN.search(text)
    logic_match = LOGIC_PREDICTION_PATTERN.search(text)

    reason = reason_match.group(1).strip() if reason_match else ""
    prediction = logic_match.group(1).strip() if logic_match else parse_logic_prediction_only(text)
    format_ok = bool(reason_match and logic_match)
    if format_ok and require_reason_before_prediction:
        format_ok = _is_reason_before_prediction(reason_match, logic_match)
    return reason, prediction, format_ok


def parse_exec_response(text: str, require_reason_before_prediction: bool = True) -> tuple[str, str, bool]:
    """
    Parse execution response and return:
    (reason_text, exec_prediction, format_ok)
    """
    text = text or ""
    reason_match = REASON_PATTERN.search(text)
    exec_match = EXEC_PREDICTION_PATTERN.search(text)

    reason = reason_match.group(1).strip() if reason_match else ""
    prediction = exec_match.group(1).strip() if exec_match else parse_exec_prediction_only(text)
    format_ok = bool(reason_match and exec_match)
    if format_ok and require_reason_before_prediction:
        format_ok = _is_reason_before_prediction(reason_match, exec_match)
    return reason, prediction, format_ok


def build_canonical_completion(code: str, reasoning: str, logic_prediction: str, exec_prediction: str) -> str:
    return (
        "<CODE>\n"
        f"{code.strip()}\n"
        "</CODE>\n"
        "<REASON>\n"
        f"{reasoning.strip()}\n"
        "<LOGIC_PREDICTION>\n"
        f"{logic_prediction.strip()}\n"
        "</LOGIC_PREDICTION>\n"
        "<EXEC_PREDICTION>\n"
        f"{exec_prediction.strip()}\n"
        "</EXEC_PREDICTION>\n"
        "</REASON>"
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

    if code_start_tag < 0 or code_end_tag < 0:
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
