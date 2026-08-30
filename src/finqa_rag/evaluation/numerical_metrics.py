"""Financial-answer normalization and controlled-condition metrics."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation

from finqa_rag.common.timing import latency_summary


NormalizedAnswer = Decimal | str
_CURRENCY = re.compile(r"[$€£¥]")


def normalize_answer(value: str | Decimal) -> NormalizedAnswer:
    """Normalize displayed financial answers without rescaling trailing percents."""

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("answer must be finite")
        return value
    text = value.strip()
    lowered = text.lower()
    if lowered in {"yes", "no"}:
        return lowered
    negative = len(text) >= 2 and text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = _CURRENCY.sub("", text).replace(",", "").strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"unparseable financial answer: {value}") from error
    if not number.is_finite():
        raise ValueError("answer must be finite")
    return -number if negative else number


def answers_match(
    predicted: str | Decimal,
    gold: str | Decimal,
    tolerance: Decimal = Decimal("0.0001"),
) -> bool:
    try:
        left = normalize_answer(predicted)
        right = normalize_answer(gold)
    except ValueError:
        return False
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    return abs(left - right) <= tolerance


def _condition_summary(
    correct: Sequence[bool],
    latencies_ms: Sequence[float],
) -> dict[str, int | float]:
    count = len(correct)
    exact = sum(correct)
    errors = count - exact
    return {
        "exact_match_count": exact,
        "exact_match_rate": exact / count,
        "arithmetic_error_count": errors,
        "arithmetic_error_rate": errors / count,
        **latency_summary(latencies_ms),
    }


def aggregate_numerical(
    records: Sequence[Mapping[str, bool | float]],
) -> dict[str, object]:
    if not records:
        raise ValueError("numerical records must not be empty")
    direct_correct = [bool(record["direct_correct"]) for record in records]
    tool_correct = [bool(record["tool_correct"]) for record in records]
    direct_latency = [float(record["direct_latency_ms"]) for record in records]
    tool_latency = [float(record["tool_latency_ms"]) for record in records]
    direct = _condition_summary(direct_correct, direct_latency)
    tool = _condition_summary(tool_correct, tool_latency)
    direct_error_rate = float(direct["arithmetic_error_rate"])
    tool_error_rate = float(tool["arithmetic_error_rate"])
    absolute_reduction = direct_error_rate - tool_error_rate
    relative_reduction = (
        None if direct_error_rate == 0 else absolute_reduction / direct_error_rate
    )
    return {
        "query_count": len(records),
        "conditions": {"direct_llm": direct, "python_tool": tool},
        "absolute_error_rate_reduction": absolute_reduction,
        "relative_error_reduction": relative_reduction,
    }
