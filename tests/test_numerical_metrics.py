from decimal import Decimal

import pytest

from finqa_rag.evaluation.numerical_metrics import (
    aggregate_numerical,
    answers_match,
    normalize_answer,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,234.50", Decimal("1234.50")),
        ("(42.5)", Decimal("-42.5")),
        ("20%", Decimal("20")),
        ("1e-05", Decimal("0.00001")),
        ("YES", "yes"),
    ],
)
def test_normalize_answer_handles_financial_formats(raw: str, expected) -> None:
    assert normalize_answer(raw) == expected


def test_answers_match_uses_explicit_tolerance() -> None:
    assert answers_match("10.00009", "10")
    assert not answers_match("10.001", "10")
    assert answers_match("Yes", "yes")
    assert not answers_match("yes", "no")


def test_aggregate_numerical_reports_error_reduction_and_latency() -> None:
    records = [
        {
            "direct_correct": True,
            "tool_correct": True,
            "direct_latency_ms": 10.0,
            "tool_latency_ms": 1.0,
        },
        {
            "direct_correct": False,
            "tool_correct": True,
            "direct_latency_ms": 20.0,
            "tool_latency_ms": 2.0,
        },
        {
            "direct_correct": False,
            "tool_correct": False,
            "direct_latency_ms": 30.0,
            "tool_latency_ms": 3.0,
        },
    ]

    summary = aggregate_numerical(records)

    assert summary["query_count"] == 3
    assert summary["conditions"]["direct_llm"]["exact_match_count"] == 1
    assert summary["conditions"]["python_tool"]["arithmetic_error_count"] == 1
    assert summary["absolute_error_rate_reduction"] == pytest.approx(1 / 3)
    assert summary["relative_error_reduction"] == pytest.approx(0.5)
    assert summary["conditions"]["direct_llm"]["p95_ms"] == 29.0


def test_relative_error_reduction_is_null_when_direct_has_no_errors() -> None:
    summary = aggregate_numerical(
        [
            {
                "direct_correct": True,
                "tool_correct": True,
                "direct_latency_ms": 1.0,
                "tool_latency_ms": 1.0,
            }
        ]
    )

    assert summary["relative_error_reduction"] is None
