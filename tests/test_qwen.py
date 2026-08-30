import pytest

from finqa_rag.generation.qwen import parse_program_response


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("divide(39, 4).", "divide(39, 4)"),
        (
            "The FinQA program is:\n```text\nsubtract(120, 100), divide(#0, 100)\n```",
            "subtract(120, 100), divide(#0, 100)",
        ),
        (
            "Use table_average(fourth quarter, none) to calculate it.",
            "table_average(fourth quarter, none)",
        ),
    ],
)
def test_parse_program_response_extracts_only_the_safe_grammar(
    raw: str,
    expected: str,
) -> None:
    assert parse_program_response(raw) == expected


def test_parse_program_response_preserves_invalid_text_for_safe_rejection() -> None:
    assert parse_program_response("I cannot calculate this.") == "I cannot calculate this."
