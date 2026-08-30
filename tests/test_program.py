from decimal import Decimal

import pytest

from finqa_rag.reasoning.program import ProgramError, execute_program
from finqa_rag.reasoning.router import route_question


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("subtract(120, 100), divide(#0, 100)", Decimal("0.2")),
        ("multiply(2.5, 4)", Decimal("10.0")),
        ("table_average(10, 20, 30)", Decimal("20")),
        ("exp(2, 3)", Decimal("8")),
        ("multiply(const_m1, const_1000000)", Decimal("-1000000")),
        ("divide(23.6%, const_100)", Decimal("0.00236")),
        ("greater(20, 10)", "yes"),
    ],
)
def test_execute_program(program: str, expected: Decimal | str) -> None:
    assert execute_program(program) == expected


def test_execute_table_program_against_original_finqa_rows() -> None:
    table = (
        ("", "first quarter", "second quarter", "third quarter", "fourth quarter"),
        ("2016", "22% ( 22 % )", "26% ( 26 % )", "26% ( 26 % )", "26% ( 26 % )"),
    )

    assert execute_program("table_average(2016, none)", table=table) == Decimal("0.25")


def test_table_program_matches_finqa_last_duplicate_row_semantics() -> None:
    table = (
        ("2014", "high", "low"),
        ("fourth quarter", "79.41", "57.96"),
        ("2013", "high", "low"),
        ("fourth quarter", "57.92", "45.68"),
    )

    assert execute_program("table_average(fourth quarter, none)", table=table) == Decimal(
        "51.80"
    )


def test_table_sum_allows_a_numeric_finqa_row_label() -> None:
    table = (
        ("for", "against", "abstained", "broker non-votes"),
        ("17695228", "963202", "155213", "0"),
    )

    result = execute_program(
        "table_sum(17695228, 963202), add(#0, 155213)",
        table=table,
    )

    assert result == Decimal("1273628")


@pytest.mark.parametrize(
    ("program", "message"),
    [
        ("__import__('os').system('whoami')", "unsupported operation"),
        ("divide(1, 0)", "division by zero"),
        ("add(#0, 1)", "forward reference"),
        ("unknown(1, 2)", "unsupported operation"),
        ("add(NaN, 1)", "malformed numeric token"),
    ],
)
def test_executor_rejects_unsafe_or_invalid_input(program: str, message: str) -> None:
    with pytest.raises(ProgramError, match=message):
        execute_program(program)


def test_route_question_uses_program_presence() -> None:
    assert route_question("divide(10, 2)") == "numerical"
    assert route_question("  ") == "text"
    assert route_question(None) == "text"
