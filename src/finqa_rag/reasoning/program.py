"""A non-evaluating parser and Decimal executor for FinQA programs."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from decimal import Decimal, DecimalException, InvalidOperation, localcontext


ProgramValue = Decimal | str


class ProgramError(ValueError):
    """Raised when a generated program is unsafe, malformed, or not executable."""


_CALL = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)\s*")
_REFERENCE = re.compile(r"#([0-9]+)\Z")
_NUMBER = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?%?\Z"
)
_WHITESPACE = re.compile(r"\s+")
_BINARY_OPERATIONS = {"add", "subtract", "multiply", "divide", "exp", "greater"}
_TABLE_OPERATIONS = {"table_sum", "table_average", "table_max", "table_min"}
_SUPPORTED_OPERATIONS = _BINARY_OPERATIONS | _TABLE_OPERATIONS
_CONSTANTS = {
    "const_m2": Decimal("-2"),
    "const_m1": Decimal("-1"),
    "const_1": Decimal("1"),
    "const_2": Decimal("2"),
    "const_3": Decimal("3"),
    "const_4": Decimal("4"),
    "const_5": Decimal("5"),
    "const_6": Decimal("6"),
    "const_7": Decimal("7"),
    "const_8": Decimal("8"),
    "const_9": Decimal("9"),
    "const_10": Decimal("10"),
    "const_100": Decimal("100"),
    "const_1000": Decimal("1000"),
    "const_10000": Decimal("10000"),
    "const_100000": Decimal("100000"),
    "const_1000000": Decimal("1000000"),
    "const_1000000000": Decimal("1000000000"),
}


def _parse_calls(program: str) -> list[tuple[str, list[str]]]:
    calls: list[tuple[str, list[str]]] = []
    position = 0
    while position < len(program):
        match = _CALL.match(program, position)
        if match is None:
            name_match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", program[position:])
            if name_match and name_match.group(1) not in _SUPPORTED_OPERATIONS:
                raise ProgramError(f"unsupported operation: {name_match.group(1)}")
            raise ProgramError("malformed program")
        operation = match.group(1)
        if operation not in _SUPPORTED_OPERATIONS:
            raise ProgramError(f"unsupported operation: {operation}")
        arguments = [part.strip() for part in match.group(2).split(",")]
        if not arguments or any(not argument for argument in arguments):
            raise ProgramError("malformed program arguments")
        calls.append((operation, arguments))
        position = match.end()
        if position == len(program):
            break
        if program[position] != ",":
            raise ProgramError("malformed program separator")
        position += 1
    if not calls:
        raise ProgramError("program must not be empty")
    return calls


def _parse_numeric_token(token: str) -> Decimal:
    if token in _CONSTANTS:
        return _CONSTANTS[token]
    if token.startswith("const_"):
        raise ProgramError(f"unsupported constant: {token}")
    cleaned = token.replace(",", "").replace("$", "").strip()
    if not _NUMBER.fullmatch(cleaned):
        raise ProgramError(f"malformed numeric token: {token}")
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    try:
        value = Decimal(cleaned)
    except InvalidOperation as error:
        raise ProgramError(f"malformed numeric token: {token}") from error
    if not value.is_finite():
        raise ProgramError(f"non-finite numeric token: {token}")
    return value / Decimal("100") if is_percent else value


def _resolve_numeric(token: str, results: Sequence[ProgramValue], step: int) -> Decimal:
    reference = _REFERENCE.fullmatch(token)
    if reference:
        index = int(reference.group(1))
        if index >= step:
            raise ProgramError(f"forward reference: {token}")
        value = results[index]
        if not isinstance(value, Decimal):
            raise ProgramError(f"non-numeric reference: {token}")
        return value
    return _parse_numeric_token(token)


def _normalized_row_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return _WHITESPACE.sub(" ", text).strip().lower()


def _table_values(
    row_name: str,
    table: Sequence[Sequence[str]],
) -> list[Decimal]:
    wanted = _normalized_row_name(row_name)
    matches = [row for row in table if row and _normalized_row_name(row[0]) == wanted]
    if not matches:
        raise ProgramError(f"table row not found: {row_name}")
    values: list[Decimal] = []
    # FinQA's official evaluator builds a dict in table order, so a repeated
    # row label deterministically resolves to its final occurrence.
    for raw_cell in matches[-1][1:]:
        cell = raw_cell.replace("$", "").strip().split("(", 1)[0].strip()
        values.append(_parse_numeric_token(cell))
    if not values:
        raise ProgramError(f"table row has no numeric values: {row_name}")
    return values


def _execute_table_operation(operation: str, values: Sequence[Decimal]) -> Decimal:
    if operation == "table_sum":
        return sum(values, Decimal("0"))
    if operation == "table_average":
        return sum(values, Decimal("0")) / Decimal(len(values))
    if operation == "table_max":
        return max(values)
    return min(values)


def execute_program(
    program: str,
    *,
    table: Sequence[Sequence[str]] | None = None,
) -> ProgramValue:
    """Execute only the fixed FinQA grammar; never evaluate Python source."""

    results: list[ProgramValue] = []
    with localcontext() as context:
        context.prec = 40
        for step, (operation, arguments) in enumerate(_parse_calls(program.strip())):
            try:
                if operation in _TABLE_OPERATIONS:
                    if table is not None:
                        if len(arguments) != 2:
                            raise ProgramError(
                                f"malformed table operation arguments: {operation}"
                            )
                        values = _table_values(arguments[0], table)
                    else:
                        values = [
                            _resolve_numeric(argument, results, step)
                            for argument in arguments
                        ]
                    result: ProgramValue = _execute_table_operation(operation, values)
                else:
                    if len(arguments) != 2:
                        raise ProgramError(f"{operation} requires exactly two arguments")
                    left = _resolve_numeric(arguments[0], results, step)
                    right = _resolve_numeric(arguments[1], results, step)
                    if operation == "add":
                        result = left + right
                    elif operation == "subtract":
                        result = left - right
                    elif operation == "multiply":
                        result = left * right
                    elif operation == "divide":
                        if right == 0:
                            raise ProgramError("division by zero")
                        result = left / right
                    elif operation == "exp":
                        result = left**right
                    else:
                        result = "yes" if left > right else "no"
            except ProgramError:
                raise
            except (DecimalException, OverflowError, ValueError) as error:
                raise ProgramError(f"calculation failed at step {step}: {operation}") from error
            if isinstance(result, Decimal) and not result.is_finite():
                raise ProgramError(f"non-finite result at step {step}")
            results.append(result)
    return results[-1]
