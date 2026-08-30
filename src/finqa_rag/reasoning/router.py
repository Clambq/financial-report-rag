"""Route FinQA examples according to the presence of a symbolic program."""

from __future__ import annotations

from typing import Literal


def route_question(program: str | None) -> Literal["text", "numerical"]:
    return "numerical" if program and program.strip() else "text"
