"""Immutable domain types for FinQA examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    evidence_id: str
    report_id: str
    split: str
    kind: Literal["text", "table"]
    text: str


@dataclass(frozen=True, slots=True)
class FinQAExample:
    example_id: str
    report_id: str
    split: str
    question: str
    answer: str
    exe_answer: str
    program: str
    table: tuple[tuple[str, ...], ...]
    evidence: tuple[EvidenceUnit, ...]
    gold_evidence_ids: tuple[str, ...]
