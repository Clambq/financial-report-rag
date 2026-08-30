"""Shared retrieval result types."""

from __future__ import annotations

from dataclasses import dataclass

from finqa_rag.data.models import EvidenceUnit


@dataclass(frozen=True, slots=True)
class ScoredEvidence:
    evidence: EvidenceUnit
    score: float
    rank: int
    method: str
