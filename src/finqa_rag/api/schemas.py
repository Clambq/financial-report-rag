"""Validated request and response contracts for the RAG service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    question: str

    @field_validator("report_id", "question")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class Citation(BaseModel):
    evidence_id: str
    evidence_type: Literal["text", "table"]
    source_text: str
    retrieval_score: float
    rank: int


class AskResponse(BaseModel):
    question: str
    answer: str
    route: Literal["text", "numerical"]
    citations: list[Citation]
    needs_human_review: bool
    error: str | None
    latency_ms: float
    generated_program: str | None = None
