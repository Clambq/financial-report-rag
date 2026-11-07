"""HTTP contracts for the FinSight agent endpoints."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from finqa_rag.agents.state import AgentIssue, NodeTrace


def _not_blank(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be blank")
    return normalized


class AgentAskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    report_id: str
    question: str
    conversation_context: list[dict[str, str]] = Field(default_factory=list)

    _validate_request_id = field_validator("request_id")(_not_blank)
    _validate_report_id = field_validator("report_id")(_not_blank)
    _validate_question = field_validator("question")(_not_blank)


class AgentAskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    answer: str | None = None
    citation_ids: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    human_review_reason: str | None = None
    issues: list[AgentIssue] = Field(default_factory=list)
    traces: list[NodeTrace] = Field(default_factory=list)

