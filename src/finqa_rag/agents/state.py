"""Pydantic state models shared by FinSight workflow nodes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Capability = Literal["research", "quant", "report"]
Route = Literal["text", "numerical", "mixed", "unsupported"]
NodeName = Literal["supervisor", "research", "quant", "verifier", "report"]


class AgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Route
    capabilities: list[Capability]
    tool_names: list[str] = Field(default_factory=list)


class AgentIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    node: NodeName | None = None
    recoverable: bool = False


class NodeTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: NodeName
    status: Literal["started", "completed", "failed"]
    latency_ms: float = Field(ge=0)


class FinSightState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    report_id: str
    question: str
    conversation_context: list[dict[str, str]] = Field(default_factory=list)
    plan: AgentPlan | None = None
    evidence: list[dict[str, object]] = Field(default_factory=list)
    calculations: list[dict[str, object]] = Field(default_factory=list)
    draft_answer: str | None = None
    final_answer: str | None = None
    citation_ids: list[str] = Field(default_factory=list)
    issues: list[AgentIssue] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    needs_human_review: bool = False
    human_review_reason: str | None = None
    traces: list[NodeTrace] = Field(default_factory=list)


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    final_answer: str | None = None
    citation_ids: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    human_review_reason: str | None = None
    issues: list[AgentIssue] = Field(default_factory=list)
    traces: list[NodeTrace] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_answer_or_review_reason(self) -> AgentResult:
        has_answer = bool(self.final_answer and self.final_answer.strip())
        has_review_reason = bool(
            self.needs_human_review
            and self.human_review_reason
            and self.human_review_reason.strip()
        )
        if not has_answer and not has_review_reason:
            raise ValueError(
                "result requires a final answer or a human-review reason"
            )
        return self
