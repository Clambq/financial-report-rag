"""Metric contracts for end-to-end FinSight agent evaluation."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from finqa_rag.agents.tools import ToolCallRecord
from finqa_rag.common.timing import latency_summary


class AgentPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    completed: bool
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    citation_valid: bool | None = None
    numerical_verified: bool | None = None
    error_category: str | None = None
    latency_ms: float = Field(ge=0)
    model_call_count: int = Field(default=0, ge=0)


def _ratio(count: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def compute_agent_metrics(
    predictions: list[AgentPrediction],
) -> dict[str, object]:
    if not predictions:
        raise ValueError("agent predictions must not be empty")
    if len({item.case_id for item in predictions}) != len(predictions):
        raise ValueError("agent prediction case IDs must be unique")

    calls = [call for item in predictions for call in item.tool_calls]
    first = [call for call in calls if call.attempt == 1]
    retries = [call for call in calls if call.attempt > 1]
    citations = [
        item.citation_valid
        for item in predictions
        if item.citation_valid is not None
    ]
    numerical = [
        item.numerical_verified
        for item in predictions
        if item.numerical_verified is not None
    ]
    errors = Counter(
        item.error_category for item in predictions if item.error_category is not None
    )
    return {
        "query_count": len(predictions),
        "task_completion": _ratio(
            sum(item.completed for item in predictions), len(predictions)
        ),
        "tool_success": _ratio(sum(call.success for call in calls), len(calls)),
        "first_attempt_success": _ratio(
            sum(call.success for call in first), len(first)
        ),
        "after_retry_success": _ratio(
            sum(call.success for call in retries), len(retries)
        ),
        "citation_validity": _ratio(sum(citations), len(citations)),
        "numerical_verification": _ratio(sum(numerical), len(numerical)),
        "error_category_counts": dict(sorted(errors.items())),
        "model_call_count": sum(item.model_call_count for item in predictions),
        "tool_call_count": len(calls),
        "latency": latency_summary([item.latency_ms for item in predictions]),
    }

