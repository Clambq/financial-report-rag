from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from finqa_rag.agents.events import AgentEvent, encode_sse_event
from finqa_rag.agents.state import AgentPlan, AgentResult, FinSightState


def test_agent_plan_rejects_unknown_capability() -> None:
    with pytest.raises(ValidationError):
        AgentPlan(route="text", capabilities=["research", "browser"])


@pytest.mark.parametrize(
    "capabilities",
    [
        ["research", "report"],
        ["research", "quant", "report"],
        ["report"],
    ],
)
def test_agent_plan_accepts_supported_capabilities(capabilities: list[str]) -> None:
    plan = AgentPlan(route="mixed", capabilities=capabilities)

    assert plan.capabilities == capabilities


def test_finsight_state_starts_without_retries_or_evidence() -> None:
    state = FinSightState(
        request_id="req-1",
        report_id="report-1",
        question="What changed?",
    )

    assert state.retry_count == 0
    assert state.evidence == []


def test_agent_result_requires_answer_or_human_review_reason() -> None:
    with pytest.raises(ValidationError):
        AgentResult(request_id="req-1")

    with pytest.raises(ValidationError):
        AgentResult(request_id="req-1", needs_human_review=True)

    completed = AgentResult(request_id="req-1", final_answer="Revenue increased.")
    review = AgentResult(
        request_id="req-2",
        needs_human_review=True,
        human_review_reason="insufficient evidence",
    )

    assert completed.final_answer == "Revenue increased."
    assert review.human_review_reason == "insufficient evidence"


@pytest.mark.parametrize("event_type", ["run_started", "run_failed"])
def test_encode_sse_event_uses_one_compact_json_data_line(event_type: str) -> None:
    event = AgentEvent(
        event_type=event_type,
        request_id="req-7",
        sequence=3,
        timestamp=datetime(2026, 11, 7, 20, 0, tzinfo=timezone.utc),
        node="supervisor",
        payload={"message": "started"},
    )

    encoded = encode_sse_event(event)
    lines = encoded.splitlines()

    assert lines[0] == "id: 3"
    assert lines[1] == f"event: {event_type}"
    assert lines[2].startswith("data: ")
    assert len([line for line in lines if line.startswith("data: ")]) == 1
    assert json.loads(lines[2].removeprefix("data: ")) == {
        "event_type": event_type,
        "request_id": "req-7",
        "sequence": 3,
        "timestamp": "2026-11-07T20:00:00Z",
        "node": "supervisor",
        "payload": {"message": "started"},
    }
    assert encoded.endswith("\n\n")


def test_agent_event_rejects_negative_sequence() -> None:
    with pytest.raises(ValidationError):
        AgentEvent(
            event_type="run_started",
            request_id="req-1",
            sequence=-1,
            payload={},
        )
