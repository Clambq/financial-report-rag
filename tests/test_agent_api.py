from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from finqa_rag.agents.events import AgentEvent
from finqa_rag.agents.state import AgentIssue, AgentResult, NodeTrace
from finqa_rag.api.agent_schemas import AgentAskRequest
from finqa_rag.api.agent_service import AgentApiService
from finqa_rag.api.app import create_app
from finqa_rag.api.schemas import AskRequest, AskResponse


class LegacyPipeline:
    def answer(self, request: AskRequest) -> AskResponse:
        return AskResponse(
            question=request.question,
            answer="legacy",
            route="text",
            citations=[],
            needs_human_review=False,
            error=None,
            latency_ms=0,
        )

    def verified_metrics(self) -> dict[str, object]:
        return {"ok": True}


class FakeAgent:
    def __init__(self, result: AgentResult) -> None:
        self.result = result

    async def ainvoke(self, request):  # type: ignore[no-untyped-def]
        assert request.report_id == "report-a"
        return self.result

    async def astream(self, request) -> AsyncIterator[AgentEvent]:  # type: ignore[no-untyped-def]
        yield AgentEvent(
            event_type="run_started",
            request_id=request.request_id,
            sequence=50,
        )
        yield AgentEvent(
            event_type="node_finished",
            request_id=request.request_id,
            sequence=50,
            node="supervisor",
        )
        yield AgentEvent(
            event_type="run_finished",
            request_id=request.request_id,
            sequence=2,
        )


def result(*, review: bool = False) -> AgentResult:
    return AgentResult(
        request_id="req-1",
        final_answer=None if review else "Revenue increased [ev-1].",
        citation_ids=[] if review else ["ev-1"],
        needs_human_review=review,
        human_review_reason="verification failed" if review else None,
        issues=(
            [
                AgentIssue(
                    code="verification_failed",
                    message="citation mismatch",
                    node="verifier",
                )
            ]
            if review
            else []
        ),
        traces=[NodeTrace(node="supervisor", status="completed", latency_ms=1.25)],
    )


def client(*, review: bool = False) -> TestClient:
    service = AgentApiService(FakeAgent(result(review=review)))
    return TestClient(create_app(LegacyPipeline(), agent_service=service))


def test_agent_request_rejects_blank_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AgentAskRequest(report_id=" ", question="What changed?")
    with pytest.raises(ValidationError):
        AgentAskRequest(report_id="report-a", question=" ")
    with pytest.raises(ValidationError):
        AgentAskRequest.model_validate(
            {"report_id": "report-a", "question": "What changed?", "extra": 1}
        )


def test_agent_json_endpoint_serializes_traces_and_human_review() -> None:
    success = client().post(
        "/agent/ask",
        json={
            "request_id": "req-1",
            "report_id": "report-a",
            "question": "What changed?",
        },
    )
    review = client(review=True).post(
        "/agent/ask",
        json={
            "request_id": "req-1",
            "report_id": "report-a",
            "question": "What changed?",
        },
    )

    assert success.status_code == 200
    assert success.json()["traces"][0]["node"] == "supervisor"
    assert success.json()["answer"] == "Revenue increased [ev-1]."
    assert review.status_code == 200
    assert review.json()["needs_human_review"] is True
    assert review.json()["human_review_reason"] == "verification failed"


def test_agent_routes_do_not_break_existing_endpoints() -> None:
    api = client()

    assert api.get("/health").json() == {"status": "ok"}
    assert api.get("/metrics").json() == {"ok": True}
    assert api.post(
        "/ask",
        json={"report_id": "report-a", "question": "What changed?"},
    ).json()["answer"] == "legacy"


def test_sse_resequences_events_and_has_one_terminal_event() -> None:
    response = client().post(
        "/agent/stream",
        json={
            "request_id": "req-1",
            "report_id": "report-a",
            "question": "What changed?",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    ids = [int(line[4:]) for line in response.text.splitlines() if line.startswith("id: ")]
    event_names = [
        line[7:] for line in response.text.splitlines() if line.startswith("event: ")
    ]
    assert ids == list(range(len(ids)))
    assert event_names[0] == "run_started"
    assert event_names[-1] == "run_finished"
    assert sum(name in {"run_finished", "run_failed"} for name in event_names) == 1


def test_disconnect_closes_agent_iterator_without_more_events() -> None:
    closed = False

    class StreamingAgent(FakeAgent):
        async def astream(self, request) -> AsyncIterator[AgentEvent]:  # type: ignore[no-untyped-def]
            nonlocal closed
            try:
                yield AgentEvent(
                    event_type="run_started",
                    request_id=request.request_id,
                    sequence=0,
                )
                yield AgentEvent(
                    event_type="tool_started",
                    request_id=request.request_id,
                    sequence=1,
                )
            finally:
                closed = True

    checks = iter([False, True])

    async def disconnected() -> bool:
        return next(checks)

    service = AgentApiService(StreamingAgent(result()))

    async def collect() -> list[str]:
        request = AgentAskRequest(
            request_id="req-1",
            report_id="report-a",
            question="What changed?",
        )
        return [chunk async for chunk in service.stream(request, disconnected)]

    chunks = asyncio.run(collect())

    assert closed is True
    assert len(chunks) == 1
    assert "event: run_started" in chunks[0]
    assert "tool_started" not in "".join(chunks)


def test_stream_heartbeat_is_comment_and_exception_is_sanitized() -> None:
    class BrokenAgent(FakeAgent):
        async def astream(self, request) -> AsyncIterator[AgentEvent]:  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.02)
            yield AgentEvent(
                event_type="run_started",
                request_id=request.request_id,
                sequence=0,
            )
            raise RuntimeError("secret stack detail")

    service = AgentApiService(BrokenAgent(result()), heartbeat_seconds=0.005)

    async def connected() -> bool:
        return False

    async def collect() -> str:
        request = AgentAskRequest(
            request_id="req-1",
            report_id="report-a",
            question="What changed?",
        )
        return "".join([chunk async for chunk in service.stream(request, connected)])

    payload = asyncio.run(collect())

    assert ": heartbeat\n\n" in payload
    assert "event: run_failed" in payload
    assert "internal agent error" in payload
    assert "secret stack detail" not in payload
    assert "Traceback" not in payload

