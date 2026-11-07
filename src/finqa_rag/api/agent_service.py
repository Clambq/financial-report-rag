"""Application service shared by FinSight JSON and SSE endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Protocol

from finqa_rag.agents.events import AgentEvent, encode_sse_event
from finqa_rag.agents.state import AgentResult, FinSightState
from finqa_rag.api.agent_schemas import AgentAskRequest, AgentAskResponse


class AgentProtocol(Protocol):
    async def ainvoke(self, request: FinSightState) -> AgentResult: ...

    def astream(self, request: FinSightState) -> AsyncIterator[AgentEvent]: ...


DisconnectCheck = Callable[[], Awaitable[bool]]


class AgentApiService:
    def __init__(
        self,
        agent: AgentProtocol,
        *,
        heartbeat_seconds: float = 15.0,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self._agent = agent
        self._heartbeat_seconds = heartbeat_seconds

    @staticmethod
    def _state(request: AgentAskRequest) -> FinSightState:
        return FinSightState(
            request_id=request.request_id,
            report_id=request.report_id,
            question=request.question,
            conversation_context=request.conversation_context,
        )

    async def ask(self, request: AgentAskRequest) -> AgentAskResponse:
        result = await self._agent.ainvoke(self._state(request))
        return AgentAskResponse(
            request_id=result.request_id,
            answer=result.final_answer,
            citation_ids=result.citation_ids,
            needs_human_review=result.needs_human_review,
            human_review_reason=result.human_review_reason,
            issues=result.issues,
            traces=result.traces,
        )

    async def stream(
        self,
        request: AgentAskRequest,
        is_disconnected: DisconnectCheck,
    ) -> AsyncIterator[str]:
        iterator = self._agent.astream(self._state(request)).__aiter__()
        next_task: asyncio.Task[AgentEvent] | None = None
        sequence = 0
        emitted_event = False
        terminal = False
        try:
            while not terminal:
                if await is_disconnected():
                    return
                if next_task is None:
                    next_task = asyncio.create_task(iterator.__anext__())
                done, _ = await asyncio.wait(
                    {next_task},
                    timeout=self._heartbeat_seconds,
                )
                if not done:
                    yield ": heartbeat\n\n"
                    continue
                try:
                    event = next_task.result()
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if not emitted_event:
                        yield encode_sse_event(
                            AgentEvent(
                                event_type="run_started",
                                request_id=request.request_id,
                                sequence=sequence,
                            )
                        )
                        sequence += 1
                    yield encode_sse_event(
                        AgentEvent(
                            event_type="run_failed",
                            request_id=request.request_id,
                            sequence=sequence,
                            payload={"error": "internal agent error"},
                        )
                    )
                    terminal = True
                    break
                finally:
                    next_task = None

                if not emitted_event and event.event_type != "run_started":
                    yield encode_sse_event(
                        AgentEvent(
                            event_type="run_started",
                            request_id=request.request_id,
                            sequence=sequence,
                        )
                    )
                    sequence += 1
                    emitted_event = True
                normalized = event.model_copy(update={"sequence": sequence})
                yield encode_sse_event(normalized)
                emitted_event = True
                sequence += 1
                terminal = event.event_type in {"run_finished", "run_failed"}

            if not terminal:
                if not emitted_event:
                    yield encode_sse_event(
                        AgentEvent(
                            event_type="run_started",
                            request_id=request.request_id,
                            sequence=sequence,
                        )
                    )
                    sequence += 1
                yield encode_sse_event(
                    AgentEvent(
                        event_type="run_failed",
                        request_id=request.request_id,
                        sequence=sequence,
                        payload={"error": "agent stream ended without terminal event"},
                    )
                )
        finally:
            if next_task is not None and not next_task.done():
                next_task.cancel()
                with suppress(asyncio.CancelledError):
                    await next_task
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

