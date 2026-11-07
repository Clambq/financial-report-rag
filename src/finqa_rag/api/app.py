"""FastAPI application factory with dependency-injected pipeline."""

from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from finqa_rag.api.agent_schemas import AgentAskRequest, AgentAskResponse
from finqa_rag.api.agent_service import AgentApiService
from finqa_rag.api.schemas import AskRequest, AskResponse


class AnswerPipeline(Protocol):
    def answer(self, request: AskRequest) -> AskResponse: ...

    def verified_metrics(self) -> dict[str, object]: ...


def create_app(
    pipeline: AnswerPipeline,
    *,
    agent_service: AgentApiService | None = None,
) -> FastAPI:
    app = FastAPI(title="Financial Report RAG", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        return pipeline.answer(request)

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        return pipeline.verified_metrics()

    if agent_service is not None:

        @app.post("/agent/ask", response_model=AgentAskResponse)
        async def agent_ask(request: AgentAskRequest) -> AgentAskResponse:
            return await agent_service.ask(request)

        @app.post("/agent/stream")
        async def agent_stream(
            request: AgentAskRequest,
            connection: Request,
        ) -> StreamingResponse:
            return StreamingResponse(
                agent_service.stream(request, connection.is_disconnected),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

    return app
