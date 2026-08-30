"""FastAPI application factory with dependency-injected pipeline."""

from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI

from finqa_rag.api.schemas import AskRequest, AskResponse


class AnswerPipeline(Protocol):
    def answer(self, request: AskRequest) -> AskResponse: ...

    def verified_metrics(self) -> dict[str, object]: ...


def create_app(pipeline: AnswerPipeline) -> FastAPI:
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

    return app
