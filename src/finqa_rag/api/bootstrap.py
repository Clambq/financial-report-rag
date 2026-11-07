"""Construct the production FastAPI app from verified project components."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from finqa_rag.api.app import create_app
from finqa_rag.api.agent_service import AgentApiService
from finqa_rag.api.schemas import AskRequest, AskResponse
from finqa_rag.agents.graph import FinSightAgent
from finqa_rag.agents.model import ModelMessage, VLLMOpenAIAdapter
from finqa_rag.agents.nodes import AgentRuntime
from finqa_rag.agents.tools import (
    ComposeGroundedReportInput,
    ComposeGroundedReportOutput,
    ExecuteFinancialProgramInput,
    ExecuteFinancialProgramOutput,
    FinancialProgramTool,
    GroundingVerifierTool,
    QueryReportTableInput,
    QueryReportTableOutput,
    ReportTableTool,
    RetrievalTool,
    RetrieveEvidenceInput,
    RetrieveEvidenceOutput,
    ToolRegistry,
    VerifyGroundingInput,
    VerifyGroundingOutput,
)
from finqa_rag.api.pipeline import RAGPipeline, ReportRetrievalIndex
from finqa_rag.data.finqa import load_split
from finqa_rag.generation.qwen import QwenGenerator


@dataclass(frozen=True, slots=True)
class FinSightSettings:
    vllm_base_url: str
    vllm_model: str
    vllm_api_key: str = "EMPTY"
    data_dir: Path = Path("data/raw/finqa")
    split: str = "test"
    device: str = "cpu"

    @classmethod
    def from_env(
        cls,
        values: Mapping[str, str] | None = None,
    ) -> FinSightSettings:
        env = os.environ if values is None else values
        base_url = env.get("FINSIGHT_VLLM_BASE_URL", "").strip()
        if not base_url:
            raise ValueError("FINSIGHT_VLLM_BASE_URL is required")
        model = env.get("FINSIGHT_VLLM_MODEL", "").strip()
        if not model:
            raise ValueError("FINSIGHT_VLLM_MODEL is required")
        return cls(
            vllm_base_url=base_url,
            vllm_model=model,
            vllm_api_key=env.get("FINSIGHT_VLLM_API_KEY", "EMPTY"),
            data_dir=Path(env.get("FINSIGHT_DATA_DIR", "data/raw/finqa")),
            split=env.get("FINSIGHT_SPLIT", "test"),
            device=env.get("FINSIGHT_DEVICE", "cpu"),
        )


class _ComposeWithModel:
    def __init__(self, model: VLLMOpenAIAdapter) -> None:
        self._model = model

    def __call__(
        self,
        request: ComposeGroundedReportInput,
    ) -> ComposeGroundedReportOutput:
        return self._model.generate(
            ComposeGroundedReportOutput,
            [
                ModelMessage(
                    role="system",
                    content=(
                        "Write a concise grounded financial answer. Use only the "
                        "provided evidence IDs and return the allowed_evidence_ids "
                        "field unchanged."
                    ),
                ),
                ModelMessage(
                    role="user",
                    content=request.model_dump_json(),
                ),
            ],
        )


class _AgentOnlyPipeline:
    def __init__(self, metrics: Mapping[str, object]) -> None:
        self._metrics = dict(metrics)

    def answer(self, request: AskRequest) -> AskResponse:
        return AskResponse(
            question=request.question,
            answer="",
            route="text",
            citations=[],
            needs_human_review=True,
            error="Use /agent/ask for the FinSight multi-agent workflow.",
            latency_ms=0,
        )

    def verified_metrics(self) -> dict[str, object]:
        return dict(self._metrics)


def _read_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_service_app(
    data_dir: Path = Path("data/raw/finqa"),
    split: str = "test",
    device: str = "cuda",
    agent_service: AgentApiService | None = None,
) -> FastAPI:
    examples = load_split(data_dir / f"{split}.json", split)
    retriever = ReportRetrievalIndex(examples, device=device)
    generator = QwenGenerator()
    metrics = {
        "retrieval": _read_metrics(Path("artifacts/retrieval/metrics_summary.json")),
        "numerical": _read_metrics(Path("artifacts/numerical/metrics_summary.json")),
    }
    pipeline = RAGPipeline.from_examples(
        examples,
        retriever=retriever,
        generator=generator,
        metrics=metrics,
    )
    return create_app(pipeline, agent_service=agent_service)


def build_finsight_app(
    settings: FinSightSettings | None = None,
) -> FastAPI:
    """Build the vLLM-backed FinSight API from explicit environment settings."""

    selected = settings or FinSightSettings.from_env()
    if selected.split not in {"train", "dev", "test"}:
        raise ValueError("FINSIGHT_SPLIT must be train, dev, or test")
    examples = load_split(
        selected.data_dir / f"{selected.split}.json",
        selected.split,
    )
    retriever = ReportRetrievalIndex(examples, device=selected.device)
    tables: dict[str, tuple[tuple[str, ...], ...]] = {}
    for example in examples:
        tables.setdefault(example.report_id, example.table)

    model = VLLMOpenAIAdapter(
        base_url=selected.vllm_base_url,
        api_key=selected.vllm_api_key,
        model=selected.vllm_model,
    )
    registry = ToolRegistry()
    registry.register(
        "retrieve_evidence",
        RetrieveEvidenceInput,
        RetrieveEvidenceOutput,
        RetrievalTool(retriever),
    )
    registry.register(
        "query_report_table",
        QueryReportTableInput,
        QueryReportTableOutput,
        ReportTableTool(tables),
    )
    registry.register(
        "execute_financial_program",
        ExecuteFinancialProgramInput,
        ExecuteFinancialProgramOutput,
        FinancialProgramTool(tables),
    )
    registry.register(
        "compose_grounded_report",
        ComposeGroundedReportInput,
        ComposeGroundedReportOutput,
        _ComposeWithModel(model),
    )
    registry.register(
        "verify_grounding",
        VerifyGroundingInput,
        VerifyGroundingOutput,
        GroundingVerifierTool(),
    )
    agent = FinSightAgent(AgentRuntime(model=model, registry=registry))
    metrics = {
        "runtime": "vllm_openai_compatible",
        "model": selected.vllm_model,
    }
    return create_app(
        _AgentOnlyPipeline(metrics),
        agent_service=AgentApiService(agent),
    )
