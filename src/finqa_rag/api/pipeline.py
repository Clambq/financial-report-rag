"""Shared evaluated retrieval, routing, generation, and safe execution pipeline."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from threading import Lock
from time import perf_counter_ns
from typing import Literal, Protocol

from finqa_rag.api.schemas import AskRequest, AskResponse, Citation
from finqa_rag.data.models import FinQAExample
from finqa_rag.generation.qwen import GenerationResult
from finqa_rag.reasoning.program import ProgramError, execute_program
from finqa_rag.retrieval.bm25 import BM25Retriever
from finqa_rag.retrieval.dense import (
    DenseRetriever,
    SentenceEncoder,
    load_sentence_encoder,
)
from finqa_rag.retrieval.fusion import reciprocal_rank_fusion
from finqa_rag.retrieval.reranker import (
    CrossEncoderReranker,
    PairScorer,
    load_cross_encoder,
)
from finqa_rag.retrieval.types import ScoredEvidence


_NUMERICAL_CUES = re.compile(
    r"\b(calculate|compute|how much|how many|percentage|percent|ratio|average|"
    r"difference|growth rate|change in|increase in|decrease in|sum|total of|maximum|minimum)\b",
    re.IGNORECASE,
)


def route_live_question(question: str) -> Literal["text", "numerical"]:
    return "numerical" if _NUMERICAL_CUES.search(question) else "text"


class EvidenceRetriever(Protocol):
    def search(self, report_id: str, question: str) -> list[ScoredEvidence]: ...


class ServiceGenerator(Protocol):
    def answer_text(
        self,
        question: str,
        evidence: Sequence[ScoredEvidence],
    ) -> GenerationResult: ...

    def generate_program(
        self,
        question: str,
        evidence: Sequence[ScoredEvidence],
    ) -> GenerationResult: ...


class ReportRetrievalIndex:
    """Lazy report-local index using the locked retrieval configuration."""

    def __init__(
        self,
        examples: Sequence[FinQAExample],
        *,
        candidate_k: int = 10,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        dense_encoder: SentenceEncoder | None = None,
        cross_encoder: PairScorer | None = None,
    ) -> None:
        self._evidence = {}
        for example in examples:
            existing = self._evidence.setdefault(example.report_id, example.evidence)
            if existing != example.evidence:
                raise ValueError(f"inconsistent evidence for report {example.report_id}")
        if not self._evidence:
            raise ValueError("retrieval index requires at least one report")
        self._candidate_k = candidate_k
        self._top_k = top_k
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight
        self._rrf_k = rrf_k
        self._dense_model_name = dense_model_name
        self._encoder = dense_encoder or load_sentence_encoder(dense_model_name, device)
        rerank_model = cross_encoder or load_cross_encoder(reranker_model_name, device)
        self._reranker = CrossEncoderReranker(
            model_name=reranker_model_name,
            device=device,
            model=rerank_model,
        )
        self._bm25: dict[str, BM25Retriever] = {}
        self._dense: dict[str, DenseRetriever] = {}
        self._lock = Lock()

    def _get_retrievers(self, report_id: str) -> tuple[BM25Retriever, DenseRetriever]:
        if report_id not in self._evidence:
            raise KeyError(report_id)
        with self._lock:
            if report_id not in self._bm25:
                evidence = self._evidence[report_id]
                self._bm25[report_id] = BM25Retriever().fit(evidence)
                self._dense[report_id] = DenseRetriever(
                    model_name=self._dense_model_name,
                    encoder=self._encoder,
                ).fit(evidence)
        return self._bm25[report_id], self._dense[report_id]

    def search(self, report_id: str, question: str) -> list[ScoredEvidence]:
        bm25, dense = self._get_retrievers(report_id)
        evidence_count = len(self._evidence[report_id])
        candidate_k = min(self._candidate_k, evidence_count)
        bm25_ranked = bm25.search(question, top_k=candidate_k)
        dense_ranked = dense.search(question, top_k=candidate_k)
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_ranked, "dense": dense_ranked},
            {"bm25": self._bm25_weight, "dense": self._dense_weight},
            rrf_k=self._rrf_k,
            top_k=candidate_k,
        )
        return self._reranker.rerank(
            question,
            fused,
            top_k=min(self._top_k, evidence_count),
        )


class RAGPipeline:
    def __init__(
        self,
        *,
        report_tables: Mapping[str, Sequence[Sequence[str]]],
        retriever: EvidenceRetriever,
        generator: ServiceGenerator,
        metrics: Mapping[str, object],
    ) -> None:
        self._tables = {
            key: tuple(tuple(cell for cell in row) for row in table)
            for key, table in report_tables.items()
        }
        self._retriever = retriever
        self._generator = generator
        self._metrics = dict(metrics)

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[FinQAExample],
        *,
        retriever: EvidenceRetriever,
        generator: ServiceGenerator,
        metrics: Mapping[str, object],
    ) -> RAGPipeline:
        tables: dict[str, Sequence[Sequence[str]]] = {}
        for example in examples:
            existing = tables.setdefault(example.report_id, example.table)
            if existing != example.table:
                raise ValueError(f"inconsistent table for report {example.report_id}")
        return cls(
            report_tables=tables,
            retriever=retriever,
            generator=generator,
            metrics=metrics,
        )

    def answer(self, request: AskRequest) -> AskResponse:
        started = perf_counter_ns()
        route = route_live_question(request.question)
        if request.report_id not in self._tables:
            return AskResponse(
                question=request.question,
                answer="",
                route=route,
                citations=[],
                needs_human_review=True,
                error=f"unknown report_id: {request.report_id}",
                latency_ms=(perf_counter_ns() - started) / 1_000_000,
            )
        try:
            ranked = self._retriever.search(request.report_id, request.question)
        except (KeyError, ValueError) as error:
            return AskResponse(
                question=request.question,
                answer="",
                route=route,
                citations=[],
                needs_human_review=True,
                error=str(error),
                latency_ms=(perf_counter_ns() - started) / 1_000_000,
            )
        citations = [
            Citation(
                evidence_id=item.evidence.evidence_id,
                evidence_type=item.evidence.kind,
                source_text=item.evidence.text,
                retrieval_score=item.score,
                rank=item.rank,
            )
            for item in ranked
        ]

        if route == "text":
            generated = self._generator.answer_text(request.question, ranked)
            answer = generated.raw_text
            generated_program = None
            needs_review = not bool(answer.strip())
            error = "model returned an empty answer" if needs_review else None
        else:
            generated = self._generator.generate_program(request.question, ranked)
            generated_program = generated.parsed_text
            try:
                value = execute_program(
                    generated.parsed_text,
                    table=self._tables[request.report_id],
                )
                answer = format(value, "f") if isinstance(value, Decimal) else value
                needs_review = False
                error = None
            except ProgramError as program_error:
                answer = ""
                needs_review = True
                error = str(program_error)

        return AskResponse(
            question=request.question,
            answer=answer,
            route=route,
            citations=citations,
            needs_human_review=needs_review,
            error=error,
            latency_ms=(perf_counter_ns() - started) / 1_000_000,
            generated_program=generated_program,
        )

    def verified_metrics(self) -> dict[str, object]:
        return dict(self._metrics)
