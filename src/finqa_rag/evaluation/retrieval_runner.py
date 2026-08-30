"""Comparable report-local retrieval evaluation."""

from __future__ import annotations

import platform
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

from finqa_rag.common.atomic import atomic_write_json, atomic_write_jsonl
from finqa_rag.common.timing import latency_summary
from finqa_rag.data.download import FINQA_REVISION
from finqa_rag.data.models import FinQAExample
from finqa_rag.evaluation.retrieval_metrics import (
    QueryRetrievalMetrics,
    aggregate_retrieval_metrics,
    evaluate_ranking,
)
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


def run_retrieval_evaluation(
    examples: Sequence[FinQAExample],
    methods: Sequence[str],
    output_dir: Path,
    top_k: int = 5,
    candidate_k: int = 20,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    rrf_k: int = 60,
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str | None = None,
    dense_encoder: SentenceEncoder | None = None,
    cross_encoder: PairScorer | None = None,
) -> dict[str, object]:
    """Evaluate configured retrievers on identical ordered examples."""

    if not examples:
        raise ValueError("retrieval examples must not be empty")
    supported = {"bm25", "dense", "hybrid", "hybrid_rerank"}
    unsupported = sorted(set(methods) - supported)
    if unsupported:
        raise ValueError(f"unsupported retrieval methods: {', '.join(unsupported)}")
    if not methods:
        raise ValueError("at least one retrieval method is required")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")

    needs_dense = bool(set(methods) & {"dense", "hybrid", "hybrid_rerank"})
    needs_reranker = "hybrid_rerank" in methods
    shared_encoder = dense_encoder
    if needs_dense and shared_encoder is None:
        shared_encoder = load_sentence_encoder(dense_model_name, device)
    shared_cross_encoder = cross_encoder
    if needs_reranker and shared_cross_encoder is None:
        shared_cross_encoder = load_cross_encoder(reranker_model_name, device)
    reranker = (
        CrossEncoderReranker(
            model_name=reranker_model_name,
            device=device,
            model=shared_cross_encoder,
        )
        if needs_reranker
        else None
    )

    bm25_cache: dict[tuple[str, tuple[str, ...]], BM25Retriever] = {}
    dense_cache: dict[tuple[str, tuple[str, ...]], DenseRetriever] = {}
    metrics_by_method: dict[str, list[QueryRetrievalMetrics]] = {
        method: [] for method in methods
    }
    latencies_by_method: dict[str, list[float]] = {method: [] for method in methods}
    predictions: list[dict[str, object]] = []

    for example in examples:
        cache_key = (
            example.report_id,
            tuple(item.evidence_id + "\0" + item.text for item in example.evidence),
        )
        if cache_key not in bm25_cache:
            retriever = BM25Retriever().fit(example.evidence)
            retriever.search("warmup", top_k=min(top_k, len(example.evidence)))
            bm25_cache[cache_key] = retriever
        bm25_retriever = bm25_cache[cache_key]
        dense_retriever = None
        if needs_dense:
            if cache_key not in dense_cache:
                dense_retriever = DenseRetriever(
                    model_name=dense_model_name,
                    device=device,
                    encoder=shared_encoder,
                ).fit(example.evidence)
                dense_retriever.search(
                    "warmup",
                    top_k=min(top_k, len(example.evidence)),
                )
                dense_cache[cache_key] = dense_retriever
            dense_retriever = dense_cache[cache_key]

        for method in methods:
            started = perf_counter_ns()
            if method == "bm25":
                ranked = bm25_retriever.search(example.question, top_k=top_k)
            elif method == "dense":
                assert dense_retriever is not None
                ranked = dense_retriever.search(example.question, top_k=top_k)
            else:
                assert dense_retriever is not None
                bm25_candidates = bm25_retriever.search(
                    example.question,
                    top_k=min(candidate_k, len(example.evidence)),
                )
                dense_candidates = dense_retriever.search(
                    example.question,
                    top_k=min(candidate_k, len(example.evidence)),
                )
                fused = reciprocal_rank_fusion(
                    {"bm25": bm25_candidates, "dense": dense_candidates},
                    {"bm25": bm25_weight, "dense": dense_weight},
                    rrf_k=rrf_k,
                    top_k=(top_k if method == "hybrid" else candidate_k),
                )
                if method == "hybrid":
                    ranked = fused
                else:
                    assert reranker is not None
                    ranked = reranker.rerank(example.question, fused, top_k=top_k)
            latency_ms = (perf_counter_ns() - started) / 1_000_000
            ranked_ids = [item.evidence.evidence_id for item in ranked]
            query_metrics = evaluate_ranking(
                ranked_ids,
                set(example.gold_evidence_ids),
                k=top_k,
            )
            metrics_by_method[method].append(query_metrics)
            latencies_by_method[method].append(latency_ms)
            predictions.append(
                {
                    "example_id": example.example_id,
                    "report_id": example.report_id,
                    "question": example.question,
                    "method": method,
                    "gold_evidence_ids": list(example.gold_evidence_ids),
                    "ranked_evidence_ids": ranked_ids,
                    "scores": [item.score for item in ranked],
                    "latency_ms": latency_ms,
                    "metrics": asdict(query_metrics),
                }
            )

    method_summaries: dict[str, dict[str, float]] = {}
    for method in methods:
        method_summaries[method] = {
            **aggregate_retrieval_metrics(
                metrics_by_method[method],
                top_k=top_k,
            ),
            **latency_summary(latencies_by_method[method]),
        }
    summary: dict[str, object] = {
        "split": examples[0].split,
        "query_count": len(examples),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "methods": method_summaries,
    }
    manifest: dict[str, object] = {
        "finqa_revision": FINQA_REVISION,
        "split": examples[0].split,
        "query_count": len(examples),
        "methods": list(methods),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "rrf_k": rrf_k,
        "dense_model_name": dense_model_name if needs_dense else None,
        "reranker_model_name": reranker_model_name if needs_reranker else None,
        "device": device,
        "python_version": platform.python_version(),
    }
    atomic_write_json(output_dir / "metrics_summary.json", summary)
    atomic_write_jsonl(output_dir / "predictions.jsonl", predictions)
    atomic_write_json(output_dir / "experiment_manifest.json", manifest)
    return summary
