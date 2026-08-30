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


def run_retrieval_evaluation(
    examples: Sequence[FinQAExample],
    methods: Sequence[str],
    output_dir: Path,
    top_k: int = 5,
) -> dict[str, object]:
    """Evaluate configured retrievers on identical ordered examples."""

    if not examples:
        raise ValueError("retrieval examples must not be empty")
    unsupported = sorted(set(methods) - {"bm25"})
    if unsupported:
        raise ValueError(f"unsupported retrieval methods: {', '.join(unsupported)}")
    if not methods:
        raise ValueError("at least one retrieval method is required")

    caches: dict[tuple[str, tuple[str, ...]], BM25Retriever] = {}
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
        if cache_key not in caches:
            retriever = BM25Retriever().fit(example.evidence)
            retriever.search("warmup", top_k=min(top_k, len(example.evidence)))
            caches[cache_key] = retriever
        retriever = caches[cache_key]

        for method in methods:
            started = perf_counter_ns()
            ranked = retriever.search(example.question, top_k=top_k)
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
        "methods": method_summaries,
    }
    manifest: dict[str, object] = {
        "finqa_revision": FINQA_REVISION,
        "split": examples[0].split,
        "query_count": len(examples),
        "methods": list(methods),
        "top_k": top_k,
        "python_version": platform.python_version(),
    }
    atomic_write_json(output_dir / "metrics_summary.json", summary)
    atomic_write_jsonl(output_dir / "predictions.jsonl", predictions)
    atomic_write_json(output_dir / "experiment_manifest.json", manifest)
    return summary
