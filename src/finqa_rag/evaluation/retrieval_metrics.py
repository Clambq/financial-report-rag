"""Retrieval metrics with explicit gold-evidence semantics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import AbstractSet


@dataclass(frozen=True, slots=True)
class QueryRetrievalMetrics:
    recall_at_k: float
    hit_at_k: float
    mrr: float
    ndcg_at_k: float


def evaluate_ranking(
    ranked_ids: Sequence[str],
    gold_ids: AbstractSet[str],
    k: int = 5,
) -> QueryRetrievalMetrics:
    if not gold_ids:
        raise ValueError("gold evidence must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    top = list(ranked_ids[:k])
    hits = [1 if evidence_id in gold_ids else 0 for evidence_id in top]
    recall = sum(hits) / len(gold_ids)
    first_rank = next(
        (rank for rank, evidence_id in enumerate(ranked_ids, start=1) if evidence_id in gold_ids),
        None,
    )
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_dcg = sum(
        1.0 / math.log2(index + 2)
        for index in range(min(len(gold_ids), k))
    )
    return QueryRetrievalMetrics(
        recall_at_k=recall,
        hit_at_k=float(any(hits)),
        mrr=0.0 if first_rank is None else 1.0 / first_rank,
        ndcg_at_k=dcg / ideal_dcg,
    )


def aggregate_retrieval_metrics(
    records: Sequence[QueryRetrievalMetrics],
    *,
    top_k: int,
) -> dict[str, float]:
    if not records:
        raise ValueError("retrieval metric records must not be empty")
    return {
        f"recall_at_{top_k}": fmean(item.recall_at_k for item in records),
        f"hit_at_{top_k}": fmean(item.hit_at_k for item in records),
        "mrr": fmean(item.mrr for item in records),
        f"ndcg_at_{top_k}": fmean(item.ndcg_at_k for item in records),
    }
