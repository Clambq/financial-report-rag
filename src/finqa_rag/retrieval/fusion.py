"""Rank-level fusion independent of incompatible retriever scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from finqa_rag.retrieval.types import ScoredEvidence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ScoredEvidence]],
    weights: Mapping[str, float],
    rrf_k: int,
    top_k: int,
) -> list[ScoredEvidence]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scores: dict[str, float] = {}
    evidence_by_id = {}
    for method, ranking in rankings.items():
        if method not in weights:
            raise ValueError(f"missing RRF weight for {method}")
        for item in ranking:
            evidence_id = item.evidence.evidence_id
            evidence_by_id.setdefault(evidence_id, item.evidence)
            scores[evidence_id] = scores.get(evidence_id, 0.0) + (
                weights[method] / (rrf_k + item.rank)
            )
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
    return [
        ScoredEvidence(evidence_by_id[evidence_id], score, rank, "hybrid")
        for rank, (evidence_id, score) in enumerate(ordered, start=1)
    ]
