"""Cross-encoder reranking of a small fused candidate set."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from finqa_rag.retrieval.types import ScoredEvidence


class PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


def load_cross_encoder(model_name: str, device: str | None = None) -> PairScorer:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        model: PairScorer | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model or load_cross_encoder(model_name, device)

    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredEvidence],
        top_k: int,
    ) -> list[ScoredEvidence]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []
        scores = np.asarray(
            self._model.predict([(query, item.evidence.text) for item in candidates]),
            dtype=float,
        )
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0].evidence.evidence_id),
        )[:top_k]
        return [
            ScoredEvidence(item.evidence, float(score), rank, "hybrid_rerank")
            for rank, (item, score) in enumerate(ordered, start=1)
        ]
