"""Dense retrieval using normalized sentence embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from finqa_rag.data.models import EvidenceUnit
from finqa_rag.retrieval.types import ScoredEvidence


class SentenceEncoder(Protocol):
    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray: ...


def load_sentence_encoder(model_name: str, device: str | None = None) -> SentenceEncoder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class DenseRetriever:
    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        encoder: SentenceEncoder | None = None,
    ) -> None:
        self.model_name = model_name
        self._encoder = encoder or load_sentence_encoder(model_name, device)
        self._evidence: tuple[EvidenceUnit, ...] = ()
        self._embeddings: np.ndarray | None = None

    def fit(self, evidence: Sequence[EvidenceUnit]) -> DenseRetriever:
        if not evidence:
            raise ValueError("dense evidence must not be empty")
        self._evidence = tuple(evidence)
        self._embeddings = np.asarray(
            self._encoder.encode(
                [item.text for item in self._evidence],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=float,
        )
        return self

    def search(self, query: str, top_k: int) -> list[ScoredEvidence]:
        if self._embeddings is None:
            raise RuntimeError("DenseRetriever.fit must be called before search")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_embedding = np.asarray(
            self._encoder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=float,
        )[0]
        scores = self._embeddings @ query_embedding
        ordered = sorted(
            zip(self._evidence, scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0].evidence_id),
        )[:top_k]
        return [
            ScoredEvidence(item, float(score), rank, "dense")
            for rank, (item, score) in enumerate(ordered, start=1)
        ]
