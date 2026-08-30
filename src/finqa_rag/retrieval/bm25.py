"""Deterministic BM25 retrieval over report-local evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from finqa_rag.data.models import EvidenceUnit
from finqa_rag.retrieval.types import ScoredEvidence


_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?%?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self) -> None:
        self._evidence: tuple[EvidenceUnit, ...] = ()
        self._index: BM25Okapi | None = None

    def fit(self, evidence: Sequence[EvidenceUnit]) -> BM25Retriever:
        if not evidence:
            raise ValueError("BM25 evidence must not be empty")
        self._evidence = tuple(evidence)
        self._index = BM25Okapi([tokenize(item.text) for item in self._evidence])
        return self

    def search(self, query: str, top_k: int) -> list[ScoredEvidence]:
        if self._index is None:
            raise RuntimeError("BM25Retriever.fit must be called before search")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self._index.get_scores(tokenize(query))
        ordered = sorted(
            zip(self._evidence, scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0].evidence_id),
        )[:top_k]
        return [
            ScoredEvidence(
                evidence=item,
                score=float(score),
                rank=rank,
                method="bm25",
            )
            for rank, (item, score) in enumerate(ordered, start=1)
        ]
