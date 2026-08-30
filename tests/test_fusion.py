import numpy as np

from finqa_rag.data.models import EvidenceUnit
from finqa_rag.retrieval.dense import DenseRetriever
from finqa_rag.retrieval.fusion import reciprocal_rank_fusion
from finqa_rag.retrieval.reranker import CrossEncoderReranker
from finqa_rag.retrieval.types import ScoredEvidence


EVIDENCE = (
    EvidenceUnit("a", "report", "test", "text", "alpha evidence"),
    EvidenceUnit("b", "report", "test", "text", "beta evidence"),
    EvidenceUnit("c", "report", "test", "text", "gamma evidence"),
    EvidenceUnit("d", "report", "test", "text", "delta evidence"),
)


class FixedEncoder:
    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        vectors = {
            "alpha evidence": [1.0, 0.0],
            "beta evidence": [0.0, 1.0],
            "gamma evidence": [-1.0, 0.0],
            "delta evidence": [0.0, -1.0],
            "beta query": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=float)


class FixedCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = {"alpha evidence": 0.1, "beta evidence": 0.9}
        return np.asarray([scores[evidence] for _, evidence in pairs], dtype=float)


def scored(evidence_id: str, score: float, rank: int, method: str) -> ScoredEvidence:
    evidence = next(item for item in EVIDENCE if item.evidence_id == evidence_id)
    return ScoredEvidence(evidence, score, rank, method)


def test_dense_retrieval_ranks_highest_cosine_similarity_first() -> None:
    retriever = DenseRetriever(encoder=FixedEncoder()).fit(EVIDENCE)

    ranked = retriever.search("beta query", top_k=2)

    assert [item.evidence.evidence_id for item in ranked] == ["b", "a"]
    assert [item.rank for item in ranked] == [1, 2]
    assert all(item.method == "dense" for item in ranked)


def test_rrf_combines_rankings_and_breaks_ties_by_evidence_id() -> None:
    fused = reciprocal_rank_fusion(
        rankings={
            "bm25": [scored("a", 3.0, 1, "bm25"), scored("b", 2.0, 2, "bm25"), scored("c", 1.0, 3, "bm25")],
            "dense": [scored("b", 0.9, 1, "dense"), scored("a", 0.8, 2, "dense"), scored("d", 0.7, 3, "dense")],
        },
        weights={"bm25": 1.0, "dense": 1.0},
        rrf_k=60,
        top_k=4,
    )

    assert [item.evidence.evidence_id for item in fused] == ["a", "b", "c", "d"]
    assert all(item.method == "hybrid" for item in fused)


def test_cross_encoder_reranks_fused_candidates() -> None:
    candidates = [scored("a", 0.9, 1, "hybrid"), scored("b", 0.8, 2, "hybrid")]
    reranker = CrossEncoderReranker(model=FixedCrossEncoder())

    ranked = reranker.rerank("beta query", candidates, top_k=2)

    assert [item.evidence.evidence_id for item in ranked] == ["b", "a"]
    assert [item.rank for item in ranked] == [1, 2]
    assert all(item.method == "hybrid_rerank" for item in ranked)
