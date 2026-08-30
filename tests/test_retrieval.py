import json
from pathlib import Path

from finqa_rag.data.finqa import load_split
from finqa_rag.data.models import EvidenceUnit
from finqa_rag.evaluation.retrieval_runner import run_retrieval_evaluation
from finqa_rag.retrieval.bm25 import BM25Retriever


FIXTURE = Path(__file__).parent / "fixtures" / "finqa_mini.json"


def test_bm25_ranks_exact_financial_terms_first() -> None:
    evidence = (
        EvidenceUnit("text_0", "report", "test", "text", "The company grew."),
        EvidenceUnit(
            "table_1",
            "report",
            "test",
            "table",
            "Metric: Operating income | 2024: 30",
        ),
        EvidenceUnit(
            "table_2",
            "report",
            "test",
            "table",
            "Metric: Revenue | 2024: 120",
        ),
    )
    retriever = BM25Retriever().fit(evidence)

    ranked = retriever.search("operating income 2024", top_k=2)

    assert ranked[0].evidence.evidence_id == "table_1"
    assert [item.rank for item in ranked] == [1, 2]
    assert all(item.method == "bm25" for item in ranked)


def test_bm25_breaks_equal_scores_by_evidence_id() -> None:
    evidence = (
        EvidenceUnit("text_b", "report", "test", "text", "unrelated"),
        EvidenceUnit("text_a", "report", "test", "text", "unrelated"),
    )
    retriever = BM25Retriever().fit(evidence)

    ranked = retriever.search("missing", top_k=2)

    assert [item.evidence.evidence_id for item in ranked] == ["text_a", "text_b"]


def test_retrieval_runner_writes_summary_manifest_and_predictions(tmp_path: Path) -> None:
    examples = load_split(FIXTURE, "test")

    summary = run_retrieval_evaluation(
        examples,
        methods=("bm25",),
        output_dir=tmp_path,
        top_k=5,
    )

    assert summary["query_count"] == 2
    assert summary["methods"]["bm25"]["recall_at_5"] == 1.0
    assert summary["methods"]["bm25"]["p50_ms"] >= 0
    predictions = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(predictions) == 2
    assert predictions[0]["method"] == "bm25"
    assert len(predictions[0]["ranked_evidence_ids"]) == 5
    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "experiment_manifest.json").exists()
