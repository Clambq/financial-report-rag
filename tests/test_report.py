import re
from pathlib import Path

from finqa_rag.evaluation.report import (
    analyze_error_artifacts,
    format_percent,
    verify_artifacts,
    write_reports,
)


RETRIEVAL_SUMMARY = {
    "query_count": 1147,
    "split": "test",
    "top_k": 5,
    "methods": {
        "bm25": {"recall_at_5": 0.7209, "mrr": 0.6217, "p95_ms": 0.36},
        "hybrid_rerank": {
            "recall_at_5": 0.8146,
            "mrr": 0.7459,
            "p95_ms": 15.18,
        },
    },
}
NUMERICAL_SUMMARY = {
    "query_count": 100,
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "conditions": {
        "direct_llm": {
            "exact_match_count": 48,
            "exact_match_rate": 0.48,
            "arithmetic_error_count": 52,
            "arithmetic_error_rate": 0.52,
        },
        "python_tool": {
            "exact_match_count": 100,
            "exact_match_rate": 1.0,
            "arithmetic_error_count": 0,
            "arithmetic_error_rate": 0.0,
        },
    },
    "absolute_error_rate_reduction": 0.52,
    "relative_error_reduction": 1.0,
}


def test_resume_bullet_uses_verified_metrics(tmp_path: Path) -> None:
    paths = write_reports(RETRIEVAL_SUMMARY, NUMERICAL_SUMMARY, tmp_path)
    text = paths.resume_bullets.read_text(encoding="utf-8")

    assert str(RETRIEVAL_SUMMARY["query_count"]) in text
    assert format_percent(
        RETRIEVAL_SUMMARY["methods"]["bm25"]["recall_at_5"]
    ) in text
    assert "52.00% to 0.00%" in text
    assert re.search(r"\[[A-Z]\]", text) is None


def test_report_does_not_claim_improvement_when_hybrid_is_lower(
    tmp_path: Path,
) -> None:
    summary = {
        **RETRIEVAL_SUMMARY,
        "methods": {
            "bm25": {"recall_at_5": 0.70, "mrr": 0.62, "p95_ms": 0.3},
            "hybrid_rerank": {"recall_at_5": 0.68, "mrr": 0.61, "p95_ms": 15.0},
        },
    }

    text = write_reports(summary, NUMERICAL_SUMMARY, tmp_path).resume_bullets.read_text(
        encoding="utf-8"
    )

    assert "improved" not in text.lower()
    assert "benchmarked" in text.lower()


def test_results_explains_controlled_numerical_scope(tmp_path: Path) -> None:
    results = write_reports(
        RETRIEVAL_SUMMARY,
        NUMERICAL_SUMMARY,
        tmp_path,
    ).results.read_text(encoding="utf-8")

    assert "gold evidence" in results.lower()
    assert "gold symbolic program" in results.lower()
    assert "end-to-end" in results.lower()


def test_verify_published_artifacts_recomputes_metrics() -> None:
    artifact_root = Path(__file__).parents[1] / "artifacts"

    verified = verify_artifacts(artifact_root)

    assert verified["retrieval_prediction_rows"] == 4588
    assert verified["numerical_prediction_rows"] == 100
    assert verified["formulas_verified"] is True
    assert verified["publication_hashes_verified"] is True


def test_error_analysis_is_derived_from_published_predictions() -> None:
    artifact_root = Path(__file__).parents[1] / "artifacts"

    analysis = analyze_error_artifacts(artifact_root)

    assert analysis["retrieval"]["improved_vs_bm25"] == 219
    assert analysis["retrieval"]["rerank_miss"] == 80
    assert analysis["numerical"]["direct_mismatches"] == 52
    assert analysis["numerical"]["percent_outputs"] == 19
    assert len(analysis["retrieval"]["representative_miss_ids"]) == 5
