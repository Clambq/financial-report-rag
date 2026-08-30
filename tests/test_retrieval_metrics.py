import math

import pytest

from finqa_rag.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_ranking,
)


def test_retrieval_metrics_use_all_gold_evidence() -> None:
    result = evaluate_ranking(
        ["text_2", "table_1", "text_9"],
        {"text_2", "table_1"},
        k=2,
    )

    assert result.recall_at_k == 1.0
    assert result.hit_at_k == 1.0
    assert result.mrr == 1.0
    assert result.ndcg_at_k == 1.0


def test_retrieval_metrics_discount_late_first_hit() -> None:
    result = evaluate_ranking(["x", "gold", "y"], {"gold", "missing"}, k=3)

    assert result.recall_at_k == 0.5
    assert result.hit_at_k == 1.0
    assert result.mrr == 0.5
    expected_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert result.ndcg_at_k == pytest.approx(expected_ndcg)


def test_aggregate_retrieval_metrics_uses_query_mean() -> None:
    first = evaluate_ranking(["gold"], {"gold"}, k=1)
    second = evaluate_ranking(["wrong"], {"gold"}, k=1)

    aggregate = aggregate_retrieval_metrics([first, second], top_k=5)

    assert aggregate == {
        "recall_at_5": 0.5,
        "hit_at_5": 0.5,
        "mrr": 0.5,
        "ndcg_at_5": 0.5,
    }
