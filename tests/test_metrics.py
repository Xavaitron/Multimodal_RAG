import pytest

from lightweight_multimodal_retrieval.metrics import (
    evaluate_run,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_basic_metrics() -> None:
    ranked = ["wrong", "relevant", "other"]
    assert recall_at_k(ranked, {"relevant"}, 1) == 0.0
    assert recall_at_k(ranked, {"relevant"}, 2) == 1.0
    assert reciprocal_rank(ranked, {"relevant"}) == 0.5
    assert ndcg_at_k(["a", "b"], {"a": 1.0}, 2) == 1.0


def test_evaluate_run_counts_missing_query_as_failure() -> None:
    result = evaluate_run({"q1": ["d1"]}, {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}})
    assert result["recall@1"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)
