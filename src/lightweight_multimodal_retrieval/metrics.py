"""Retrieval metrics with explicit relevance sets and deterministic ranking."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(ranked, 1) if item in relevant), 0.0)


def ndcg_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")

    def dcg(gains: Iterable[float]) -> float:
        return sum((2.0**gain - 1.0) / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))

    actual = dcg(relevance.get(item, 0.0) for item in ranked[:k])
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def evaluate_run(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Macro-average metrics over qrels queries; missing rankings count as failures."""
    if not qrels:
        raise ValueError("qrels cannot be empty")
    totals = {f"recall@{k}": 0.0 for k in ks}
    totals["mrr"] = 0.0
    totals["ndcg@5"] = 0.0
    for query_id, judgments in qrels.items():
        ranked = rankings.get(query_id, ())
        relevant = {doc_id for doc_id, gain in judgments.items() if gain > 0}
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
        totals["mrr"] += reciprocal_rank(ranked, relevant)
        totals["ndcg@5"] += ndcg_at_k(ranked, judgments, 5)
    return {name: value / len(qrels) for name, value in totals.items()}
