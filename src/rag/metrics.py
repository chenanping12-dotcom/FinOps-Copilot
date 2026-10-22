"""Ranking metrics and error-analysis helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def reciprocal_rank(result_ids: Iterable[str], gold_ids: set[str], k: int) -> float:
    for rank, result_id in enumerate(list(result_ids)[:k], start=1):
        if result_id in gold_ids:
            return 1.0 / rank
    return 0.0


def evaluate_rankings(
    questions: list[dict[str, Any]], rankings: dict[str, dict[str, list[str]]], k: int = 5
) -> dict[str, dict[str, float | int]]:
    answerable = [question for question in questions if question.get("gold_chunk_ids")]
    unanswerable = [question for question in questions if not question.get("gold_chunk_ids")]
    results: dict[str, dict[str, float | int]] = {}
    for method, method_rankings in rankings.items():
        hits = []
        reciprocal_ranks = []
        for question in answerable:
            gold = set(question["gold_chunk_ids"])
            retrieved = method_rankings.get(question["question_id"], [])
            hits.append(len(set(retrieved[:k]) & gold) / len(gold))
            reciprocal_ranks.append(reciprocal_rank(retrieved, gold, k))
        results[method] = {
            "recall_at_5" if k == 5 else f"recall_at_{k}": sum(hits) / max(len(hits), 1),
            "mrr_at_5" if k == 5 else f"mrr_at_{k}": sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1),
            "answerable_questions": len(answerable),
            "excluded_unanswerable": len(unanswerable),
            "unanswerable_questions": len(unanswerable),
        }
    return results
