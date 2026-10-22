"""Score fusion utilities."""

from __future__ import annotations

from collections import defaultdict

from .models import RetrievalHit


def reciprocal_rank_fusion(
    rankings: dict[str, list[RetrievalHit]], rrf_k: int = 60, limit: int = 100
) -> list[RetrievalHit]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    scores: dict[str, float] = defaultdict(float)
    components: dict[str, dict[str, float]] = defaultdict(dict)
    records: dict[str, RetrievalHit] = {}
    for method, hits in rankings.items():
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            records[chunk_id] = hit
            contribution = 1.0 / (rrf_k + rank)
            scores[chunk_id] += contribution
            components[chunk_id][f"{method}_rrf"] = contribution
            components[chunk_id][f"{method}_score"] = hit.score
            # Record the per-arm rank too. It is not needed to compute the fused
            # score, but it is the most trustworthy signal left when a later
            # reranker saturates and its scores all tie (see
            # rerank.rerank_sort_key, and EVAL-116 in
            # reports/stage2_retrieval_findings.md).
            components[chunk_id][f"{method}_rank"] = float(rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return [
        RetrievalHit(
            chunk=records[chunk_id].chunk,
            score=scores[chunk_id],
            rank=rank,
            method="rrf",
            components=components[chunk_id] | {"rrf_score": scores[chunk_id]},
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]
