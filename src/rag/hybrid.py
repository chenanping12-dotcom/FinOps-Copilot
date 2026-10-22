"""Hybrid BM25 + dense retrieval with RRF and second-stage reranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bm25 import BM25Retriever
from .corpus import ParentChildCorpus
from .dense import DenseRetriever, build_embedder
from .fusion import reciprocal_rank_fusion
from .models import RetrievalHit
from .rerank import Reranker, build_reranker


class HybridRetriever:
    def __init__(self, corpus: ParentChildCorpus, config: dict[str, Any], root: str | None = None) -> None:
        self.corpus = corpus
        retrieval = config.get("retrieval", config)
        self.bm25 = BM25Retriever(corpus, **{key: retrieval.get("bm25", {}).get(key) for key in ("k1", "b") if key in retrieval.get("bm25", {})})
        device = str(retrieval.get("device", "cpu"))
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"
        project_root = Path(root) if root else Path(__file__).resolve().parents[2]
        model_cache_value = retrieval.get("model_cache")
        model_cache = str((project_root / model_cache_value).resolve()) if model_cache_value else None
        self.embedder = build_embedder(retrieval.get("dense", {}), model_cache, device)
        self.dense = DenseRetriever(corpus, self.embedder)
        self.reranker: Reranker = build_reranker(retrieval.get("reranker", {}), model_cache, device)
        self.rrf_k = int(retrieval.get("fusion", {}).get("rrf_k", 60))
        self.candidate_k = int(retrieval.get("fusion", {}).get("candidate_k", 100))
        self.rerank_candidate_k = int(retrieval.get("reranker", {}).get("candidate_k", self.candidate_k))

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def search_bm25(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        return self.bm25.search(query, top_k=top_k)

    def search_dense(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        return self.dense.search(query, top_k=top_k)

    def search_hybrid(self, query: str, top_k: int = 5, rerank: bool = True) -> list[RetrievalHit]:
        candidate_k = min(self.candidate_k, len(self.corpus))
        bm25_hits = self.bm25.search(query, top_k=candidate_k)
        dense_hits = self.dense.search(query, top_k=candidate_k)
        return self.search_hybrid_from_candidates(query, bm25_hits, dense_hits, top_k, rerank)

    def search_hybrid_from_candidates(
        self,
        query: str,
        bm25_hits: list[RetrievalHit],
        dense_hits: list[RetrievalHit],
        top_k: int = 5,
        rerank: bool = True,
        rerank_candidate_k: int | None = None,
    ) -> list[RetrievalHit]:
        """Fuse two ranked lists with RRF and optionally rerank the fused head.

        Why the reranker only sees a slice: a Cross-Encoder scores every
        (query, passage) pair through the full transformer, so its cost is linear in
        the number of candidates. Fusing at ``candidate_k`` and then reranking a
        smaller head keeps the latency budget while still letting the pair model
        fix the ordering of everything that could plausibly reach ``top_k``.

        Recall and latency trade off directly through ``rerank_candidate_k``: the
        reranker can only promote a document that survived RRF, so shrinking this
        window caps the achievable Recall@k no matter how good the reranker is.
        ``rerank_candidate_k`` therefore defaults to ``self.rerank_candidate_k``
        (config) but can be overridden per call for ablation.
        """
        window = self.rerank_candidate_k if rerank_candidate_k is None else rerank_candidate_k
        # The arm names here must match the `method` labels set by BM25Retriever
        # ("bm25") and DenseRetriever ("dense"): reciprocal_rank_fusion keys its
        # per-arm components as f"{method}_rank", and rerank_sort_key reads
        # "bm25_rank" / "dense_rank" out of them.
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_hits, "dense": dense_hits}, rrf_k=self.rrf_k, limit=window
        )
        if rerank:
            return self.reranker.rerank(query, fused)[:top_k]
        return [
            RetrievalHit(
                chunk=hit.chunk,
                score=hit.score,
                rank=rank,
                method="hybrid",
                components=hit.components,
            )
            for rank, hit in enumerate(fused[:top_k], start=1)
        ]

    def search(self, query: str, mode: str = "hybrid_rerank", top_k: int = 5) -> list[RetrievalHit]:
        if mode == "bm25":
            return self.search_bm25(query, top_k)
        if mode in {"dense", "vector"}:
            return self.search_dense(query, top_k)
        if mode in {"hybrid", "hybrid_rerank"}:
            return self.search_hybrid(query, top_k, rerank=mode == "hybrid_rerank")
        raise ValueError(f"Unknown retrieval mode: {mode}")
