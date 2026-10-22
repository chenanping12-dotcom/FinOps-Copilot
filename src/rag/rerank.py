"""Optional cross-encoder and a transparent offline reranking fallback."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol

from .bm25 import tokenize
from .hf_cache import enable_offline_if_cached
from .models import RetrievalHit


class Reranker(Protocol):
    backend_name: str
    model_name: str

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]: ...


def rerank_sort_key(hit: RetrievalHit) -> tuple[float, float, int, int, str]:
    """Order reranked hits, degrading through progressively weaker signals.

    The first key is the reranker score. Everything after it only decides ties,
    and the tie-break choice turned out to matter more than expected: the
    Cross-Encoder can *saturate*, and on this corpus it does. For EVAL-116 the
    100 candidates spanned a score band of 0.0034, so at float precision nearly
    all of them tie and the remaining keys decide the order outright.

    Falling back to `chunk_id` made the final ordering alphabetical -- arbitrary
    for this purpose. It is tempting to blame the tie-break alone, but recomputing
    the order over the recorded scores with four different sorting keys leaves gold
    `POL-068-V2-C04` at rank 8 every time: six near-duplicate siblings differing
    only by a scope clause outscore it by ~1e-6, so the tie-break never gets to
    decide. The rank-2 figure quoted elsewhere was measured *before* reranking, on
    a different curve, and is not evidence that the tie-break pushed gold from 2 to
    8. What removes those siblings from the candidate set is narrowing the window
    (100 -> 20); switching the fallback from `chunk_id` to `rrf_score` keeps the
    lexical/vector consensus as the deciding signal and is the supporting fix,
    after which gold returns to rank 4 and Recall@5 recovers to 1.0000.

    The per-arm ranks come next (preserving whichever retriever was confident),
    with `chunk_id` last purely so the order is total and deterministic -- a
    stable order is needed for reproducible metrics, not because the ID carries
    meaning.

    Both reranker implementations sort through this function so the two cannot
    drift apart.
    """
    components = hit.components

    def as_int(key: str) -> int:
        value = components.get(key)
        return int(value) if isinstance(value, (int, float)) else 10**9

    return (
        -float(hit.score),
        -float(components.get("rrf_score", 0.0) or 0.0),
        as_int("bm25_rank"),
        as_int("dense_rank"),
        hit.chunk_id,
    )


class LexicalReranker:
    """A deterministic pair scorer for offline runs without PyTorch."""

    backend_name = "transparent_lexical_fallback"
    model_name = "local://token-overlap-cross-encoder"

    @staticmethod
    def _score(query: str, hit: RetrievalHit) -> float:
        query_tokens = tokenize(query)
        doc_tokens = tokenize(hit.chunk.retrieval_text)
        q_counter = {token: query_tokens.count(token) for token in set(query_tokens)}
        d_counter = {token: doc_tokens.count(token) for token in set(doc_tokens)}
        overlap = sum(min(count, d_counter.get(token, 0)) for token, count in q_counter.items())
        query_mass = max(sum(q_counter.values()), 1)
        overlap_ratio = overlap / query_mass
        exact_phrase = 1.0 if query.strip() and query.strip() in hit.chunk.text else 0.0
        number_matches = len(set(re.findall(r"\d+", query)) & set(re.findall(r"\d+", hit.chunk.text)))
        version_bonus = 0.25 if "当前" in query and hit.chunk.version == "V2" else 0.0
        # Numeric and negation cues are high-value for policy questions but never use gold IDs.
        return 4.0 * overlap_ratio + 1.5 * exact_phrase + 0.35 * number_matches + version_bonus

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        rescored = []
        for hit in hits:
            score = self._score(query, hit)
            rescored.append(
                RetrievalHit(
                    chunk=hit.chunk,
                    score=score,
                    rank=0,
                    method="rerank",
                    components=hit.components | {"rerank_score": score, "rrf_score": hit.score},
                )
            )
        rescored.sort(key=rerank_sort_key)
        for rank, hit in enumerate(rescored, start=1):
            hit.rank = rank
        return rescored


class SentenceTransformerReranker:
    """Optional BGE CrossEncoder implementation, imported only when available."""

    backend_name = "sentence_transformers_cross_encoder"

    def __init__(
        self,
        model_name: str,
        cache_folder: str | None = None,
        batch_size: int = 8,
        device: str = "cpu",
        max_length: int = 384,
        revision: str | None = None,
    ) -> None:
        # 同 dense.py：模型已在本地就切离线模式，避免断网时 transformers 抛
        # ValueError 打挂构造。
        enable_offline_if_cached(cache_folder, model_name)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for the CrossEncoder backend") from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.revision = revision
        self.model = CrossEncoder(
            model_name,
            revision=revision,
            cache_folder=cache_folder,
            device=device,
            max_length=max_length,
        )
        model_config = getattr(getattr(self.model, "model", None), "config", None)
        self.resolved_revision = getattr(model_config, "_commit_hash", None) or revision

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        rescored = [
            RetrievalHit(
                chunk=hit.chunk,
                score=float(score),
                rank=0,
                method="rerank",
                components=hit.components | {"rerank_score": float(score), "rrf_score": hit.score},
            )
            for hit, score in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=rerank_sort_key)
        for rank, hit in enumerate(rescored, start=1):
            hit.rank = rank
        return rescored


def build_reranker(config: dict[str, Any], model_cache: str | None, device: str) -> Reranker:
    requested = str(config.get("backend", "auto"))
    if requested in {"auto", "sentence_transformers"}:
        try:
            return SentenceTransformerReranker(
                str(config.get("model_id", "BAAI/bge-reranker-base")),
                cache_folder=model_cache,
                batch_size=int(config.get("batch_size", 8)),
                device=device,
                max_length=int(config.get("max_length", 384)),
                revision=config.get("revision"),
            )
        except (ImportError, RuntimeError, OSError):
            if requested == "sentence_transformers":
                raise
    return LexicalReranker()
