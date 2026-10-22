"""Dense retrieval backends with a deterministic no-dependency fallback."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Protocol

from .bm25 import tokenize
from .corpus import ParentChildCorpus
from .models import RetrievalHit


class Embedder(Protocol):
    backend_name: str
    model_name: str

    def encode(self, texts: list[str], is_query: bool = False) -> list[Any]: ...


def _hash_bucket(feature: str, dimensions: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


def _sparse_dot(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


class HashedNgramEmbedder:
    """Deterministic sparse vector encoder used when model dependencies are unavailable."""

    backend_name = "hashed_ngrams"
    model_name = "local://hashed-cjk-ngrams"

    def __init__(self, dimensions: int = 8192) -> None:
        self.dimensions = dimensions

    def encode(self, texts: list[str], is_query: bool = False) -> list[dict[int, float]]:
        encoded: list[dict[int, float]] = []
        for text in texts:
            features = Counter(tokenize(text))
            # Add word-boundary character trigrams to make the vector path less tied to BM25's IDF.
            compact = re.sub(r"\s+", "", text.lower())
            features.update(compact[index : index + 3] for index in range(max(0, len(compact) - 2)))
            vector: dict[int, float] = {}
            for feature, frequency in features.items():
                bucket = _hash_bucket(feature, self.dimensions)
                vector[bucket] = vector.get(bucket, 0.0) + math.log1p(frequency)
            norm = math.sqrt(sum(value * value for value in vector.values()))
            if norm:
                vector = {index: value / norm for index, value in vector.items()}
            encoded.append(vector)
        return encoded


class SentenceTransformerEmbedder:
    """BGE encoder; imported lazily so the lexical baseline stays dependency-free."""

    backend_name = "sentence_transformers"

    def __init__(
        self,
        model_name: str,
        cache_folder: str | None = None,
        batch_size: int = 48,
        device: str = "cpu",
        query_instruction: str = "",
        max_seq_length: int = 512,
        revision: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for the BGE backend") from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.revision = revision
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            cache_folder=cache_folder,
            device=device,
        )
        self.model.max_seq_length = max_seq_length
        self.resolved_revision = self._resolved_model_revision()

    def _resolved_model_revision(self) -> str | None:
        for module in self.model:
            auto_model = getattr(module, "auto_model", None)
            commit = getattr(getattr(auto_model, "config", None), "_commit_hash", None)
            if commit:
                return str(commit)
        return self.revision

    def encode(self, texts: list[str], is_query: bool = False) -> Any:
        prefix = self.query_instruction if is_query else ""
        return self.model.encode(
            [prefix + text for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def build_embedder(config: dict[str, Any], model_cache: str | None, device: str) -> Embedder:
    requested = str(config.get("backend", "auto"))
    model_name = str(config.get("model_id", "BAAI/bge-small-zh-v1.5"))
    if requested in {"auto", "sentence_transformers"}:
        try:
            return SentenceTransformerEmbedder(
                model_name,
                cache_folder=model_cache,
                batch_size=int(config.get("batch_size", 48)),
                device=device,
                query_instruction=str(config.get("query_instruction", "")),
                max_seq_length=int(config.get("max_seq_length", 512)),
                revision=config.get("revision"),
                normalize_embeddings=bool(config.get("normalize_embeddings", True)),
            )
        except (ImportError, RuntimeError, OSError):
            if requested == "sentence_transformers":
                raise
    return HashedNgramEmbedder(dimensions=int(config.get("fallback_dimensions", 8192)))


class DenseRetriever:
    def __init__(self, corpus: ParentChildCorpus, embedder: Embedder) -> None:
        self.corpus = corpus
        self.embedder = embedder
        self.document_vectors = embedder.encode([chunk.retrieval_text for chunk in corpus.chunks])

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        return self.search_many([query], top_k=top_k)[0]

    def search_many(self, queries: list[str], top_k: int = 10) -> list[list[RetrievalHit]]:
        if top_k <= 0:
            return [[] for _ in queries]
        if not queries:
            return []

        query_vectors = self.embedder.encode(queries, is_query=True)
        if hasattr(self.document_vectors, "shape"):
            score_matrix = query_vectors @ self.document_vectors.T
            all_scores = score_matrix.tolist()
        else:
            all_scores = [
                [_sparse_dot(query_vector, vector) for vector in self.document_vectors]
                for query_vector in query_vectors
            ]

        results: list[list[RetrievalHit]] = []
        for scores in all_scores:
            ranked = sorted(
                range(len(scores)),
                key=lambda index: (-float(scores[index]), self.corpus.chunks[index].chunk_id),
            )
            results.append(
                [
                    RetrievalHit(
                        chunk=self.corpus.chunks[index],
                        score=float(scores[index]),
                        rank=rank,
                        method="dense",
                        components={"dense_score": float(scores[index])},
                    )
                    for rank, index in enumerate(ranked[:top_k], start=1)
                ]
            )
        return results
