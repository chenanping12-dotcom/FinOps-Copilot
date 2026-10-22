"""Dependency-free BM25 retrieval for the synthetic Chinese rule corpus."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from .corpus import ParentChildCorpus
from .models import RetrievalHit


_ASCII_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese characters into unigrams/bigrams plus ASCII words/numbers."""
    lowered = text.lower()
    tokens = [match.group(0) for match in _ASCII_RE.finditer(lowered)]
    for run_match in _CJK_RUN_RE.finditer(lowered):
        run = run_match.group(0)
        tokens.extend(_CJK_RE.findall(run))
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


class BM25Retriever:
    def __init__(self, corpus: ParentChildCorpus, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.corpus = corpus
        self.k1 = float(k1)
        self.b = float(b)
        self.document_tokens = [tokenize(chunk.retrieval_text) for chunk in corpus.chunks]
        self.document_lengths = [len(tokens) for tokens in self.document_tokens]
        self.avgdl = sum(self.document_lengths) / max(len(self.document_lengths), 1)
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        document_frequency: Counter[str] = Counter()
        for index, tokens in enumerate(self.document_tokens):
            frequencies = Counter(tokens)
            for token, frequency in frequencies.items():
                self.postings[token][index] = frequency
            document_frequency.update(frequencies.keys())
        self.idf = {
            token: math.log(1.0 + (len(corpus) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def score_all(self, query: str) -> list[float]:
        query_terms = Counter(tokenize(query))
        scores = [0.0] * len(self.corpus)
        for term, query_frequency in query_terms.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for document_index, term_frequency in postings.items():
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * self.document_lengths[document_index] / max(self.avgdl, 1e-9)
                )
                scores[document_index] += idf * term_frequency * (self.k1 + 1.0) / denominator
        return scores

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        scores = self.score_all(query)
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], self.corpus.chunks[index].chunk_id))
        hits: list[RetrievalHit] = []
        for rank, index in enumerate(ranked[:top_k], start=1):
            hits.append(
                RetrievalHit(
                    chunk=self.corpus.chunks[index],
                    score=scores[index],
                    rank=rank,
                    method="bm25",
                    components={"bm25_score": scores[index]},
                )
            )
        return hits
