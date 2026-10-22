import unittest

from src.rag.bm25 import BM25Retriever, tokenize
from src.rag.corpus import ParentChildCorpus
from src.rag.dense import DenseRetriever, HashedNgramEmbedder
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.metrics import evaluate_rankings
from src.rag.models import RetrievalHit
from src.rag.rerank import rerank_sort_key


def record(chunk_id: str, page: int, text: str, line: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "POL-001-V2",
        "version": "V2",
        "effective_from": "2025-07-01",
        "page": page,
        "line_start": line,
        "line_end": line,
        "policy_key": "refund:global",
        "policy_family": "refund_accounting",
        "scope": "global",
        "rule_type": "refund_boundary",
        "text": text,
    }


class RAGUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            record("C1", 1, "退款申请须在签收后七个自然日内提交，第七日仍可受理。", 10),
            record("C2", 1, "平台承担的运费补贴不得计入买家实付金额。", 12),
            record("C3", 2, "商家结算差异超过一百分时转人工复核。", 20),
        ]
        self.corpus = ParentChildCorpus.from_records(self.records)

    def test_chinese_bigram_tokenizer_keeps_latin_terms(self) -> None:
        terms = tokenize("V2退款窗口为15个自然日")
        self.assertIn("v2", terms)
        self.assertIn("15", terms)
        self.assertIn("退款", terms)
        self.assertIn("窗口", terms)
        self.assertIn("自然", terms)

    def test_bm25_returns_matching_chunk_first(self) -> None:
        hits = BM25Retriever(self.corpus, k1=1.5, b=0.75).search("退款申请在第七个自然日是否可受理", top_k=2)
        self.assertEqual(hits[0].chunk.chunk_id, "C1")
        self.assertEqual([hit.rank for hit in hits], [1, 2])

    def test_parent_child_mapping_and_citation_preserve_exact_source(self) -> None:
        child = self.corpus.by_id["C1"]
        self.assertEqual(child.parent_id, "POL-001-V2:P1")
        self.assertIn(self.records[1]["text"], child.parent_text)
        self.assertEqual(child.citation, {"doc_id": "POL-001-V2", "page": 1, "line_start": 10, "line_end": 10, "chunk_id": "C1"})

    def test_rrf_combines_independent_rankings(self) -> None:
        retriever = BM25Retriever(self.corpus)
        hit_a = retriever.search("退款", top_k=1)[0]
        hit_b = retriever.search("运费补贴", top_k=1)[0]
        fused = reciprocal_rank_fusion({"bm25": [hit_a, hit_b], "dense": [hit_b, hit_a]}, rrf_k=60, limit=2)
        self.assertEqual(fused[0].chunk.chunk_id, hit_a.chunk.chunk_id)
        self.assertEqual(fused[0].score, fused[1].score)
        self.assertEqual(fused[0].rank, 1)

    def test_dense_batch_search_matches_single_query_results(self) -> None:
        retriever = DenseRetriever(self.corpus, HashedNgramEmbedder(dimensions=256))
        batched = retriever.search_many(["退款申请 第七日", "平台运费补贴"], top_k=2)
        single = retriever.search("退款申请 第七日", top_k=2)
        self.assertEqual([hit.chunk_id for hit in batched[0]], [hit.chunk_id for hit in single])
        self.assertEqual(len(batched), 2)
        self.assertEqual([hit.rank for hit in batched[1]], [1, 2])

    def test_rrf_records_per_arm_rank_for_tie_break(self) -> None:
        """RRF must expose each arm's rank, not just its score.

        rerank.rerank_sort_key falls back to these when the Cross-Encoder
        saturates and every candidate ties. Without them the tie-break silently
        degrades to alphabetical chunk_id order (the EVAL-116 regression).
        """
        corpus = ParentChildCorpus.from_records(self.records)
        bm25_hits = BM25Retriever(corpus).search("退款申请 第七日", top_k=3)
        dense_hits = DenseRetriever(corpus, HashedNgramEmbedder(dimensions=256)).search(
            "退款申请 第七日", top_k=3
        )
        fused = reciprocal_rank_fusion({"bm25": bm25_hits, "dense": dense_hits}, limit=3)
        self.assertTrue(fused)
        for hit in fused:
            self.assertIn("bm25_rank", hit.components)
            self.assertIn("dense_rank", hit.components)
            self.assertGreaterEqual(hit.components["bm25_rank"], 1.0)

    def test_rerank_tie_break_prefers_rrf_over_chunk_id_order(self) -> None:
        """Equal reranker scores must fall back to RRF order, not alphabetically.

        This is the regression guard for the EVAL-116 failure: the gold chunk had
        an alphabetically late ID, so a chunk_id tie-break buried it behind
        near-duplicates that the Cross-Encoder scored identically.
        """
        corpus = ParentChildCorpus.from_records(self.records)
        # "C2" is the gold: alphabetically late, but it leads on RRF. "C1" is an
        # alphabetically-early decoy carrying a weak RRF score.
        lead = RetrievalHit(
            chunk=corpus.by_id["C2"],
            score=0.99,
            rank=1,
            method="rrf",
            components={"rrf_score": 0.9},
        )
        decoy = RetrievalHit(
            chunk=corpus.by_id["C1"],
            score=0.99,
            rank=2,
            method="rrf",
            components={"rrf_score": 0.1},
        )
        # Identical rerank score -> only the fallback keys can decide. The old
        # chunk_id tie-break would put C1 first; RRF order must put C2 first.
        self.assertLess("C1", "C2")  # the decoy really does win on raw ID order
        ordered = sorted([decoy, lead], key=rerank_sort_key)
        self.assertEqual(ordered[0].chunk_id, "C2")

    def test_metrics_exclude_unanswerable_items(self) -> None:
        questions = [
            {"question_id": "Q1", "gold_chunk_ids": ["C2"], "must_abstain": False},
            {"question_id": "Q2", "gold_chunk_ids": ["missing"], "must_abstain": False},
            {"question_id": "Q3", "gold_chunk_ids": [], "must_abstain": True},
        ]
        rankings = {
            "bm25": {"Q1": ["C1", "C2"], "Q2": ["C1"], "Q3": ["C1"]},
            "dense": {"Q1": ["C2"], "Q2": ["C1"], "Q3": ["C1"]},
        }
        metrics = evaluate_rankings(questions, rankings, k=5)
        self.assertEqual(metrics["bm25"]["recall_at_5"], 0.5)
        self.assertEqual(metrics["bm25"]["mrr_at_5"], 0.25)
        self.assertEqual(metrics["bm25"]["answerable_questions"], 2)
        self.assertEqual(metrics["bm25"]["excluded_unanswerable"], 1)


if __name__ == "__main__":
    unittest.main()
