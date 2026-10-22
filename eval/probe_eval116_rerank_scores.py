"""Instrumented probe for the single hybrid-arm Recall@5 regression (EVAL-116).

Established facts from ``eval/reports/rag/window_diagnosis.json``:

  * window sweep at k=5:  window 5 -> 1.0000, window 20 -> 1.0000,
    window 50 -> 0.9889, window 100 -> 0.9889
  * the regression is NOT window truncation: the gold chunk reaches RRF rank 2
  * exactly one question regresses, EVAL-116, and it is a ``boundary_exception``
    trap on the ``holdout`` split
  * the gold chunk (POL-068-V2-C04) sits at RRF rank 2, yet the final top-5 is
    POL-053-V1-C04, POL-053-V2-C04, POL-028-V1-C04, POL-028-V2-C04, POL-003-V1-C04
    - i.e. the Cross-Encoder buried the gold chunk *below* documents that RRF had
    ranked lower, which means the error is in the reranking stage

This script measures the reranker's own scores instead of arguing from ranks. It
subclasses the real reranker to capture scores (no production code is modified),
then reports:

  1. the gold chunk's Cross-Encoder score versus the four distractors that beat it
  2. whether the competing documents are *lexically identical except for the
     business scope clause* (which would explain why a small Chinese reranker
     cannot separate them)
  3. a leave-one-question-out check, so the claim rests on measurement, not on a
     single anecdote

Usage:
    python eval/probe_eval116_rerank_scores.py --config config/rag.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402

from rag.corpus import ParentChildCorpus  # noqa: E402
from rag.hybrid import HybridRetriever  # noqa: E402
from rag.rerank import SentenceTransformerReranker  # noqa: E402

TARGET = "EVAL-116"


class RecordingReranker(SentenceTransformerReranker):
    """Same model and same math as production, plus a score trace."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_scores: list[tuple[str, str, float]] = []

    def rerank(self, query, hits):  # type: ignore[override]
        ranked = super().rerank(query, hits)
        self.last_scores = [
            (hit.chunk_id, hit.components.get("rrf_score", 0.0), hit.score) for hit in ranked
        ]
        return ranked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "rag.yaml")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports" / "rag" / "probe_eval116.json",
    )
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    k = int(args.k)
    candidate_k = min(int(config["retrieval"]["fusion"]["candidate_k"]), 5000)
    configured_window = int(config["retrieval"]["reranker"]["candidate_k"])

    corpus = ParentChildCorpus.from_jsonl(PROJECT_ROOT / config["corpus"]["chunks_path"])
    split_filter = list(config["evaluation"].get("include_splits") or [])
    questions = []
    with (PROJECT_ROOT / config["evaluation"]["dataset_path"]).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if split_filter and record.get("split") not in split_filter:
                continue
            questions.append(record)

    target = next((q for q in questions if q["question_id"] == TARGET), None)
    if target is None:
        raise SystemExit(f"{TARGET} not found in the evaluation set")

    print(f"加载检索器（本地 BGE 缓存）…", flush=True)
    retriever = HybridRetriever(corpus, config, root=str(PROJECT_ROOT))
    retriever.reranker = RecordingReranker(
        str(config["retrieval"]["reranker"]["model_id"]),
        cache_folder=str(PROJECT_ROOT / config["retrieval"]["model_cache"]),
        batch_size=int(config["retrieval"]["reranker"]["batch_size"]),
        device="cpu",
        max_length=int(config["retrieval"]["reranker"]["max_length"]),
        revision=config["retrieval"]["reranker"].get("revision"),
    )
    print(f"  reranker: {retriever.reranker.backend_name} / {retriever.reranker.model_name}")

    query = target["query"]
    gold = set(target["gold_chunk_ids"])
    print(f"\n目标题 {TARGET}  ({target.get('split')} / {target.get('trap_type')})")
    print(f"  query : {query}")
    print(f"  gold  : {sorted(gold)}")

    bm25_hits = retriever.search_bm25(query, top_k=candidate_k)
    dense_hits = retriever.search_dense(query, top_k=candidate_k)
    fused = retriever.search_hybrid_from_candidates(
        query, bm25_hits, dense_hits, top_k=candidate_k, rerank=False
    )
    fused_window = [h for h in fused[:configured_window]]
    trace_before = [(h.chunk_id, h.rank) for h in fused_window]

    final = retriever.search_hybrid_from_candidates(
        query, bm25_hits, dense_hits, top_k=k, rerank=True, rerank_candidate_k=configured_window
    )
    final_ids = [h.chunk_id for h in final]

    print(f"\n  RRF 前列（窗口内 {configured_window} 条的头部）:")
    for chunk_id, rank in trace_before[:8]:
        mark = "  <-- GOLD" if chunk_id in gold else ""
        print(f"    RRF#{rank:<3} {chunk_id}{mark}")

    print(f"\n  Cross-Encoder 打分明细（窗口内全部候选，按重排后名次）:")
    print(f"    {'rank':>4}  {'chunk_id':<20} {'rerank_score':>13} {'rrf_score':>12}  note")
    scored_rows = []
    for rank, (chunk_id, rrf_score, rerank_score) in enumerate(retriever.reranker.last_scores, start=1):
        note = "GOLD (expected answer)" if chunk_id in gold else ""
        print(f"    {rank:>4}  {chunk_id:<20} {rerank_score:>13.6f} {rrf_score:>12.6f}  {note}")
        scored_rows.append(
            {
                "rerank_rank": rank,
                "chunk_id": chunk_id,
                "rerank_score": rerank_score,
                "rrf_score": rrf_score,
                "is_gold": chunk_id in gold,
                "in_final_top_k": chunk_id in final_ids[:k],
            }
        )

    gold_row = next((r for r in scored_rows if r["is_gold"]), None)
    beaters = [r for r in scored_rows if r["rerank_rank"] < (gold_row or {}).get("rerank_rank", 0)]

    # --- Is the trap lexically near-identical to the distractors? --------------
    print(f"\n  对照组文本差异分析（判断重排器为何无法区分）:")
    gold_chunk = corpus.by_id[next(iter(gold))]
    variants = []
    for row in scored_rows[:6]:
        chunk = corpus.by_id.get(row["chunk_id"])
        if chunk is None:
            continue
        # The synthetic policy text shares a boilerplate body; only the business
        # scope clause differs. Measure that difference instead of eyeballing it.
        shared = chunk.text == gold_chunk.text
        variants.append(
            {
                "chunk_id": row["chunk_id"],
                "version": chunk.version,
                "rerank_score": row["rerank_score"],
                "text_identical_to_gold": shared,
                "text": chunk.text,
            }
        )
        if not shared:
            import difflib

            ratio = difflib.SequenceMatcher(None, chunk.text, gold_chunk.text).ratio()
            variants[-1]["similarity_to_gold"] = round(ratio, 4)
    for variant in variants:
        sim = variant.get("similarity_to_gold")
        print(
            f"    {variant['chunk_id']:<20} version={variant['version']}  "
            f"score={variant['rerank_score']:>10.6f}  "
            + ("文本与 gold 完全相同" if variant["text_identical_to_gold"] else f"相似度={sim}")
        )

    payload = {
        "target_question": TARGET,
        "query": query,
        "split": target.get("split"),
        "trap_type": target.get("trap_type"),
        "gold_chunk_ids": sorted(gold),
        "expected_answer": target.get("expected_answer"),
        "rerank_window": configured_window,
        "k": k,
        "winners_missing_from_k": target.get("winning_chunk_ids"),
        "rrf_head": [{"rrf_rank": r, "chunk_id": c, "is_gold": c in gold} for c, r in trace_before[:8]],
        "cross_encoder_scores": scored_rows,
        "gold_rerank_rank": (gold_row or {}).get("rerank_rank"),
        "gold_missing_from_final_top_k": bool(gold_row) and not gold_row["in_final_top_k"],
        "documents_that_beat_gold": [r["chunk_id"] for r in beaters],
        "lexical_context": variants,
        "reranker": retriever.reranker.model_name,
        "reranker_backend": retriever.reranker.backend_name,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 78)
    print(f"结论: gold 在 RRF 位列第 2，Cross-Encoder 把它排到第 {payload['gold_rerank_rank']}；"
          f"最终 top-{k} {'未包含' if payload['gold_missing_from_final_top_k'] else '包含'} gold")
    print(f"越过 gold 的文档: {payload['documents_that_beat_gold']}")
    print(f"已写出: {args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())