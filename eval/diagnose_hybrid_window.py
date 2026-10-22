"""Diagnose *why* the hybrid arm lost two questions that pure BM25 got right.

The ablation run reports ``Recall@5: bm25 1.0000 / hybrid_rerank 0.9889``. A
regression like that is only defensible if the mechanism is understood, so this
script decomposes every question into the ranks each stage produced:

    bm25 rank -> dense rank -> RRF rank -> (rerank window?) -> final rank

Two hypotheses are tested:

  H1  WINDOW TRUNCATION - the gold chunk survived RRF but sat outside the
      rerank window, so the Cross-Encoder never saw it and could not promote it.
      Fix: raise the window; the cost is latency, not correctness.

  H2  RERANKER ERROR - the gold chunk was inside the window and the pair model
      still ranked it below k. Raising the window cannot help; the ceiling is
      the reranker itself.

The script reports which hypothesis holds and at what window size the regression
disappears, so the fix is chosen from evidence rather than by guessing.

Usage:
    python eval/diagnose_hybrid_window.py --config config/rag.yaml
    python eval/diagnose_hybrid_window.py --config config/rag.yaml --windows 5 20 50 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402

from rag.corpus import ParentChildCorpus  # noqa: E402
from rag.hybrid import HybridRetriever  # noqa: E402
from rag.metrics import evaluate_rankings  # noqa: E402


def rank_of(gold: set[str], ranked: list[str], k: int) -> int | None:
    for index, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in gold:
            return index
    return None


def _own_top(record, retriever, candidate_k, k, window):  # pragma: no cover - harness helper
    """Recompute this script's own top-k for one question (used only for the diff)."""
    hits = retriever.search_hybrid_from_candidates(
        record["query"],
        retriever.search_bm25(record["query"], top_k=candidate_k),
        retriever.search_dense(record["query"], top_k=candidate_k),
        top_k=k,
        rerank=True,
        rerank_candidate_k=window,
    )
    return [h.chunk_id for h in hits]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "rag.yaml")
    parser.add_argument("--windows", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval" / "reports" / "rag" / "window_diagnosis.json")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    k = int(args.k)

    corpus_path = PROJECT_ROOT / config["corpus"]["chunks_path"]
    questions_path = PROJECT_ROOT / config["evaluation"]["dataset_path"]
    include_splits = list(config["evaluation"].get("include_splits") or [])

    # Load the exact order the ablation harness scores in. Order matters here: the
    # RRF tie-break is (score desc, chunk_id asc), so a stable ordering is what
    # makes this script's numbers comparable to ablation.json run for run.
    corpus = ParentChildCorpus.from_jsonl(corpus_path)
    all_questions = []
    with questions_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if include_splits and record.get("split") not in include_splits:
                continue
            all_questions.append(record)
    questions = list(all_questions)

    answerable = [q for q in questions if q.get("gold_chunk_ids")]
    answerable_qids = {q["question_id"] for q in answerable}
    print(f"语料切片 {len(corpus)}   可回答题 {len(answerable)} / 全部题 {len(questions)}")
    print("加载检索器（复用本地 BGE 缓存）...", flush=True)
    retriever = HybridRetriever(corpus, config, root=str(PROJECT_ROOT))
    print(f"  embedder: {retriever.embedder.backend_name} / {retriever.embedder.model_name}")
    print(f"  reranker: {retriever.reranker.backend_name} / {retriever.reranker.model_name}")

    candidate_k = min(int(config["retrieval"]["fusion"]["candidate_k"]), len(corpus))
    configured_window = int(config["retrieval"]["reranker"]["candidate_k"])

    # --- Stage A: capture every stage's ranking for each question ---------------
    print("\n[1/3] 逐题采集各阶段排名 ...", flush=True)
    per_question: list[dict] = []
    started = time.perf_counter()
    for position, question in enumerate(questions, start=1):
        query = question["query"]
        gold = set(question["gold_chunk_ids"])
        bm25_hits = retriever.search_bm25(query, top_k=candidate_k)
        dense_hits = retriever.search_dense(query, top_k=candidate_k)
        fused = retriever.search_hybrid_from_candidates(
            query, bm25_hits, dense_hits, top_k=candidate_k, rerank=False
        )
        bm25_ranked = [h.chunk_id for h in bm25_hits]
        dense_ranked = [h.chunk_id for h in dense_hits]
        fused_ranked = [h.chunk_id for h in fused]

        per_question.append(
            {
                "question_id": question["question_id"],
                "split": question.get("split"),
                "trap_type": question.get("trap_type"),
                "query": query,
                "gold_chunk_ids": sorted(gold),
                "is_answerable": bool(gold),
                "bm25_rank": rank_of(gold, bm25_ranked, candidate_k) if gold else None,
                "dense_rank": rank_of(gold, dense_ranked, candidate_k) if gold else None,
                "rrf_rank": rank_of(gold, fused_ranked, candidate_k) if gold else None,
                "bm25_top1": bm25_ranked[0] if bm25_ranked else None,
                "fused_top1": fused_ranked[0] if fused_ranked else None,
                "rrf_ranked": fused_ranked,
            }
        )
        if position % 30 == 0:
            print(f"      {position}/{len(questions)}  已用 {time.perf_counter() - started:.0f}s", flush=True)
    print(f"      采集完成，用时 {time.perf_counter() - started:.0f}s")

    # --- Stage B: window sweep ------------------------------------------------
    # Arm 1 is this script's own re-run at the configured window; arm 2 reads the
    # ablation harness's frozen rankings for the same questions. Scoring both with
    # the identical metric function is what proves (or refutes) that the two
    # pipelines agree - if they disagree, the disagreement IS the finding.
    print("\n[2/3] 窗口扫描 + 与消融报告对账 ...", flush=True)
    window_results: dict[str, dict] = {}
    for window in sorted(set(args.windows + [configured_window])):
        rankings: dict[str, list[str]] = {}
        sweep_started = time.perf_counter()
        for record in per_question:
            hits = retriever.search_hybrid_from_candidates(
                record["query"],
                retriever.search_bm25(record["query"], top_k=candidate_k),
                retriever.search_dense(record["query"], top_k=candidate_k),
                top_k=k,
                rerank=True,
                rerank_candidate_k=window,
            )
            rankings[record["question_id"]] = [h.chunk_id for h in hits]
        elapsed = time.perf_counter() - sweep_started
        metrics = evaluate_rankings(questions, {"hybrid_rerank": rankings}, k)["hybrid_rerank"]
        window_results[str(window)] = {
            "recall_at_k": metrics[f"recall_at_{k}"],
            "mrr_at_k": metrics[f"mrr_at_{k}"],
            "answerable_questions": metrics["answerable_questions"],
            "elapsed_seconds": elapsed,
            "avg_latency_ms": elapsed / max(len(per_question), 1) * 1000,
            "is_configured_value": window == configured_window,
        }
        print(
            f"      窗口 {window:>4}: Recall@{k}={metrics[f'recall_at_{k}']:.4f}  "
            f"MRR@{k}={metrics[f'mrr_at_{k}']:.4f}  "
            f"({elapsed:.0f}s / {elapsed / max(len(per_question), 1) * 1000:.0f}ms each)"
            + ("   <- config 当前值" if window == configured_window else "")
        )

    # Reconcile against the frozen ablation report, if it exists.
    reconciliation: dict | None = None
    frozen_rankings: dict[str, list[str]] = {}
    ablation_path = PROJECT_ROOT / config["output"]["directory"] / config["output"]["comparison_filename"]
    ablation_rankings_path = PROJECT_ROOT / config["output"]["directory"] / config["output"]["rankings_filename"]
    if ablation_path.exists() and ablation_rankings_path.exists():
        report = json.loads(ablation_path.read_text(encoding="utf-8"))
        frozen: dict[str, list[str]] = {}
        with ablation_rankings_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["arm"] == "hybrid_rerank":
                    frozen[record["question_id"]] = record["ranked_chunk_ids"]
        frozen_rankings = frozen
        frozen_metrics = evaluate_rankings(questions, {"hybrid_rerank": frozen}, k)["hybrid_rerank"]
        own = window_results[str(configured_window)]
        differing = [
            record["question_id"]
            for record in per_question
            if frozen.get(record["question_id"], [])[:k] != _own_top(record, retriever, candidate_k, k, configured_window)
        ]
        reconciliation = {
            "ablation_generated_at": report.get("generated_at"),
            "frozen": {
                "recall_at_k": frozen_metrics[f"recall_at_{k}"],
                "mrr_at_k": frozen_metrics[f"mrr_at_{k}"],
            },
            "this_script_at_configured_window": {
                "recall_at_k": own["recall_at_k"],
                "mrr_at_k": own["mrr_at_k"],
            },
            "reproduced": (
                abs(frozen_metrics[f"recall_at_{k}"] - own["recall_at_k"]) < 1e-12
                and abs(frozen_metrics[f"mrr_at_{k}"] - own["mrr_at_k"]) < 1e-12
            ),
            "differing_questions": differing,
        }
        print(
            f"      对账: 消融报告 Recall@{k}={frozen_metrics[f'recall_at_{k}']:.4f} "
            f"MRR@{k}={frozen_metrics[f'mrr_at_{k}']:.4f}  |  "
            f"本脚本(窗口{configured_window}) Recall@{k}={own['recall_at_k']:.4f} "
            f"MRR@{k}={own['mrr_at_k']:.4f}  -> "
            + ("一致" if reconciliation["reproduced"] else "不一致")
        )

    bm25_reference_rankings = {
        r["question_id"]: ([r["bm25_top1"]] if r["bm25_top1"] else []) for r in per_question
    }
    bm25_only_reference = evaluate_rankings(answerable, {"bm25": bm25_reference_rankings}, k)

    # --- Stage C: classify each regression as truncation vs reranker error ------
    print("\n[3/3] 归因分析 ...", flush=True)
    # Attribute against the arm that the ablation report actually scored, so the
    # explanation applies to the published number rather than to a re-run.
    reference = frozen_rankings if frozen_rankings else {
        record["question_id"]: _own_top(record, retriever, candidate_k, k, configured_window)
        for record in per_question
    }
    reference_label = "消融报告(ablation.json/rankings.jsonl)" if frozen_rankings else "本脚本重跑"

    truncated: list[dict] = []
    reranker_errors: list[dict] = []
    bm25_wins_lost: list[dict] = []
    for record in per_question:
        if not record["is_answerable"]:
            continue
        gold = set(record["gold_chunk_ids"])
        qid = record["question_id"]
        final_ranked = reference.get(qid, [])
        final_rank = rank_of(gold, final_ranked, k)
        bm25_rank = record["bm25_rank"]
        rrf_rank = record["rrf_rank"]

        # Only questions BM25 got right but the hybrid arm got wrong are regressions.
        if bm25_rank is None or final_rank is not None:
            continue
        entry = {
            "question_id": qid,
            "query": record["query"],
            "split": record["split"],
            "trap_type": record["trap_type"],
            "bm25_rank": bm25_rank,
            "dense_rank": record["dense_rank"],
            "rrf_rank": rrf_rank,
            "final_rank_within_k": final_rank,
            "gold_chunk_ids": record["gold_chunk_ids"],
            "rrf_top5": record["rrf_ranked"][:k],
            "fused_top1": record["fused_top1"],
            "final_top5": final_ranked[:k],
            "gold_in_rerank_window": rrf_rank is not None and rrf_rank <= configured_window,
        }
        bm25_wins_lost.append(entry)
        if rrf_rank is None:
            reranker_errors.append({**entry, "note": "gold 未进入 RRF 候选池（召回问题，非重排问题）"})
        elif rrf_rank > configured_window:
            truncated.append(entry)
        else:
            reranker_errors.append({**entry, "note": "gold 在重排窗口内，仍被挤出 top-k（重排器判断错误）"})

    best_window = max(
        window_results.items(), key=lambda kv: (kv[1]["recall_at_k"], kv[1]["mrr_at_k"])
    )

    payload = {
        "k": k,
        "candidate_k": candidate_k,
        "configured_rerank_window": configured_window,
        "answerable_questions": len(answerable),
        "total_questions": len(questions),
        "embedder": retriever.embedder.model_name,
        "embedder_backend": retriever.embedder.backend_name,
        "reranker": retriever.reranker.model_name,
        "reranker_backend": retriever.reranker.backend_name,
        "window_sweep": window_results,
        "best_window_by_recall_then_mrr": int(best_window[0]),
        "reconciliation_with_ablation": reconciliation,
        "attribution_reference": reference_label,
        "bm25_selfcheck": {
            "note": "仅用于确认本次采集与消融报告一致；这是 BM25 top1 的参考值，不是完整 BM25 臂。",
            "metrics": bm25_only_reference,
        },
        "attribution": {
            "regressions_bm25_won_but_hybrid_lost": {
                "count": len(bm25_wins_lost),
                "cases": bm25_wins_lost,
            },
            "hypothesis_H1_window_truncation": {
                "verdict": "支持" if truncated else "不支持",
                "count": len(truncated),
                "cases": truncated,
            },
            "hypothesis_H2_reranker_error": {
                "verdict": "支持" if reranker_errors else "不支持",
                "count": len(reranker_errors),
                "cases": reranker_errors,
            },
        },
        "per_question": per_question,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 78)
    print("Hybrid 排名回退归因")
    print("=" * 78)
    print(f"配置的 rerank 窗口: {configured_window}   候选池: {candidate_k}")
    print(f"H1 窗口截断（gold 在 RRF 里但窗口外）: {len(truncated)} 例 -> {payload['attribution']['hypothesis_H1_window_truncation']['verdict']}")
    print(f"H2 重排错误（gold 在窗口内仍被压到 k 名之外）: {len(reranker_errors)} 例 -> {payload['attribution']['hypothesis_H2_reranker_error']['verdict']}")
    for entry in truncated + reranker_errors:
        print(f"   - {entry['question_id']}  bm25_rank={entry.get('bm25_rank')} "
              f"rrf_rank={entry.get('rrf_rank')}  q={entry.get('query')}")
    print(f"\n窗口扫描最优值: {best_window[0]}  -> Recall@{k}={best_window[1]['recall_at_k']:.4f}")
    print(f"已写出: {args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())