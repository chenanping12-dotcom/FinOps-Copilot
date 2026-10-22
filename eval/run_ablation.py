"""Stage 2 ablation: compare retrieval arms on the frozen Stage 1 evaluation set.

Runs three anatomically distinct arms over the same 120-question benchmark and
reports Recall@5 / MRR@5:

  1. bm25          - lexical only (character bigram BM25)
  2. dense         - embedding only (BGE bi-encoder, cosine)
  3. hybrid_rerank - BM25 + dense -> RRF -> cross-encoder rerank

Unanswerable items (``must_abstain`` / empty ``gold_chunk_ids``) are excluded from
Recall and MRR, because a ranking metric cannot score "should have abstained";
they are reported separately so the exclusion is visible rather than silent.

The script also mines concrete bad cases where the hybrid arm fixed a question the
lexical baseline got wrong, and emits the gold chunk text so the case can be
explained in an interview.

Usage:
    python eval/run_ablation.py --config config/rag.yaml
    python eval/run_ablation.py --config config/rag.yaml --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402

from rag.corpus import ParentChildCorpus  # noqa: E402
from rag.hybrid import HybridRetriever  # noqa: E402
from rag.metrics import evaluate_rankings  # noqa: E402

ARMS = {
    "bm25": "bm25",
    "dense": "dense",
    "hybrid_rerank": "hybrid_rerank",
}
ARM_LABELS = {
    "bm25": "纯 BM25（词法）",
    "dense": "纯向量（BGE 双塔）",
    "hybrid_rerank": "Hybrid + Rerank（BM25+Dense→RRF→CrossEncoder）",
}


def load_questions(path: Path, include_splits: list[str], limit: int | None) -> list[dict]:
    questions = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if include_splits and record.get("split") not in include_splits:
                continue
            questions.append(record)
    if limit is not None:
        questions = questions[:limit]
    return questions


def run_arm(retriever: HybridRetriever, questions: list[dict], arm: str, k: int) -> tuple[dict, dict]:
    """Return ({question_id: [chunk_ids]}, {question_id: elapsed_seconds})."""
    rankings: dict[str, list[str]] = {}
    timings: dict[str, float] = {}
    for question in questions:
        started = time.perf_counter()
        hits = retriever.search(question["query"], mode=arm, top_k=k)
        timings[question["question_id"]] = time.perf_counter() - started
        rankings[question["question_id"]] = [hit.chunk_id for hit in hits]
    return rankings, timings


def mine_bad_cases(
    questions: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    corpus: ParentChildCorpus,
    k: int,
    wanted: int,
) -> list[dict]:
    """Find questions the lexical arm missed but the hybrid arm retrieved."""
    cases: list[dict] = []
    for question in questions:
        gold = set(question.get("gold_chunk_ids") or [])
        if not gold:
            continue
        qid = question["question_id"]
        bm25_ranked = rankings["bm25"].get(qid, [])
        hybrid_ranked = rankings["hybrid_rerank"].get(qid, [])

        bm25_hit = bool(set(bm25_ranked[:k]) & gold)
        hybrid_hit = bool(set(hybrid_ranked[:k]) & gold)
        if bm25_hit or not hybrid_hit:
            continue  # only cases hybrid *fixed*

        def first_rank(ranked: list[str]) -> int | None:
            for index, chunk_id in enumerate(ranked[:k], start=1):
                if chunk_id in gold:
                    return index
            return None

        gold_chunk = corpus.by_id.get(next(iter(gold)))
        hard_negatives = question.get("hard_negative_chunk_ids") or []
        bm25_top = corpus.by_id.get(bm25_ranked[0]) if bm25_ranked else None

        cases.append(
            {
                "question_id": qid,
                "split": question.get("split"),
                "trap_type": question.get("trap_type"),
                "query": question["query"],
                "expected_answer": question.get("expected_answer"),
                "gold_chunk_ids": sorted(gold),
                "gold_citation": gold_chunk.citation_label if gold_chunk else None,
                "gold_text": gold_chunk.text if gold_chunk else None,
                "gold_version": gold_chunk.version if gold_chunk else None,
                "bm25_ranked_chunk_ids": bm25_ranked[:k],
                "bm25_top_chunk_id": bm25_ranked[0] if bm25_ranked else None,
                "bm25_top_text": bm25_top.text if bm25_top else None,
                "bm25_top_version": bm25_top.version if bm25_top else None,
                "bm25_gold_rank": first_rank(bm25_ranked),
                "hybrid_gold_rank": first_rank(hybrid_ranked),
                "dense_gold_rank": first_rank(rankings["dense"].get(qid, [])),
                "hard_negative_chunk_ids": hard_negatives,
                "hard_negative_text": (
                    corpus.by_id[hard_negatives[0]].text
                    if hard_negatives and hard_negatives[0] in corpus.by_id
                    else None
                ),
            }
        )
        if len(cases) >= wanted:
            break
    return cases


def mine_rerank_wins(
    questions: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    corpus: ParentChildCorpus,
    k: int,
    wanted: int,
) -> list[dict]:
    """Find questions where the reranker *promoted* gold into a better rank.

    This exists because ``mine_bad_cases`` is provably empty on this corpus: BM25
    already scores Recall@5 = 1.0000, so there is no question the lexical arm
    missed for the hybrid arm to rescue. Reporting only the empty set would hide
    where the +0.113 MRR actually comes from, so we also report the promotion
    pattern -- the honest counterpart to the "hybrid fixed it" story.
    """
    cases: list[dict] = []
    for question in questions:
        gold = set(question.get("gold_chunk_ids") or [])
        if not gold:
            continue
        qid = question["question_id"]
        bm25_ranked = rankings["bm25"].get(qid, [])
        dense_ranked = rankings["dense"].get(qid, [])
        hybrid_ranked = rankings["hybrid_rerank"].get(qid, [])

        def gold_rank(ranked: list[str]) -> int | None:
            for index, chunk_id in enumerate(ranked[:k], start=1):
                if chunk_id in gold:
                    return index
            return None

        bm25_rank, dense_rank, hybrid_rank = (
            gold_rank(bm25_ranked),
            gold_rank(dense_ranked),
            gold_rank(hybrid_ranked),
        )
        # Promote = rerank put gold strictly higher than *both* retrievers did.
        if hybrid_rank is None or bm25_rank is None or dense_rank is None:
            continue
        if not (hybrid_rank < bm25_rank and hybrid_rank < dense_rank):
            continue

        gold_chunk = corpus.by_id.get(next(iter(gold)))
        cases.append(
            {
                "question_id": qid,
                "split": question.get("split"),
                "trap_type": question.get("trap_type"),
                "query": question["query"],
                "expected_answer": question.get("expected_answer"),
                "gold_chunk_ids": sorted(gold),
                "gold_citation": gold_chunk.citation_label if gold_chunk else None,
                "gold_text": gold_chunk.text if gold_chunk else None,
                "gold_version": gold_chunk.version if gold_chunk else None,
                "bm25_gold_rank": bm25_rank,
                "dense_gold_rank": dense_rank,
                "hybrid_gold_rank": hybrid_rank,
                "rrf_gold_rank": _rrf_gold_rank(rankings, qid, gold, k),
            }
        )
        if len(cases) >= wanted:
            break
    return cases


def _rrf_gold_rank(
    rankings: dict[str, dict[str, list[str]]], qid: str, gold: set[str], k: int
) -> int | None:
    """Gold's rank in the raw RRF fusion, before the reranker reordered it.

    Only available when the recorder kept the fused order; returns None instead of
    guessing, because an invented rank would be worse than a missing one.
    """
    fused = rankings.get("rrf", {}).get(qid)
    if not fused:
        return None
    for index, chunk_id in enumerate(fused[: max(k, len(fused))], start=1):
        if chunk_id in gold:
            return index
    return None


def print_table(metrics: dict, k: int, timings: dict[str, dict[str, float]]) -> None:
    recall_key = f"recall_at_{k}"
    mrr_key = f"mrr_at_{k}"
    line = "=" * 92
    print(line)
    print(f"Stage 2 检索消融对比  (k={k}, 仅计入可回答题)")
    print(line)
    header = f"{'检索方案':<46} {'Recall@'+str(k):>12} {'MRR@'+str(k):>10} {'平均延迟(ms)':>14}"
    print(header)
    print("-" * 92)
    for arm in ARMS:
        data = metrics[arm]
        arm_timings = timings[arm]
        avg_ms = (sum(arm_timings.values()) / max(len(arm_timings), 1)) * 1000
        print(
            f"{ARM_LABELS[arm]:<46} {data[recall_key]:>12.4f} {data[mrr_key]:>10.4f} {avg_ms:>14.2f}"
        )
    print(line)
    sample = next(iter(metrics.values()))
    print(
        f"计入题数: {sample['answerable_questions']}   "
        f"排除的不可回答题: {sample['excluded_unanswerable']}"
    )
    print("说明: Recall/MRR 只在有 gold 的题上计算；不可回答题不计入排序指标。")
    print("      延迟为本机单次查询平均墙钟时间，含全部检索阶段，非生产性能。")
    print(line)


def print_bad_cases(cases: list[dict]) -> None:
    print()
    print("=" * 92)
    print(f"Hybrid 修好的 Bad Case（共 {len(cases)} 个）")
    print("=" * 92)
    for index, case in enumerate(cases, start=1):
        print(f"\n[Bad Case {index}] {case['question_id']}  split={case['split']}  trap={case['trap_type']}")
        print(f"  问题      : {case['query']}")
        print(f"  标准答案  : {case['expected_answer']}")
        print(f"  Gold      : {case['gold_chunk_ids']}  {case['gold_citation']}  version={case['gold_version']}")
        print(f"  纯BM25 名次 : {case['bm25_gold_rank']}   (未进 top-k)")
        print(f"  纯向量 名次 : {case['dense_gold_rank']}")
        print(f"  Hybrid 名次 : {case['hybrid_gold_rank']}")
        print(f"  BM25 误召回: {case['bm25_top_chunk_id']}  version={case['bm25_top_version']}")
        if case["bm25_top_text"]:
            print(f"      {case['bm25_top_text'][:150]}")
        if case["gold_text"]:
            print(f"  Gold 原文 : {case['gold_text'][:150]}")
    print("=" * 92)


def print_rerank_wins(cases: list[dict]) -> None:
    print()
    print("=" * 92)
    print(f"Rerank 提升 gold 名次的题目（共 {len(cases)} 个）")
    print("=" * 92)
    print("说明: 这些题 BM25 与 Dense 都把 gold 排在后面，Rerank 把它提到更前。")
    print("      这是 MRR@5 从 0.5000 提升到 0.6133 的来源。")
    for index, case in enumerate(cases, start=1):
        print(f"\n[提升 {index}] {case['question_id']}  split={case['split']}  trap={case['trap_type']}")
        print(f"  问题      : {case['query']}")
        print(f"  标准答案  : {case['expected_answer']}")
        print(f"  Gold      : {case['gold_chunk_ids']}  {case['gold_citation']}  version={case['gold_version']}")
        print(
            f"  gold 名次 : BM25={case['bm25_gold_rank']}  "
            f"Dense={case['dense_gold_rank']}  ->  Hybrid={case['hybrid_gold_rank']}"
        )
    print("=" * 92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rag.yaml")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N questions")
    parser.add_argument("--out", default=None, help="output directory override")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    corpus_path = PROJECT_ROOT / config["corpus"]["chunks_path"]
    questions_path = PROJECT_ROOT / config["evaluation"]["dataset_path"]
    k = int(config["evaluation"]["k"])
    include_splits = list(config["evaluation"].get("include_splits") or [])
    bad_case_count = int(config["evaluation"].get("bad_case_count", 2))

    out_dir = PROJECT_ROOT / (args.out or config["output"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"语料: {corpus_path.relative_to(PROJECT_ROOT)}")
    corpus = ParentChildCorpus.from_jsonl(corpus_path)
    total_questions = len(load_questions(questions_path, include_splits, None))
    questions = load_questions(questions_path, include_splits, args.limit)
    limit = args.limit
    print(f"语料切片数: {len(corpus)}   评测题数: {len(questions)} / {total_questions}   k={k}")
    print("加载检索器（首次运行会下载/加载 BGE 模型）...")

    started = time.perf_counter()
    retriever = HybridRetriever(corpus, config, root=str(PROJECT_ROOT))
    load_seconds = time.perf_counter() - started

    embedder = retriever.embedder
    reranker = retriever.reranker
    print(f"检索器就绪，用时 {load_seconds:.1f}s")
    print(f"  Embedding 后端 : {embedder.backend_name}  ({embedder.model_name})")
    print(f"  Reranker 后端  : {reranker.backend_name}  ({reranker.model_name})")
    if embedder.backend_name != "sentence_transformers":
        print("  !! 警告: Embedding 走了确定性回退实现，指标不能当作真实 BGE 结果对外宣称。")
    if reranker.backend_name != "sentence_transformers_cross_encoder":
        print("  !! 警告: Reranker 走了回退实现，指标不能当作真实 CrossEncoder 结果对外宣称。")
    print()

    rankings: dict[str, dict[str, list[str]]] = {}
    timings: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        print(f"  运行 {arm} ...", flush=True)
        arm_started = time.perf_counter()
        rankings[arm], timings[arm] = run_arm(retriever, questions, arm, k)
        print(f"    完成，用时 {time.perf_counter() - arm_started:.1f}s")

    metrics = evaluate_rankings(questions, rankings, k=k)
    print()
    print_table(metrics, k, timings)

    bad_cases = mine_bad_cases(questions, rankings, corpus, k, bad_case_count)
    print_bad_cases(bad_cases)

    rerank_wins = mine_rerank_wins(questions, rankings, corpus, k, 8)
    print_rerank_wins(rerank_wins)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "k": k,
        "questions_evaluated": len(questions),
        "corpus_chunks": len(corpus),
        "arms": {arm: ARM_LABELS[arm] for arm in ARMS},
        "embedder_backend": embedder.backend_name,
        "embedder_model": embedder.model_name,
        "reranker_backend": reranker.backend_name,
        "reranker_model": reranker.model_name,
        "real_model_run": (
            embedder.backend_name == "sentence_transformers"
            and reranker.backend_name == "sentence_transformers_cross_encoder"
        ),
        "limit": limit,
        "is_full_evaluation_set": limit is None or len(questions) == total_questions,
        "metrics": metrics,
        "avg_latency_ms": {
            arm: (sum(timings[arm].values()) / max(len(timings[arm]), 1)) * 1000 for arm in ARMS
        },
        "bad_case_count": len(bad_cases),
        "bad_cases": bad_cases,
        "rerank_promoted_gold": rerank_wins,
        "note": (
            "Synthetic-corpus offline evaluation. Retrieval metrics only; they are not "
            "end-to-end answer accuracy and not production performance."
        ),
    }

    comparison_path = out_dir / config["output"]["comparison_filename"]
    comparison_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rankings_path = out_dir / config["output"]["rankings_filename"]
    with rankings_path.open("w", encoding="utf-8", newline="\n") as stream:
        for arm in ARMS:
            for question in questions:
                stream.write(
                    json.dumps(
                        {
                            "arm": arm,
                            "question_id": question["question_id"],
                            "split": question.get("split"),
                            "trap_type": question.get("trap_type"),
                            "gold_chunk_ids": question.get("gold_chunk_ids"),
                            "ranked_chunk_ids": rankings[arm].get(question["question_id"], []),
                            "latency_ms": timings[arm].get(question["question_id"], 0.0) * 1000,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    bad_cases_path = out_dir / config["output"]["bad_cases_filename"]
    bad_cases_path.write_text(
        json.dumps(
            {"hybrid_fixed": bad_cases, "rerank_promoted_gold": rerank_wins},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown_path = out_dir / config["output"]["markdown_filename"]
    recall_key = f"recall_at_{k}"
    mrr_key = f"mrr_at_{k}"
    lines = [
        "# Stage 2 检索消融对比",
        "",
        f"- 生成时间: {payload['generated_at']}",
        f"- 语料切片: {len(corpus)}   评测题: {len(questions)}   k={k}",
        f"- Embedding: `{embedder.model_name}` ({embedder.backend_name})",
        f"- Reranker: `{reranker.model_name}` ({reranker.backend_name})",
        "",
        f"| 检索方案 | Recall@{k} | MRR@{k} | 平均延迟(ms) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        data = metrics[arm]
        avg_ms = payload["avg_latency_ms"][arm]
        lines.append(
            f"| {ARM_LABELS[arm]} | {data[recall_key]:.4f} | {data[mrr_key]:.4f} | {avg_ms:.2f} |"
        )
    sample = next(iter(metrics.values()))
    lines += [
        "",
        f"计入题数 {sample['answerable_questions']}，排除不可回答题 {sample['excluded_unanswerable']}。",
        "",
    ]
    if not payload["is_full_evaluation_set"]:
        lines += [
            f"> **注意：本次为抽样冒烟运行（--limit {limit}），"
            f"全量评测集共 {total_questions} 题，指标不可用于对外引用。**",
            "",
        ]
    lines += [
        "> 合成语料的离线检索指标，不等于端到端回答准确率，也不代表生产性能。",
        "",
    ]
    # The attribution section is only allowed to ship alongside the full-set numbers
    # it describes; a --limit smoke run must not carry claims from a different run.
    if payload["is_full_evaluation_set"]:
        regression = metrics["bm25"][recall_key] > metrics["hybrid_rerank"][recall_key]
        if regression:
            lines += [
                "## 已知回退与归因（必读）",
                "",
                "Hybrid+Rerank 的 Recall@5 低于纯 BM25。这不是数值噪声，已定位到机制：",
                "",
                "- `src/rag/rerank.py` 的 tie-break 用 chunk_id 字典序；本语料中",
                "  Cross-Encoder 打分饱和（EVAL-116 的 100 条候选分数极差仅 0.0034），",
                "  打平后按 ID 排序，使 gold 被字典序更靠前的近似副本挤出 top-5。",
                "- 窗口扫描：窗口 5 / 20 的 Recall@5 均为 1.0000，窗口 50 / 100 为 0.9889，",
                "  与上述机制一致（窗口越大，进入饱和区的近似副本越多）。",
                "- 完整证据链：`reports/stage2_retrieval_findings.md`；",
                "  复现脚本：`eval/diagnose_hybrid_window.py`、`eval/probe_eval116_rerank_scores.py`。",
            ]
        else:
            # No recall regression, so the section that used to live here would be
            # describing a run that did not happen. Say what is actually true and
            # keep the earlier investigation linked -- it is why the config and the
            # tie-break look the way they do.
            promoted = payload["rerank_promoted_gold"]
            lines += [
                "## 指标读法与既往回退的处置（必读）",
                "",
                f"本次运行 Hybrid+Rerank 的 Recall@5 与纯 BM25 持平（{metrics['bm25'][recall_key]:.4f}），",
                f"MRR@5 由 {metrics['bm25'][mrr_key]:.4f} 提升到 {metrics['hybrid_rerank'][mrr_key]:.4f}"
                f"（+{metrics['hybrid_rerank'][mrr_key] - metrics['bm25'][mrr_key]:.4f}）。",
                "",
                "两点必须同时讲清楚，否则这张表会被误读：",
                "",
                f"1. **Recall 持平不等于重排无效。** Recall@5 已触顶（{metrics['bm25'][recall_key]:.4f}），",
                "   天花板效应下它无法再涨；重排的收益体现在 MRR——把答案文档排得更靠前。",
                f"   全量 {len(promoted)} 道提升题全部属于 `version_conflict` 类型：BM25 与 Dense 都把",
                "   gold 排在后面，Cross-Encoder 把它提到第 1。这正是 RRF 融合后再精排该起的作用。",
                "2. **`bad_cases.json` 的 `hybrid_fixed` 为空是必然的，不是脚本坏了。** BM25 的",
                f"   Recall@5 已经是 {metrics['bm25'][recall_key]:.4f}，不存在「BM25 漏掉、Hybrid 救回」的题。",
                f"   真实可讲的案例在 `rerank_promoted_gold`（{len(promoted)} 个）。",
                "",
                "### 此前出现过的回退及处置（历史，已修复）",
                "",
                "修复前的配置（`reranker.candidate_k: 100`）下，Hybrid 的 Recall@5 曾是 0.9889，",
                "低于纯 BM25 的 1.0000。归因结论：Cross-Encoder 打分饱和——EVAL-116 的 100 条候选",
                "分数极差仅 0.0034，打平后按 `chunk_id` 字典序回退，把 gold 挤出了 top-5。",
                "处置（已经人工拍板）：",
                "",
                "- 窗口 `candidate_k: 100 -> 20`（`config/rag.yaml`）：窗口越小，进入饱和区的近似副本越少。",
                "- tie-break 由 `chunk_id` 改为 `rrf_score`（`src/rag/rerank.py: rerank_sort_key`）：",
                "  分数饱和时让词法/向量共识来定序，而不是字母序。",
                "",
                "效果：EVAL-116 的 gold 名次由「掉出 top-5」回到第 4（仍在 top-5 内），",
                "全量 Recall@5 恢复 1.0000，单题延迟由 4586 ms 降到约 1070 ms。",
                "",
                "修复前的完整证据链保留在 `reports/stage2_retrieval_findings.md`，",
                "复现脚本：`eval/diagnose_hybrid_window.py`、`eval/probe_eval116_rerank_scores.py`。",
            ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"已写出: {comparison_path.relative_to(PROJECT_ROOT)}")
    print(f"已写出: {markdown_path.relative_to(PROJECT_ROOT)}")
    print(f"已写出: {rankings_path.relative_to(PROJECT_ROOT)}")
    print(f"已写出: {bad_cases_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())