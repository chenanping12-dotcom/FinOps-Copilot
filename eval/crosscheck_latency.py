# -*- coding: utf-8 -*-
"""Measure hybrid_rerank latency at window=20 vs window=100 in one process.

Purpose: the ablation run reported 771 ms/query at window=20 while the window
sweep reported 1069 ms/query at the same window. Same model, same corpus, same
90 answerable questions. This script isolates whether the gap is (a) a real
warm-up/caching effect, or (b) a stale ablation run.

Method: load the retriever once, sweep windows in increasing order, and between
sweeps run a fixed warm-up pass so both windows see an equally warm reranker.
Report per-window median and mean latency, plus the first-10 vs last-10 spread
that would reveal warm-up contamination.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402
from rag.corpus import ParentChildCorpus  # noqa: E402
from rag.hybrid import HybridRetriever  # noqa: E402


def load_questions(path: Path, splits: list[str]) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("split") in splits:
            rows.append(row)
    return rows


def measure(retriever, questions, window: int, k: int = 5) -> dict:
    # Force the rerank window for this measurement via the documented per-call
    # override on search_hybrid_from_candidates (search_hybrid has no such knob).
    latencies = []
    candidate_k = retriever.candidate_k
    for q in questions:
        started = time.perf_counter()
        bm25_hits = retriever.bm25.search(q["query"], top_k=candidate_k)
        dense_hits = retriever.dense.search(q["query"], top_k=candidate_k)
        retriever.search_hybrid_from_candidates(
            q["query"], bm25_hits, dense_hits, top_k=k, rerank_candidate_k=window
        )
        latencies.append((time.perf_counter() - started) * 1000)

    ordered = sorted(latencies)
    return {
        "window": window,
        "n": len(latencies),
        "mean_ms": round(statistics.fmean(latencies), 1),
        "median_ms": round(statistics.median(latencies), 1),
        "p90_ms": round(ordered[int(len(ordered) * 0.9) - 1], 1),
        "first10_mean_ms": round(statistics.fmean(latencies[:10]), 1),
        "last10_mean_ms": round(statistics.fmean(latencies[-10:]), 1),
    }


def main() -> int:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "rag.yaml").read_text(encoding="utf-8"))
    corpus_path = PROJECT_ROOT / config["corpus"]["chunks_path"]
    dataset_path = PROJECT_ROOT / config["evaluation"]["dataset_path"]
    splits = config["evaluation"]["include_splits"]

    corpus = ParentChildCorpus.from_jsonl(corpus_path)
    questions = [q for q in load_questions(dataset_path, splits) if q.get("gold_chunk_ids")]
    print(f"corpus={len(corpus)}  answerable questions={len(questions)}")

    # Match run_ablation.py exactly: it passes config["retrieval"], not the whole
    # file, plus root=PROJECT_ROOT. Passing the whole config would make
    # HybridRetriever look for retrieval.* at the top level and silently fall
    # back to a hashed-ngram embedder.
    retrieval_config = {"retrieval": config["retrieval"]}
    retriever = HybridRetriever(corpus, retrieval_config, root=str(PROJECT_ROOT))
    print(f"embedder={retriever.embedder.backend_name}  reranker={retriever.reranker.backend_name}")
    if retriever.embedder.backend_name != "sentence_transformers":
        print("!! embedder fell back; aborting so the numbers are not misread")
        return 1

    # Warm-up: exercise both window widths once so neither sweep pays first-touch cost.
    print("warm-up ...")
    for w in (20, 100):
        measure(retriever, questions[:5], w)

    results = []
    # Measure each window twice, alternating order, to expose ordering effects.
    for label, windows in (("pass-A", (20, 100)), ("pass-B", (100, 20))):
        for w in windows:
            m = measure(retriever, questions, w)
            m["pass"] = label
            results.append(m)
            print(
                f"  {label} window={w:3d}  mean={m['mean_ms']:8.1f}  median={m['median_ms']:8.1f}  "
                f"p90={m['p90_ms']:8.1f}  first10={m['first10_mean_ms']:8.1f}  last10={m['last10_mean_ms']:8.1f}"
            )

    out = PROJECT_ROOT / "eval" / "reports" / "rag" / "latency_window_crosscheck.json"
    out.write_text(
        json.dumps(
            {
                "purpose": "reconcile ablation.json hybrid latency vs window_diagnosis.json sweep",
                "answerable_questions": len(questions),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())