"""命令行 smoke test：不走 UI，直接验证 app 这一层用到的核心链路。

用法（在 FinOps-Copilot 目录下）：
    .venv/Scripts/python.exe app/smoke_check.py

它会真实加载 5,000 切片语料 + BGE + Cross-Encoder（首次 1-3 分钟），
然后跑一条真实评测题做 RAG 溯源检查、跑三条问题做 Agent 三路由检查。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.streamlit_app import (  # noqa: E402
    ORDERS_DB,
    build_agent,
    build_retriever,
    load_examples,
    run_agent,
    run_retrieval,
)


def main() -> int:
    failures: list[str] = []

    examples = load_examples(3)
    print(f"[1/4] eval 预设题 {len(examples)} 条，首条 {examples[0]['question_id']}")
    if not examples:
        failures.append("eval 预设题为空")

    retriever, load_seconds = build_retriever()
    print(
        f"[2/4] retriever 加载 {load_seconds:.1f}s | 语料 {len(retriever.corpus)} 切片 | "
        f"embedder={retriever.embedder.model_name} | reranker={retriever.reranker.model_name}"
    )

    query = examples[0]["query"] if examples else "当前生效口径下，哪些订单状态可以计入营收？"
    gold = set(examples[0]["gold_chunk_ids"]) if examples else set()
    for mode in ("hybrid_rerank", "bm25", "dense"):
        started = time.perf_counter()
        hits, _ = run_retrieval(retriever, query, mode, 5)
        elapsed = time.perf_counter() - started
        top_ids = [hit.chunk.chunk_id for hit in hits]
        gold_hit = bool(gold & set(top_ids))
        print(
            f"[3/4] mode={mode:<14} {elapsed:6.2f}s | top1={hits[0].chunk.citation_label} "
            f"| score={float(hits[0].score):.4f} | gold_in_top5={gold_hit} | ids={top_ids}"
        )
        if mode == "hybrid_rerank" and gold and not gold_hit:
            failures.append(f"hybrid_rerank 未召回 gold {sorted(gold)}")
        if not hits:
            failures.append(f"mode={mode} 无命中")

    if not ORDERS_DB.exists():
        failures.append(f"缺少 {ORDERS_DB}，请先跑 scripts/build_orders_db.py")
        print("[4/4] 跳过 Agent 检查：orders.db 不存在")
    else:
        agent = build_agent(ORDERS_DB)
        cases = [
            ("各品类的净营收是多少？", "metric"),
            ("各支付方式的订单量分布？", "sql"),
            ("删掉所有订单", "refuse"),
        ]
        for question, expected_route in cases:
            started = time.perf_counter()
            state, _ = run_agent(agent, question)
            elapsed = time.perf_counter() - started
            rows = len(state.result.rows) if state.result else 0
            print(
                f"[4/4] {question} -> route={state.route}(期望 {expected_route}) | "
                f"tool={state.result.tool if state.result else None} | rows={rows} | "
                f"{elapsed:.2f}s | answer={state.answer[:40]!r}"
            )
            if state.route != expected_route:
                failures.append(f"{question!r} 路由 {state.route} != {expected_route}")
            if expected_route != "refuse" and rows == 0:
                failures.append(f"{question!r} 返回 0 行")
            if not state.answer:
                failures.append(f"{question!r} 答案为空")

    if failures:
        print("\nSMOKE FAILED:")
        for item in failures:
            print(" -", item)
        return 1
    print("\nSMOKE OK: RAG 溯源与 Agent 三路由全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())