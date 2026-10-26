"""复现用户反馈：点「开始分析（同时跑 RAG 与 Agent）」之后到底渲染了什么。

用法（在 FinOps-Copilot 目录下）：
    .venv/Scripts/python.exe app/probe_combined.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def dump(app: AppTest, label: str) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"exception      : {[e.value for e in app.exception]}")
    print(f"error          : {[e.value for e in app.error]}")
    print(f"warning        : {[w.value for w in app.warning]}")
    print(f"success 条数   : {len(app.success)}")
    for s in app.success:
        print(f"  success: {s.value[:200]!r}")
    print(f"info 条数      : {len(app.info)}")
    for i in app.info:
        print(f"  info   : {i.value[:120]!r}")
    print(f"markdown 条数  : {len(app.markdown)}")
    for m in app.markdown:
        v = m.value
        if any(k in v for k in ("最终答案", "命中", "路由", "答案", "错误", "回答")):
            print(f"  markdown: {v[:160]!r}")

    rag = app.session_state.get("rag_result")
    if rag:
        print(f"RAG   : 命中 {len(rag['hits'])} 条 | 耗时 {rag['elapsed']:.2f}s")
        print(f"        排名 1 引用: {rag['hits'][0].chunk.citation_label}")
        print(f"        排名 1 正文: {rag['hits'][0].chunk.text[:90]!r}")
    else:
        print("RAG   : session_state 里没有 rag_result")

    agent = app.session_state.get("agent_result")
    if agent:
        st = agent["state"]
        print(f"Agent : route={st.route} | rows={len(st.result.rows) if st.result else 0} "
              f"| retries={getattr(st, 'retries', '?')}")
        print(f"        问题={st.question!r}")
        print(f"        答案={st.answer!r}")
    else:
        print("Agent : session_state 里没有 agent_result")


def find_button_by_label(app: AppTest, needle: str):
    for b in app.button:
        if needle in b.label:
            return b
    return None


def main() -> int:
    app = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=1200)
    app.run()
    print("首页渲染完成")
    combined = find_button_by_label(app, "开始分析")
    if combined is None:
        print("!! 找不到「开始分析」按钮")
        return 1

    scenarios = [
        ("A：知识库预设题（只点预设，不点开始分析）", "EVAL-001"),
        ("B：指标预设题（只点预设）", "各品类的净营收"),
        ("C：删掉所有订单（只点预设）", "删掉所有订单"),
    ]
    for label, needle in scenarios:
        btn = find_button_by_label(app, needle)
        print(f"\n>>> 只点预设题：{btn.label[:44] if btn else '(没找到 ' + needle + ')'}")
        if btn is None:
            continue
        btn.click().run()
        dump(app, f"场景 {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())