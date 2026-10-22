"""用 Streamlit 官方 AppTest 在真实脚本运行时里跑一遍 UI，验证无异常。

用法（在 FinOps-Copilot 目录下）：
    .venv/Scripts/python.exe app/apptest_check.py

它会：渲染首页 → 点一个指标预设题 → 跑 Agent → 切到纯 BM25 → 点一个知识库预设题
→ 跑检索，并断言全程 ``AppTest.exception`` 为空、答案/命中非空。
首次运行要真实加载检索器（约 1-3 分钟）。
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def find_button(app: AppTest, key: str):
    for button in app.button:
        if getattr(button, "key", None) == key:
            return button
    raise AssertionError(f"button key={key} not found; keys={[b.key for b in app.button]}")


def check(app: AppTest, step: str) -> None:
    if len(app.exception):
        for element in app.exception:
            print(f"  !! {step} 抛出异常: {element.value}")
        raise SystemExit(f"APPTEST FAILED at {step}")
    print(f"  ok: {step}")


def main() -> int:
    app = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=900)
    app.run()
    check(app, "首页渲染")
    print(f"  标题: {app.title[0].value if app.title else '(无)'} | 按钮数: {len(app.button)}")

    app.radio[0].set_value("bm25").run()
    check(app, "侧边栏切换检索模式 -> bm25")

    find_button(app, "example_metric_0").click().run()
    check(app, "点击指标预设题")
    find_button(app, "run_agent_only").click().run()
    check(app, "运行 Agent")
    agent_state = app.session_state["agent_result"]["state"]
    print(
        f"  Agent: route={agent_state.route} | rows={len(agent_state.result.rows) if agent_state.result else 0} "
        f"| answer={agent_state.answer[:50]!r}"
    )
    if not agent_state.answer:
        raise SystemExit("APPTEST FAILED: Agent 答案为空")

    find_button(app, "example_rag_0").click().run()
    check(app, "点击知识库预设题")
    find_button(app, "run_rag_only").click().run()
    check(app, "运行 RAG 检索（含首次模型加载）")
    rag_result = app.session_state["rag_result"]
    print(f"  RAG: 命中 {len(rag_result['hits'])} 条 | 耗时 {rag_result['elapsed']:.2f}s")
    if not rag_result["hits"]:
        raise SystemExit("APPTEST FAILED: RAG 无命中")

    if app.warning:
        print(f"  页面 warning: {[w.value for w in app.warning]}")
    print("\nAPPTEST OK: UI 全程无异常，Agent 与 RAG 两条链路都出结果")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())