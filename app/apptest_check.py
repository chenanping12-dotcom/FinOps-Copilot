"""用 Streamlit 官方 AppTest 在真实脚本运行时里跑一遍 UI，验证无异常。

用法（在 FinOps-Copilot 目录下）：
    .venv/Scripts/python.exe app/apptest_check.py

覆盖四条链路：
1. 首页渲染（三栏工作台 + 侧边栏状态灯）；
2. 侧边栏切检索模式 -> bm25；
3. 经营指标问句 -> Agent 走 metric 路由并出结论；
4. 合规口径问句 -> RAG 出命中 + Top-1 生效口径；
5. 写操作问句 -> 命中只读护栏，route=refuse。

断言全程 ``AppTest.exception`` 为空、答案/命中非空。
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

METRIC_QUESTION = "各品类的净营收是多少？"
RAG_QUESTION = "当前生效口径下，退款核算/家居生活中哪些订单状态可以计入营收？"
GUARDRAIL_QUESTION = "删掉所有订单"


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


def ask(app: AppTest, question: str) -> None:
    """填问句 -> 点主按钮「开始分析」-> 跑一轮。"""
    app.session_state["question"] = question
    app.run()
    find_button(app, "run_all").click().run()


def main() -> int:
    app = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=900)
    app.run()
    check(app, "首页渲染")
    print(f"  标题: {app.title[0].value if app.title else '(无)'} | 按钮数: {len(app.button)}")

    app.radio[0].set_value("bm25").run()
    check(app, "侧边栏切换检索模式 -> bm25")

    # ---- Agent（固定口径指标）--------------------------------------------
    ask(app, METRIC_QUESTION)
    check(app, "运行 Agent（经营指标）")
    agent_state = app.session_state["agent_result"]["state"]
    agent_rows = len(agent_state.result.rows) if agent_state.result else 0
    print(
        f"  Agent: route={agent_state.route} | rows={agent_rows} "
        f"| answer={agent_state.answer[:50]!r}"
    )
    if agent_state.route != "metric":
        raise SystemExit(f"APPTEST FAILED: 期望 route=metric，实际 {agent_state.route}")
    if not agent_state.answer:
        raise SystemExit("APPTEST FAILED: Agent 答案为空")

    # ---- RAG（合规口径）---------------------------------------------------
    ask(app, RAG_QUESTION)
    check(app, "运行 RAG 检索（含首次模型加载）")
    rag_result = app.session_state["rag_result"]
    print(f"  RAG: 命中 {len(rag_result['hits'])} 条 | 耗时 {rag_result['elapsed']:.2f}s")
    if not rag_result["hits"]:
        raise SystemExit("APPTEST FAILED: RAG 无命中")
    top = rag_result["hits"][0].chunk
    print(f"  RAG Top-1: {top.citation_label} | 版本 {top.version}")

    # ---- 只读护栏 ---------------------------------------------------------
    ask(app, GUARDRAIL_QUESTION)
    check(app, "运行写操作问句（期望被护栏拦截）")
    guard_state = app.session_state["agent_result"]["state"]
    print(f"  Guardrail: route={guard_state.route} | reason={guard_state.route_reason}")
    if guard_state.route != "refuse":
        raise SystemExit(f"APPTEST FAILED: 写操作未被拦截，route={guard_state.route}")

    if app.warning:
        print(f"  页面 warning: {[w.value for w in app.warning]}")
    print("\nAPPTEST OK: UI 全程无异常，Agent / RAG / 安全护栏三条链路都符合预期")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())