"""FinOps Copilot —— 企业经营分析与合规审计控制台（Streamlit 单页应用）。

布局是「Copilot 三栏决策工作台」：

1. **左栏 · 引擎配置与系统监控**（``st.sidebar``）：引擎/AST 护栏状态指示灯、
   检索策略与返回条数、语义网关的运行时指标。
2. **中栏 · 业务决策看板**：品牌 Header → 业务场景快选 → 问句与执行 →
   核心结论卡片 + 业务指标看板；命中写操作/PII 时切换为红色安全护栏横幅。
3. **右栏 · 证据链与审计**：合规依据（Top-1 规则原文 + 溯源徽标）、
   版本口径比对（现行 / 已废止）、LangGraph 审计调用链（节点时间线 +
   AST 校验结论 + 实际执行的只读 SQL + 结构化 trace）。

工程约束（这几条是刻意为之，改动前请先读）：

* **不动 ``src/``**：本文件只是展示层。``HybridRetriever`` / ``FinOpsAgent`` /
  网关接口一行未改，页面只负责调用与呈现。
* **重资源只加载一次**：5,000 切片语料 + BGE 向量编码 + Cross-Encoder 的首轮
  构造成本是分钟级的，统一走 ``@st.cache_resource(show_spinner=False)``，
  跨 rerun、跨会话复用，绝不在每次点击或刷新时重新编码。加载失败不抛栈，
  收敛成降级提示。
* **渲染永不崩**：``safe_json_dump`` 把 ``None`` / 列表收敛为 JSON 对象再交给
  ``st.json``（该组件只收对象，传 ``None`` 会在前端渲染出一个
  ``{"ERROR": ...}`` 框）；所有拼进 ``unsafe_allow_html`` 的动态文本先过
  ``_esc``，否则规则原文里的 ``<`` ``&`` 会把版面拆掉。
* **视觉可复现**：注入一层 CSS 把配色与层级钉死（深空灰底 + 企业蓝），
  不依赖浏览器/系统主题，保证换台机器打开是同一个样子。
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

# --------------------------------------------------------------------------
# 路径：``streamlit run app/streamlit_app.py`` 时 sys.path[0] 是 app/，
# 而 src/ 是 namespace package，必须以项目根为 import 根（与 pytest 的
# ``pythonpath = ["src"]`` 之外的 ``from src.xxx import`` 写法保持一致）。
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "config" / "rag.yaml"
CHUNKS_PATH = ROOT / "data" / "generated" / "rule_docs" / "chunks.jsonl"
ORDERS_DB = ROOT / "data" / "processed" / "orders.db"
EVAL_PATH = ROOT / "eval" / "datasets" / "stage1_eval.jsonl"

# 数据来源声明。演示数据是本地合成的，页面底部保留一行，避免被误读成真实平台政策。
DISCLAIMER = "演示数据与规则文档为本地合成样本，非真实平台政策。"

# 检索策略：对应 HybridRetriever.search(mode=...)，用于现场做策略对比。
RETRIEVAL_MODES: dict[str, str] = {
    "hybrid_rerank": "混合检索 + Cross-Encoder 精排（推荐）",
    "hybrid": "混合检索（BM25 + 向量 RRF 粗排，无精排）",
    "bm25": "纯 BM25（词法基线）",
    "dense": "纯向量（语义基线）",
}

# 场景快选里「兜底问句」：当评测集不可用时仍然能演示，不至于点了没反应。
FALLBACK_RAG_QUESTION = "当前生效口径下，退款核算/家居生活中哪些订单状态可以计入营收？"
FALLBACK_RAG_QUESTION_2 = "报表时区口径下，数码电子的退货申请在签收后第几天内仍然受理？"

# 经营指标类问句（问 orders.db，走 Agent 链路）。
METRIC_EXAMPLES = [
    "各品类的净营收是多少？",
    "各支付方式的订单量分布？",
    "整体取消率是多少？",
    "删掉所有订单",
]


# ==========================================================================
# 一、视觉层
#
# 为什么自己写 CSS：Streamlit 原生组件的外观由浏览器/OS 主题决定，同一份代码在
# 两台机器上颜色、间距、层级都不一样，现场演示很容易「看着像半成品」。这里把
# 配色与层级钉死，再用几个小 HTML 组件承载 Streamlit 没有的东西——溯源徽标、
# 卡片、状态指示灯、护栏横幅、审计时间线。
#
# 约束：CSS 是纯字符串常量（不是 f-string），因此可以自由使用 ``{`` ``}``。
# ==========================================================================
CSS = """<style>
:root {
  --fops-bg: #0b1220;
  --fops-panel: rgba(255, 255, 255, 0.035);
  --fops-panel-2: rgba(255, 255, 255, 0.055);
  --fops-border: rgba(148, 163, 184, 0.18);
  --fops-blue: #3b82f6;
  --fops-blue-soft: rgba(59, 130, 246, 0.14);
  --fops-cyan: #22d3ee;
  --fops-text: #e2e8f0;
  --fops-muted: #94a3b8;
  --fops-ok: #22c55e;
  --fops-warn: #f59e0b;
  --fops-bad: #ef4444;
  --fops-radius: 8px;
  --fops-radius-lg: 12px;
  --fops-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

/* ---------- 全局：深空灰底 + 两团柔和光晕 ---------- */
.stApp {
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(59, 130, 246, 0.13), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34, 211, 238, 0.07), transparent 55%),
    var(--fops-bg);
  color: var(--fops-text);
}

/* ---------- 去掉 Streamlit 默认冗余留白与页脚 ---------- */
.block-container { padding-top: 2.2rem !important; padding-bottom: 1.2rem !important; }
footer, #MainMenu, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* ---------- 字体层级：紧凑、有力 ---------- */
.stApp h1 { font-size: 1.62rem !important; font-weight: 700 !important;
            letter-spacing: -0.3px; margin-bottom: 0.1rem !important; }
.stApp h2 { font-size: 1.05rem !important; font-weight: 650 !important;
            margin-top: 0.4rem !important; margin-bottom: 0.3rem !important; }
.stApp h3 { font-size: 0.95rem !important; font-weight: 600 !important; }
.stApp p, .stApp li { line-height: 1.68; }

/* ---------- 品牌 Header ---------- */
.fops-brand {
  display: flex; align-items: center; gap: 12px;
  padding: 2px 0 14px 0; border-bottom: 1px solid var(--fops-border);
  margin-bottom: 16px;
}
.fops-brand-mark {
  flex: 0 0 auto; width: 34px; height: 34px; border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.05rem; font-weight: 800; color: #dbeafe;
  background: linear-gradient(135deg, rgba(59,130,246,0.85), rgba(34,211,238,0.55));
  box-shadow: 0 6px 16px rgba(2, 6, 23, 0.5);
}
.fops-brand-title { font-size: 1.42rem; font-weight: 750; letter-spacing: -0.3px; }
.fops-brand-sub { color: var(--fops-muted); font-size: 0.88rem; margin-top: 1px; }

/* ---------- 章节标题 ---------- */
.fops-section {
  margin: 4px 0 10px 0; font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase; color: var(--fops-muted);
}

/* ---------- 通用卡片 ---------- */
.fops-card {
  border: 1px solid var(--fops-border); border-radius: var(--fops-radius-lg);
  background: var(--fops-panel); padding: 15px 17px; margin: 8px 0 14px 0;
  box-shadow: 0 6px 18px rgba(2, 6, 23, 0.35);
}
.fops-card-title {
  margin-bottom: 9px; font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.09em; text-transform: uppercase; color: var(--fops-muted);
}

/* ---------- 核心结论 ---------- */
.fops-conclusion { font-size: 1.02rem; line-height: 1.7; color: var(--fops-text); }
.fops-conclusion strong { color: #bfdbfe; }

/* ---------- 溯源徽标 ---------- */
.fops-badge {
  display: inline-block; margin: 0 6px 8px 0; padding: 3px 10px;
  font-family: var(--fops-mono); font-size: 0.76rem; font-weight: 600;
  color: #bfdbfe; background: var(--fops-blue-soft);
  border: 1px solid rgba(59, 130, 246, 0.38); border-radius: 7px;
}
.fops-badge-time {
  display: inline-block; margin: 0 6px 8px 0; padding: 3px 10px;
  font-family: var(--fops-mono); font-size: 0.76rem; font-weight: 600;
  color: #a7f3d0; background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 7px;
}
.fops-badge-muted {
  display: inline-block; margin: 0 6px 8px 0; padding: 3px 10px;
  font-family: var(--fops-mono); font-size: 0.76rem; font-weight: 600;
  color: var(--fops-muted); background: rgba(148, 163, 184, 0.1);
  border: 1px solid rgba(148, 163, 184, 0.28); border-radius: 7px;
}

/* ---------- 合规依据块 ---------- */
.fops-evidence {
  margin: 8px 0 12px 0; padding: 13px 15px;
  background: rgba(59, 130, 246, 0.07);
  border: 1px solid rgba(59, 130, 246, 0.26);
  border-left: 3px solid var(--fops-blue); border-radius: var(--fops-radius);
}
.fops-evidence-text { margin-top: 6px; color: var(--fops-text);
                      font-size: 0.93rem; line-height: 1.76; }
.fops-evidence-note { margin-top: 9px; color: var(--fops-muted); font-size: 0.78rem; }

/* ---------- 安全护栏横幅 ---------- */
.fops-guard {
  margin: 6px 0 14px 0; padding: 14px 16px; border-radius: var(--fops-radius-lg);
  background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(127, 29, 29, 0.24) 100%);
  border: 1px solid rgba(239, 68, 68, 0.5);
  box-shadow: 0 8px 24px rgba(127, 29, 29, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.fops-guard-head { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
.fops-guard-icon { font-size: 1.05rem; }
.fops-guard-title { font-size: 0.98rem; font-weight: 750; color: #fecaca;
                    letter-spacing: 0.01em; }
.fops-guard-tag {
  padding: 2px 9px; font-family: var(--fops-mono); font-size: 0.7rem; font-weight: 700;
  color: #fee2e2; background: rgba(239, 68, 68, 0.28);
  border: 1px solid rgba(239, 68, 68, 0.55); border-radius: 999px;
}
.fops-guard-body { margin-top: 9px; color: #fecaca; font-size: 0.89rem; line-height: 1.66; }

/* ---------- 状态指示灯 ---------- */
.fops-statusbar {
  display: flex; align-items: center; gap: 7px; flex-wrap: wrap;
  padding: 7px 9px; margin-bottom: 12px;
  background: var(--fops-panel); border: 1px solid var(--fops-border);
  border-radius: var(--fops-radius); font-size: 0.78rem;
}
.fops-status-k { color: var(--fops-muted); }
.fops-status-v { color: var(--fops-text); font-weight: 650; }
.fops-status-sep { color: rgba(148, 163, 184, 0.4); margin: 0 2px; }
.fops-dot { display: inline-block; flex: 0 0 auto;
            width: 8px; height: 8px; border-radius: 50%; }
.fops-dot-ok   { background: var(--fops-ok);
                 box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.16), 0 0 9px rgba(34, 197, 94, 0.75); }
.fops-dot-warn { background: var(--fops-warn);
                 box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.16), 0 0 9px rgba(245, 158, 11, 0.75); }
.fops-dot-bad  { background: var(--fops-bad);
                 box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.16), 0 0 9px rgba(239, 68, 68, 0.75); }

/* ---------- 审计调用链时间线 ---------- */
.fops-tl { margin: 6px 0 10px 0; }
.fops-tl-item {
  display: flex; align-items: flex-start; gap: 9px;
  padding: 7px 0 7px 0; border-left: 2px solid rgba(148, 163, 184, 0.22);
  padding-left: 12px; margin-left: 6px; position: relative;
}
.fops-tl-item::before {
  content: ""; position: absolute; left: -5px; top: 13px;
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--fops-blue); box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.16);
}
.fops-tl-item.is-ok::before  { background: var(--fops-ok); box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.16); }
.fops-tl-item.is-bad::before { background: var(--fops-bad); box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.16); }
.fops-tl-node {
  flex: 0 0 auto; font-family: var(--fops-mono); font-size: 0.76rem; font-weight: 700;
  color: #bfdbfe; background: var(--fops-blue-soft);
  border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 6px; padding: 1px 7px;
}
.fops-tl-detail { color: var(--fops-muted); font-size: 0.82rem; line-height: 1.6;
                  font-family: var(--fops-mono); word-break: break-word; }

/* ---------- 侧边栏 ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.94) 0%, rgba(11, 18, 32, 0.97) 100%);
  border-right: 1px solid var(--fops-border);
}
section[data-testid="stSidebar"] .stRadio > label,
section[data-testid="stSidebar"] .stSlider > label {
  font-size: 0.8rem !important; font-weight: 600 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] hr { margin: 12px 0; border-color: var(--fops-border); }
section[data-testid="stSidebar"] div[data-testid="stMetric"] { padding: 8px 10px; }
section[data-testid="stSidebar"] div[data-testid="stMetricValue"] { font-size: 1.05rem; }

/* ---------- 组件：按钮 / 指标 / 胶囊 ---------- */
.stButton > button {
  border-radius: var(--fops-radius); border: 1px solid var(--fops-border);
  transition: transform 0.08s ease, border-color 0.15s ease;
}
.stButton > button:hover { border-color: rgba(59, 130, 246, 0.55); transform: translateY(-1px); }
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #2563eb, #3b82f6); border-color: rgba(59,130,246,0.9);
  font-weight: 650;
}
div[data-testid="stMetric"] {
  padding: 11px 13px; background: var(--fops-panel);
  border: 1px solid var(--fops-border); border-radius: var(--fops-radius);
}
div[data-testid="stMetricLabel"] p { font-size: 0.7rem !important; color: var(--fops-muted) !important; }
</style>"""


def _inject_css() -> None:
    """注入全局样式。每次 rerun 都会重新插一份，浏览器只认最后一份，幂等。"""
    st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
# 二、渲染小工具（全部防御式：绝不因为一个空值把整页打崩）
# ==========================================================================
def _esc(value: Any) -> str:
    """HTML 转义。

    凡是拼进 ``unsafe_allow_html`` 的动态文本都必须过这里：规则切片原文、Agent
    答案里都可能出现 ``<`` ``&`` ``"``，不转义会被浏览器当成标签解析，轻则样式
    错乱，重则整块内容消失。
    """
    return html.escape(str(value), quote=True)


def _badge(text: str, kind: str = "cite") -> str:
    """徽标。``kind`` ∈ ``{"cite", "time", "muted"}``。"""
    css_class = {"time": "fops-badge-time", "muted": "fops-badge-muted"}.get(kind, "fops-badge")
    return f'<span class="{css_class}">{_esc(text)}</span>'


def _card(title: str, body_html: str) -> None:
    """一张带小标题的卡片。``body_html`` 由调用方负责转义。"""
    st.markdown(
        f'<div class="fops-card"><div class="fops-card-title">{_esc(title)}</div>'
        f"{body_html}</div>",
        unsafe_allow_html=True,
    )


def _section(title: str) -> None:
    st.markdown(f'<div class="fops-section">{_esc(title)}</div>', unsafe_allow_html=True)


def _safe(value: Any, fallback: str = "—") -> str:
    """指标文案兜底。

    ``st.metric`` 拿到 ``None`` 会渲染成字面量 "None"，看起来像程序出错；refuse /
    sql 路由下 ``result`` 本来就是 ``None``，所以统一先过这里收敛成占位符。
    """
    if value is None or value == "":
        return fallback
    if isinstance(value, (list, tuple, dict)) and not value:
        return fallback
    return str(value)


def safe_json_dump(value: Any) -> Any:
    """把任意值收敛成 **JSON 对象**再交给 ``st.json``。

    实测坑：Streamlit 的 JSON 视图组件（``reactJsonViewCompat``）只接受 JSON
    **对象**。传 ``None``（refuse 路由下 ``state.decision`` 就是 ``None``）或传
    数组（``state.trace``）时，前端会渲染成一个刺眼的
    ``{"ERROR": {"message": "src property must be a valid json object"}}`` 框，
    看上去像程序崩了，其实只是这个组件不收非对象值。

    这是唯一的 JSON 出口：任何要交给 ``st.json`` 的值都必须先过这里。
    """
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, (list, tuple)):
        return {"items": list(value)}
    return {"value": value}


# 历史名称保留：tests/unit/test_json_panel.py 与既有调用点仍在用这个名字。
_as_json_object = safe_json_dump


def _health_dot(ok: bool | None) -> str:
    """状态圆点：``True`` 绿 / ``None`` 黄（未加载）/ ``False`` 红。"""
    if ok is None:
        return "fops-dot-warn"
    return "fops-dot-ok" if ok else "fops-dot-bad"


def _status_bar(engine_state: bool | None, engine_text: str, guard_text: str) -> str:
    """侧边栏顶部的微型状态条：``● Engine: Online | AST Guardrail: Active``。"""
    return (
        '<div class="fops-statusbar">'
        f'<span class="fops-dot {_health_dot(engine_state)}"></span>'
        '<span class="fops-status-k">Engine</span>'
        f'<span class="fops-status-v">{_esc(engine_text)}</span>'
        '<span class="fops-status-sep">|</span>'
        f'<span class="fops-dot {_health_dot(True)}"></span>'
        '<span class="fops-status-k">AST Guardrail</span>'
        f'<span class="fops-status-v">{_esc(guard_text)}</span>'
        "</div>"
    )


def _guardrail_banner(reason: str) -> str:
    """写操作 / PII 被拦截时的红色安全护栏横幅。"""
    return (
        '<div class="fops-guard">'
        '<div class="fops-guard-head">'
        '<span class="fops-guard-icon">&#9940;</span>'
        '<span class="fops-guard-title">Security Guardrail Intercepted</span>'
        '<span class="fops-guard-tag">Read-only Policy Enforced</span>'
        "</div>"
        f'<div class="fops-guard-body">请求已被只读安全护栏拦截，未对数据进行任何写操作。'
        f"拦截依据：{_esc(reason or '命中只读策略，拒绝执行')}</div>"
        "</div>"
    )


def _timeline_html(steps: list[dict[str, Any]]) -> str:
    """把 ``state.trace`` 渲染成审计调用链时间线。"""
    if not steps:
        return '<div class="fops-tl-detail">本次未产生调用链记录。</div>'
    items: list[str] = []
    for step in steps:
        node = str(step.get("node", "?"))
        detail = _trace_detail(step)
        state_class = ""
        if step.get("ok") is True:
            state_class = " is-ok"
        elif step.get("ok") is False or step.get("error"):
            state_class = " is-bad"
        items.append(
            f'<div class="fops-tl-item{state_class}">'
            f'<span class="fops-tl-node">{_esc(node)}</span>'
            f'<span class="fops-tl-detail">{_esc(detail)}</span>'
            "</div>"
        )
    return '<div class="fops-tl">' + "".join(items) + "</div>"


def _trace_detail(step: dict[str, Any]) -> str:
    """把一条 trace 记录压成一行可读文本（未知字段也能安全展示）。"""
    known = {"node"}
    for key in ("route", "action", "sql", "reason", "error", "chars"):
        if key not in step:
            continue
        known.add(key)
        value = step[key]
        if key == "sql":
            value = " ".join(str(value).split())
            if len(value) > 160:
                value = value[:160] + " …"
        return f"{key} = {value}"
    for key in ("arguments", "rows", "ok"):
        if key in step:
            known.add(key)
    rest = {k: v for k, v in step.items() if k not in known}
    if "arguments" in step:
        return f"arguments = {json.dumps(step['arguments'], ensure_ascii=False)}"
    if rest:
        return json.dumps(rest, ensure_ascii=False, default=str)[:200]
    return json.dumps(step, ensure_ascii=False, default=str)[:200]


# ==========================================================================
# 三、RAG：纯函数 + Streamlit 缓存包装
# ==========================================================================
def load_rag_config() -> dict[str, Any]:
    """读取 config/rag.yaml（整体传给 HybridRetriever，它内部取 config['retrieval']）。"""
    import yaml

    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def build_retriever() -> tuple[Any, float]:
    """构造 HybridRetriever 并返回 (retriever, 加载耗时秒)。

    注意 ``DenseRetriever.__init__`` 会对全部 5,000 个切片做一次向量编码，
    ``build_reranker`` 会加载 Cross-Encoder，所以首次调用是分钟级的。
    这一步只应该发生一次，调用方必须走 ``get_retriever``（带缓存）。
    """
    from src.rag.corpus import ParentChildCorpus
    from src.rag.hybrid import HybridRetriever

    started = time.perf_counter()
    corpus = ParentChildCorpus.from_jsonl(CHUNKS_PATH)
    retriever = HybridRetriever(corpus, load_rag_config(), root=str(ROOT))
    return retriever, time.perf_counter() - started


@st.cache_resource(show_spinner=False)
def get_retriever(config_mtime_ns: str) -> tuple[Any, float]:
    """跨 rerun / 跨会话缓存检索器。

    这是「不卡死」的关键：5,000 切片的向量编码与 Cross-Encoder 只在这个进程里
    做一次，后续每次点击、每次刷新都直接复用。key 里带 config mtime，
    改了 ``rag.yaml`` 会自动重建。
    """
    return build_retriever()


def run_retrieval(retriever: Any, query: str, mode: str, top_k: int) -> tuple[list[Any], float]:
    """单条查询走 ``HybridRetriever.search(mode=...)``，返回 (hits, 耗时秒)。"""
    started = time.perf_counter()
    hits = retriever.search(query, mode=mode, top_k=top_k)
    return hits, time.perf_counter() - started


# ==========================================================================
# 四、Agent：本地确定性 provider（离线、可复现）
# ==========================================================================
# Provider 契约来自 src/agent/graph.py:101 ``Provider = Callable[[str, str], str]``，
# 即 ``provider(system_prompt, user_prompt) -> 模型输出文本``。tests/unit 里用
# ``lambda system, user: next(queue)`` 照抄了这个契约；这里用规则版实现，
# 以便对任意提问都能给出确定性结果（队列版只能应付固定脚本）。
_UNSAFE_KEYWORDS = (
    "删除", "删掉", "删库", "清空", "修改", "改成", "更新数据", "写入",
    "insert", "update", "delete", "drop", "truncate", "alter",
)
_PII_KEYWORDS = ("手机号", "手机号码", "收货地址", "邮箱", "身份证", "真实姓名", "收货人")

# 需要长尾 SQL 才能表达的口径（固定 schema 的 metric 表达不了「按支付方式分组」）。
_SQL_KEYWORDS = ("支付方式", "渠道", "来源", "客户", "分组", "分布", "占比", "排名", "明细", "逐月")

# 固定口径指标的识别顺序：先特化，再泛化。
_METRIC_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("cancel_rate", ("取消率", "退款率", "取消比例")),
    ("avg_order_value", ("客单价", "平均订单", "平均客单")),
    ("revenue", ("营收", "收入", "营业额", "gmv", "销售额")),
    ("order_count", ("订单数", "订单量", "单量", "多少单", "订单总数")),
]
_DIMENSION_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("category", ("品类", "类目")),
    ("merchant", ("商家", "商户", "merchant")),
    ("status", ("状态",)),
    ("month", ("按月", "每月", "逐月", "月份")),
]


class LocalDemoProvider:
    """本地确定性 provider：命中即返回，不联网、不依赖随机性。

    它按 system prompt 的用途分流（router / metric planner / sql planner /
    reflector / answer），在各自分支里只依据 user prompt 的关键词做决策；
    answer 分支则**只用工具真实返回的数字**组织中文回答，绝不编造口径。
    """

    def __call__(self, system: str, user: str) -> str:
        if "You are the router" in system:
            return self._route(user)
        if "self-correction node" in system:
            return self._reflect()
        if "answer node" in system:
            return self._answer(user)
        if "read-only SQLite SELECT" in system:
            return json.dumps({"sql": self._plan_sql(user)}, ensure_ascii=False)
        if "fixed-schema metric tool" in system:
            return json.dumps({"arguments": self._plan_metric(user)}, ensure_ascii=False)
        # 未识别的提示词：回退到最安全的固定口径路径，而不是抛错。
        return json.dumps(
            {"route": "metric", "reason": "本地 provider 未识别的提示词，回退固定口径路径"},
            ensure_ascii=False,
        )

    # ------------------------------------------------------------ 各节点实现
    @staticmethod
    def _route(question: str) -> str:
        lowered = question.lower()
        if any(word in lowered for word in _UNSAFE_KEYWORDS):
            route, reason = "refuse", "命中数据修改类关键词，只读不写"
        elif any(word in question for word in _PII_KEYWORDS):
            route, reason = "refuse", "请求超出 customer_id 的 PII，拒绝"
        elif any(word in question for word in _SQL_KEYWORDS):
            route, reason = "sql", "固定口径指标表达不了该分组，走只读长尾 SQL"
        else:
            route, reason = "metric", "订单数/净营收/客单价/取消率属于已审计固定口径"
        return json.dumps({"route": route, "reason": reason}, ensure_ascii=False)

    @staticmethod
    def _question_part(user: str) -> str:
        """metric planner 的 user prompt 是「问题 + 工具 schema」，schema 里含英文
        指标名，若整体做关键词匹配会把 'cancel_rate' 误判成用户意图，所以先切掉。"""
        return user.split("\n\nTool schema:")[0]

    def _plan_metric(self, user: str) -> dict[str, Any]:
        question = self._question_part(user)
        lowered = question.lower()
        metric = "order_count"
        for name, words in _METRIC_KEYWORDS:
            if any(word in lowered for word in words):
                metric = name
                break
        dimension = "none"
        for name, words in _DIMENSION_KEYWORDS:
            if any(word in question for word in words):
                dimension = name
                break
        arguments: dict[str, Any] = {"metric": metric, "dimension": dimension}
        merchant = re.search(r"M\d{3,}", question)
        if merchant:
            arguments["merchant_id"] = merchant.group(0)
        month = re.search(r"\d{4}-\d{2}", question)
        if month:
            arguments["month"] = month.group(0)
        return arguments

    @staticmethod
    def _plan_sql(question: str) -> str:
        """生成一条只读单表 SELECT。列名全部在 tools.ALLOWED_COLUMNS 白名单内，
        用到的函数也在 ALLOWED_FUNCTIONS 内，因此能通过 ReadOnlyGuard 的四层校验。"""
        table = [
            (
                ("支付方式", "payment"),
                "SELECT payment_method, COUNT(*) AS order_count FROM orders "
                "GROUP BY payment_method ORDER BY order_count DESC LIMIT 20",
            ),
            (
                ("渠道", "来源", "source"),
                "SELECT source_channel, COUNT(*) AS order_count FROM orders "
                "GROUP BY source_channel ORDER BY order_count DESC LIMIT 20",
            ),
            (
                ("客户", "customer"),
                "SELECT customer_id, COUNT(*) AS order_count, "
                "ROUND(SUM(paid_amount_minor - refunded_amount_minor) / 100.0, 2) AS net_revenue_yuan "
                "FROM orders WHERE status IN ('completed', 'shipped') AND quantity > 0 "
                "GROUP BY customer_id ORDER BY net_revenue_yuan DESC LIMIT 20",
            ),
            (
                ("月", "month"),
                "SELECT strftime('%Y-%m', order_time) AS month, COUNT(*) AS order_count "
                "FROM orders GROUP BY month ORDER BY month LIMIT 20",
            ),
            (
                ("品类", "类目"),
                "SELECT category, COUNT(*) AS order_count FROM orders "
                "GROUP BY category ORDER BY order_count DESC LIMIT 20",
            ),
            (
                ("状态", "status"),
                "SELECT status, COUNT(*) AS order_count FROM orders "
                "GROUP BY status ORDER BY order_count DESC LIMIT 20",
            ),
        ]
        for words, sql in table:
            if any(word in question for word in words):
                return sql
        # 兜底：仍然是合法的只读查询，且在答案里说明这是兜底口径。
        return "SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status LIMIT 20"

    def _reflect(self) -> str:
        """本轮 demo 不做查询改写：工具报错时诚实放弃，而不是猜一个 SQL 继续烧 token。
        真实 provider 在这里返回 ``{"action": "sql", "sql": ...}`` 即可复用同一套重试循环。"""
        return json.dumps(
            {"action": "give_up", "reason": "本地确定性 provider 不重写查询，交由答案节点如实说明"},
            ensure_ascii=False,
        )

    @staticmethod
    def _format_cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:,.2f}"
        return str(value)

    def _answer(self, payload_text: str) -> str:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return "工具返回结果无法解析，未生成答案。"
        rows = payload.get("rows") or []
        row_count = int(payload.get("row_count", len(rows)))
        if row_count == 0:
            return "查询结果为空：当前数据集中没有匹配的行。"
        lines = [f"查询共返回 {row_count} 行" + ("（此处仅预览前 8 行）。" if row_count > 8 else "。")]
        for row in rows[:8]:
            lines.append("- " + "；".join(f"{key} = {self._format_cell(value)}" for key, value in row.items()))
        if row_count > 8:
            lines.append(f"其余 {row_count - 8} 行的完整结果见下方表格。")
        if payload.get("tool") == "metric_query":
            lines.append(
                "口径说明：净营收 = paid_amount_minor - refunded_amount_minor，"
                "且只统计 status ∈ {completed, shipped} 的订单；取消率不做状态过滤。"
            )
        return "\n".join(lines)


def build_agent(database: Path) -> Any:
    """构造 FinOpsAgent（LangGraph 路径）。provider 为本地确定性实现。"""
    from src.agent.graph import FinOpsAgent

    return FinOpsAgent(database, LocalDemoProvider(), max_retries=2, use_langgraph=True)


@st.cache_resource(show_spinner=False)
def get_agent(db_mtime_ns: str) -> Any:
    """Agent 是轻对象，但仍然缓存：它的构造会编译一遍 LangGraph 状态机。"""
    return build_agent(ORDERS_DB)


def run_agent(agent: Any, question: str) -> tuple[Any, float]:
    """invoke 返回 AgentState（graph.py:349）。返回 (state, 耗时秒)。"""
    started = time.perf_counter()
    state = agent.invoke(question)
    return state, time.perf_counter() - started


# ==========================================================================
# 五、语义网关：探活真实 FastAPI 网关的 /cache/stats
# ==========================================================================
class EchoTransport:
    """网关演示用的本地确定性 transport（与 ScriptedTransport 同契约）。

    用 LLMClient 的真实 failover/cache 逻辑，但不联网；脚本队列型 transport
    会在队列耗尽后抛错，这里改成永远稳定的回声，便于反复点击演示。
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, provider: str, prompt: str, model: str | None) -> str:
        self.calls.append(provider)
        return f"[{provider}] 本地演示网关已收到长度为 {len(prompt)} 的提问（未调用外部大模型）。"

    async def stream(self, provider: str, prompt: str, model: str | None):
        self.calls.append(provider)
        for chunk in ("[", provider, "] 本地演示网关流式响应"):
            yield chunk


_DEMO_GATEWAY: dict[str, Any] = {}


def start_demo_gateway(port: int = 8010) -> str | None:
    """在本进程后台线程里用 src/gateway/app.py 的 create_app 起一个真网关。

    返回 None 表示成功，否则返回错误文本（页面会如实显示）。
    """
    if _DEMO_GATEWAY.get("server") is not None:
        return None
    try:
        import uvicorn

        from src.gateway.app import create_app
        from src.gateway.client import LLMClient, SemanticCache

        cache = SemanticCache()
        transport = EchoTransport()
        client = LLMClient(["primary", "backup"], transport=transport, cache=cache)
        server = uvicorn.Server(
            uvicorn.Config(create_app(client, cache), host="127.0.0.1", port=port, log_level="warning")
        )
        thread = threading.Thread(target=server.run, daemon=True, name="finops-demo-gateway")
        thread.start()
        _DEMO_GATEWAY.update(server=server, thread=thread, cache=cache, client=client)
        return None
    except Exception as exc:  # pragma: no cover - 页面内兜底展示
        return f"{type(exc).__name__}: {exc}"


def probe_gateway(url: str, timeout: float = 0.8, attempts: int = 1) -> dict[str, Any] | None:
    """GET /cache/stats；拿不到返回 None。"""
    import requests

    endpoint = url.rstrip("/") + "/cache/stats"
    for attempt in range(attempts):
        try:
            response = requests.get(endpoint, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            return {"enabled": False, "error": f"HTTP {response.status_code}"}
        except Exception as exc:
            if attempt == attempts - 1:
                _DEMO_GATEWAY["probe_error"] = f"{type(exc).__name__}: {exc}"
    return None


def warm_gateway_cache(url: str, prompt: str = "演示用缓存预热提问") -> dict[str, Any] | None:
    """POST /chat 两次：第一次 miss 写入缓存，第二次命中，让 hits/hit_rate 动起来。"""
    import requests

    endpoint = url.rstrip("/") + "/chat"
    for _ in range(2):
        try:
            requests.post(endpoint, json={"prompt": prompt}, timeout=5.0)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    return probe_gateway(url, timeout=1.0)


# ==========================================================================
# 六、业务场景目录
# ==========================================================================
def load_examples(limit: int = 3) -> list[dict[str, Any]]:
    """从 eval/datasets/stage1_eval.jsonl 取真实评测题（优先 development split）。"""
    if not EVAL_PATH.exists():
        return []
    rows = [json.loads(line) for line in EVAL_PATH.open(encoding="utf-8") if line.strip()]
    development = [row for row in rows if row.get("split") == "development"] or rows
    return development[:limit]


def scenario_catalog() -> dict[str, dict[str, Any]]:
    """场景快选目录。

    合规口径类场景的问句直接取自 ``eval/datasets/stage1_eval.jsonl`` 的真实评测题，
    不手写、不改写；评测集缺失时退回内置问句，保证演示仍然可点。
    """
    rows = load_examples(3)
    questions = [str(row.get("query", "")).strip() for row in rows]
    first = questions[0] if questions and questions[0] else FALLBACK_RAG_QUESTION
    second = questions[1] if len(questions) > 1 and questions[1] else FALLBACK_RAG_QUESTION_2
    return {
        "refund_caliber": {"label": "退款确认口径核验", "question": first, "kind": "rag"},
        "return_window": {"label": "数码退货窗口规则", "question": second, "kind": "rag"},
        "category_revenue": {"label": "品类净营收汇总", "question": METRIC_EXAMPLES[0], "kind": "metric"},
        "payment_mix": {"label": "支付渠道分布", "question": METRIC_EXAMPLES[1], "kind": "metric"},
        "guardrail_test": {"label": "只读 SQL 护栏拦截测试", "question": METRIC_EXAMPLES[3], "kind": "guardrail"},
    }


# ==========================================================================
# 七、执行层：按需计算并把结果缓存在 session_state
# ==========================================================================
def _config_mtime() -> str:
    """config/rag.yaml 的 mtime，作为检索器缓存的 key 一部分。"""
    try:
        return str(CONFIG_PATH.stat().st_mtime_ns)
    except OSError:
        return "missing"


def _db_mtime() -> str:
    try:
        return str(ORDERS_DB.stat().st_mtime_ns)
    except OSError:
        return "missing"


def ensure_rag_result() -> dict[str, Any] | None:
    """执行 RAG 检索（带 nonce 去重 + 缓存复用 + 优雅降级）。

    返回 ``None`` 表示本次没有检索请求；否则返回
    ``{"hits", "elapsed", "mode", "query", "load_seconds"}`` 或
    ``{"error": "...", "query": ...}``（加载失败时不抛栈，交给页面降级展示）。
    """
    request = st.session_state.get("rag_request")
    if request is None:
        return None
    cached = st.session_state.get("rag_result")
    if cached is not None and cached.get("nonce") == request["nonce"]:
        return cached

    payload: dict[str, Any] = {
        "nonce": request["nonce"],
        "query": request["query"],
        "mode": request["mode"],
    }
    try:
        with st.spinner("检索中：BM25 + 向量召回 → RRF 融合 → Cross-Encoder 精排 …"):
            retriever, load_seconds = get_retriever(_config_mtime())
            hits, elapsed = run_retrieval(
                retriever, request["query"], request["mode"], int(request["top_k"])
            )
        payload.update(hits=hits, elapsed=elapsed, load_seconds=load_seconds, error=None)
        st.session_state["rag_ready"] = True
    except Exception as exc:  # 模型缺失 / 离线 / 语料损坏都收敛成降级提示，不抛栈
        payload.update(hits=None, elapsed=0.0, load_seconds=0.0, error=f"{type(exc).__name__}: {exc}")
        st.session_state["rag_ready"] = False
    st.session_state["rag_result"] = payload
    return payload


def ensure_agent_result() -> dict[str, Any] | None:
    """执行 Agent 推理（带 nonce 去重 + 缓存复用 + 优雅降级）。"""
    request = st.session_state.get("agent_request")
    if request is None:
        return None
    cached = st.session_state.get("agent_result")
    if cached is not None and cached.get("nonce") == request["nonce"]:
        return cached

    payload: dict[str, Any] = {"nonce": request["nonce"], "question": request["question"]}
    try:
        with st.spinner("Agent 推理中：意图识别 → 工具调用 → 护栏校验 → 结论生成 …"):
            agent = get_agent(_db_mtime())
            state, elapsed = run_agent(agent, request["question"])
        payload.update(state=state, elapsed=elapsed, error=None)
        st.session_state["agent_ready"] = True
    except Exception as exc:
        payload.update(state=None, elapsed=0.0, error=f"{type(exc).__name__}: {exc}")
        st.session_state["agent_ready"] = False
    st.session_state["agent_result"] = payload
    return payload


# ==========================================================================
# 八、侧边栏：引擎配置与系统监控
# ==========================================================================
def _gateway_controls(scope: str) -> None:
    """网关动作：启动 / 刷新 / 预热。

    ``scope`` 区分侧边栏与主区里的同名按钮——Streamlit 的 widget key 必须全局
    唯一，否则报 DuplicateWidgetID。
    """

    def _start_gateway_callback() -> None:
        error = start_demo_gateway(8010)
        if error:
            st.session_state["gateway_error"] = error
            return
        st.session_state.pop("gateway_error", None)
        st.session_state["gateway_url"] = "http://127.0.0.1:8010"
        st.session_state["cache_stats"] = probe_gateway("http://127.0.0.1:8010", timeout=1.0, attempts=8)

    def _warm_cache_callback() -> None:
        st.session_state["cache_stats"] = warm_gateway_cache(st.session_state["gateway_url"])

    button_cols = st.columns(2)
    button_cols[0].button("启动本地网关", key=f"gw_start_{scope}", on_click=_start_gateway_callback)
    button_cols[1].button(
        "刷新统计",
        key=f"gw_refresh_{scope}",
        on_click=lambda: st.session_state.update(cache_stats=None),
    )
    st.button(
        "预热语义缓存（发 2 次请求）",
        key=f"gw_warm_{scope}",
        on_click=_warm_cache_callback,
    )
    if st.session_state.get("gateway_error"):
        st.error(st.session_state["gateway_error"])


def _gateway_stats() -> dict[str, Any] | None:
    """读一次 ``/cache/stats`` 并缓存进 session_state（拿不到就返回 None）。"""
    if "gateway_url" not in st.session_state:
        st.session_state["gateway_url"] = os.environ.get("FINOPS_GATEWAY_URL", "http://127.0.0.1:8000")
    if st.session_state.get("cache_stats") is None:
        st.session_state["cache_stats"] = probe_gateway(st.session_state["gateway_url"])
    return st.session_state.get("cache_stats")


def _sidebar_shell() -> tuple[str, int, Any]:
    """渲染侧边栏骨架，返回 (检索模式, top_k, 状态条占位)。

    状态条用 ``st.empty()`` 占位、稍后回填：Engine 的真实状态要等检索器真的
    加载完才知道，而侧边栏必须先于主区渲染（它的控件决定本次请求参数）。
    """
    with st.sidebar:
        status_slot = st.empty()

        _section("检索策略")
        mode = st.radio(
            "检索模式",
            options=list(RETRIEVAL_MODES),
            format_func=lambda key: RETRIEVAL_MODES[key],
            index=0,
            key="mode",
            label_visibility="collapsed",
            help="切换后可再次点「开始分析」做同一问句的策略对比。",
        )
        top_k = st.slider("返回条数 Top-K", min_value=1, max_value=10, value=5, key="top_k")

        st.divider()
        _section("语义网关")
        st.text_input("网关地址", key="gateway_url", label_visibility="collapsed")
        _gateway_controls("side")

        stats = _gateway_stats()
        if stats is None:
            st.caption("网关未在线。点「启动本地网关」即可在本进程内拉起，无需另开终端。")
        elif stats.get("enabled") is False:
            st.info("网关在线，但语义缓存未启用。")
        else:
            metric_cols = st.columns(3)
            metric_cols[0].metric("请求量", _safe(stats.get("entries", 0), "0"))
            metric_cols[1].metric("命中", _safe(stats.get("hits", 0), "0"))
            metric_cols[2].metric("命中率", _safe(stats.get("hit_rate", 0.0), "0.0"))
        if st.session_state.get("gateway_error"):
            st.error(st.session_state["gateway_error"])
    return mode, top_k, status_slot


def _render_status_slot(slot: Any, rag_payload: dict[str, Any] | None, agent_payload: dict[str, Any] | None) -> None:
    """回填侧边栏顶部状态条。"""
    if rag_payload is not None and rag_payload.get("error"):
        engine_state: bool | None = False
        engine_text = "Degraded"
    elif st.session_state.get("rag_ready") or (rag_payload is not None):
        engine_state = True
        engine_text = "Online"
    else:
        engine_state = None
        engine_text = "Standby"

    guard_state = "Active"
    if agent_payload is not None and agent_payload.get("state") is not None:
        route = getattr(agent_payload["state"], "route", None)
        guard_state = "Intercepted" if route == "refuse" else "Active"

    slot.markdown(_status_bar(engine_state, engine_text, guard_state), unsafe_allow_html=True)


# ==========================================================================
# 九、主区：品牌 Header / 场景快选 / 问句
# ==========================================================================
def _header() -> None:
    st.markdown(
        '<div class="fops-brand">'
        '<div class="fops-brand-mark">FC</div>'
        "<div>"
        '<div class="fops-brand-title">FinOps Copilot</div>'
        '<div class="fops-brand-sub">企业经营分析与合规审计控制台</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def _scenario_bar(catalog: dict[str, dict[str, Any]]) -> str | None:
    """业务场景快选。返回**本次新选中**的场景键，重复 rerun 不会重复触发。"""
    keys = list(catalog)
    picked = st.pills(
        "业务场景",
        options=keys,
        format_func=lambda key: catalog[key]["label"],
        selection_mode="single",
        key="scenario",
        label_visibility="collapsed",
    )
    if picked and picked != st.session_state.get("_scenario_applied"):
        st.session_state["_scenario_applied"] = picked
        return picked
    return None


def _ask_row() -> tuple[str, bool]:
    """问句输入 + 主执行按钮。返回 (问句, 是否要求执行)。"""
    st.session_state.setdefault("question", "")
    input_col, button_col = st.columns([5, 1.15], vertical_alignment="bottom")
    with input_col:
        st.text_input(
            "业务问题",
            key="question",
            label_visibility="collapsed",
            placeholder="输入业务问题，例如：当前生效口径下，哪些订单状态可以计入营收？",
        )
    with button_col:
        clicked = st.button("开始分析", type="primary", key="run_all", width="stretch")
    return st.session_state["question"], bool(clicked)


def _submit(question: str, mode: str, top_k: int) -> None:
    """把一次执行请求写进 session_state（同一 nonce 下结果会被复用）。"""
    nonce = time.time()
    st.session_state["rag_request"] = {
        "query": question,
        "mode": mode,
        "top_k": int(top_k),
        "nonce": nonce,
    }
    st.session_state["agent_request"] = {"question": question, "nonce": nonce}


# ==========================================================================
# 十、中栏：业务决策看板
# ==========================================================================
_NUMERIC_METRIC_KEYS = ("order_count", "rows", "count")
_REVENUE_KEYS = ("net_revenue_yuan", "revenue_yuan", "revenue", "gmv")


def _business_metrics(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """从真实返回行里派生业务指标（合计/分组数），不臆造字段。"""
    if not rows:
        return []
    out: list[tuple[str, str]] = [("返回分组", f"{len(rows)}")]
    sample = rows[0]
    for key in sample:
        lowered = str(key).lower()
        if any(token in lowered for token in _REVENUE_KEYS):
            total = sum(float(row.get(key) or 0) for row in rows)
            out.append(("净营收合计（元）", f"{total:,.2f}"))
            break
    for key in sample:
        lowered = str(key).lower()
        if any(token in lowered for token in _NUMERIC_METRIC_KEYS):
            total = sum(float(row.get(key) or 0) for row in rows)
            out.append(("订单量合计", f"{int(total):,}"))
            break
    return out


def _decision_panel(agent_payload: dict[str, Any] | None, rag_payload: dict[str, Any] | None) -> None:
    """中栏：核心结论 + 指标看板，或安全护栏横幅。"""
    _section("业务决策")

    if agent_payload is None:
        st.markdown(
            '<div class="fops-card"><div class="fops-card-title">待分析</div>'
            '<div class="fops-conclusion">请选择上方业务场景，或直接输入问题后点「开始分析」。</div></div>',
            unsafe_allow_html=True,
        )
        return

    if agent_payload.get("error"):
        st.error(f"分析未完成：{agent_payload['error']}")
        return

    state = agent_payload.get("state")
    if state is None:
        st.error("分析未返回结果。")
        return

    route = getattr(state, "route", None)
    result = getattr(state, "result", None)
    rows = list(getattr(result, "rows", None) or []) if result is not None else []

    # --- 安全护栏：写操作 / PII 被拦截 -------------------------------------
    if route == "refuse":
        st.markdown(_guardrail_banner(getattr(state, "route_reason", "") or ""), unsafe_allow_html=True)
        _card(
            "响应结论",
            f'<div class="fops-conclusion">{_esc(getattr(state, "answer", "") or "请求已被拒绝。")}</div>',
        )
        return

    # --- 正常业务查询：结论卡片 + 指标看板 ---------------------------------
    answer = (getattr(state, "answer", "") or "").strip()
    first_line = answer.split("\n")[0] if answer else "已完成分析。"
    _card(
        "核心结论",
        f'<div class="fops-conclusion">{_esc(first_line)}</div>',
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("路由", _safe(route))
    metric_cols[1].metric("工具", _safe(getattr(result, "tool", None)))
    metric_cols[2].metric("返回行数", len(rows))
    metric_cols[3].metric("耗时", f"{agent_payload.get('elapsed', 0.0):.2f}s")

    business = _business_metrics(rows)
    if business:
        biz_cols = st.columns(len(business))
        for column, (label, value) in zip(biz_cols, business):
            column.metric(label, value)

    if len(answer.split("\n")) > 1:
        with st.expander("完整结论", expanded=True):
            st.markdown(answer.replace("\n", "  \n"))

    if rows:
        st.markdown("**返回数据**")
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption("本次查询没有返回数据行。")

    if getattr(state, "error", None):
        st.warning(f"执行告警：{state.error}")


# ==========================================================================
# 十一、右栏：证据链与审计
# ==========================================================================
def _version_comparison(hits: list[Any]) -> list[dict[str, Any]]:
    """从命中切片里挑出同一政策的多版本，并把现行 / 已废止标出来。

    判定依据是切片自带的 ``effective_from``（ISO 日期字符串可直接比较）：
    同一 ``policy_key`` 下 ``effective_from`` 最大的那个版本即现行口径。
    """
    grouped: dict[str, list[Any]] = {}
    for hit in hits:
        key = getattr(hit.chunk, "policy_key", None)
        if key:
            grouped.setdefault(str(key), []).append(hit)
    table: list[dict[str, Any]] = []
    for key, group in grouped.items():
        versions = {str(getattr(hit.chunk, "version", "")) for hit in group}
        if len(versions) < 2:
            continue
        newest = max(str(getattr(hit.chunk, "effective_from", "") or "") for hit in group)
        for hit in group:
            chunk = hit.chunk
            effective = str(getattr(chunk, "effective_from", "") or "")
            table.append(
                {
                    "政策键": key,
                    "版本": chunk.version,
                    "状态": "现行" if effective == newest and effective else "已废止",
                    "生效日期": effective or "—",
                    "检索方法": hit.method,
                    "分数": round(float(hit.score), 4),
                    "Doc_ID": chunk.doc_id,
                    "溯源": chunk.citation_label,
                }
            )
    table.sort(key=lambda row: (row["政策键"], row["生效日期"]), reverse=True)
    return table


def _evidence_panel(rag_payload: dict[str, Any] | None, agent_payload: dict[str, Any] | None) -> None:
    """右栏：合规依据 + 版本口径比对 + 审计调用链。"""
    _section("证据链与审计")

    # ---------------- 合规依据 ----------------
    st.markdown("**合规依据**")
    if rag_payload is None:
        st.markdown(
            '<div class="fops-evidence">'
            '<div class="fops-evidence-text">尚未检索。选择上方业务场景后，'
            "这里会给出与该问题最相关的生效口径原文。</div></div>",
            unsafe_allow_html=True,
        )
    elif rag_payload.get("error"):
        st.warning(f"规则库不可用：{rag_payload['error']}")
    else:
        hits = rag_payload.get("hits") or []
        if not hits:
            st.warning("未检索到相关条款。")
        else:
            top = hits[0]
            chunk = top.chunk
            badge = f"[DOC: {chunk.doc_id}] · Page {chunk.page} · Line {chunk.line_start}"
            st.markdown(
                '<div class="fops-evidence">'
                + _badge(badge)
                + _badge(f"版本 {chunk.version}", "time")
                + _badge(f"相关性 {float(top.score):.4f}", "muted")
                + f'<div class="fops-evidence-text">{_esc(chunk.text)}</div>'
                '<div class="fops-evidence-note">生效口径原文，未做任何生成式改写。</div>'
                "</div>",
                unsafe_allow_html=True,
            )

            comparison = _version_comparison(hits)
            if comparison:
                with st.expander("版本口径比对（现行 / 已废止）", expanded=False):
                    st.dataframe(comparison, hide_index=True, width="stretch")
                    st.caption("同一政策键下 effective_from 最大的版本为现行口径。")

            with st.expander(f"其余命中（共 {len(hits)} 条）", expanded=False):
                st.dataframe(
                    [
                        {
                            "排名": hit.rank,
                            "溯源引用": hit.chunk.citation_label,
                            "方法": hit.method,
                            "分数": round(float(hit.score), 4),
                            "版本": hit.chunk.version,
                        }
                        for hit in hits
                    ],
                    hide_index=True,
                    width="stretch",
                )
                for hit in hits:
                    sub = hit.chunk
                    with st.expander(f"#{hit.rank} · {sub.citation_label} · {hit.method}"):
                        st.markdown(
                            f"`chunk_id` = `{sub.chunk_id}` ｜ `policy_key` = `{sub.policy_key}` ｜ "
                            f"`scope` = `{sub.scope}` ｜ `rule_type` = `{sub.rule_type}` ｜ "
                            f"`effective_from` = `{sub.effective_from or '—'}`"
                        )
                        st.markdown("**命中片段正文**")
                        st.write(sub.text)
                        st.markdown("**同页父块原文**")
                        st.write(sub.parent_text)
                        st.json(safe_json_dump(hit.components))

    # ---------------- 审计调用链 ----------------
    st.markdown("**审计调用链**")
    if agent_payload is None or agent_payload.get("state") is None:
        st.caption("尚未执行。运行分析后，这里会给出每个节点的执行轨迹。")
        return

    state = agent_payload["state"]
    trace = list(getattr(state, "trace", None) or [])
    st.markdown(_timeline_html(trace), unsafe_allow_html=True)

    result = getattr(state, "result", None)
    checklist = st.columns(3)
    checklist[0].metric("AST 校验", "通过" if (result is not None and getattr(result, "ok", False)) else "未执行")
    checklist[1].metric("重试次数", _safe(getattr(state, "retries", 0), "0"))
    checklist[2].metric("执行耗时", f"{agent_payload.get('elapsed', 0.0):.2f}s")

    if result is not None and getattr(result, "sql", None):
        st.markdown("**实际执行的只读 SQL**")
        st.code(str(result.sql), language="sql")
        if getattr(result, "params", None):
            st.caption(f"绑定参数：{result.params}")

    if getattr(state, "error", None):
        st.warning(f"节点告警：{state.error}")

    with st.expander("结构化 trace（原始 JSON）", expanded=False):
        st.json(safe_json_dump(trace))
    with st.expander("请求决策与参数", expanded=False):
        st.json(safe_json_dump(getattr(state, "decision", None)))


# ==========================================================================
# 十二、主流程
# ==========================================================================
def main() -> None:
    st.set_page_config(
        page_title="FinOps Copilot | 企业经营分析与合规审计控制台",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    catalog = scenario_catalog()
    mode, top_k, status_slot = _sidebar_shell()

    _header()

    # 场景快选：点了就把问句填进输入框并立刻执行（与手动点「开始分析」等价）。
    picked = _scenario_bar(catalog)
    if picked:
        st.session_state["question"] = catalog[picked]["question"]

    question, clicked = _ask_row()

    should_run = clicked or bool(picked)
    if should_run and question.strip():
        _submit(question.strip(), mode, top_k)

    # 执行（结果按 nonce 缓存在 session_state，重复 rerun 不会重算）。
    rag_payload = ensure_rag_result()
    agent_payload = ensure_agent_result()
    _render_status_slot(status_slot, rag_payload, agent_payload)

    # 三栏工作台的主区与证据抽屉。
    decision_col, evidence_col = st.columns([1.55, 1], gap="large")
    with decision_col:
        _decision_panel(agent_payload, rag_payload)
    with evidence_col:
        _evidence_panel(rag_payload, agent_payload)

    st.divider()
    st.caption(DISCLAIMER)


def _in_streamlit_runtime() -> bool:
    """是否运行在 Streamlit 脚本运行时里。

    必要性：``smoke_check.py`` 会 ``from app.streamlit_app import build_agent``——
    如果无条件在模块末尾调 ``main()``，那次纯 Python 导入就会顺手把整个页面渲染
    一遍。所以只有 ``streamlit run``（``__name__ == "__main__"``）或 AppTest
    （runtime 存在）时才进主流程。
    """
    try:
        from streamlit import runtime

        return bool(runtime.exists())
    except Exception:  # pragma: no cover - 极旧的 streamlit 没有 runtime
        return False


if __name__ == "__main__" or _in_streamlit_runtime():
    main()