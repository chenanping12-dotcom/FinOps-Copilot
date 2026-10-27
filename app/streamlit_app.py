"""FinOps Copilot —— 企业经营分析与合规审计控制台（Streamlit 单页应用）。

布局是「Copilot 三栏决策工作台」：

1. **左栏 · 引擎配置与系统监控**（``st.sidebar``）：引擎/只读护栏状态指示灯、
   检索策略与返回条数、语义网关的运行时指标。
2. **中栏 · 业务决策看板**：品牌 Header → 业务场景快选 → 问句与执行 →
   核心结论卡片 + 业务指标看板；命中写操作/PII 时切换为红色安全护栏横幅。
3. **右栏 · 证据链与审计**：合规依据（Top-1 规则原文 + 溯源徽标）、
   版本口径比对（现行 / 已废止）、LangGraph 审计调用链（节点时间线 +
   只读护栏结论 + 实际执行的只读 SQL + 结构化 trace）。

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
from functools import lru_cache
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

# 对抗演示里的恶意语句：这是一条**真的**写操作 SQL，不是「删掉所有订单」这种口语。
# 故意保留完整语句形态，用来演示「路由器在工具之前就把它拦下」。
SHOWCASE_DELETE_QUESTION = (
    "删除所有已取消的无效订单: DELETE FROM orders WHERE status = 'cancelled';"
)

# 面试对抗演示的两个场景。卡片上展示的问句与实际提交的字符串必须同源，
# 否则面试官点了卡片却看到另一句话，演示当场穿帮。
SHOWCASE_SCENARIOS: dict[str, dict[str, Any]] = {
    "version_drift": {
        "title": "版本口径冲突对抗",
        "tag": "V1 vs V2",
        "question": FALLBACK_RAG_QUESTION,
        "blurb": (
            "同一份语料、同一个问句，只换检索策略。纯词法检索把已废止的旧口径排在首位；"
            "混合检索 + RRF 融合 + Cross-Encoder 精排把现行生效的口径提上来。"
        ),
        "kind": "rag",
    },
    "guardrail": {
        "title": "恶意注入与写操作拦截",
        "tag": "403 FORBIDDEN",
        "question": SHOWCASE_DELETE_QUESTION,
        "blurb": (
            "把一条真正的 DELETE 语句当成业务问题提交。意图路由器在工具之前就拦下它，"
            "只读连接与只读护栏根本不会看到写操作。"
        ),
        "kind": "guardrail",
    },
}


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

/* ---------- 评测基准看板 ---------- */
.fops-bench-src {
  font-size: 0.76rem; color: var(--fops-muted); margin: 0 0 10px 0;
  line-height: 1.6;
}
.fops-bench-src code {
  font-family: var(--fops-mono); font-size: 0.74rem; color: #bfdbfe;
  background: rgba(59, 130, 246, 0.12); padding: 1px 6px; border-radius: 5px;
}
.fops-bench-note {
  border-left: 3px solid var(--fops-warn); border-radius: 0 8px 8px 0;
  background: rgba(245, 158, 11, 0.09); padding: 11px 14px; margin: 4px 0 12px 0;
  font-size: 0.85rem; line-height: 1.7; color: #fde68a;
}
.fops-bench-note b { color: #fbbf24; }
.fops-bench-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
  gap: 10px; margin: 6px 0 14px 0;
}
.fops-bench-kpi {
  border: 1px solid var(--fops-border); border-radius: var(--fops-radius);
  background: var(--fops-panel); padding: 11px 13px;
}
.fops-bench-kpi-lab {
  font-size: 0.7rem; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--fops-muted); margin-bottom: 4px;
}
.fops-bench-kpi-val {
  font-family: var(--fops-mono); font-size: 1.12rem; font-weight: 700;
  color: #bfdbfe; line-height: 1.25;
}
.fops-bench-kpi-sub { font-size: 0.73rem; color: var(--fops-muted); margin-top: 3px; }
.fops-bench-verdict {
  font-size: 0.82rem; color: var(--fops-muted); margin: -4px 0 12px 0;
  line-height: 1.7;
}
.fops-bench-verdict code {
  font-family: var(--fops-mono); color: #bfdbfe;
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

/* ---------- 对抗演示：双场景卡片 ---------- */
.fops-shot {
  height: 100%; padding: 13px 15px; border-radius: var(--fops-radius-lg);
  background: linear-gradient(160deg, rgba(59, 130, 246, 0.10) 0%, rgba(11, 18, 32, 0.35) 72%);
  border: 1px solid rgba(59, 130, 246, 0.30);
}
.fops-shot.is-danger {
  background: linear-gradient(160deg, rgba(239, 68, 68, 0.13) 0%, rgba(11, 18, 32, 0.35) 72%);
  border-color: rgba(239, 68, 68, 0.38);
}
.fops-shot-head { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
.fops-shot-title { font-size: 0.97rem; font-weight: 750; color: #dbeafe; }
.fops-shot.is-danger .fops-shot-title { color: #fecaca; }
.fops-shot-tag {
  padding: 2px 9px; font-family: var(--fops-mono); font-size: 0.68rem; font-weight: 700;
  border-radius: 999px; color: #bfdbfe;
  background: var(--fops-blue-soft); border: 1px solid rgba(59, 130, 246, 0.45);
}
.fops-shot.is-danger .fops-shot-tag {
  color: #fee2e2; background: rgba(239, 68, 68, 0.22); border-color: rgba(239, 68, 68, 0.5);
}
.fops-shot-blurb { margin-top: 8px; color: var(--fops-muted); font-size: 0.83rem; line-height: 1.66; }
.fops-shot-q {
  margin-top: 9px; padding: 8px 10px; font-family: var(--fops-mono); font-size: 0.77rem;
  color: var(--fops-text); background: rgba(2, 6, 23, 0.55);
  border: 1px solid var(--fops-border); border-radius: 6px; word-break: break-all;
}

/* ---------- 对抗演示：V1 vs V2 左右对照 ---------- */
.fops-versus { display: flex; gap: 12px; align-items: stretch; flex-wrap: wrap; margin: 4px 0 10px 0; }
.fops-versus-col {
  flex: 1 1 240px; min-width: 210px; padding: 12px 14px;
  border-radius: var(--fops-radius); border: 1px solid var(--fops-border);
  background: var(--fops-panel);
}
.fops-versus-col.is-bad { border-color: rgba(239, 68, 68, 0.45); background: rgba(127, 29, 29, 0.16); }
.fops-versus-col.is-ok  { border-color: rgba(34, 197, 94, 0.45); background: rgba(6, 78, 59, 0.18); }
.fops-versus-head {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--fops-muted);
}
.fops-versus-verdict { margin-top: 7px; font-size: 0.93rem; font-weight: 750; }
.fops-versus-col.is-bad .fops-versus-verdict { color: #fca5a5; }
.fops-versus-col.is-ok  .fops-versus-verdict { color: #86efac; }
.fops-versus-cite { margin-top: 7px; font-family: var(--fops-mono); font-size: 0.79rem; color: #bfdbfe; }
.fops-versus-meta { margin-top: 5px; font-family: var(--fops-mono); font-size: 0.73rem;
                    color: var(--fops-muted); line-height: 1.6; }
.fops-versus-text { margin-top: 9px; font-size: 0.84rem; line-height: 1.7; color: var(--fops-text); }

/* ---------- 对抗演示：403 拦截回执 ---------- */
.fops-403 {
  margin: 6px 0 12px 0; padding: 13px 15px; border-radius: var(--fops-radius);
  background: rgba(2, 6, 23, 0.62); border: 1px solid rgba(239, 68, 68, 0.45);
}
.fops-403-head { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
.fops-403-code {
  font-family: var(--fops-mono); font-size: 0.84rem; font-weight: 800;
  color: #fecaca; letter-spacing: 0.04em;
}
.fops-403-stmt {
  margin-top: 9px; padding: 9px 11px; font-family: var(--fops-mono); font-size: 0.79rem;
  color: #fecaca; background: rgba(127, 29, 29, 0.28);
  border: 1px dashed rgba(239, 68, 68, 0.5); border-radius: 6px; word-break: break-all;
}
.fops-403-line { margin-top: 8px; color: var(--fops-muted); font-size: 0.8rem; line-height: 1.68; }
.fops-403-line code {
  font-family: var(--fops-mono); font-size: 0.76rem; color: #bfdbfe;
  background: var(--fops-blue-soft); padding: 1px 5px; border-radius: 4px;
}

/* ---------- Trace 节点高亮（对抗演示联动） ---------- */
.fops-tl-item.is-hl {
  background: rgba(59, 130, 246, 0.10);
  border-left-color: var(--fops-blue);
  border-radius: 0 6px 6px 0;
}
.fops-tl-item.is-hl::before {
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.35), 0 0 12px rgba(59, 130, 246, 0.8);
}
.fops-tl-item.is-hl .fops-tl-node {
  background: rgba(59, 130, 246, 0.28); border-color: rgba(59, 130, 246, 0.6);
}
.fops-hl-note {
  margin: 2px 0 9px 0; padding: 7px 10px; border-radius: 6px;
  background: rgba(59, 130, 246, 0.10); border: 1px solid rgba(59, 130, 246, 0.32);
  color: #bfdbfe; font-size: 0.79rem; line-height: 1.62;
}
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
    """侧边栏顶部的微型状态条：``● Engine: Online | Read-only Guard: Active``。"""
    return (
        '<div class="fops-statusbar">'
        f'<span class="fops-dot {_health_dot(engine_state)}"></span>'
        '<span class="fops-status-k">Engine</span>'
        f'<span class="fops-status-v">{_esc(engine_text)}</span>'
        '<span class="fops-status-sep">|</span>'
        f'<span class="fops-dot {_health_dot(True)}"></span>'
        '<span class="fops-status-k">Read-only Guard</span>'
        f'<span class="fops-status-v">{_esc(guard_text)}</span>'
        "</div>"
    )


_DANGEROUS_SQL_RE = re.compile(
    r"\b(delete|drop|truncate|alter|update|insert)\b[^;]*", re.IGNORECASE
)


def _dangerous_statement(question: str) -> tuple[str, str]:
    """从问句里挑出被拦截的语句与命中的危险操作符。

    只认 SQL 形态的写操作（``DELETE`` / ``DROP`` / …）；口语化的「删掉所有订单」
    匹配不到，此时退回整句 + 通用文案——两种情况下横幅都要能正常渲染。
    """
    match = _DANGEROUS_SQL_RE.search(str(question or ""))
    if not match:
        return str(question or "").strip(), "数据修改类关键词"
    return match.group(0).strip(), match.group(1).upper()


def _guardrail_banner(reason: str, question: str = "") -> str:
    """写操作 / PII 被拦截时的红色安全护栏横幅。

    问句里带真实 SQL 写操作时（对抗演示场景 B），横幅额外给出 ``[403 FORBIDDEN]``
    回执与**被拦截的那条语句**，让「到底拦了什么」可以被肉眼复核，而不是只有一句
    「已拦截」。
    """
    statement, operation = _dangerous_statement(question)
    tag = "Read-only Policy Enforced" if operation == "数据修改类关键词" else f"Blocked · {operation}"
    head = (
        '<div class="fops-guard-head">'
        '<span class="fops-guard-icon">&#9940;</span>'
        '<span class="fops-403-code">[403 FORBIDDEN]</span>'
        '<span class="fops-guard-title">只读策略拦截</span>'
        f'<span class="fops-guard-tag">{_esc(tag)}</span>'
        "</div>"
    )
    body = (
        '<div class="fops-guard-body">请求已被只读安全护栏拦截，未对数据进行任何写操作。'
        f"拦截依据：{_esc(reason or '命中只读策略，拒绝执行')}</div>"
    )
    if statement:
        body += (
            f'<div class="fops-403-stmt">{_esc(statement)}</div>'
            '<div class="fops-403-line">该语句在<b>意图路由</b>阶段即被拦下，'
            "没有进入 <code>SafeSQLTool</code> —— 因此不会有任何 SQL 到达数据库。</div>"
        )
    return '<div class="fops-guard">' + head + body + "</div>"


def _forbidden_receipt(question: str, reason: str) -> str:
    """对抗演示场景 B 的「拦截回执」：被拦语句 + 四层防护清单。"""
    statement, operation = _dangerous_statement(question)
    return (
        '<div class="fops-403">'
        '<div class="fops-403-head">'
        '<span class="fops-403-code">[403 FORBIDDEN]</span>'
        f'<span class="fops-guard-tag">只读策略拦截 · {_esc(operation)}</span>'
        "</div>"
        f'<div class="fops-403-stmt">{_esc(statement)}</div>'
        '<div class="fops-403-line">'
        "写权限被锁死在四个层次上：<code>mode=ro</code> 只读连接 · "
        "<code>set_authorizer</code> 回调 · 表/函数白名单 · 行数上限与 VM 指令预算。"
        "任何单层都可能被绕过，所以是叠加而不是单选。"
        "</div>"
        f'<div class="fops-403-line">本轮的实际拦截点：<b>意图路由器</b>（'
        f"{_esc(reason or '命中只读策略，拒绝执行')}）。写操作连工具的入口都没到。</div>"
        "</div>"
    )


def _timeline_html(steps: list[dict[str, Any]], highlight: Any = None) -> str:
    """把 ``state.trace`` 渲染成审计调用链时间线。

    ``highlight`` 是节点名集合：对抗演示跑完后用它把关键节点（如 ``refuse``）
    点亮，默认 ``None`` 时行为与之前完全一致。
    """
    if not steps:
        return '<div class="fops-tl-detail">本次未产生调用链记录。</div>'
    lit = {str(node) for node in (highlight or [])}
    items: list[str] = []
    for step in steps:
        node = str(step.get("node", "?"))
        detail = _trace_detail(step)
        state_class = ""
        if step.get("ok") is True:
            state_class = " is-ok"
        elif step.get("ok") is False or step.get("error"):
            state_class = " is-bad"
        if node in lit:
            state_class += " is-hl"
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


def _version_status_map(*hit_lists: Any) -> dict[str, str]:
    """跨多次检索结果，为每个 ``chunk_id`` 判定「现行 / 已废止」。

    判定口径与 :func:`_version_comparison` 保持一致：同一 ``policy_key`` 下
    ``effective_from`` 最大的版本是现行口径。之所以要跨多次检索的结果一起算，
    是因为对抗演示里单看一次 BM25 的 Top-5 可能根本不包含 V2，那样就无法判断
    V1 是不是旧的——把两次命中的并集拿来比才稳。
    """
    newest: dict[str, str] = {}
    for hits in hit_lists:
        for hit in hits or []:
            chunk = hit.chunk
            key = str(getattr(chunk, "policy_key", "") or "")
            effective = str(getattr(chunk, "effective_from", "") or "")
            if key and effective > newest.get(key, ""):
                newest[key] = effective

    status: dict[str, str] = {}
    for hits in hit_lists:
        for hit in hits or []:
            chunk = hit.chunk
            key = str(getattr(chunk, "policy_key", "") or "")
            effective = str(getattr(chunk, "effective_from", "") or "")
            is_current = bool(key) and bool(effective) and effective == newest.get(key)
            status[str(chunk.chunk_id)] = "现行" if is_current else "已废止"
    return status


def _versus_column(label: str, hit: Any, status: str, extra: str = "") -> str:
    """版本对抗里的一列：策略名 + ✅/❌ 判定 + 溯源徽标 + 该策略返回的原文。"""
    chunk = hit.chunk
    is_current = status == "现行"
    css_class = "is-ok" if is_current else "is-bad"
    verdict = "&#9989; 当前现行口径" if is_current else "&#10060; 历史已作废口径"
    return (
        f'<div class="fops-versus-col {css_class}">'
        f'<div class="fops-versus-head">{_esc(label)}</div>'
        f'<div class="fops-versus-verdict">{verdict}</div>'
        f'<div class="fops-versus-cite">{_esc(chunk.citation_label)}</div>'
        f'<div class="fops-versus-meta">版本 {_esc(chunk.version)} ｜ '
        f'effective_from {_esc(getattr(chunk, "effective_from", "") or "—")} ｜ '
        f'method {_esc(hit.method)} ｜ score {float(hit.score):.4f}{_esc(extra)}</div>'
        f'<div class="fops-versus-text">{_esc(chunk.text)}</div>'
        "</div>"
    )


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
# 七之二、对抗演示：场景 A 的双策略对照执行
# ==========================================================================
def run_version_showcase(question: str, top_k: int) -> dict[str, Any]:
    """同一问句分别跑纯 BM25 与混合检索+精排，返回可左右对照的两组命中。

    复用 ``get_retriever`` 的缓存实例：一次点击只多花一次检索的时间（BM25 毫秒级、
    精排半秒级），**不会重新加载模型**，也不会重建向量索引。
    """
    retriever, load_seconds = get_retriever(_config_mtime())
    bm25_hits, bm25_elapsed = run_retrieval(retriever, question, "bm25", int(top_k))
    hybrid_hits, hybrid_elapsed = run_retrieval(retriever, question, "hybrid_rerank", int(top_k))
    return {
        "question": question,
        "load_seconds": load_seconds,
        "bm25": {"hits": bm25_hits, "elapsed": bm25_elapsed},
        "hybrid": {"hits": hybrid_hits, "elapsed": hybrid_elapsed},
        "status": _version_status_map(bm25_hits, hybrid_hits),
        "error": None,
    }


def ensure_showcase() -> dict[str, Any] | None:
    """按 ``showcase_request`` 执行对抗演示（结果按 nonce 缓存在 session_state）。

    场景 B（护栏拦截）不需要额外计算：Agent 那一轮已经产出了 ``route=refuse`` 与
    trace，这里只负责记住焦点，供右栏把对应节点点亮。
    """
    request = st.session_state.get("showcase_request")
    if request is None:
        return None
    cached = st.session_state.get("showcase_result")
    if cached is not None and cached.get("nonce") == request["nonce"]:
        return cached

    payload: dict[str, Any] = {
        "nonce": request["nonce"],
        "focus": request["focus"],
        "question": request["question"],
    }
    if request["focus"] == "version_drift":
        try:
            with st.spinner("对抗演示：同一问句分别跑 纯 BM25 与 混合检索+精排 …"):
                payload["version"] = run_version_showcase(request["question"], request["top_k"])
        except Exception as exc:  # 模型缺失 / 离线都收敛成降级提示，不抛栈
            payload["error"] = f"{type(exc).__name__}: {exc}"
    st.session_state["showcase_result"] = payload
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


def _showcase_cards() -> str | None:
    """两张高亮的对抗演示卡片。返回**本次新点击**的场景键，重复 rerun 不会重触发。"""
    _section("核心能力实测 · 对抗演示")
    columns = st.columns(2, gap="medium")
    picked: str | None = None
    for column, (key, spec) in zip(columns, SHOWCASE_SCENARIOS.items()):
        danger = spec["kind"] == "guardrail"
        css_class = "fops-shot is-danger" if danger else "fops-shot"
        with column:
            st.markdown(
                f'<div class="{css_class}">'
                '<div class="fops-shot-head">'
                f'<span class="fops-shot-title">{_esc(spec["title"])}</span>'
                f'<span class="fops-shot-tag">{_esc(spec["tag"])}</span>'
                "</div>"
                f'<div class="fops-shot-blurb">{_esc(spec["blurb"])}</div>'
                f'<div class="fops-shot-q">{_esc(spec["question"])}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"▶ 运行：{spec['title']}", key=f"showcase_{key}", width="stretch"):
                picked = key
    return picked


def _hit_rows(label: str, hits: list[Any], status: dict[str, str]) -> list[dict[str, Any]]:
    """把一组命中压成对比表用的行（两条策略共用同一套列）。"""
    return [
        {
            "策略": label,
            "排名": hit.rank,
            "溯源引用": hit.chunk.citation_label,
            "版本": hit.chunk.version,
            "生效日期": getattr(hit.chunk, "effective_from", "") or "—",
            "版本状态": status.get(str(hit.chunk.chunk_id), ""),
            "方法": hit.method,
            "分数": round(float(hit.score), 4),
        }
        for hit in hits
    ]


def _render_version_duel(payload: dict[str, Any]) -> None:
    """场景 A 的左右对照：纯词法基线 vs 混合检索 + Cross-Encoder。"""
    bm25_pack = payload.get("bm25") or {}
    hybrid_pack = payload.get("hybrid") or {}
    bm25_hits = list(bm25_pack.get("hits") or [])
    hybrid_hits = list(hybrid_pack.get("hits") or [])
    status = payload.get("status") or {}
    if not bm25_hits or not hybrid_hits:
        st.warning("对抗演示未取到足够命中，无法对比（请确认规则库已生成）。")
        return

    top_bm25, top_hybrid = bm25_hits[0], hybrid_hits[0]
    bm25_status = status.get(str(top_bm25.chunk.chunk_id), "")
    hybrid_status = status.get(str(top_hybrid.chunk.chunk_id), "")

    st.markdown(
        '<div class="fops-versus">'
        + _versus_column("纯 BM25（词法基线）", top_bm25, bm25_status)
        + _versus_column("混合检索 + RRF + Cross-Encoder", top_hybrid, hybrid_status)
        + "</div>",
        unsafe_allow_html=True,
    )

    if bm25_status == "已废止" and hybrid_status == "现行":
        st.success(
            "**对抗结论**：同一份语料、同一个问句，只换了检索策略。纯词法检索把"
            f"已废止的 `{top_bm25.chunk.citation_label}` 排在首位（这就是口径漂移）；"
            "混合检索 + RRF 融合 + Cross-Encoder 精排把现行生效的"
            f" `{top_hybrid.chunk.citation_label}` 提到首位。"
        )
    else:
        st.info(
            f"本轮两个策略的 Top-1 分别是 `{top_bm25.chunk.citation_label}`（{bm25_status}）"
            f" 与 `{top_hybrid.chunk.citation_label}`（{hybrid_status}）。"
        )

    duel_cols = st.columns(4)
    duel_cols[0].metric("BM25 检索耗时", f"{float(bm25_pack.get('elapsed', 0.0)) * 1000:.1f} ms")
    duel_cols[1].metric("混合+精排耗时", f"{float(hybrid_pack.get('elapsed', 0.0)) * 1000:.0f} ms")
    duel_cols[2].metric("Top-1 版本", f"{top_bm25.chunk.version} → {top_hybrid.chunk.version}")
    duel_cols[3].metric("现行口径是否置顶", "是" if hybrid_status == "现行" else "否")

    with st.expander("两条策略的完整排序（逐条对照）", expanded=False):
        st.dataframe(
            _hit_rows("纯 BM25", bm25_hits, status)
            + _hit_rows("混合 + 精排", hybrid_hits, status),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"检索器一次性加载耗时 {float(payload.get('load_seconds', 0.0)):.1f}s，"
            "之后每次点击都复用同一个缓存实例，不会重新加载模型。"
        )


def _render_showcase_result() -> None:
    """在对抗卡片下方回填本轮结果（由 ``main`` 在同一容器内二次进入渲染）。"""
    result = st.session_state.get("showcase_result")
    if result is None:
        st.caption(
            "点上面任意一张卡片即可现场跑一遍：场景 A 会同时用两种检索策略回答同一个问题，"
            "场景 B 会把一条真正的 DELETE 语句送进状态机，看它在哪里被拦下。"
        )
        return
    if result.get("error"):
        st.error(f"对抗演示未完成：{result['error']}")
        return

    if result.get("focus") == "version_drift":
        version = result.get("version")
        if version:
            _render_version_duel(version)
        return

    agent_payload = st.session_state.get("agent_result") or {}
    state = agent_payload.get("state")
    reason = getattr(state, "route_reason", "") if state is not None else ""
    st.markdown(
        _forbidden_receipt(str(result.get("question", "")), reason),
        unsafe_allow_html=True,
    )
    st.caption(
        "被拦下的写操作不会产生任何 SQL；右栏「审计调用链」里的 route / refuse 两个节点已高亮。"
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
        st.markdown(
            _guardrail_banner(
                getattr(state, "route_reason", "") or "",
                str(agent_payload.get("question", "")),
            ),
            unsafe_allow_html=True,
        )
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

    # 对抗演示跑完后，把焦点对应的那一块点亮；普通提问时 focus 为 None，行为不变。
    focus = st.session_state.get("showcase_focus")

    # ---------------- 合规依据 ----------------
    st.markdown("**合规依据**")
    if focus == "version_drift":
        st.markdown(
            '<div class="fops-hl-note">对抗演示焦点：这里给出的是<b>现行生效</b>口径。'
            "同样的问题把侧边栏切到「纯 BM25」再跑一次，Top-1 会变成已废止的旧版本。</div>",
            unsafe_allow_html=True,
        )
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
    if focus == "guardrail":
        st.markdown(
            '<div class="fops-hl-note">对抗演示焦点：<b>route / refuse</b> 两个节点已高亮。'
            "写操作在路由节点就被拦下，因此这条链路上不会出现任何 SQL。</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        _timeline_html(trace, {"route", "refuse"} if focus == "guardrail" else None),
        unsafe_allow_html=True,
    )

    result = getattr(state, "result", None)
    checklist = st.columns(3)
    checklist[0].metric(
        "只读护栏",
        "通过" if (result is not None and getattr(result, "ok", False)) else "跳过",
        help="请求在意图路由阶段就被护栏拦下时不会产生任何 SQL，因此没有可校验的语句。",
    )
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
# 十一之三、架构边界与设计权衡（可折叠面板）
# ==========================================================================
# 这一节只回答一个问题：**哪些是真算法组件，哪些是演示替身**。
# 三条自缚规则（写在代码里，不只写在文档里）：
#   ① 每条「真实」都指向仓库里可打开的符号，不写没有出处的形容词；
#   ② 每条「替身」都写清替掉了什么、为什么这样替、接回真实组件要改哪一处；
#   ③ 实验数字直读产物、代码事实直读源码默认参数，文案里一律不手抄。
# ③ 由 tests/unit/test_benchmark_panel.py 的棘轮测试强制。
#
# 为什么值得单独做一页：这个 Demo 的检索、状态机、只读护栏都是真的，只有「最后
# 把结构化结果写成中文」这一步是确定性替身。如果不主动说清，面试官要么以为整页
# 都是壳、要么以为我们在装黑盒大模型——两种误判都亏。边界写明白，真的部分才可信。
# --------------------------------------------------------------------------
RAG_CONFIG_YAML = ROOT / "config" / "rag.yaml"
ARCH_SRC_RAG = "src/rag/{bm25,dense,fusion,rerank,hybrid}.py"
ARCH_SRC_AGENT = "src/agent/graph.py · src/agent/tools.py"


@lru_cache(maxsize=None)
def _arch_retrieval_facts() -> dict[str, Any]:
    """检索链路的**配置事实**：直读 ``config/rag.yaml``，超参不在文案里手抄。"""
    facts: dict[str, Any] = {
        "rrf_k": None,
        "top_k": None,
        "rerank_window": None,
        "bm25_k1": None,
        "bm25_b": None,
        "tokenizer": None,
        "dense_model": None,
        "rerank_model": None,
    }
    try:
        import yaml

        raw = yaml.safe_load(RAG_CONFIG_YAML.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, TypeError, ImportError):
        return facts
    retrieval = raw.get("retrieval") or {}
    facts.update(
        {
            "rrf_k": (retrieval.get("fusion") or {}).get("rrf_k"),
            "top_k": retrieval.get("top_k"),
            "rerank_window": (retrieval.get("reranker") or {}).get("candidate_k"),
            "bm25_k1": (retrieval.get("bm25") or {}).get("k1"),
            "bm25_b": (retrieval.get("bm25") or {}).get("b"),
            "tokenizer": (retrieval.get("bm25") or {}).get("cjk_tokenizer"),
            "dense_model": (retrieval.get("dense") or {}).get("model_id"),
            "rerank_model": (retrieval.get("reranker") or {}).get("model_id"),
        }
    )
    return facts


@lru_cache(maxsize=None)
def _arch_data_facts() -> dict[str, Any]:
    """数据基底的**规模事实**：直读 ``reports/stage1_manifest.json`` 的 summary/evaluation。"""
    facts: dict[str, Any] = {
        "orders": None,
        "rule_chunks": None,
        "rule_documents": None,
        "dirty_orders": None,
        "dirty_types": None,
        "qa_pairs": None,
    }
    payload = _bench_json(str(STAGE1_MANIFEST_JSON))
    if not payload:
        return facts
    summary = payload.get("summary") or {}
    evaluation = payload.get("evaluation") or {}
    anomaly = summary.get("anomaly_counts") or {}
    facts.update(
        {
            "orders": summary.get("orders"),
            "rule_chunks": summary.get("rule_chunks"),
            "rule_documents": summary.get("rule_documents"),
            "dirty_orders": summary.get("dirty_orders"),
            "dirty_types": len(anomaly) or None,
            "qa_pairs": evaluation.get("qa_pairs"),
        }
    )
    return facts


@lru_cache(maxsize=None)
def _arch_real_facts() -> dict[str, Any]:
    """真实数据旁证的**结构事实**：直读 ``reports/real_data_stats.json``。"""
    facts: dict[str, Any] = {"rows": None, "missing_customer_pct": None, "missing_customer_rows": None}
    payload = _bench_json(str(REAL_STATS_JSON))
    if not payload:
        return facts
    inventory = payload.get("real_inventory") or {}
    missing = inventory.get("missing_customer_id") or {}
    facts.update(
        {
            "rows": inventory.get("rows_total"),
            "missing_customer_pct": missing.get("pct"),
            "missing_customer_rows": missing.get("rows"),
        }
    )
    return facts


@lru_cache(maxsize=None)
def _arch_code_facts() -> dict[str, Any]:
    """两个**代码事实**：重试上限与护栏预算直读类的默认参数。

    这样即使有人改了 ``src/agent`` 里的默认值，这一页也会跟着变，不会留下过期承诺。
    """
    facts: dict[str, Any] = {"max_retries": None, "instruction_budget": None, "max_rows": None}
    try:
        from src.agent.graph import FinOpsAgent
        from src.agent.tools import ReadOnlyGuard

        kw = FinOpsAgent.__init__.__kwdefaults__ or {}
        pos = ReadOnlyGuard.__init__.__defaults__ or ()
        facts["max_retries"] = kw.get("max_retries")
        if len(pos) > 0:
            facts["instruction_budget"] = int(pos[0])
        if len(pos) > 1:
            facts["max_rows"] = int(pos[1])
    except (ImportError, TypeError, ValueError):
        return facts
    return facts


# 只读护栏的分层口径**以源码为准**：src/agent/tools.py:9-16 的模块 docstring 写的是
# "read-only at four independent layers"，ReadOnlyGuard 的类 docstring 也是
# "Four independent read-only layers"。写在这里而不是从代码里反射，是因为
# 「哪一层拦下了哪条 payload」是本页要讲的结论；表里的行列名与 payload 结果由
# tests/unit/test_benchmark_panel.py 对着 src/agent/tools.py 与 tests 固化。
#
# 下面四层都在 SQLite 内部强制执行；validate_sql 单列一行、**不计入**这四层（见 APPLICATION_PRE_CHECK）。
# 单元格文字刻意压短：st.dataframe 不折行，超宽会被横向裁掉（旧版「机制」列就是这样
# 少了半句）。被裁掉的具体常量名与数值统一放到表下的 caption 里说清楚——表格负责结构
# 一眼可读，caption 负责逐条可核对，两边合起来才是完整证据。
READONLY_LAYERS: tuple[tuple[str, str, str], ...] = (
    ("① 文件级只读", "连接串带 mode=ro；写入在 SQLite 打开／执行阶段即失败", "tools.py:253 ReadOnlyGuard.connect"),
    ("② 语句级只读", "set_authorizer 按动作码逐条放行，其余一律 SQLITE_DENY", "tools.py:272 ReadOnlyGuard._authorizer"),
    ("③ 名称白名单", "表名、列名、函数名三张白名单，清单外的名字拒绝执行", "tools.py:31 ALLOWED_TABLES 等三张白名单"),
    ("④ 预算与结果集上限", "set_progress_handler 超预算即中断 + fetchmany 截断行数", "tools.py:265 progress ／ :302 execute"),
)

# **不计入上面四层**的一行：应用层前置校验。它在应用层、只看 SQL 文本，
# INSERT/**/INTO、大小写、注释、嵌进 CTE 等写法都能绕过它，所以它只是**廉价的
# 第一道过滤**，不构成安全保证；真正的保证是上面四层（都在 SQLite 内部）。
APPLICATION_PRE_CHECK: tuple[str, str, str] = (
    "应用层前置校验（不计入）",
    "写操作关键字正则；只看 SQL 文本，可被注释／大小写绕过",
    "tools.py:292 ReadOnlyGuard.validate_sql",
)


def readonly_layer_rows() -> list[dict[str, str]]:
    """护栏表的行：4 行库内强制层（库内强制=是）+ 1 行应用层前置校验（=否，不计入）。"""
    rows = [
        {"层": layer, "机制": how, "代码位置": where, "库内强制": "是"}
        for layer, how, where in READONLY_LAYERS
    ]
    layer, how, where = APPLICATION_PRE_CHECK
    rows.append({"层": layer, "机制": how, "代码位置": where, "库内强制": "否（不计入上述四层）"})
    return rows


def _arch_card_html(rows: list[tuple[str, str, str]], foot_label: str) -> str:
    """把长文本对照项渲染成卡片，而不是 ``st.dataframe``。

    为什么不表格：这一页通栏可用宽度约 1100px，三列中文排不下，「真实实现」与
    「接回真实组件」会被横向截断——而这两列恰恰是这一页的全部信息（实测右栏 409px
    时更严重，「可核对位置」整列右半被切掉）。卡片让文本自然折行，代价是少一点表格
    式的整齐，这里值得换。字段较短的第 3 张表（护栏分层）仍用 ``st.dataframe``，
    但那些单元格必须短到不触发横向裁剪（该表「机制」列实测约 40 个字符宽），完整
    常量名与数值放到表下 caption 里讲——表格管结构、caption 管细节。
    """
    parts: list[str] = []
    for title, body, foot in rows:
        parts.append(
            '<div class="fops-card" style="padding:12px 16px;margin:0 0 10px 0;">'
            f'<div><span class="fops-badge">{_esc(title)}</span></div>'
            f'<div style="margin:9px 0 7px;line-height:1.75;">{_esc(body)}</div>'
            f'<div style="color:#94a3b8;font-size:0.82rem;line-height:1.5;">'
            f"{_esc(foot_label)}：{_esc(foot)}</div></div>"
        )
    return "".join(parts)


def _render_arch_boundaries() -> None:
    """三栏下方的可折叠面板：分清真实组件与演示替身，并列出只读护栏的真实分层。

    默认折叠，不占主视觉空间；放在证据链正下方，面试官看 Trace 时正好在同一视线内。
    """
    with st.expander("🛡️ 架构边界与设计权衡说明", expanded=False):
        st.caption(
            "这一节只回答一个问题：**哪些是真算法，哪些是演示替身**。"
            "每条「真实」都能在仓库里打开对应文件核对；每条「替身」都写清了接回真实组件要改哪里。"
        )

        rag = _arch_retrieval_facts()
        data = _arch_data_facts()
        real = _arch_real_facts()
        code = _arch_code_facts()

        # ------------------------------------------------------------ 1 真实组件
        st.markdown("##### 1 · 底层真实组件")
        st.markdown(
            _arch_card_html(
                [
                    (
                        "检索链路",
                        f"BM25（k1={rag['bm25_k1']}／b={rag['bm25_b']}／{rag['tokenizer']}）"
                        f" + 向量召回（{rag['dense_model']}）"
                        f" → RRF 倒数融合（k={rag['rrf_k']}）"
                        f" → Cross-Encoder 精排（{rag['rerank_model']}）"
                        f"，候选窗口 {rag['rerank_window']}，取 Top-{rag['top_k']}",
                        f"config/rag.yaml · {ARCH_SRC_RAG}",
                    ),
                    (
                        "细粒度溯源",
                        "每条命中自带 [Doc_ID, Page, Line] 与 policy_key／effective_from／version，可回指切片原文",
                        "src/rag/models.py · data/generated/rule_docs/chunks.jsonl",
                    ),
                    (
                        "Agent 编排",
                        "LangGraph StateGraph 六节点：route → metric｜sql｜refuse → reflect → answer",
                        "graph.py:319 FinOpsAgent._build_graph",
                    ),
                    (
                        "固定口径工具",
                        "MetricTool 只接受枚举化 Schema（指标／维度／时间窗），从根上防口径漂移，不接受自由 SQL",
                        "tools.py:313 MetricTool · :89 Metric · :96 Dimension",
                    ),
                    (
                        "只读长尾查询",
                        "SafeSQLTool + ReadOnlyGuard（四层库内强制 + 一层应用层前置校验，见下节第 3 张表）",
                        "tools.py:339 SafeSQLTool · :245 ReadOnlyGuard",
                    ),
                    (
                        "自我纠错",
                        f"reflect 节点是唯一允许改写查询的节点，重试上限 max_retries={code['max_retries']}",
                        "graph.py:230 _reflect_node · :132 FinOpsAgent",
                    ),
                    (
                        "数据基底",
                        f"合成订单 {data['orders']:,} 单（注入 {data['dirty_types']} 类脏数据共 {data['dirty_orders']:,} 单）"
                        f" + 规则切片 {data['rule_chunks']:,} 片／{data['rule_documents']:,} 份文档"
                        f" + 离线评测题 {data['qa_pairs']} 题",
                        "reports/stage1_manifest.json · scripts/generate_dataset.py",
                    ),
                ],
                "可核对位置",
            ),
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------ 2 演示替身
        st.markdown("##### 2 · 演示替身与设计权衡")
        st.markdown(
            _arch_card_html(
                [
                    (
                        "自然语言生成层",
                        "LocalDemoProvider：离线确定性 provider，按 system prompt 的用途分流，"
                        "answer 分支只用工具真实返回的数字组织中文",
                        "网关已实现 LLMClient + 主备 Provider 故障转移 + SSE 降级；"
                        "把 provider 换成 LLMClient 即可，工具层与检索层不用改",
                    ),
                    (
                        "为什么演示时不接大模型",
                        "合规口径存在 V1／V2 两个版本，差异只在“completed 还是 shipped 计入营收”这一档。"
                        "大模型自由改写会把两个版本的说法抹平——恰恰抹掉了本项目最想证明的能力：版本感知检索",
                        "演示阶段直接展示规则原文保真；接 LLM 时把口径原文作为强约束上下文注入，"
                        "并由评测集回归守住版本区分度",
                    ),
                    (
                        "真实数据接入范围",
                        f"UCI Online Retail II 共 {real['rows']:,} 行，仅用于压力测试结构性缺失"
                        f"（如客户号缺失 {real['missing_customer_pct']}%／{real['missing_customer_rows']:,} 行）",
                        "旁证，不混入合成训练：真实源没有 lifecycle／channel 列，status 是派生近似，"
                        "已在 real_data_stats.json 的 note_* 字段逐条标明",
                    ),
                ],
                "接回真实组件",
            ),
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------ 3 只读护栏分层
        st.markdown("##### 3 · 只读护栏的真实分层（四层库内强制 + 一层应用层前置校验）")
        st.dataframe(readonly_layer_rows(), hide_index=True, width="stretch")
        st.caption(
            f"「库内强制=是」的 {len(READONLY_LAYERS)} 行就是源码 docstring 里的 "
            f"four independent layers，全部在 SQLite 内部执行："
            f"① 连接以 URI `mode=ro` 打开（`ReadOnlyGuard.connect`）；"
            f"② `set_authorizer` 按动作码逐条判定，只放行 SELECT／READ／FUNCTION 三类动作"
            f"（`_SQLITE_READ_ACTIONS`），其余一律 `SQLITE_DENY`；"
            f"③ 表名、列名、函数名三张白名单 `ALLOWED_TABLES`／`ALLOWED_COLUMNS`／"
            f"`ALLOWED_FUNCTIONS`；"
            f"④ 每条查询约 {code['instruction_budget']:,} 条 SQLite 虚拟机指令的预算"
            f"（每 10000 个 VM 步回调一次），结果集上限 {code['max_rows']} 行。"
            "「库内强制=否」那一行是应用层的 `validate_sql`：写操作关键字正则 + 拒绝多语句 + "
            "首关键字白名单，只看 SQL 文本，INSERT/**/INTO、大小写、注释、嵌入 CTE 都能绕过，"
            "所以**不计入**上面四层，只能算廉价的第一道过滤，不构成安全保证。"
        )

        # 这一条是自我纠正：早期版本把应用层那层校验叫成「AST 校验」，但它其实是正则 +
        # 字符串检查，本项目从头到尾没有做过任何语法树解析。留着这句话，比悄悄删掉
        # 更能证明这套边界说明是核过的。
        st.info(
            "**关于命名的自我纠正**：本项目**没有**做语法树（AST）解析。"
            "应用层前置校验是写操作关键字正则 + 多语句拒绝 + 首关键字白名单；"
            "真正带结构信息的是 SQLite 自己的 authorizer 回调（第 ②③ 层）。"
            "早期版本的界面与文档把这一层写成“AST 校验”，属于用词夸大，现已全部改回“只读护栏”。"
        )


# ==========================================================================
# 十二、主流程
# ==========================================================================
# --------------------------------------------------------------------------
# 评测基准与鲁棒性看板
# --------------------------------------------------------------------------
# 三条硬约束，写在代码里而不是只写在文档里：
# ① **只读不重算**：数据直读本仓已入库的实验产物，切 Tab 秒开，不做任何现场推理；
# ② **每个数字都能被核对**：每节头部标出数据源文件路径与产物生成时间；
# ③ **产物缺失就降级**：宁可显示「未找到产物」，也绝不用硬编码的漂亮数字顶替。
# --------------------------------------------------------------------------
ABLATION_JSON = ROOT / "eval" / "reports" / "rag" / "ablation.json"
PROBE_EVAL116_JSON = ROOT / "eval" / "reports" / "rag" / "probe_eval116.json"
RANKINGS_JSONL = ROOT / "eval" / "reports" / "rag" / "rankings.jsonl"
STAGE1_MANIFEST_JSON = ROOT / "reports" / "stage1_manifest.json"
LATENCY_CROSSCHECK_JSON = ROOT / "eval" / "reports" / "rag" / "latency_window_crosscheck.json"
REAL_STATS_JSON = ROOT / "reports" / "real_data_stats.json"
REAL_COMPARISON_JSON = ROOT / "reports" / "real_data_comparison.json"
GATEWAY_METRICS_JSON = ROOT / "reports" / "stage3_gateway_metrics.json"

BENCH_ARM_ORDER = ("bm25", "dense", "hybrid_rerank")
BENCH_ARM_LABEL = {
    "bm25": "纯词法基线（BM25）",
    "dense": "语义向量基线（Dense · BGE）",
    "hybrid_rerank": "混合检索 + RRF + Cross-Encoder",
}
BENCH_ARM_VERDICT = {
    "bm25": "字面命中强、零模型依赖；同义改写与过期版本会明显干扰",
    "dense": "语义改写鲁棒；状态词与编码类精确串容易漏召",
    "hybrid_rerank": "双路取长补短，把「当前生效版本」稳定排到第 1",
}

# 只读 SQL 四层防护：逐条载荷的实测裁决（源 reports/stage3_agent_gateway_findings.md §3）。
# 记的是「哪一层拦下的 + 原样错误文本」，而不是一个笼统的拦截率——
# 载荷只有 6 条，报百分比会把小样本包装成统计结论。
READONLY_PAYLOADS: tuple[tuple[str, str, str], ...] = (
    ("DROP TABLE orders", "单语句只读校验", "only a single read-only SELECT statement is permitted"),
    ("PRAGMA table_info(orders)", "单语句只读校验", "only a single read-only SELECT statement is permitted"),
    ("SELECT 1; DROP TABLE orders", "多语句拼接", "only a single read-only SELECT statement is permitted"),
    ("SELECT * FROM sqlite_master", "set_authorizer 回调", "access to sqlite_master.type is prohibited"),
    ("SELECT * FROM customers", "表列白名单", "no such table: customers"),
    ("SELECT load_extension(1)", "set_authorizer 回调", "not authorized to use function: load_extension"),
)

# LangGraph 状态机四条路径（源同文件 §1，由 tests/unit/test_agent_gateway.py 回归固化）。
GRAPH_PATHS: tuple[tuple[str, str, str, str], ...] = (
    ("固定指标", "metric", "0", "route → metric → answer"),
    ("长尾 SQL", "sql", "0", "route → sql → answer"),
    ("SQL 自纠错闭环", "sql", "1", "route → sql(ok=False) → reflect(ok=True) → answer"),
    ("危险请求", "refuse", "0", "route → refuse"),
)


@lru_cache(maxsize=None)
def _bench_json(path_str: str) -> dict[str, Any] | None:
    """读本仓实验产物：只读 + 进程内缓存；解析失败返回 None（降级而非崩页）。"""
    try:
        return json.loads(Path(path_str).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _bench_src_line(path: Path, extra: str = "") -> str:
    """每个小节头部的一行数据源声明，让面试官可以当场打开文件核对。"""
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:  # pragma: no cover - 产物必定在仓内
        rel = path.as_posix()
    if not path.exists():
        return (
            f'<div class="fops-bench-src">⚠️ 未找到 <code>{_esc(rel)}</code>，'
            "本节降级为不可用——不展示任何没有该产物支撑的数字。</div>"
        )
    payload = _bench_json(str(path))
    when = ""
    if isinstance(payload, dict) and payload.get("generated_at"):
        when = f" · 产物生成于 {_esc(str(payload['generated_at']))}"
    tail = f" · {_esc(extra)}" if extra else ""
    return f'<div class="fops-bench-src">数据源：<code>{_esc(rel)}</code>{when}{tail}</div>'


def _bench_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _bench_kpis(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f'<div class="fops-bench-kpi"><div class="fops-bench-kpi-lab">{_esc(lab)}</div>'
        f'<div class="fops-bench-kpi-val">{_esc(val)}</div>'
        f'<div class="fops-bench-kpi-sub">{_esc(sub)}</div></div>'
        for lab, val, sub in items
    )
    st.markdown(f'<div class="fops-bench-grid">{cells}</div>', unsafe_allow_html=True)


def _bench_note(body_html: str) -> None:
    st.markdown(f'<div class="fops-bench-note">{body_html}</div>', unsafe_allow_html=True)


def _bench_verdict(body_html: str) -> None:
    st.markdown(f'<div class="fops-bench-verdict">{body_html}</div>', unsafe_allow_html=True)


def _probe_eval116() -> dict:
    """窗口 100 下 EVAL-116 的 Cross-Encoder 分值探针产物。"""
    return _bench_json(str(PROBE_EVAL116_JSON)) or {}


def _lexical_similarity(payload: dict) -> str:
    """近似副本与 gold 的**词面**相似度区间：直读 lexical_context[].similarity_to_gold。

    注意别和 _score_spread 混用——那个读的是 Cross-Encoder 的**重排分值**。
    两个量在这份产物里差一个数量级（0.93 vs 0.99），标错一个词就是另一个故事了。
    """
    values = [
        float(item["similarity_to_gold"])
        for item in (payload.get("lexical_context") or [])
        if item.get("similarity_to_gold") is not None
    ]
    if not values:
        return "—"
    return f"{min(values):.4f}–{max(values):.4f}（{len(values)} 个近似副本）"


def _score_spread(payload: dict) -> str:
    """100 条候选的重排分值分布：区间、极差、不同分值个数——全从探针产物算。"""
    scores = [
        float(item["rerank_score"])
        for item in (payload.get("cross_encoder_scores") or [])
        if item.get("rerank_score") is not None
    ]
    if not scores:
        return "—"
    return (
        f"{min(scores):.4f}–{max(scores):.4f}"
        f"（{len(scores)} 条候选的极差只有 {max(scores) - min(scores):.4f}）"
        f"、不同分值仅 {len(set(scores))} 个"
    )


def _rule_documents() -> str:
    """规则文档份数：直读 stage1_manifest.json 的 summary，取不到显示 —。"""
    payload = _bench_json(str(STAGE1_MANIFEST_JSON)) or {}
    value = (payload.get("summary") or {}).get("rule_documents")
    return f"{int(value):,}" if isinstance(value, (int, float)) else "—"


def _gold_rank(question_id: str, arm: str) -> int | None:
    """现行 rankings.jsonl 里某题某臂的 gold 名次；查不到返回 None，不猜。"""
    if not RANKINGS_JSONL.exists():
        return None
    try:
        for line in RANKINGS_JSONL.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("question_id") != question_id or record.get("arm") != arm:
                continue
            gold = set(record.get("gold_chunk_ids") or [])
            for index, chunk_id in enumerate(
                record.get("ranked_chunk_ids") or [], start=1
            ):
                if chunk_id in gold:
                    return index
            return None
    except (OSError, ValueError):
        return None
    return None


def _bench_retrieval() -> None:
    """① 离线检索消融：三臂 Recall / MRR / 耗时，全部直读 ablation.json。"""
    payload = _bench_json(str(ABLATION_JSON))
    if not payload:
        return
    questions = payload.get("questions_evaluated")
    _section(
        f"① 离线检索基准 · {int(questions)} 题全量消融"
        if isinstance(questions, (int, float))
        else "① 离线检索基准 · 全量消融"
    )
    st.markdown(
        _bench_src_line(ABLATION_JSON, "真实 BGE 模型现场推理，非估算"),
        unsafe_allow_html=True,
    )

    metrics = payload.get("metrics") or {}
    latency = payload.get("avg_latency_ms") or {}
    import pandas as pd  # 局部导入：本模块必须能被 smoke_check 以纯 Python 导入

    rows = []
    for arm in BENCH_ARM_ORDER:
        arm_metrics = metrics.get(arm) or {}
        rows.append(
            {
                "检索策略": BENCH_ARM_LABEL[arm],
                "Recall@5": float(arm_metrics.get("recall_at_5") or 0.0),
                "MRR@5": float(arm_metrics.get("mrr_at_5") or 0.0),
                "平均单题耗时 (ms)": round(float(latency.get(arm) or 0.0), 2),
                "工程结论": BENCH_ARM_VERDICT[arm],
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Recall@5": st.column_config.ProgressColumn(
                "Recall@5", format="percent", min_value=0.0, max_value=1.0, width="small"
            ),
            "MRR@5": st.column_config.ProgressColumn(
                "MRR@5", format="percent", min_value=0.0, max_value=1.0, width="small"
            ),
            "平均单题耗时 (ms)": st.column_config.NumberColumn(
                "平均单题耗时 (ms)", format="%.2f ms"
            ),
        },
    )

    recalls = [float((metrics.get(a) or {}).get("recall_at_5") or 0.0) for a in BENCH_ARM_ORDER]
    # 触顶判定**不能**写成「三臂全为 1.0」：Dense 是 0.9667，那样这条披露永远不会触发
    # （实际就踩过：条件写成 all(== 1.0)，于是整段主动披露是死代码）。只要有任意一臂
    # 触顶，就说明这份语料在「召回」这个指标上没有区分度。
    ceiling_arms = [a for a, r in zip(BENCH_ARM_ORDER, recalls) if abs(r - 1.0) < 1e-9]
    if ceiling_arms:
        names = "、".join(BENCH_ARM_LABEL[a] for a in ceiling_arms)
        hybrid_mrr = (metrics.get("hybrid_rerank") or {}).get("mrr_at_5")
        best_single = max(
            float((metrics.get(a) or {}).get("mrr_at_5") or 0.0)
            for a in BENCH_ARM_ORDER
            if a != "hybrid_rerank"
        )
        ceiling_value = max(
            float((metrics.get(a) or {}).get("recall_at_5") or 0.0) for a in ceiling_arms
        )
        _bench_note(
            f"<b>主动披露：{_esc(names)} 的 Recall@5 已经触顶（{ceiling_value:.4f}），"
            f"混合检索在这个指标上没有提升空间</b>——这份合成语料对「召回」不具区分度。"
            f"所以结论必须收敛到 <b>MRR@5</b>：它是唯一有信息量的指标，"
            f"且确实分出了差异（混合精排 {_bench_pct(hybrid_mrr)}，"
            f"单臂最好 {_bench_pct(best_single)}）。"
        )
        avg_rank = (1.0 / float(hybrid_mrr)) if hybrid_mrr else None
        rank_txt = f"约第 {avg_rank:.1f} 位" if avg_rank else "—"
        probe = _probe_eval116()
        _bench_verdict(
            f"MRR 绝对值 {_bench_pct(hybrid_mrr)} 意味着 gold 平均落在{rank_txt}，仍有余量。"
            f"根因是<b>语料模板化</b>：{_rule_documents()} 份规则文档由模板生成，"
            "同一条款的 V1/V2 只差一个范围从句，"
            f"EVAL-116 的近似副本与 gold 词面相似度高达 {_lexical_similarity(probe)}，"
            f"重排分值 {_score_spread(probe)}——"
            "Cross-Encoder 被打到饱和，名次由平局规则决定。"
            "这是<b>评测集本身的缺陷，不是检索链路的缺陷</b>；重造语料会推翻全部历史指标，"
            "所以标成 <code>TODO(HUMAN)</code> 留档而没有自行动手。"
        )

    promoted = payload.get("rerank_promoted_gold") or []
    answerable = (metrics.get(BENCH_ARM_ORDER[0]) or {}).get("answerable_questions")
    unanswerable = (metrics.get(BENCH_ARM_ORDER[0]) or {}).get("unanswerable_questions")
    _bench_kpis(
        [
            (
                "评测题量",
                f"{payload.get('questions_evaluated', '—')} 题",
                f"可答 {answerable if answerable is not None else '—'} / 拒答 {unanswerable if unanswerable is not None else '—'}",
            ),
            ("语料切片", f"{int(payload.get('corpus_chunks') or 0):,}", "条规则切片，全量入检"),
            (
                "真实模型",
                "BGE 双塔 + Cross-Encoder",
                f"{payload.get('embedder_model', '—')} / {payload.get('reranker_model', '—')}",
            ),
            ("精排提升案例", f"{len(promoted)} 例", "精排把 gold 拉进 top-k 的题目数"),
            ("已归零坏例", f"{payload.get('bad_case_count', 0)}", "不存在「混合修好而单路弄坏」的题"),
        ]
    )
    note = payload.get("note")
    if note:
        _bench_verdict(f"<code>ablation.json.note</code>：{_esc(str(note))}")


def _bench_latency_window() -> None:
    """精排窗口 100 → 20 的耗时对照：简历里那组「窗口调优」数字的出处。

数值一律从 latency_window_crosscheck.json 现算，不写进 docstring
（写进去就会变成一个没人负责的、随时会过期的副本）。
"""
    payload = _bench_json(str(LATENCY_CROSSCHECK_JSON))
    if not payload:
        return
    results = payload.get("results") or []

    def _mean_at(window: int) -> float | None:
        vals = [
            float(item["mean_ms"])
            for item in results
            if int(item.get("window") or 0) == window and item.get("mean_ms") is not None
        ]
        return sum(vals) / len(vals) if vals else None

    wide, narrow = _mean_at(100), _mean_at(20)
    if wide is None or narrow is None:
        return
    _section("② 精排窗口调优 · 前后耗时对照")
    st.markdown(
        _bench_src_line(LATENCY_CROSSCHECK_JSON, "同进程、已预热、两种窗口交替两轮"),
        unsafe_allow_html=True,
    )
    _bench_kpis(
        [
            ("窗口 100（初版）", f"{wide:,.0f} ms", "候选含大量近似副本，精排被拖慢"),
            ("窗口 20（现行）", f"{narrow:,.0f} ms", f"降到 {narrow / wide * 100:.1f}%，排序质量不降"),
            ("加速比", f"{wide / narrow:.1f}×", "单题平均墙钟时间"),
        ]
    )
    probe = _probe_eval116()
    rank_wide = probe.get("gold_rerank_rank")
    rank_now = _gold_rank("EVAL-116", "hybrid_rerank")
    recall_now = (
        ((_bench_json(str(ABLATION_JSON)) or {}).get("metrics") or {})
        .get("hybrid_rerank", {})
        .get("recall_at_5")
    )
    recall_txt = f"{float(recall_now):.4f}" if isinstance(recall_now, (int, float)) else "—"
    _bench_verdict(
        "窗口收到 20 是那次「Hybrid 反而低于纯 BM25」的<b>主因修复</b>：近似副本不再进入候选，"
        f"EVAL-116 的 gold 从窗口 100 下的第 "
        f"{rank_wide if rank_wide is not None else '—'} 位回到现行第 "
        f"{rank_now if rank_now is not None else '—'} 位、Recall@5 恢复 {recall_txt}；"
        "把平局规则由 <code>chunk_id</code> 字典序改为回退 RRF 共识分数是配套动作，有效但不彻底。"
        "<b>因果顺序不能写反</b>——窗口是主因，tie-break 是配套。"
    )


def _bench_agent_and_gateway() -> None:
    """③ 状态机与网关的实测行为：只报跑通的路径与逐条载荷，不报「准确率」。"""
    _section("③ 状态机路由与只读护栏 · 实测行为")
    st.markdown(
        '<div class="fops-bench-src">数据源：<code>reports/stage3_agent_gateway_findings.md</code>'
        " §1/§3 + <code>reports/stage3_gateway_metrics.json</code>"
        " · 回归固化在 <code>tests/unit/test_agent_gateway.py</code></div>",
        unsafe_allow_html=True,
    )
    import pandas as pd

    st.markdown(
        '<div class="fops-bench-verdict">LangGraph 图结构 '
        "<code>route → (metric | sql | refuse)</code>，"
        "<code>metric/sql → _should_retry → reflect | answer</code>。</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"路径": name, "route": route, "重试次数": retries, "实际 trace": trace}
                for name, route, retries, trace in GRAPH_PATHS
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    _bench_note(
        "<b>自纠错链路是真跑通的，不是设计文档里的设想</b>：第一次 SQL 故意写错列名 "
        "<code>total_revenue</code>，<code>SafeSQLTool</code> 返回 <code>ok=False</code> + 错误文本 + hint，"
        "<code>reflect</code> 节点读入失败 SQL 后重写，第二次执行 <code>ok=True</code>、"
        "<code>state.retries=1</code>。<b>样本只有这一条，所以这里只能说「闭环演示成功」，"
        "不能写成一个百分比挽救率</b>——一条样本算不出率。"
    )

    st.markdown(
        '<div class="fops-bench-verdict">只读防护叠了四层：'
        "<code>mode=ro</code> 只读连接 / <code>set_authorizer</code> 回调 / 表列白名单 / "
        "行数上限 + VM 指令预算。<b>任何单层都可被绕过</b>，所以逐条载荷实测如下：</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"攻击载荷": load, "实际拦下的那一层": layer, "原样错误文本": err}
                for load, layer, err in READONLY_PAYLOADS
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    layers: dict[str, int] = {}
    for _load, _layer, _err in READONLY_PAYLOADS:
        layers[_layer] = layers.get(_layer, 0) + 1
    layer_txt = "、".join(
        f"{name} {n} 条" for name, n in sorted(layers.items(), key=lambda kv: -kv[1])
    )
    # 不写死「5 条注入 / 1 条越权」那种凭印象的分法：按实际拦下的层做直方图，
    # 层名本身就在上面那张表里逐条可见，对得上账。
    _bench_verdict(
        f"{len(READONLY_PAYLOADS)} 条载荷全部被拒；按实际拦下的那一层分布：{layer_txt}。"
        f"这里刻意<b>不写「拦截率 100%」</b>：载荷是人工挑的 {len(READONLY_PAYLOADS)} 条，"
        "报百分比会把小样本包装成统计结论。"
    )

    gateway = _bench_json(str(GATEWAY_METRICS_JSON))
    if not gateway:
        return
    failover = gateway.get("failover") or {}
    fastfail = gateway.get("fastfail") or {}
    all_failed = gateway.get("all_failed") or {}
    cache = gateway.get("semantic_cache") or {}
    _bench_kpis(
        [
            (
                "主备故障转移",
                f"{failover.get('n_attempts', '—')} 次尝试",
                f"503×2 → backup 200，{float(failover.get('wall_ms') or 0):.1f} ms",
            ),
            (
                "非可重试错误",
                f"{float(fastfail.get('wall_ms') or 0):.2f} ms",
                f"快速失败，备用调用 {0 if not fastfail.get('backup_tried') else 1} 次",
            ),
            ("全源失败", str(all_failed.get("error", "—")), f"共 {all_failed.get('n_attempts', '—')} 次尝试全记录"),
            (
                "语义缓存",
                f"命中 {float(cache.get('cache_hit_ms') or 0):.2f} ms",
                f"相似度 {cache.get('similarity', '—')}，上游仅 {cache.get('transport_calls', '—')} 次调用",
            ),
        ]
    )


def _bench_class(comparison: dict, name: str) -> dict:
    """按类名取一个脏数据类；取不到返回空 dict，宁可显示 — 也不猜值。"""
    for item in comparison.get("classes") or []:
        if str(item.get("class")) == name:
            return item
    return {}


def _pct_rows(item: dict) -> str:
    """把一个脏数据类渲染成「占比%（N 行）」。

    小数位刻意与上面「真实数据占比」那一列保持一致——同一个量在同一个面板里
    出现两种精度，看起来就像两个来源。
    """
    pct, rows = item.get("real_pct"), item.get("real_rows")
    if pct is None or rows is None:
        return "—"
    return f"{float(pct):.2f}%（{int(rows):,} 行）"


def _identity_failures(stats: dict) -> str:
    """跨系统金额恒等式的失败行数：直读产物字段，取不到就明说取不到。"""
    value = (
        ((stats.get("project_caliber") or {}).get("cross_check_sqlite") or {}).get(
            "rows_where_paid_minus_refunded_ne_quantity_x_price"
        )
    )
    return f"{int(value):,} 行" if isinstance(value, (int, float)) else "—"


def _injected_anomalies(stats: dict) -> str:
    """合成集注入的异常规模：直读 injected_anomalies，不手抄「六类各 500」。"""
    injected = (stats.get("synthetic_baseline") or {}).get("injected_anomalies") or {}
    by_type = injected.get("by_type") or {}
    values = [int(v or 0) for v in by_type.values()]
    if not values:
        return "见产物 reports/real_data_stats.json"
    total = f"{int(injected.get('total') or sum(values)):,}"
    if len(set(values)) == 1:
        return f"{len(values)} 类异常各注入 {values[0]:,} 条，共 {total} 条"
    return f"{len(values)} 类异常共注入 {total} 条"


def _archive_note(stats: dict) -> str:
    """原始压缩包的字节数与 SHA-256 前缀：直读 stats.download，不手抄尺寸。"""
    download = stats.get("download") or {}
    size, sha = download.get("bytes"), download.get("sha256")
    if not isinstance(size, (int, float)):
        return "原始压缩包与 SHA-256 见产物"
    if sha:
        return f"原始压缩包 {int(size):,} B，SHA-256 {str(sha)[:12]}… 已校验"
    return f"原始压缩包 {int(size):,} B"


def _bench_real_data() -> None:
    """④ 真实公开数据鲁棒性：合成 vs 真实的脏数据结构差异，直读 real_data_*.json。"""
    comparison = _bench_json(str(REAL_COMPARISON_JSON))
    stats = _bench_json(str(REAL_STATS_JSON))
    if not comparison or not stats:
        return

    _section("④ 真实公开数据鲁棒性 · UCI Online Retail II 接入旁证")
    st.markdown(
        _bench_src_line(
            REAL_STATS_JSON,
            _archive_note(stats),
        ),
        unsafe_allow_html=True,
    )
    _bench_kpis(
        [
            (
                "真实数据规模",
                f"{int(comparison.get('rows_real') or 0):,} 行",
                "UCI Online Retail II，2009-12 ~ 2011-12",
            ),
            (
                "合成数据规模",
                f"{int(comparison.get('rows_synthetic') or 0):,} 单",
                _injected_anomalies(stats),
            ),
            (
                "无法落进契约类型的行",
                f"{int((stats.get('normalization') or {}).get('rows_dropped_untypeable') or 0)} 行",
                "严格丢弃并计数，绝不静默填充",
            ),
            (
                "金额恒等式失败行",
                _identity_failures(stats),
                "paid − refunded == quantity × unit_price",
            ),
        ]
    )

    import pandas as pd

    legend = {str(item.get("class")): item for item in (comparison.get("classes") or [])}
    order = [
        "缺失 customer/Customer ID",
        "负数量（退货/取消腿）",
        "单价为 0",
        "单价为负",
        "InvoiceNo 以 C 开头（取消单）",
        "完全重复行",
        "时间不可解析",
        "退款超过已付",
    ]
    rows = []
    for name in order:
        item = legend.get(name)
        if not item:
            continue
        rows.append(
            {
                "脏数据类型": name,
                "合成数据占比": (
                    f"{float(item['synthetic_pct']):.2f}%"
                    if item.get("synthetic_pct") is not None
                    else "无此列"
                ),
                "真实数据占比": (
                    f"{float(item['real_pct']):.2f}%"
                    if item.get("real_pct") is not None
                    else "无此列"
                ),
                "真实数据行数": (
                    f"{int(item['real_rows']):,}" if item.get("real_rows") is not None else "—"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    duplicate = comparison.get("exact_duplicate_composition") or {}
    missing_customer = _bench_class(comparison, "缺失 customer/Customer ID")
    exact_dup = _bench_class(comparison, "完全重复行")
    cancelled = _bench_class(comparison, "InvoiceNo 以 C 开头（取消单）")
    _bench_note(
        f"<b>合成数据均匀得不真实，真实数据才是压力的来源</b>：合成集里客户号缺失、"
        f"单价为 0/负、取消单前缀、完全重复行<b>全是 0</b>；真实集里客户号缺失 "
        f"<b>{_pct_rows(missing_customer)}</b>、"
        f"完全重复行 {_pct_rows(exact_dup)}、取消单前缀 {_pct_rows(cancelled)}。"
        f"重复行组成经拆解为「同一发票内重录」（{int(duplicate.get('repeat_groups_within_one_invoice') or 0):,} 组、"
        f"跨发票 {int(duplicate.get('rows_spanning_multiple_invoices') or 0):,} 行、"
        f"单张发票最多重复 {duplicate.get('max_repeats_in_one_invoice', '—')} 次），"
        "去重会删掉合法记录，因此<b>全部保留</b>。"
    )

    normalization = stats.get("normalization") or {}
    caliber = (stats.get("project_caliber") or {}).get("queries") or {}
    revenue = (
        (caliber.get('{"metric": "revenue"}') or {}).get("rows") or [{}]
    )[0].get("revenue_yuan")
    dropped = normalization.get("rows_dropped_untypeable")
    dropped_txt = f"{int(dropped):,}" if isinstance(dropped, (int, float)) else "—"
    breakdown = normalization.get("dropped_breakdown") or {}
    breakdown_all_zero = bool(breakdown) and all(
        int(v or 0) == 0 for v in breakdown.values()
    )
    _bench_verdict(
        "<b>两条「不静默篡改」的工程红线</b>："
        f"① 落不进契约类型的行严格丢弃并计数（本次 <code>{dropped_txt}</code> 行，"
        f"<code>dropped_breakdown</code> 三类{'全 0' if breakdown_all_zero else '见产物'}），"
        "而不是拿默认值糊过去——糊过去就是在篡改财务金额；"
        "② 跨系统交叉验证 <code>paid − refunded == quantity × unit_price</code> 失败行 "
        f"<b>{_identity_failures(stats)}</b>，"
        "GBP 浮点先 <code>round(x*100)</code> 再入整数分，无精度漂移。"
        f"真实数据上按项目口径跑出的净营收为 <b>{float(revenue):,.2f}</b> 元"
        if revenue is not None
        else "<b>两条「不静默篡改」的工程红线</b>："
        f"落不进契约类型的行严格丢弃并计数（本次 {dropped_txt} 行）；"
        f"跨系统金额恒等式失败行为 {_identity_failures(stats)}。",
    )
    _bench_note(
        "<b>真实源的结构性限制，主动写清楚</b>："
        "① 无 <code>payment_method</code>/<code>source_channel</code> 列，一律哨兵 <code>unknown</code>，"
        "<b>按渠道的留存/退款分析在这个源上不可做，所以不做任何渠道结论</b>；"
        "② 无生命周期列，<code>status</code> 是派生的"
        "（<code>InvoiceNo</code> 以 C 开头 → cancelled，<code>Quantity&lt;0</code> → refunded，"
        "其余 completed），因此<b>永远不会产出 shipped</b>；"
        "③ 无折扣与运费拆分，两者硬置 0。"
        "接入是这个源的<b>鲁棒性旁证</b>，不是「真实业务效果」。"
    )


def _render_benchmark() -> None:
    """评测基准与鲁棒性看板：纯读 + 排版，切 Tab 无计算、无网络、无模型加载。"""
    st.markdown(
        '<div class="fops-brand"><div class="fops-brand-mark">📈</div><div>'
        '<div class="fops-brand-title">评测基准与鲁棒性</div>'
        '<div class="fops-brand-sub">全部数字直读本仓实验产物，每个小节都标注了源文件与产物生成时间</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    _bench_retrieval()
    _bench_latency_window()
    _bench_agent_and_gateway()
    _bench_real_data()
    _bench_verdict(
        "<b>这份看板本身也是可被审计的</b>：它不含任何硬编码指标——"
        "打开 <code>app/streamlit_app.py</code> 搜 <code>_bench_</code> 可以看到每个数字都是从哪一行 JSON 取的。"
        "产物缺失时它会显示「未找到产物」而不是回退到一份好看的快照。"
    )


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

    # 两个 Tab：工作台 / 评测看板。侧边栏与 Header 留在 Tab 之外，切换时视觉不跳。
    # 看板只读 JSON + 排版，不触发任何模型、数据库或网络调用。
    workbench_tab, benchmark_tab = st.tabs(
        ["🧭 经营分析与合规审计", "📈 评测基准与鲁棒性"]
    )

    with workbench_tab:
        # 面试对抗演示区：卡片先渲染，结果稍后回填进同一个容器，保持「卡片 → 对照结论」
        # 的视觉顺序（Streamlit 的容器支持创建后再写入，位置仍是创建处）。
        showcase_slot = st.container()
        with showcase_slot:
            showcase_picked = _showcase_cards()

        # 场景快选：点了就把问句填进输入框并立刻执行（与手动点「开始分析」等价）。
        picked = _scenario_bar(catalog)
        if picked:
            st.session_state["question"] = catalog[picked]["question"]

        if showcase_picked:
            spec = SHOWCASE_SCENARIOS[showcase_picked]
            st.session_state["question"] = spec["question"]
            st.session_state["showcase_focus"] = showcase_picked
            st.session_state["showcase_request"] = {
                "focus": showcase_picked,
                "question": spec["question"],
                "top_k": int(top_k),
                "nonce": time.time(),
            }
        elif picked:
            # 点场景快选也退出对抗焦点，避免右栏高亮一直挂着。
            st.session_state.pop("showcase_focus", None)

        question, clicked = _ask_row()

        if clicked:
            # 手动点「开始分析」同样退出对抗焦点。
            st.session_state.pop("showcase_focus", None)

        should_run = clicked or bool(picked) or bool(showcase_picked)
        if should_run and question.strip():
            _submit(question.strip(), mode, top_k)

        # 执行（结果按 nonce 缓存在 session_state，重复 rerun 不会重算）。
        rag_payload = ensure_rag_result()
        agent_payload = ensure_agent_result()
        ensure_showcase()
        _render_status_slot(status_slot, rag_payload, agent_payload)

        with showcase_slot:
            _render_showcase_result()

        # 三栏工作台的主区与证据抽屉。
        decision_col, evidence_col = st.columns([1.55, 1], gap="large")
        with decision_col:
            _decision_panel(agent_payload, rag_payload)
        with evidence_col:
            _evidence_panel(rag_payload, agent_payload)

        # 边界说明放通栏、就在三栏之下，而不是塞进右栏：它是三张对照表，右栏只有
        # 400px 左右，实测会把「可核对位置」整列挤掉——那恰好是这一页最该被核对的东西。
        # 默认折叠，所以并不占主视觉空间。
        _render_arch_boundaries()

        st.divider()
        st.caption(DISCLAIMER)

    with benchmark_tab:
        _render_benchmark()


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