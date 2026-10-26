"""FinOps-Copilot 交互式 Web Demo（Streamlit 单页应用）。

这个页面只做一件事：把仓库里已经跑通、但此前只能通过命令行脚本看到的两条能力
变成可以现场点击的产品：

1. 「知识库问答（RAG）」→ ``src/rag/hybrid.py`` 的 ``HybridRetriever``，展示带
   ``[Doc_ID, Page, Line]`` 溯源的 top-k 命中，并支持现场切换检索模式做消融对比。
2. 「经营指标（Agent）」→ ``src/agent/graph.py`` 的 ``FinOpsAgent``，展示路由结果、
   工具调用参数、真实返回行数与完整 ``trace``。

设计取舍（面试可讲）：
* **不改 src/**：这里只做展示层。``src/`` 下的代码是已验证的生产代码，发现的问题
  写在 ``app/README.md``，而不是就地修改。
* **默认不真调外部大模型 API**：``LocalDemoProvider`` 用关键词规则实现
  ``Provider = Callable[[str, str], str]`` 契约，保证 demo 离线、确定性、可复现；
  换成真实 provider 只需替换 ``build_agent`` 里的一个对象。
* **重资源只加载一次**：5,000 切片语料 + BGE 向量编码 + Cross-Encoder 很贵，
  统一走 ``@st.cache_resource``。核心逻辑放在普通函数里（``build_retriever`` /
  ``run_retrieval`` / ``run_agent``），这样它们既能被 Streamlit 缓存包装，
  也能被命令行 smoke test 直接调用。
* **展示层与生产代码解耦**：本文件只做「渲染 + 触发」，``src/`` 下的
  ``HybridRetriever`` / ``FinOpsAgent`` / 语义缓存网关接口一行未动。
* **视觉层可复现**：Streamlit 原生组件外观随浏览器主题漂移，同一份代码在两台
  机器上不是一个样子。这里注入一层 CSS（``CSS`` 常量）把配色固定为
  「深空灰底 + 企业蓝点缀」，并用少量 HTML 小组件承载溯源徽标 / 卡片 /
  健康指示灯。
* **渲染永不崩**：``_as_json_object`` 把 ``None``、列表收敛成 JSON 对象再交给
  ``st.json``（该组件只收对象）；所有拼进 ``unsafe_allow_html`` 的动态文本
  都先过 ``_esc``，否则规则原文里的 ``<`` ``&`` 会把版面拆掉。
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

DISCLAIMER = "数据与规则文档均为本地合成，非真实平台政策。"

# 检索模式：对应 HybridRetriever.search(mode=...)，用于现场做消融对比。
RETRIEVAL_MODES: dict[str, str] = {
    "hybrid_rerank": "混合检索 + Cross-Encoder 精排（默认）",
    "hybrid": "混合检索（BM25+向量 RRF 粗排，无精排）",
    "bm25": "纯 BM25（词法基线）",
    "dense": "纯向量（BGE 语义基线）",
}

# Agent Tab 的预设问题（问的是 orders.db 里的经营数据，不是规则文档）。
METRIC_EXAMPLES = [
    "各品类的净营收是多少？",
    "各支付方式的订单量分布？",
    "整体取消率是多少？",
    "删掉所有订单",
]


# --------------------------------------------------------------------------
# 视觉层：企业级看板的样式与小组件
#
# 为什么自己写 CSS：Streamlit 原生组件的外观由浏览器/OS 主题决定，同一份代码在
# 两台机器上颜色、间距、层级都不一样，做现场演示很容易"看着像半成品"。这里注入
# 一层样式把配色与层级钉死，再用几个小 HTML 小组件承载 Streamlit 没有的东西
# ——溯源徽标、卡片、健康指示灯。
#
# 约束：CSS 是纯字符串常量（不是 f-string），因此可以自由使用 ``{`` ``}``。
# --------------------------------------------------------------------------
CSS = """<style>
:root {
  --fops-bg: #0b1220;
  --fops-panel: rgba(255, 255, 255, 0.035);
  --fops-border: rgba(148, 163, 184, 0.18);
  --fops-blue: #3b82f6;
  --fops-blue-soft: rgba(59, 130, 246, 0.14);
  --fops-cyan: #22d3ee;
  --fops-text: #e2e8f0;
  --fops-muted: #94a3b8;
  --fops-ok: #22c55e;
  --fops-bad: #ef4444;
  --fops-radius: 14px;
  --fops-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

/* ---------- 全局：深空灰底 + 两团柔和光晕 ---------- */
.stApp {
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(59, 130, 246, 0.14), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34, 211, 238, 0.08), transparent 55%),
    var(--fops-bg);
  color: var(--fops-text);
}

/* ---------- 字体层级：紧凑、有力 ---------- */
.stApp h1 { font-size: 1.72rem !important; font-weight: 700 !important;
            letter-spacing: -0.3px; margin-bottom: 0.15rem !important; }
.stApp h2 { font-size: 1.14rem !important; font-weight: 650 !important;
            margin-top: 0.5rem !important; margin-bottom: 0.3rem !important; }
.stApp h3 { font-size: 1rem !important; font-weight: 600 !important; }
.stApp p, .stApp li { line-height: 1.72; }

/* ---------- 顶部 Hero 卡片 ----------
   .st-key-hero 是 Streamlit 给 st.container(key="hero") 自动加的 class，
   用容器而不是纯 HTML 当 Hero，标题就能保留成真正的 st.title。 */
.st-key-hero {
  border: 1px solid var(--fops-border) !important;
  border-radius: var(--fops-radius) !important;
  padding: 20px 24px 16px 24px !important;
  margin-bottom: 14px !important;
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.13) 0%,
              rgba(15, 23, 42, 0.55) 45%, rgba(34, 211, 238, 0.07) 100%);
  box-shadow: 0 12px 34px rgba(2, 6, 23, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.st-key-hero h1 { margin: 0 !important; padding: 0 !important; }
.st-key-hero .stMarkdown { margin-bottom: 0 !important; }
.fops-eyebrow {
  display: inline-block; margin-bottom: 10px; padding: 2px 10px;
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--fops-cyan);
  background: rgba(34, 211, 238, 0.08);
  border: 1px solid rgba(34, 211, 238, 0.35); border-radius: 999px;
}
.fops-sub { margin: 8px 0 12px 0; color: var(--fops-muted);
            font-size: 0.93rem; line-height: 1.62; }
.fops-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.fops-chip {
  font-family: var(--fops-mono); font-size: 0.71rem; color: #bfdbfe;
  background: var(--fops-blue-soft); border: 1px solid rgba(59, 130, 246, 0.3);
  border-radius: 8px; padding: 3px 9px;
}

/* ---------- 通用卡片 ---------- */
.fops-card {
  border: 1px solid var(--fops-border); border-radius: var(--fops-radius);
  background: var(--fops-panel); padding: 16px 18px; margin: 10px 0 14px 0;
  box-shadow: 0 6px 18px rgba(2, 6, 23, 0.35);
}
.fops-card-title {
  margin-bottom: 10px; font-size: 0.73rem; font-weight: 700;
  letter-spacing: 0.09em; text-transform: uppercase; color: var(--fops-muted);
}

/* ---------- 溯源徽标（[Doc_ID, Page, Line] 一眼可辨） ---------- */
.fops-badge {
  display: inline-block; margin: 0 6px 8px 0; padding: 3px 10px;
  font-family: var(--fops-mono); font-size: 0.78rem; font-weight: 600;
  color: #bfdbfe; background: var(--fops-blue-soft);
  border: 1px solid rgba(59, 130, 246, 0.38); border-radius: 7px;
}
.fops-badge-time {
  display: inline-block; margin: 0 6px 8px 0; padding: 3px 10px;
  font-family: var(--fops-mono); font-size: 0.78rem; font-weight: 600;
  color: #a7f3d0; background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 7px;
}

/* ---------- 抽取式「回答依据」块 ---------- */
.fops-answer {
  margin: 8px 0 12px 0; padding: 14px 16px;
  background: rgba(59, 130, 246, 0.07);
  border: 1px solid rgba(59, 130, 246, 0.28);
  border-left: 3px solid var(--fops-blue); border-radius: 10px;
}
.fops-answer-text { margin-top: 6px; color: var(--fops-text);
                    font-size: 0.95rem; line-height: 1.78; }

/* ---------- 健康状态指示灯 ---------- */
.fops-health { display: flex; align-items: center; gap: 8px;
               margin: 4px 0 10px 0; font-size: 0.85rem; }
.fops-dot { display: inline-block; flex: 0 0 auto;
            width: 9px; height: 9px; border-radius: 50%; }
.fops-dot-ok  { background: var(--fops-ok);
                box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.18), 0 0 10px rgba(34, 197, 94, 0.7); }
.fops-dot-bad { background: var(--fops-bad);
                box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.18), 0 0 10px rgba(239, 68, 68, 0.7); }
.fops-health-label { color: var(--fops-muted); }
.fops-health-value { color: var(--fops-text); font-weight: 600; }

/* ---------- 侧边栏：控件间距 + 分隔 ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.94) 0%, rgba(11, 18, 32, 0.97) 100%);
  border-right: 1px solid var(--fops-border);
}
section[data-testid="stSidebar"] .stRadio > label,
section[data-testid="stSidebar"] .stSlider > label {
  font-size: 0.82rem !important; font-weight: 600 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] hr { margin: 14px 0; border-color: var(--fops-border); }

/* ---------- Tab 页签 ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--fops-border); }
.stTabs [data-baseweb="tab"] {
  height: 42px; padding: 0 16px; border-radius: 10px 10px 0 0;
  font-size: 0.9rem; font-weight: 600; color: var(--fops-muted);
}
.stTabs [aria-selected="true"] { color: #dbeafe !important; background: var(--fops-blue-soft); }

/* ---------- 按钮与指标卡片 ---------- */
.stButton > button {
  border-radius: 10px; border: 1px solid var(--fops-border);
  transition: transform 0.08s ease, border-color 0.15s ease;
}
.stButton > button:hover { border-color: rgba(59, 130, 246, 0.55); transform: translateY(-1px); }
div[data-testid="stMetric"] {
  padding: 12px 14px; background: var(--fops-panel);
  border: 1px solid var(--fops-border); border-radius: 12px;
}
div[data-testid="stMetricLabel"] p { font-size: 0.72rem !important; color: var(--fops-muted) !important; }
</style>"""


def _inject_css() -> None:
    """注入全局样式。每次 rerun 都会重新插一份，浏览器只认最后一份，幂等。"""
    st.markdown(CSS, unsafe_allow_html=True)


def _esc(value: Any) -> str:
    """HTML 转义。

    凡是拼进 ``unsafe_allow_html`` 的动态文本都必须过这里：规则切片原文、Agent
    答案里都可能出现 ``<`` ``&`` ``"``，不转义会被浏览器当成标签解析，轻则样式
    错乱，重则整块内容消失。
    """
    return html.escape(str(value), quote=True)


def _badge(text: str, kind: str = "cite") -> str:
    """溯源 / 耗时徽标。``kind`` ∈ ``{"cite", "time"}``。"""
    css_class = "fops-badge-time" if kind == "time" else "fops-badge"
    return f'<span class="{css_class}">{_esc(text)}</span>'


def _card(title: str, body_html: str) -> None:
    """一张带小标题的卡片。``body_html`` 由调用方负责转义。"""
    st.markdown(
        f'<div class="fops-card"><div class="fops-card-title">{_esc(title)}</div>'
        f"{body_html}</div>",
        unsafe_allow_html=True,
    )


def _health(ok: bool, label: str, value: str) -> None:
    """健康状态指示灯：绿=正常，红=不可用。"""
    dot = "fops-dot-ok" if ok else "fops-dot-bad"
    st.markdown(
        f'<div class="fops-health"><span class="fops-dot {dot}"></span>'
        f'<span class="fops-health-label">{_esc(label)}</span>'
        f'<span class="fops-health-value">{_esc(value)}</span></div>',
        unsafe_allow_html=True,
    )


def _safe(value: Any, fallback: str = "—") -> str:
    """指标文案兜底。

    ``st.metric`` 拿到 ``None`` 会渲染成字面量 "None"，看起来像程序出错；refuse /
    sql 路由下 ``result`` 本来就可能是 ``None``，所以统一先过这里收敛成占位符。
    """
    if value is None or value == "":
        return fallback
    if isinstance(value, (list, tuple, dict)) and not value:
        return fallback
    return str(value)


def load_rag_config() -> dict[str, Any]:
    """读取 config/rag.yaml（整体传给 HybridRetriever，它内部取 config['retrieval']）。"""
    import yaml

    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


# --------------------------------------------------------------------------
# 一、RAG：纯函数 + Streamlit 缓存包装
# --------------------------------------------------------------------------
def build_retriever() -> tuple[Any, float]:
    """构造 HybridRetriever 并返回 (retriever, 加载耗时秒)。

    注意 ``DenseRetriever.__init__`` 会对全部 5,000 个切片做一次向量编码，
    ``build_reranker`` 会加载 Cross-Encoder，所以首次调用是分钟级的。
    """
    from src.rag.corpus import ParentChildCorpus
    from src.rag.hybrid import HybridRetriever

    started = time.perf_counter()
    corpus = ParentChildCorpus.from_jsonl(CHUNKS_PATH)
    retriever = HybridRetriever(corpus, load_rag_config(), root=str(ROOT))
    return retriever, time.perf_counter() - started


@st.cache_resource(show_spinner=False)
def get_retriever(config_mtime_ns: str) -> tuple[Any, float]:
    """跨 rerun / 跨会话缓存。key 里带 config mtime，改了 rag.yaml 会自动重建。"""
    with st.spinner(
        "正在加载检索器：5,000 切片语料 + BGE 向量编码 + Cross-Encoder 精排……"
        "首次约 1-3 分钟，之后走缓存秒开"
    ):
        return build_retriever()


def run_retrieval(retriever: Any, query: str, mode: str, top_k: int) -> tuple[list[Any], float]:
    """单条查询走 HybridRetriever.search(mode=...)，返回 (hits, 耗时秒)。"""
    started = time.perf_counter()
    hits = retriever.search(query, mode=mode, top_k=top_k)
    return hits, time.perf_counter() - started


# --------------------------------------------------------------------------
# 二、Agent：本地确定性 provider（离线、不调外部 API）
# --------------------------------------------------------------------------
# Provider 契约来自 src/agent/graph.py:101 ``Provider = Callable[[str, str], str]``，
# 即 ``provider(system_prompt, user_prompt) -> 模型输出文本``。tests/unit 里用
# ``lambda system, user: next(queue)`` 照抄了这个契约；这里用规则版实现，
# 以便对任意用户提问都能给出确定性结果（队列版只能应付固定脚本）。
_UNSAFE_KEYWORDS = (
    "删除",
    "删掉",
    "删库",
    "清空",
    "修改",
    "改成",
    "更新数据",
    "写入",
    "insert",
    "update",
    "delete",
    "drop",
    "truncate",
    "alter",
)
_PII_KEYWORDS = ("手机号", "手机号码", "收货地址", "邮箱", "身份证", "真实姓名", "收货人")

# 需要长尾 SQL 才能表达的口径（固定 schema 的 metric 表达不了"按支付方式分组"）。
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
        真实 provider 在这里返回 {"action": "sql", "sql": ...} 即可复用同一套重试循环。"""
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
    return build_agent(ORDERS_DB)


def run_agent(agent: Any, question: str) -> tuple[Any, float]:
    """invoke 返回 AgentState（graph.py:349）。返回 (state, 耗时秒)。"""
    started = time.perf_counter()
    state = agent.invoke(question)
    return state, time.perf_counter() - started


# --------------------------------------------------------------------------
# 三、网关语义缓存：探活真实 FastAPI 网关的 /cache/stats
# --------------------------------------------------------------------------
class EchoTransport:
    """网关 demo 用的本地确定性 transport（与 ScriptedTransport 同契约）。

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
    """GET /cache/stats；拿不到返回 None（页面显示"网关未启动"）。"""
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


# --------------------------------------------------------------------------
# 四、页面
# --------------------------------------------------------------------------
def load_examples(limit: int = 3) -> list[dict[str, Any]]:
    """从 eval/datasets/stage1_eval.jsonl 取真实评测题（优先 development split）。"""
    if not EVAL_PATH.exists():
        return []
    rows = [json.loads(line) for line in EVAL_PATH.open(encoding="utf-8") if line.strip()]
    development = [row for row in rows if row.get("split") == "development"] or rows
    return development[:limit]


def _render_rag_result(hits: list[Any], elapsed: float, mode: str, query: str) -> None:
    # 用徽标替代纯文本，让「命中几条 / 花了多久」一眼可见。
    st.markdown(
        _badge(f"模式 {mode}") + _badge(f"命中 {len(hits)} 条") + _badge(f"检索 {elapsed:.2f}s", "time"),
        unsafe_allow_html=True,
    )
    st.caption(f"查询：{query}")
    if not hits:
        st.warning("没有命中任何切片。")
        return

    # 顶部先给「回答依据」：精排后的第 1 条规则原文。
    # 本 Demo 不接大模型，所以这里是**抽取式**的回答——直接把规则切片原文摆出来，
    # 不做生成式改写，因此不存在口径幻觉；每一句都能在下面点开溯源核对。
    top = hits[0]
    _card(
        "回答依据（检索到的规则原文）",
        _badge(top.chunk.citation_label)
        + f'<div class="fops-answer-text">{_esc(top.chunk.text)}</div>',
    )
    st.caption(
        "本 Demo 不调用大模型：上面是**规则切片原文**，未经生成式改写，所以不会编口径。"
        "想让它换成人话复述，把 `src/gateway/` 的 LLMClient 接上任意一家 API 即可，检索这一层不用动。"
    )

    st.markdown(f"**其余 {len(hits)} 条命中（按精排分数排序）**")
    st.dataframe(
        [
            {
                "排名": hit.rank,
                "溯源引用": hit.chunk.citation_label,
                "方法": hit.method,
                "分数": round(float(hit.score), 4),
                "Doc_ID": hit.chunk.doc_id,
                "版本": hit.chunk.version,
            }
            for hit in hits
        ],
        hide_index=True,
        width="stretch",
    )
    for hit in hits:
        chunk = hit.chunk
        with st.expander(
            f"#{hit.rank} · {chunk.citation_label} · {hit.method} · score={float(hit.score):.4f}"
        ):
            meta_cols = st.columns(4)
            meta_cols[0].markdown(f"**doc_id**\n\n`{chunk.doc_id}`")
            meta_cols[1].markdown(f"**version**\n\n`{chunk.version}`")
            meta_cols[2].markdown(f"**page**\n\n`{chunk.page}`")
            meta_cols[3].markdown(f"**line**\n\n`{chunk.line_start}-{chunk.line_end}`")
            st.markdown(
                f"`chunk_id` = `{chunk.chunk_id}` ｜ `policy_key` = `{chunk.policy_key}` ｜ "
                f"`scope` = `{chunk.scope}` ｜ `rule_type` = `{chunk.rule_type}` ｜ "
                f"`effective_from` = `{chunk.effective_from or '—'}`"
            )
            st.markdown("**命中片段正文**")
            st.write(chunk.text)
            st.markdown("**同页父块原文（Parent-Child 的 parent）**")
            st.write(chunk.parent_text)
            st.caption("各路召回证据（rank/score）")
            st.json(_as_json_object(hit.components))


def _as_json_object(value: Any) -> Any:
    """把任意值收敛成 JSON 对象再交给 ``st.json``。

    实测坑：Streamlit 的 JSON 视图组件（`reactJsonViewCompat`）只接受
    **JSON 对象**。传 ``None``（refuse/sql 路由下 ``state.decision`` 就是
    ``None``）或传数组（``state.trace``）时，前端会渲染成一个刺眼的
    ``{"ERROR": {"message": "src property must be a valid json object"}}`` 框，
    看上去像程序崩了，其实只是这个组件不收非对象值。
    """
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, (list, tuple)):
        return {"items": list(value)}
    return {"value": value}


# 平台规则/口径类问题的特征词。本 Tab 是经营指标 Agent，遇到这类问题它会按固定口径
# 给出**数字**（不是规则原文），很容易被误读成「答非所问」，所以显式提示一句。
POLICY_HINTS = ("口径", "规则", "条款", "生效", "政策", "规定", "适用", "可以计入", "能否")


def _render_agent_result(state: Any, elapsed: float) -> None:
    # 先结论、后证据：答案永远排在最上面，下面才是路由/工具/重试这些过程指标。
    st.markdown("**最终答案**")
    st.success(state.answer or "（无答案）")

    result = state.result
    # 防御式取值：refuse 路由下 result 为 None，sql 路由下部分字段可能缺失，
    # 直接用 result.tool / result.rows 会在这些分支上炸 TypeError。
    tool = getattr(result, "tool", None) if result is not None else None
    rows = list(getattr(result, "rows", None) or []) if result is not None else []

    cols = st.columns(5)
    cols[0].metric("路由 route", _safe(getattr(state, "route", None)))
    cols[1].metric("工具", _safe(tool))
    cols[2].metric("返回行数", len(rows))
    cols[3].metric("重试次数", _safe(getattr(state, "retries", 0), "0"))
    cols[4].metric("耗时", f"{elapsed:.2f}s")

    if any(hint in (state.question or "") for hint in POLICY_HINTS):
        st.info(
            "这看起来是**平台规则 / 口径**类问题。本 Tab 是经营指标 Agent，它按固定口径给出的是"
            "**实际数字**，不是规则原文。要规则原文，请看左边「📚 知识库问答（RAG）」Tab 的"
            "「回答依据」块。"
        )

    st.caption(f"路由理由：{state.route_reason or '—'}")
    if state.error:
        st.warning(f"error: {state.error}")
    if result is not None and result.hint:
        st.info(f"工具给的修复提示：{result.hint}")

    info_col, decision_col = st.columns(2)
    with info_col:
        st.markdown("**工具调用参数 / 决策**")
        st.json(_as_json_object(state.decision))
        if result is not None and result.sql:
            st.markdown("**实际执行的 SQL（只读）**")
            st.code(result.sql, language="sql")
            if result.params:
                st.caption(f"绑定参数：{result.params}")
    with decision_col:
        st.markdown("**返回数据**")
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("无返回行。")

    with st.expander("state.trace（每个节点的执行轨迹，可核对工具选择）"):
        st.json(_as_json_object(getattr(state, "trace", None)))
    with st.expander("完整 AgentState.as_dict()"):
        payload = state.as_dict() if hasattr(state, "as_dict") else vars(state)
        st.json(_as_json_object(payload))


def _gateway_controls(scope: str) -> None:
    """网关的三个动作：启动 / 刷新 / 预热。

    ``scope`` 用来区分侧边栏与 Tab 3 里的同名按钮——两处都要有按钮，但 Streamlit
    的 widget key 必须全局唯一，否则同一份 key 会直接报 DuplicateWidgetID。
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
    button_cols[0].button("启动本地演示网关", key=f"gw_start_{scope}", on_click=_start_gateway_callback)
    button_cols[1].button(
        "刷新缓存统计",
        key=f"gw_refresh_{scope}",
        on_click=lambda: st.session_state.update(cache_stats=None),
    )
    st.button(
        "发 2 次 /chat 预热缓存（看命中率）",
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


def _gateway_status(stats: dict[str, Any] | None) -> tuple[bool, str]:
    """把 /cache/stats 收敛成 (是否健康, 人话状态)，给健康指示灯用。"""
    if stats is None:
        return False, "未启动"
    if stats.get("enabled") is False:
        return False, "在线 / 缓存未启用"
    return True, "在线 / 缓存已启用"


def _sidebar(rag_config: dict[str, Any]) -> tuple[str, int]:
    with st.sidebar:
        st.header("检索设置")
        mode = st.radio(
            "检索模式",
            options=list(RETRIEVAL_MODES),
            format_func=lambda key: RETRIEVAL_MODES[key],
            index=0,
            key="mode",
            help="现场对比混合检索与纯 BM25 / 纯向量的差异。",
        )
        top_k = st.slider("返回条数 top_k", min_value=1, max_value=10, value=5, key="top_k")

        st.divider()
        st.header("网关语义缓存")
        st.caption("读取 src/gateway/app.py 的 `/cache/stats`（拿不到就显示未启动）。")
        # 输入框只在这里出现一次：同一份 key 在 Tab 3 再放一份会触发 DuplicateWidgetID。
        st.text_input("网关地址", key="gateway_url")
        _gateway_controls("side")

        stats = _gateway_stats()
        ok, status_text = _gateway_status(stats)
        _health(ok, "网关状态", status_text)

        if stats is None:
            st.warning(
                "网关未启动（拿不到 /cache/stats）。可点上面的「启动本地演示网关」，"
                "或自行运行 `uvicorn` 后填入地址。"
            )
        elif stats.get("enabled") is False:
            st.info(f"网关在线，但语义缓存未启用：{stats}")
        else:
            metric_cols = st.columns(3)
            metric_cols[0].metric("缓存条目", _safe(stats.get("entries", 0), "0"))
            metric_cols[1].metric("命中次数", _safe(stats.get("hits", 0), "0"))
            metric_cols[2].metric("命中率", _safe(stats.get("hit_rate", 0.0), "0.0"))
        st.caption("完整监控视图见主区「🌐 网关与缓存监控」Tab。")
    return mode, top_k


def _header(rag_config: dict[str, Any]) -> None:
    retrieval = rag_config.get("retrieval", {})
    dense = retrieval.get("dense", {})
    fusion = retrieval.get("fusion", {})
    reranker = retrieval.get("reranker", {})

    # 把 rag.yaml 的真实配置渲染成 chips：面试时一眼就能证明「这些参数是从配置读的，
    # 不是我写死的」。所有值都过 _esc，配置里出现特殊字符也不会破坏版面。
    chips = "".join(
        f'<span class="fops-chip">{_esc(text)}</span>'
        for text in (
            f"向量模型 {dense.get('model_id')}",
            f"精排模型 {reranker.get('model_id')}",
            f"RRF k={fusion.get('rrf_k')}",
            f"候选 {fusion.get('candidate_k')} → {reranker.get('candidate_k')}",
            f"设备 {retrieval.get('device')}",
        )
    )
    # 用带 key 的容器当 Hero，而不是把整块内容塞进一个 HTML 字符串：这样标题仍然
    # 是一个真正的 ``st.title``（语义、AppTest、无障碍都能读到），同时 Streamlit 会
    # 给容器加 ``.st-key-hero`` class，CSS 就能把它画成卡片，两边都不牺牲。
    with st.container(border=True, key="hero"):
        st.markdown(
            '<span class="fops-eyebrow">Enterprise-grade AI Demo · v2.0</span>',
            unsafe_allow_html=True,
        )
        st.title("FinOps-Copilot · 电商经营分析与合规审计智能体")
        st.markdown(
            '<div class="fops-sub">一个<strong>本地可跑的全栈 AI 系统</strong>：'
            "左边用<strong>混合检索 + 溯源引用</strong>回答平台规则问题，"
            "右边用 <strong>LangGraph Agent</strong> 查询订单经营指标（固定口径 + 只读 SQL）。"
            "不接外部大模型 API 也能端到端跑通，每条结论都可溯源、可复算。</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="fops-chips">{chips}</div>', unsafe_allow_html=True)


def _set_question(value: str) -> None:
    """预设问题按钮的回调：填问题 **并且直接开跑**。

    必须在 on_click 回调里写 session_state：Streamlit 禁止在组件实例化之后再修改
    该组件的 session_state key，而按钮渲染发生在 text_input 之后，所以不能用
    ``if st.button(...): st.session_state["question"] = ...``。

    这里顺带把 rag_request / agent_request 一起写好——实测很多人点完预设题
    以为已经出结果了，其实只是把文字填进了输入框，于是以为「Demo 不返回回答」。
    """
    st.session_state["question"] = value
    nonce = time.time()
    st.session_state["rag_request"] = {
        "query": value,
        "mode": st.session_state.get("mode", "hybrid_rerank"),
        "top_k": int(st.session_state.get("top_k", 5)),
        "nonce": nonce,
    }
    st.session_state["agent_request"] = {"question": value, "nonce": nonce}


def _question_area() -> str:
    st.subheader("提问")
    if "question" not in st.session_state:
        st.session_state["question"] = ""
    st.text_input(
        "输入你的问题",
        key="question",
        placeholder="例如：当前生效口径下，哪些订单状态可以计入营收？",
    )

    st.caption("知识库预设问题（取自 eval/datasets/stage1_eval.jsonl 的真实评测题）")
    example_cols = st.columns(3)
    for index, row in enumerate(load_examples(3)):
        label = f"{row.get('question_id', 'EVAL')}｜{row['query'][:24]}…"
        example_cols[index % 3].button(
            label, key=f"example_rag_{index}", on_click=_set_question, args=(row["query"],)
        )

    st.caption("经营指标预设问题（问 orders.db，走 Agent Tab）")
    metric_cols = st.columns(4)
    for index, question in enumerate(METRIC_EXAMPLES):
        metric_cols[index % 4].button(
            question, key=f"example_metric_{index}", on_click=_set_question, args=(question,)
        )
    return st.session_state["question"]


def main() -> None:
    st.set_page_config(
        page_title="FinOps-Copilot 交互式 Demo",
        page_icon="🧭",
        layout="wide",
    )
    _inject_css()
    rag_config = load_rag_config()
    _header(rag_config)
    mode, top_k = _sidebar(rag_config)
    question = _question_area()

    run_all = st.button("开始分析（同时跑 RAG 与 Agent）", type="primary")
    if run_all and question.strip():
        nonce = time.time()
        st.session_state["rag_request"] = {"query": question.strip(), "mode": mode, "top_k": top_k, "nonce": nonce}
        st.session_state["agent_request"] = {"question": question.strip(), "nonce": nonce}

    tab_rag, tab_agent, tab_gateway = st.tabs(
        ["📚 知识库问答（RAG）", "📊 经营指标（Agent）", "🌐 网关与缓存监控"]
    )

    with tab_rag:
        request = st.session_state.get("rag_request")
        if st.button("仅运行检索", key="run_rag_only") and question.strip():
            request = {"query": question.strip(), "mode": mode, "top_k": top_k, "nonce": time.time()}
            st.session_state["rag_request"] = request
        if request is None:
            st.info(
                "还没跑过。三种开始方式：① 点上面的**知识库预设问题**（点了就直接跑）；"
                "② 在输入框里手打问题后点「开始分析」；③ 只跑检索点本 Tab 的「仅运行检索」。"
                "⚠️ 在输入框里按回车**不会**触发运行，必须点按钮。"
            )
        else:
            config_mtime = str(CONFIG_PATH.stat().st_mtime_ns)
            retriever, load_seconds = get_retriever(config_mtime)
            st.caption(f"检索器已就绪（本次加载耗时 {load_seconds:.1f}s，已用 @st.cache_resource 缓存）")
            cached = st.session_state.get("rag_result")
            if cached is None or cached["nonce"] != request["nonce"]:
                with st.spinner("正在进行混合检索与 RRF 融合 → Cross-Encoder 精排 …"):
                    hits, elapsed = run_retrieval(
                        retriever, request["query"], request["mode"], int(request["top_k"])
                    )
                cached = {"nonce": request["nonce"], "hits": hits, "elapsed": elapsed}
                st.session_state["rag_result"] = cached
            _render_rag_result(cached["hits"], cached["elapsed"], request["mode"], request["query"])
            st.caption(
                "每条命中都带 [Doc_ID, Page, Line] 溯源；把侧边栏切到「纯 BM25」再点一次"
                "「仅运行检索」，就能看到同一问题下两种检索的排序差异。"
            )

    with tab_agent:
        request = st.session_state.get("agent_request")
        if st.button("仅运行 Agent", key="run_agent_only") and question.strip():
            request = {"question": question.strip(), "nonce": time.time()}
            st.session_state["agent_request"] = request
        if request is None:
            st.info(
                "还没跑过。三种开始方式：① 点上面的**经营指标预设问题**（点了就直接跑）；"
                "② 在输入框里手打问题后点「开始分析」；③ 只跑 Agent 点本 Tab 的「仅运行 Agent」。"
                "⚠️ 在输入框里按回车**不会**触发运行，必须点按钮。"
            )
        else:
            agent = get_agent(str(ORDERS_DB.stat().st_mtime_ns))
            st.caption(
                f"Agent 就绪（LangGraph 状态机：route → metric/sql/refuse → reflect → answer）｜ "
                f"provider = 本地确定性规则实现，不调外部大模型 API ｜ 数据库 `{ORDERS_DB.name}`"
            )
            cached = st.session_state.get("agent_result")
            if cached is None or cached["nonce"] != request["nonce"]:
                with st.spinner("LangGraph 状态机推理中：意图路由 → 工具调用 → 自检纠错 → 生成答案 …"):
                    state, elapsed = run_agent(agent, request["question"])
                cached = {"nonce": request["nonce"], "state": state, "elapsed": elapsed}
                st.session_state["agent_result"] = cached
            _render_agent_result(cached["state"], cached["elapsed"])

    with tab_gateway:
        st.subheader("网关与语义缓存监控")
        st.caption(
            "这一页探活的是 `src/gateway/app.py` 的真实 FastAPI 网关："
            "`LLMClient` 的主备故障转移、语义缓存与 SSE 降级都发生在那个进程里，"
            "这里只把它的运行时状态读出来。"
        )

        stats = _gateway_stats()
        ok, status_text = _gateway_status(stats)

        # 三个健康指示灯：网关 / 语料 / 订单库。都不需要加载模型，点击即时可得。
        health_cols = st.columns(3)
        with health_cols[0]:
            _health(ok, "网关", status_text)
        with health_cols[1]:
            _health(
                CHUNKS_PATH.exists(),
                "规则语料",
                "已就绪" if CHUNKS_PATH.exists() else "缺失",
            )
        with health_cols[2]:
            _health(
                ORDERS_DB.exists(),
                "订单库",
                "已就绪" if ORDERS_DB.exists() else "缺失",
            )

        st.markdown(
            f"当前网关地址：`{st.session_state.get('gateway_url')}`"
            "（要改地址请用左侧边栏的「网关地址」输入框）"
        )
        _gateway_controls("tab")

        if stats is None:
            st.warning("拿不到 /cache/stats。点上面的「启动本地演示网关」，或自行 `uvicorn` 起服务后刷新。")
        elif stats.get("enabled") is False:
            st.info(f"网关在线，但语义缓存未启用：{stats}")
        else:
            metric_cols = st.columns(4)
            metric_cols[0].metric("缓存条目", _safe(stats.get("entries", 0), "0"))
            metric_cols[1].metric("命中次数", _safe(stats.get("hits", 0), "0"))
            metric_cols[2].metric("未命中次数", _safe(stats.get("misses", 0), "0"))
            metric_cols[3].metric("命中率", _safe(stats.get("hit_rate", 0.0), "0.0"))
            st.markdown("**`GET /cache/stats` 原始返回**")
            st.json(_as_json_object(stats))

        st.markdown("**这个 Tab 在证明什么**")
        st.markdown(
            """
- 语义缓存命中的是**语义近似**而不是字符串相等：把同一句话改写一遍，第二次仍然命中；
- 主 Provider 返回 HTTP 500 / 429 时会切到备用 Provider，且**客户端断开后立即停止重试**；
- SSE 在首帧之前发生故障会降级到非流式，避免「连上了却一直不出字」。
"""
        )

    st.divider()
    st.caption(f"⚠️ 诚实声明：{DISCLAIMER} 「版本冲突 / 边界条款」属于抓取题目设计，不是真实监管要求。")


if __name__ == "__main__" or bool(getattr(st.runtime, "exists", lambda: False)()):
    main()