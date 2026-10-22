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
"""

from __future__ import annotations

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
    st.caption(f"模式 `{mode}` ｜ 命中 {len(hits)} 条 ｜ 检索耗时 {elapsed:.2f}s ｜ 查询：{query}")
    if not hits:
        st.warning("没有命中任何切片。")
        return
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
            st.json(hit.components)


def _render_agent_result(state: Any, elapsed: float) -> None:
    st.markdown("**最终答案**")
    st.success(state.answer or "（无答案）")

    result = state.result
    cols = st.columns(5)
    cols[0].metric("路由 route", state.route or "—")
    cols[1].metric("工具", result.tool if result else "—")
    cols[2].metric("返回行数", len(result.rows) if result else 0)
    cols[3].metric("重试次数", state.retries)
    cols[4].metric("耗时", f"{elapsed:.2f}s")

    st.caption(f"路由理由：{state.route_reason or '—'}")
    if state.error:
        st.warning(f"error: {state.error}")
    if result is not None and result.hint:
        st.info(f"工具给的修复提示：{result.hint}")

    info_col, decision_col = st.columns(2)
    with info_col:
        st.markdown("**工具调用参数 / 决策**")
        st.json(state.decision)
        if result is not None and result.sql:
            st.markdown("**实际执行的 SQL（只读）**")
            st.code(result.sql, language="sql")
            if result.params:
                st.caption(f"绑定参数：{result.params}")
    with decision_col:
        st.markdown("**返回数据**")
        if result is not None and result.rows:
            st.dataframe(result.rows, hide_index=True, width="stretch")
        else:
            st.caption("无返回行。")

    with st.expander("state.trace（每个节点的执行轨迹，可核对工具选择）"):
        st.json(state.trace)
    with st.expander("完整 AgentState.as_dict()"):
        st.json(state.as_dict())


def _sidebar(rag_config: dict[str, Any]) -> tuple[str, int]:
    with st.sidebar:
        st.header("检索设置")
        mode = st.radio(
            "检索模式",
            options=list(RETRIEVAL_MODES),
            format_func=lambda key: RETRIEVAL_MODES[key],
            index=0,
            help="现场对比混合检索与纯 BM25 / 纯向量的差异。",
        )
        top_k = st.slider("返回条数 top_k", min_value=1, max_value=10, value=5)

        st.divider()
        st.header("网关语义缓存")
        st.caption("读取 src/gateway/app.py 的 `/cache/stats`（拿不到就显示未启动）。")
        if "gateway_url" not in st.session_state:
            st.session_state["gateway_url"] = os.environ.get(
                "FINOPS_GATEWAY_URL", "http://127.0.0.1:8000"
            )

        def _start_gateway_callback() -> None:
            error = start_demo_gateway(8010)
            if error:
                st.session_state["gateway_error"] = error
                return
            st.session_state.pop("gateway_error", None)
            st.session_state["gateway_url"] = "http://127.0.0.1:8010"
            st.session_state["cache_stats"] = probe_gateway(
                "http://127.0.0.1:8010", timeout=1.0, attempts=8
            )

        def _warm_cache_callback() -> None:
            st.session_state["cache_stats"] = warm_gateway_cache(st.session_state["gateway_url"])

        st.text_input("网关地址", key="gateway_url")
        button_cols = st.columns(2)
        button_cols[0].button("启动本地演示网关", on_click=_start_gateway_callback)
        button_cols[1].button("刷新缓存统计", on_click=lambda: st.session_state.update(cache_stats=None))
        st.button("发 2 次 /chat 预热缓存（看命中率）", on_click=_warm_cache_callback)
        if st.session_state.get("gateway_error"):
            st.error(st.session_state["gateway_error"])

        if st.session_state.get("cache_stats") is None:
            st.session_state["cache_stats"] = probe_gateway(st.session_state["gateway_url"])
        stats = st.session_state.get("cache_stats")
        if stats is None:
            st.warning(
                "网关未启动（拿不到 /cache/stats）。可点上面的「启动本地演示网关」，"
                "或自行运行 `uvicorn` 后填入地址。"
            )
        elif stats.get("enabled") is False:
            st.info(f"网关在线，但语义缓存未启用：{stats}")
        else:
            metric_cols = st.columns(3)
            metric_cols[0].metric("缓存条目", stats.get("entries", 0))
            metric_cols[1].metric("命中次数", stats.get("hits", 0))
            metric_cols[2].metric("命中率", stats.get("hit_rate", 0.0))
            st.json(stats)
    return mode, top_k


def _header(rag_config: dict[str, Any]) -> None:
    retrieval = rag_config.get("retrieval", {})
    dense = retrieval.get("dense", {})
    fusion = retrieval.get("fusion", {})
    reranker = retrieval.get("reranker", {})
    st.title("FinOps-Copilot · 电商经营分析与合规审计智能体")
    st.markdown(
        "一个本地可跑的全栈 AI 系统：左边用 **混合检索 + 溯源引用** 回答平台规则问题，"
        "右边用 **LangGraph Agent** 查询订单经营指标（固定口径 + 只读 SQL）。"
    )
    st.caption(
        f"检索配置（config/rag.yaml）：向量模型 `{dense.get('model_id')}` ｜ "
        f"精排模型 `{reranker.get('model_id')}` ｜ RRF `k={fusion.get('rrf_k')}` ｜ "
        f"融合候选 `candidate_k={fusion.get('candidate_k')}` → 精排候选 "
        f"`reranker.candidate_k={reranker.get('candidate_k')}` ｜ 设备 `{retrieval.get('device')}`"
    )


def _set_question(value: str) -> None:
    """预设问题按钮的回调。

    必须在 on_click 回调里写 session_state：Streamlit 禁止在组件实例化之后再修改
    该组件的 session_state key，而按钮渲染发生在 text_input 之后，所以不能用
    ``if st.button(...): st.session_state["question"] = ...``。
    """
    st.session_state["question"] = value


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
    rag_config = load_rag_config()
    _header(rag_config)
    mode, top_k = _sidebar(rag_config)
    question = _question_area()

    run_all = st.button("开始分析（同时跑 RAG 与 Agent）", type="primary")
    if run_all and question.strip():
        nonce = time.time()
        st.session_state["rag_request"] = {"query": question.strip(), "mode": mode, "top_k": top_k, "nonce": nonce}
        st.session_state["agent_request"] = {"question": question.strip(), "nonce": nonce}

    tab_rag, tab_agent = st.tabs(["📚 知识库问答（RAG）", "📊 经营指标（Agent）"])

    with tab_rag:
        request = st.session_state.get("rag_request")
        if st.button("仅运行检索", key="run_rag_only") and question.strip():
            request = {"query": question.strip(), "mode": mode, "top_k": top_k, "nonce": time.time()}
            st.session_state["rag_request"] = request
        if request is None:
            st.info("先在下方输入问题（或用上方预设问题按钮），再点「开始分析」或「仅运行检索」。")
        else:
            config_mtime = str(CONFIG_PATH.stat().st_mtime_ns)
            retriever, load_seconds = get_retriever(config_mtime)
            st.caption(f"检索器已就绪（本次加载耗时 {load_seconds:.1f}s，已用 @st.cache_resource 缓存）")
            cached = st.session_state.get("rag_result")
            if cached is None or cached["nonce"] != request["nonce"]:
                with st.spinner("检索中：BM25 + 向量召回 → RRF 融合 → 精排 …"):
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
            st.info("输入一个经营指标问题（如「各品类净营收是多少？」），再点「开始分析」或「仅运行 Agent」。")
        else:
            agent = get_agent(str(ORDERS_DB.stat().st_mtime_ns))
            st.caption(
                f"Agent 就绪（LangGraph 状态机：route → metric/sql/refuse → reflect → answer）｜ "
                f"provider = 本地确定性规则实现，不调外部大模型 API ｜ 数据库 `{ORDERS_DB.name}`"
            )
            cached = st.session_state.get("agent_result")
            if cached is None or cached["nonce"] != request["nonce"]:
                with st.spinner("Agent 执行中：路由 → 工具调用 → 自我纠错/回答 …"):
                    state, elapsed = run_agent(agent, request["question"])
                cached = {"nonce": request["nonce"], "state": state, "elapsed": elapsed}
                st.session_state["agent_result"] = cached
            _render_agent_result(cached["state"], cached["elapsed"])

    st.divider()
    st.caption(f"⚠️ 诚实声明：{DISCLAIMER} 「版本冲突 / 边界条款」属于抓取题目设计，不是真实监管要求。")


if __name__ == "__main__" or bool(getattr(st.runtime, "exists", lambda: False)()):
    main()