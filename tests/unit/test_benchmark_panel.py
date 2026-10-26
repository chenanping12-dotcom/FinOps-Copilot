"""守卫「📈 评测基准与鲁棒性」Tab 的口径：页面上的数字必须来自仓库产物。

这个测试存在的意义：看板最容易腐化的方式是「有人嫌读 JSON 麻烦，直接把指标抄成
常量」。那样页面看着没变，但产物更新后界面就成了假数据，而面试官又要拿它去对账。

所以这里不比对任何硬编码期望值，而是**现读产物、再断言页面渲染出了同样的值**：

    产物改了 → 页面必须跟着改，否则红；
    页面被手抄成别的值 → 红。

运行（在 FinOps-Copilot 目录下）：
    .venv/Scripts/python.exe -m pytest tests/unit/test_benchmark_panel.py -q
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

ABLATION = ROOT / "eval" / "reports" / "rag" / "ablation.json"
REAL_STATS = ROOT / "reports" / "real_data_stats.json"
GATEWAY_METRICS = ROOT / "reports" / "stage3_gateway_metrics.json"
PROBE = ROOT / "eval" / "reports" / "rag" / "probe_eval116.json"

ARMS = ("bm25", "dense", "hybrid_rerank")
ARM_HINT = {"bm25": "BM25", "dense": "Dense", "hybrid_rerank": "Cross-Encoder"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value) -> float:
    """把「22.77%」这类展示串还原成数字。"""
    text = str(value).strip().rstrip("%").replace(",", "")
    return float(text)


@pytest.fixture(scope="module")
def app() -> AppTest:
    """渲染一次页面。Tab 内容在 Streamlit 里是全量下发的，不需要真的去点 Tab。"""
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=900)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture(scope="module")
def markdown_text(app: AppTest) -> str:
    return "\n".join(str(m.value) for m in app.markdown)


def _frame_with(app: AppTest, column: str):
    for frame in app.dataframe:
        value = frame.value
        if column in getattr(value, "columns", []):
            return value
    raise AssertionError(f"页面上找不到含列「{column}」的表格")


def test_页面本身不能崩(app: AppTest) -> None:
    assert not app.exception, "页面渲染抛异常"


# --------------------------------------------------------------------------
# ① 离线检索基准：三臂 Recall / MRR / 耗时
# --------------------------------------------------------------------------
def test_检索基准表与_ablation_json_逐格一致(app: AppTest) -> None:
    payload = _load(ABLATION)
    metrics = payload["metrics"]
    latency = payload["avg_latency_ms"]
    frame = _frame_with(app, "Recall@5")

    assert len(frame) == 3, "应当是 BM25 / Dense / 混合精排三臂"

    for arm in ARMS:
        row = frame[frame["检索策略"].str.contains(ARM_HINT[arm], regex=False)]
        assert len(row) == 1, f"{arm} 这一臂没在表里"
        assert row.iloc[0]["Recall@5"] == pytest.approx(metrics[arm]["recall_at_5"])
        assert row.iloc[0]["MRR@5"] == pytest.approx(metrics[arm]["mrr_at_5"])
        assert row.iloc[0]["平均单题耗时 (ms)"] == pytest.approx(
            round(latency[arm], 2), abs=0.01
        )


def test_触顶必须主动披露而不是当成卖点(app: AppTest, markdown_text: str) -> None:
    """只要有臂的 Recall 触顶，页面就必须写明该指标没有区分度。

    这里刻意不用「三臂全 1.0」当触发条件 —— Dense 是 0.9667，那样写这段披露会变成
    死代码（本项目实际踩过这个坑，所以单独测它）。
    """
    metrics = _load(ABLATION)["metrics"]
    ceiling = [a for a in ARMS if abs(float(metrics[a]["recall_at_5"]) - 1.0) < 1e-9]
    if not ceiling:
        pytest.skip("没有臂在 Recall 上触顶，本断言前提不成立")

    assert "主动披露" in markdown_text, f"{ceiling} 已触顶，但页面没有主动披露"
    assert "MRR" in markdown_text, "披露里必须把结论导向 MRR"


# --------------------------------------------------------------------------
# 每个数字都要能当场核对：源文件路径 + 产物生成时间
# --------------------------------------------------------------------------
def test_每节的源文件路径都印在页面上(app: AppTest, markdown_text: str) -> None:
    for path in (ABLATION, REAL_STATS, GATEWAY_METRICS):
        rel = path.relative_to(ROOT).as_posix()
        assert rel in markdown_text, f"页面上没标出数据源 {rel}"


def test_产物生成时间印在页面上(app: AppTest, markdown_text: str) -> None:
    when = _load(ABLATION).get("generated_at")
    assert when, "ablation.json 里没有 generated_at"
    assert when in markdown_text, "页面上没印产物生成时间，面试官无法拿它去 git log 对账"


# --------------------------------------------------------------------------
# ④ 真实公开数据：行数与脏数据占比
# --------------------------------------------------------------------------
def test_真实数据行数与_stats_json_一致(app: AppTest, markdown_text: str) -> None:
    inventory = _load(REAL_STATS)["real_inventory"]
    total = inventory["rows_total"]
    # KPI 卡片走的是 st.markdown（自定义 HTML），不是 st.metric，所以查 markdown。
    assert f"{total:,}" in markdown_text, f"页面上的真实行数不是 {total:,}"


def test_脏数据占比与_stats_json_逐行一致(app: AppTest) -> None:
    inventory = _load(REAL_STATS)["real_inventory"]
    frame = _frame_with(app, "脏数据类型")
    column = next((c for c in frame.columns if "真实数据占比" in str(c)), None)
    assert column, f"对照表缺少「真实数据占比」列，实际列：{list(frame.columns)}"

    # 页面上每行的中文标签 → 产物里的键名
    mapping = {
        "缺失": "missing_customer_id",
        "负数量": "negative_quantity",
        "单价为0": "price_zero",
        "重复": "exact_duplicate_rows",
        "以C开头": "cancel_invoice_c",
    }
    checked = 0
    for label, key in mapping.items():
        if key not in inventory or "pct" not in inventory[key]:
            continue
        row = frame[frame["脏数据类型"].astype(str).str.contains(label, regex=False)]
        if row.empty:
            continue
        assert _num(row.iloc[0][column]) == pytest.approx(
            float(inventory[key]["pct"]), abs=5e-3
        ), f"{label} 这一行的占比与 real_data_stats.json 对不上"
        checked += 1
    assert checked >= 3, "对照表里没核到几行，映射可能失效了"


def test_契约外行数必须写明是零而不是悄悄填充(app: AppTest, markdown_text: str) -> None:
    normalization = _load(REAL_STATS)["normalization"]
    dropped = normalization["rows_dropped_untypeable"]
    assert dropped == 0
    assert "无法落进契约类型的行" in markdown_text, "缺少「不静默篡改」的说明"
    assert "严格丢弃并计数" in markdown_text or "绝不静默填充" in markdown_text


# --------------------------------------------------------------------------
# 反向守卫：展示层不许再出现「手抄的数据数字」
# --------------------------------------------------------------------------
BENCH_RENDER_FUNCS = {
    "_bench_src_line",
    "_bench_kpis",
    "_bench_note",
    "_bench_verdict",
    "_bench_retrieval",
    "_bench_latency_window",
    "_bench_state_machine",
    "_bench_real_data",
    "_render_benchmark",
    "_pct_rows",
    "_injected_anomalies",
    "_identity_failures",
    "_archive_note",
    "_score_spread",
    "_rule_documents",
    "_gold_rank",
    # 架构边界面板与看板同属「对外自证」的展示层，同样不许手抄数字。
    "_render_arch_boundaries",
}

DATA_NUMBER = (
    re.compile(r"\d{1,3}(?:,\d{3})+"),            # 千分位：243,007
    re.compile(r"\d+(?:\.\d+)?\s*%"),             # 百分比：3.217%
    re.compile(r"\d+\.\d{2,}"),                   # 精确小数：0.9368
    re.compile(r"\d+\s*(?:ms|条|行|题|单|次)"),   # 带单位：3450 ms
)

# 允许保留的：键为「函数名:命中串」，值为理由。目前为空——所有数据数字都应直读产物。
ALLOWED_HANDWRITTEN: dict[str, str] = {}


def _rendered_string_literals() -> list[tuple[int, str, str]]:
    """扫出看板渲染路径里的字符串常量（跳过 docstring），连同命中的裸数字。"""
    tree = ast.parse((ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8"))
    hits: list[tuple[int, str, str]] = []
    for fn in [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in BENCH_RENDER_FUNCS
    ]:
        doc = ast.get_docstring(fn, clean=False)
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if doc is not None and node.value == doc:
                continue
            for pattern in DATA_NUMBER:
                for match in pattern.finditer(node.value):
                    hits.append((node.lineno, fn.name, match.group(0).strip()))
    return hits


def test_展示层不得再出现手抄的数据数字() -> None:
    """任何「看起来是实验结果」的数字，都必须来自产物，不能是字符串常量。

    本项目实际踩过这个坑（所以单独测它）：
      * KPI 卡片写死 `"0 行"`，同一个列表里的兄弟卡片却是直读的；
      * 正文写死 `243,007 行` / `3.217%（34,337 行）` / `1.83%（19,494 行）`；
      * 正文写死 `0.9964–0.9999`（重排分值）——它同时**错标成了「文本相似度」**，
        而产物里 `lexical_context[].similarity_to_gold` 才是文本相似度（0.9368–0.9468）。
        两个量差一个数量级，混用就是把结论说反，见下一个测试。

    数字一旦抄进字符串，产物更新后就没人会发现它已经过期，面试官拿去对账就会翻车。
    这个测试是「棘轮」：新增的手抄数字必须先进 ALLOWED_HANDWRITTEN 并写明理由。
    """
    hits = [
        hit
        for hit in _rendered_string_literals()
        if f"{hit[1]}:{hit[2]}" not in ALLOWED_HANDWRITTEN
    ]
    assert not hits, (
        "看板渲染路径里出现了手抄的数据数字（应改为从产物直读）：\n"
        + "\n".join(f"  L{lineno} {name}()  「{hit}」" for lineno, name, hit in sorted(hits))
    )

def test_词面相似度与重排分值不得混为一谈(app: AppTest, markdown_text: str) -> None:
    """EVAL-116 那段里有两个量，差一个数量级，标错一个词结论就反了。

      * `lexical_context[].similarity_to_gold` → **词面**相似度（0.93 量级）
      * `cross_encoder_scores[].rerank_score`  → **重排**分值（0.99 量级）

    本项目实际踩过：把重排分值的区间当成「文本相似度」写进正文。所以这里要求
    页面**同时**渲染两个量，且各自对得上产物——只写一个就说明它们又被合并了。
    """
    probe = _load(PROBE)
    lexical = [float(item["similarity_to_gold"]) for item in probe["lexical_context"]]
    rerank = [float(item["rerank_score"]) for item in probe["cross_encoder_scores"]]
    assert lexical and rerank, "产物结构变了，本测试的前提要跟着改"
    assert max(lexical) < min(rerank), (
        "两个量本该差一个数量级；若产物真的变了，请先确认语义再改这个断言"
    )

    assert f"{min(lexical):.4f}–{max(lexical):.4f}" in markdown_text, (
        "词面相似度区间没有按产物渲染（lexical_context[].similarity_to_gold）"
    )
    assert f"{min(rerank):.4f}–{max(rerank):.4f}" in markdown_text, (
        "重排分值区间没有按产物渲染（cross_encoder_scores[].rerank_score）"
    )


# ---------------------------------------------------------------------------
# 架构边界页：这一页的卖点就是「主动交代边界」，所以它自己最不能说过头的话
# ---------------------------------------------------------------------------

MANIFEST = ROOT / "reports" / "stage1_manifest.json"


def _frame_text(frame) -> str:
    return "\n".join(str(value) for value in frame.to_numpy().ravel())


def test_架构边界页把五层只读护栏逐层列清(app: AppTest) -> None:
    """护栏不是一句「我们做了只读」，是五层独立机制；页面必须逐层给机制和代码位置。"""
    from app.streamlit_app import READONLY_LAYERS

    assert len(READONLY_LAYERS) == 5, "护栏层数变了：先确认 src/agent/tools.py 再改这里"
    frame = _frame_with(app, "层")
    assert list(frame["层"]) == [layer for layer, _, _ in READONLY_LAYERS]
    text = _frame_text(frame)
    for layer, how, where in READONLY_LAYERS:
        assert how[:10] in text, f"{layer} 只给了名词，没给机制"
        assert where in text, f"{layer} 没标出代码位置，面试官没法当场核对"


def test_护栏表引用的符号必须真的存在于_tools_py(app: AppTest) -> None:
    """表里写的是「可核对位置」，那就必须真能核对到——符号名要在源码里找得到。"""
    tree = ast.parse((ROOT / "src" / "agent" / "tools.py").read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    defined |= {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    frame = _frame_with(app, "层")
    for where in frame["代码位置"]:
        for symbol in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(where)):
            if symbol in {"tools", "py"}:
                continue
            assert symbol in defined, f"面板引用的符号在 tools.py 里不存在：{symbol}"


def test_架构边界页的数字与产物一致(markdown_text: str) -> None:
    """真实组件那一节报的规模，必须与 reports/stage1_manifest.json 一致。"""
    manifest = _load(MANIFEST)
    summary = manifest["summary"]
    stats = _load(REAL_STATS)["real_inventory"]

    assert f"{summary['orders']:,}" in markdown_text, "合成订单量没有按 manifest 渲染"
    assert f"{summary['rule_chunks']:,}" in markdown_text, "规则切片数没有按 manifest 渲染"
    assert f"{summary['rule_documents']:,}" in markdown_text, "规则文档数没有按 manifest 渲染"
    assert str(manifest["evaluation"]["qa_pairs"]) in markdown_text, "评测题数没有按 manifest 渲染"
    assert str(stats["missing_customer_id"]["pct"]) in markdown_text, (
        "真实数据缺失率没有按 real_data_stats.json 渲染"
    )


def test_架构边界页的长文本不得退回_dataframe(app: AppTest) -> None:
    """三列中文在 1100px 通栏里排不下，``st.dataframe`` 会横向截断「真实实现」。

    实测：右栏（409px）时「可核对位置」整列右半被切掉；通栏也会截断检索链路那行的
    模型名与 RRF k 值。这两列正是这一页的全部信息，所以必须走会折行的卡片。
    """
    leaked = [
        list(getattr(frame.value, "columns", []))
        for frame in app.dataframe
        if {"环节", "边界"} & set(getattr(frame.value, "columns", []))
    ]
    assert not leaked, f"架构边界的长文本表又变回了 st.dataframe，会被横向截断：{leaked}"


def test_不得再自称做过_AST_解析() -> None:
    """本项目从未做过语法树解析：第 ④ 层是正则 + 字符串检查。

    唯一的例外是那条自我纠正说明——留着它比删掉更能证明这页是核过的。
    """
    offenders = [
        f"L{i}: {line.strip()}"
        for i, line in enumerate(
            (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8").splitlines(), 1
        )
        if "AST" in line and "自我纠正" not in line and "用词夸大" not in line
    ]
    assert not offenders, (
        "页面又出现了 AST 宣称（真正的结构校验在 SQLite authorizer，不在我们代码里）：\n"
        + "\n".join(offenders)
    )


def test_边界页必须自己承认哪一层是替身(markdown_text: str) -> None:
    """只讲「我们都做了什么」，不讲「哪一层是替身」，这一页就没有意义。"""
    assert "LocalDemoProvider" in markdown_text, "没点名自然语言生成层是本地确定性替身"
    assert "接回真实组件" in markdown_text, "替身表没有「怎么接回真实组件」这层说明"
