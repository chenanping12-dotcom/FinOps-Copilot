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

import json
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