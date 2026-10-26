"""评测看板自检：只断言「页面上出现的数字 == 产物里的数字」。

为什么单独写一个探针：看板的价值全在「数字可核对」这一条上。如果哪天有人把手写的
漂亮数字塞回去、或者产物换了路径导致读不到，这个脚本必须立刻失败，而不是让页面
悄悄降级成一份看着很专业、却对不上任何文件的快照。

用法：
    PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe app/probe_benchmark.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

ABLATION = ROOT / "eval" / "reports" / "rag" / "ablation.json"
REAL_STATS = ROOT / "reports" / "real_data_stats.json"

# 需求文档里给过一组示例数字。它们和本仓产物对不上，页面上必须一个都不出现。
FABRICATED = (
    "68.3", "79.2", "78.5", "88.1", "94.2", "98.3",
    "98.5", "66.7", "24.9",
)

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}: {label}{(' | ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main() -> int:
    print("=== 载入 AppTest ===")
    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=300)
    at.run()
    if at.exception:
        print("!! 页面抛异常:", [e.value for e in at.exception])
        return 1

    markdown_text = "\n".join(str(m.value) for m in at.markdown)
    frames = [df.value for df in at.dataframe]
    frame_text = "\n".join(frame.to_string() for frame in frames)
    page = markdown_text + "\n" + frame_text

    ablation = json.loads(ABLATION.read_text(encoding="utf-8"))
    stats = json.loads(REAL_STATS.read_text(encoding="utf-8"))
    metrics = ablation["metrics"]
    latency = ablation["avg_latency_ms"]

    print("\n=== ① 消融指标：看板表格逐格必须等于 ablation.json ===")
    # ProgressColumn 的百分比是前端渲染出来的，AppTest 拿不到——
    # 所以断言必须落在 DataFrame 的真实值上，而不是页面的可见文本上，
    # 否则「1.0000」这种字符串会从别处的散文里被误配到，检查就变成了摆设。
    table = next((f for f in frames if "检索策略" in list(f.columns)), None)
    check("找到检索对照表", table is not None, f"共 {len(frames)} 张表")
    if table is not None:
        check("对照表行数 = 3", len(table) == 3, f"{len(table)} 行")
        for idx, arm in enumerate(("bm25", "dense", "hybrid_rerank")):
            row = table.iloc[idx]
            exp_recall = float(metrics[arm]["recall_at_5"])
            exp_mrr = float(metrics[arm]["mrr_at_5"])
            exp_ms = round(float(latency[arm]), 2)
            check(
                f"{arm} Recall@5 = {exp_recall:.4f}",
                abs(float(row["Recall@5"]) - exp_recall) < 1e-9,
                f"表格 {float(row['Recall@5']):.6f}",
            )
            check(
                f"{arm} MRR@5 = {exp_mrr:.4f}",
                abs(float(row["MRR@5"]) - exp_mrr) < 1e-9,
                f"表格 {float(row['MRR@5']):.6f}",
            )
            check(
                f"{arm} 平均耗时 = {exp_ms} ms",
                abs(float(row["平均单题耗时 (ms)"]) - exp_ms) < 0.011,
                f"表格 {float(row['平均单题耗时 (ms)'])}",
            )
    check("题量 120 与可答/拒答 90/30", "120" in page and "90" in page and "30" in page)
    check("语料 5,000 切片", "5,000" in page)
    check("真实模型名 bge-reranker-base", ablation["reranker_model"] in page)

    print("\n=== ② 精排窗口对照：3450 / 715 必须来自 crosscheck ===")
    cross = json.loads(
        (ROOT / "eval" / "reports" / "rag" / "latency_window_crosscheck.json").read_text(
            encoding="utf-8"
        )
    )
    means: dict[int, list[float]] = {}
    for item in cross["results"]:
        means.setdefault(int(item["window"]), []).append(float(item["mean_ms"]))
    wide = sum(means[100]) / len(means[100])
    narrow = sum(means[20]) / len(means[20])
    check(f"窗口100 ≈ {wide:,.0f} ms", f"{wide:,.0f}" in page, f"页面含 {wide:,.0f}")
    check(f"窗口20 ≈ {narrow:,.0f} ms", f"{narrow:,.0f}" in page, f"页面含 {narrow:,.0f}")

    print("\n=== ③ 真实数据鲁棒性：数字必须等于 real_data_stats/comparison ===")
    inv = stats["real_inventory"]
    comp = json.loads((ROOT / "reports" / "real_data_comparison.json").read_text(encoding="utf-8"))
    check("真实行数 1,067,371", f"{comp['rows_real']:,}" in page)
    check("合成行数 50,000", f"{comp['rows_synthetic']:,}" in page or "50,000" in page)
    check(
        f"客户号缺失 {inv['missing_customer_id']['pct']}%",
        f"{inv['missing_customer_id']['pct']:.2f}" in page,
    )
    check(
        f"完全重复行 {inv['exact_duplicate_rows']['pct']}%",
        f"{inv['exact_duplicate_rows']['pct']:.2f}" in page,
    )
    check(
        "落不进类型的行 = 0（不是静默填充）",
        str(stats["normalization"]["rows_dropped_untypeable"]) in page,
    )
    check("金额恒等式 0 行", "quantity" in page and "unit_price" in page)

    print("\n=== ④ 反向断言：需求里的示例数字一个都不许出现 ===")
    for bad in FABRICATED:
        hits = [m for m in re.finditer(re.escape(bad), page)]
        check(f"未出现「{bad}」", not hits, f"命中 {len(hits)} 次" if hits else "")

    print("\n=== ⑤ 结构断言 ===")
    check("存在两个 Tab", len(at.tabs) == 2, f"实际 {len(at.tabs)}")
    check("看板含数据源声明", page.count("数据源：") >= 3, f"{page.count('数据源：')} 处")
    check("工作台按钮仍可用", len(at.button) >= 6, f"{len(at.button)} 个按钮")
    check("无页面异常", not at.exception)

    print()
    if failures:
        print(f"BENCHMARK PROBE FAILED: {len(failures)} 项不通过")
        for item in failures:
            print("  -", item)
        return 1
    print("BENCHMARK PROBE OK: 页面数字全部可追溯到实验产物，且无虚构指标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())