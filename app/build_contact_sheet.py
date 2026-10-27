"""把 Demo 截图拼成一张总览图，便于放进 README / 直接给人看。

面板有两种排布：普通面板两列并排；``wide=True`` 的面板（比如整页长图的
「评测基准」看板）独占一行、按整幅宽度缩放 —— 否则它的高度会被当成所有行的
行高，把其它格子撑出大片空白。
"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(os.environ.get("DEMO_OUT", Path(__file__).resolve().parents[1] / "docs" / "demo"))
OUT = HERE / "00-Demo总览.png"

# (文件名, 说明, 是否独占整行)
PANELS = [
    ("01-首页.png", "① 首页：三栏工作台（左=引擎配置与监控 / 中=业务决策 / 右=证据链与审计）", False),
    ("02-RAG检索结果.png", "② 合规依据：命中生效口径原文，带 [DOC_ID] · Page · Line 溯源徽标与相关性", False),
    ("03-溯源明细展开.png", "③ 证据链：切片的 policy_key / scope / effective_from 与同页父块原文", False),
    ("04-Agent业务决策.png", "④ 业务决策：LangGraph 路由指标卡 + 业务指标看板 + 实际执行的只读 SQL", False),
    ("05-安全护栏拦截.png", "⑤ 安全护栏：问「删掉所有订单」→ refuse 路由，不产出任何写 SQL", False),
    ("06-纯BM25排序.png", "⑥ 策略对照：纯 BM25 把已废止的 POL-013-V1 排在第一位", False),
    ("07-混合检索精排排序.png", "⑦ 策略对照：混合检索+精排把现行生效的 POL-013-V2 提到第一位", False),
    ("08-对抗A-版本口径冲突.png", "⑧ 对抗演示 A：同一问句下 V1(已废止) vs V2(现行) 并排，附耗时与置顶结论", False),
    ("09-对抗B-写操作拦截.png", "⑨ 对抗演示 B：真实 DELETE 语句 → 403 回执 + 右栏 route/refuse 节点高亮", False),
    (
        "10-评测基准看板.png",
        "⑩ 评测基准与鲁棒性：120 题全量消融的 Recall@5/MRR@5/耗时（真实 BGE 现推现算，非估算）"
        " · 精排窗口调优 · 状态机与只读护栏 · UCI 真实数据鲁棒性（每个数字都标了源文件）",
        True,
    ),
    (
        "11-架构边界与设计权衡.png",
        "⑪ 架构边界与设计权衡（展开态）：逐条分清哪些是真算法、哪些是演示替身，"
        "并列出只读护栏的四层库内强制机制与代码位置（应用层前置校验单列一行，不计入这四层）",
        True,
    ),
]

COLS = 2
THUMB_W = 660
CAP_H = 46
PAD = 16
BG = (248, 249, 251)
CAP_BG = (255, 255, 255)
CAP_FG = (32, 34, 38)


def load_font(size: int):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> int:
    sheet_w = COLS * THUMB_W + (COLS + 1) * PAD
    wide_w = sheet_w - 2 * PAD

    # 先把每个面板缩放到目标宽度，再按「普通两列 / 整行独占」分组
    loaded = []
    for fname, caption, wide in PANELS:
        src = HERE / fname
        if not src.exists():
            print(f"  跳过（不存在）：{fname}")
            continue
        im = Image.open(src).convert("RGB")
        target_w = wide_w if wide else THUMB_W
        ratio = target_w / im.width
        im = im.resize((target_w, int(im.height * ratio)), Image.LANCZOS)
        loaded.append((im, caption, wide))
        print(f"  已载入 {fname}  {im.width}x{im.height}{'  [整行]' if wide else ''}")

    if not loaded:
        print("没有任何截图")
        return 1

    # 普通行的高度由普通面板里的最高者决定；整行面板不吃这个约束
    normal_h = max((t[0].height for t in loaded if not t[2]), default=0) + CAP_H
    rows: list[tuple[str, list, int]] = []
    buf: list = []
    for tile in loaded:
        if tile[2]:
            if buf:
                rows.append(("normal", buf, normal_h))
                buf = []
            rows.append(("wide", [tile], tile[0].height + CAP_H))
        else:
            buf.append(tile)
            if len(buf) == COLS:
                rows.append(("normal", buf, normal_h))
                buf = []
    if buf:
        rows.append(("normal", buf, normal_h))

    sheet_h = sum(h for _, _, h in rows) + (len(rows) + 1) * PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)
    font = load_font(19)

    y = PAD
    for kind, tiles, row_h in rows:
        for idx, (im, caption, _wide) in enumerate(tiles):
            x = PAD if kind == "wide" else PAD + idx * (THUMB_W + PAD)
            draw.rectangle([x, y, x + im.width, y + CAP_H], fill=CAP_BG)
            draw.text((x + 12, y + 12), caption, fill=CAP_FG, font=font)
            sheet.paste(im, (x, y + CAP_H))
        y += row_h + PAD

    sheet.save(OUT, optimize=True)
    print(f"\n总览图 -> {OUT}  ({sheet.width}x{sheet.height}, {OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())