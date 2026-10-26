"""把 Demo 截图拼成一张总览图，便于放进 README / 直接给人看。"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(os.environ.get("DEMO_OUT", Path(__file__).resolve().parents[1] / "docs" / "demo"))
OUT = HERE / "00-Demo总览.png"

PANELS = [
    ("01-首页.png", "① 首页：三栏工作台（左=引擎配置与监控 / 中=业务决策 / 右=证据链与审计）"),
    ("02-RAG检索结果.png", "② 合规依据：命中生效口径原文，带 [DOC_ID] · Page · Line 溯源徽标与相关性"),
    ("03-溯源明细展开.png", "③ 证据链：切片的 policy_key / scope / effective_from 与同页父块原文"),
    ("04-Agent业务决策.png", "④ 业务决策：LangGraph 路由指标卡 + 业务指标看板 + 实际执行的只读 SQL"),
    ("05-安全护栏拦截.png", "⑤ 安全护栏：问「删掉所有订单」→ refuse 路由，不产出任何写 SQL"),
    ("06-纯BM25排序.png", "⑥ 策略对照：纯 BM25 把已废止的 POL-013-V1 排在第一位"),
    ("07-混合检索精排排序.png", "⑦ 策略对照：混合检索+精排把现行生效的 POL-013-V2 提到第一位"),
    ("08-对抗A-版本口径冲突.png", "⑧ 对抗演示 A：同一问句下 V1(已废止) vs V2(现行) 并排，附耗时与置顶结论"),
    ("09-对抗B-写操作拦截.png", "⑨ 对抗演示 B：真实 DELETE 语句 → 403 回执 + 右栏 route/refuse 节点高亮"),
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
    tiles = []
    font = load_font(19)
    for fname, caption in PANELS:
        src = HERE / fname
        if not src.exists():
            print(f"  跳过（不存在）：{fname}")
            continue
        im = Image.open(src).convert("RGB")
        ratio = THUMB_W / im.width
        im = im.resize((THUMB_W, int(im.height * ratio)), Image.LANCZOS)
        tiles.append((im, caption))
        print(f"  已载入 {fname}  {im.width}x{im.height}")

    if not tiles:
        print("没有任何截图")
        return 1

    rows = (len(tiles) + COLS - 1) // COLS
    tile_h = max(t[0].height for t in tiles) + CAP_H
    sheet_w = COLS * THUMB_W + (COLS + 1) * PAD
    sheet_h = rows * tile_h + (rows + 1) * PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)

    for idx, (im, caption) in enumerate(tiles):
        r, c = divmod(idx, COLS)
        x = PAD + c * (THUMB_W + PAD)
        y = PAD + r * (tile_h + PAD)
        draw.rectangle([x, y, x + THUMB_W, y + CAP_H], fill=CAP_BG)
        draw.text((x + 12, y + 12), caption, fill=CAP_FG, font=font)
        sheet.paste(im, (x, y + CAP_H))

    sheet.save(OUT, optimize=True)
    print(f"\n总览图 -> {OUT}  ({sheet.width}x{sheet.height}, {OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())