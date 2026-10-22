"""第二轮截图：只读 SQL 拦截演示 + 检索模式对比。

- 演示 A：「删掉所有订单」-> Agent 走 refuse 路由，不生成任何写操作 SQL
- 演示 B：侧边栏切到「纯 BM25」再跑同一问题 -> 与混合检索对比排序差异
"""

import asyncio
import os
import base64
import json
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).parent))
from capture_demo import BODY_TEXT, HAS_TABLE, Chrome, Session  # noqa: E402

OUT_DIR = Path(os.environ.get("DEMO_OUT", Path(__file__).resolve().parents[1] / "docs" / "demo"))

CLICK_BY_TEXT = """
(() => {
  const want = %s;
  const nodes = [...document.querySelectorAll('button, [role="tab"], label')];
  const hit = nodes.find(b => (b.innerText || '').replace(/\\s+/g, ' ').trim().includes(want));
  if (!hit) return false;
  hit.scrollIntoView({block: 'center'});
  hit.click();
  return true;
})()
"""

RADIO_CHECKED = """
(() => {
  const want = %s;
  const lab = [...document.querySelectorAll('label')].find(l => l.innerText.includes(want));
  if (!lab) return null;
  const inp = lab.querySelector('input');
  return inp ? inp.checked : null;
})()
"""

FIRST_ROW = """
(() => {
  const t = document.querySelector('[data-testid="stDataFrame"]');
  return t ? t.innerText.slice(0, 300).replace(/\\n+/g, ' | ') : '';
})()
"""


async def shot(s: Session, name: str) -> None:
    res = await s.call("Page.captureScreenshot", format="png", captureBeyondViewport=True)
    path = OUT_DIR / name
    path.write_bytes(base64.b64decode(res["data"]))
    print(f"    已截图 {name}  ({path.stat().st_size / 1024:.0f} KB)")


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chrome = Chrome()
    try:
        ws_url = chrome.wait_ready()
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
            s = Session(ws)
            await s.call("Page.enable")
            await s.call("Runtime.enable")
            await s.wait_for(f"{BODY_TEXT}.includes('开始分析')", timeout=120, label="首页")

            # ---------- 演示 A：写操作被拒 ----------
            print("[A1] 点「删掉所有订单」...")
            print("    已点:", await s.js(CLICK_BY_TEXT % json.dumps("删掉所有订单")))
            await asyncio.sleep(2)
            print("[A2] 点「开始分析」...")
            print("    已点:", await s.js(CLICK_BY_TEXT % json.dumps("开始分析")))
            ok = await s.wait_for(
                f"{BODY_TEXT}.includes('read-only') || {BODY_TEXT}.includes('refuse')",
                timeout=240,
                label="拒绝路由",
            )
            print("    拒绝路由出现:", ok)
            await s.js(CLICK_BY_TEXT % json.dumps("经营指标"))
            await asyncio.sleep(4)
            await shot(s, "05-写操作被拒绝.png")

            text = await s.js(BODY_TEXT)
            idx = text.find("最终答案")
            print("    答案片段:", text[idx : idx + 220].replace("\n", " ") if idx >= 0 else "（未找到）")

            # ---------- 演示 B：检索模式对比 ----------
            print("[B1] 点 EVAL-001 预设题 ...")
            await s.js(CLICK_BY_TEXT % json.dumps("EVAL-001"))
            await asyncio.sleep(2)
            await s.js(CLICK_BY_TEXT % json.dumps("知识库问答"))
            await asyncio.sleep(1)

            print("[B2] 侧边栏切「纯 BM25」...")
            print("    已点:", await s.js(CLICK_BY_TEXT % json.dumps("纯 BM25")))
            await s.wait_for(f"{RADIO_CHECKED} === true" % json.dumps("纯 BM25"), timeout=30, label="BM25 选中")
            await asyncio.sleep(3)

            print("[B3] 点「仅运行检索」...")
            print("    已点:", await s.js(CLICK_BY_TEXT % json.dumps("仅运行检索")))
            await asyncio.sleep(10)
            await shot(s, "06-纯BM25排序.png")
            print("    BM25 首行:", await s.js(FIRST_ROW))

            print("[B4] 切回「混合检索 + Cross-Encoder 精排」...")
            await s.js(CLICK_BY_TEXT % json.dumps("精排（默认）"))
            await asyncio.sleep(3)
            await s.js(CLICK_BY_TEXT % json.dumps("仅运行检索"))
            await asyncio.sleep(12)
            await shot(s, "07-混合检索精排排序.png")
            print("    混合 首行:", await s.js(FIRST_ROW))

            (OUT_DIR / "页面文本2.txt").write_text(await s.js(BODY_TEXT), encoding="utf-8")
    finally:
        chrome.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))