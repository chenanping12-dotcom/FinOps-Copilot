"""第二轮截图：同一问句在「纯 BM25」与「混合检索 + Cross-Encoder 精排」下的排序对比。

演示逻辑：
1. 点业务场景胶囊「退款确认口径核验」-> 填问句并执行（默认 hybrid_rerank）；
2. 侧边栏切到「纯 BM25（词法基线）」-> 重跑同一问句 -> 存档 06；
3. 侧边栏切回「混合检索 + Cross-Encoder 精排（推荐）」-> 重跑 -> 存档 07。

对照关系：BM25 更偏向词面重合的旧版本（V1），精排能把当前生效版本（V2）顶到第 1。
用法（先跑 `capture_demo.py` 或另开终端把 Streamlit 跑在 8501）：
    .venv/Scripts/python.exe app/capture_demo2.py
"""

import asyncio
import os
import base64
import json
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).parent))
from capture_demo import (  # noqa: E402
    BODY_TEXT,
    EXPAND_MORE_HITS,
    HAS_TABLE,
    MORE_HITS_OPEN,
    SCENARIO_RAG,
    Chrome,
    Session,
    click_text,
)

OUT_DIR = Path(os.environ.get("DEMO_OUT", Path(__file__).resolve().parents[1] / "docs" / "demo"))

# Streamlit 的 radio 选项是 <label> 包 <input>，不在 button 选择器里，得单独点。
CLICK_LABEL = """
(() => {
  const want = %s;
  const hit = [...document.querySelectorAll('label')]
    .find(l => (l.innerText || '').replace(/\\s+/g, ' ').trim().includes(want));
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

# 合规依据卡片顶部的溯源徽标，是策略对比最直接的证据。
EVIDENCE_BADGE = """
(() => {
  const m = document.body.innerText.match(/\\[DOC: [^\\]]+\\][^\\n]*/);
  return m ? m[0] : '';
})()
"""

MODE_BM25 = "纯 BM25（词法基线）"
MODE_HYBRID = "Cross-Encoder 精排"


async def shot(s: Session, name: str) -> None:
    # 与 capture_demo.Session.shot 同理：不加 captureBeyondViewport，避免侧边栏被压窄。
    res = await s.call("Page.captureScreenshot", format="png")
    path = OUT_DIR / name
    path.write_bytes(base64.b64decode(res["data"]))
    print(f"    已截图 {name}  ({path.stat().st_size / 1024:.0f} KB)")


async def switch_mode(s: Session, mode_text: str) -> None:
    """切检索模式并确认选中。"""
    clicked = await s.js(CLICK_LABEL % json.dumps(mode_text))
    print(f"    点击「{mode_text}」: {clicked}")
    if not clicked:
        labs = await s.js("[...document.querySelectorAll('label')].map(l => l.innerText.trim()).filter(Boolean).join(' | ')")
        print(f"    [!] 页面上的 label: {labs}")
    await s.wait_for(f"{RADIO_CHECKED} === true" % json.dumps(mode_text), timeout=30, label=mode_text)


async def rerun_and_shot(s: Session, name: str) -> None:
    """点「开始分析」-> 等结果 -> 展开「其余命中」让排序表可见 -> 截图。"""
    print("    已点开始分析:", await click_text(s, "开始分析"))
    await s.wait_for(f"{HAS_TABLE} > 0", timeout=300, label="结果表")
    await asyncio.sleep(3)
    await s.js(EXPAND_MORE_HITS)
    await s.wait_for(MORE_HITS_OPEN, timeout=30, label="其余命中展开")
    await asyncio.sleep(4)
    print("    合规依据徽标:", await s.js(EVIDENCE_BADGE))
    await shot(s, name)


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chrome = Chrome()
    try:
        ws_url = chrome.wait_ready()
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
            s = Session(ws)
            await s.call("Page.enable")
            await s.call("Runtime.enable")
            await s.wait_for(
                f"{BODY_TEXT}.includes('开始分析') && {BODY_TEXT}.includes('业务场景')",
                timeout=120,
                label="首页",
            )

            print(f"[B1] 点业务场景「{SCENARIO_RAG}」（默认 hybrid_rerank）...")
            print("    已点:", await click_text(s, SCENARIO_RAG))
            await s.wait_for(f"{HAS_TABLE} > 0", timeout=600, label="首次检索")
            await asyncio.sleep(3)
            print("    合规依据徽标:", await s.js(EVIDENCE_BADGE))

            print("[B2] 侧边栏切「纯 BM25」...")
            await switch_mode(s, MODE_BM25)
            print("[B3] 重跑同一问句（纯 BM25）...")
            await rerun_and_shot(s, "06-纯BM25排序.png")

            print("[B4] 切回「混合检索 + Cross-Encoder 精排」...")
            await switch_mode(s, MODE_HYBRID)
            print("[B5] 重跑同一问句（精排）...")
            await rerun_and_shot(s, "07-混合检索精排排序.png")

            (OUT_DIR / "页面文本2.txt").write_text(await s.js(BODY_TEXT), encoding="utf-8")
    finally:
        chrome.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))