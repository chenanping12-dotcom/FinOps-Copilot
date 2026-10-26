"""第三轮截图：对抗演示区（版本口径冲突 V1 vs V2 / 写操作拦截 403）。

两张卡片不是快捷提问，是两条已经用真实数据跑出来的技术主张，所以单独存图。

用法（先把 Streamlit 跑在 8501，见 docs/Demo设计说明.md §6）：
    .venv/Scripts/python.exe app/capture_showcase.py
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).parent))
from capture_demo import BODY_TEXT, Chrome, Session, click_text  # noqa: E402

OUT_DIR = Path(os.environ.get("DEMO_OUT", Path(__file__).resolve().parents[1] / "docs" / "demo"))

SCENE_A = "版本口径冲突对抗"
SCENE_B = "恶意注入与写操作拦截"

# 场景 A 的判定：左右两栏必须同时出现「已废止」与「现行」，且各带一条溯源徽标。
DUEL_READY = "(() => { const t = document.body.innerText; return t.includes('已废止') && t.includes('现行') && (t.match(/\\[POL-\\d+-V\\d+/g) || []).length >= 2; })()"

# 场景 B 的判定：403 回执 + 空 SQL + 锁死。
GUARD_READY = "(() => { const t = document.body.innerText; return t.includes('403 FORBIDDEN') && t.includes('锁死'); })()"

# 右栏高亮轨迹：审计时间线里被标红的节点。
HIGHLIGHTED = "(() => document.querySelectorAll('.fops-tl-item.is-hl').length)()"

BUTTONS = "[...document.querySelectorAll('button')].map(b => (b.innerText || '').replace(/\\s+/g, ' ').trim()).filter(Boolean).join(' | ')"

# 页面「画完了没有」。Streamlit 是流式下发 DOM 的：对战结论的文字先到，下面的
# st.metric / st.dataframe 还是灰色骨架屏。只等文字会出现拍到一排空框——
# 上一版 08/09 两张图就是这么拍坏的。这里要求骨架屏清零且每张 metric 都有文字。
SETTLED = """
(() => {
  if (document.querySelectorAll('[data-testid="stSkeleton"]').length) return false;
  const ms = [...document.querySelectorAll('[data-testid="stMetric"]')];
  if (!ms.length) return false;
  return ms.every(n => (n.innerText || '').trim().length > 0);
})()
"""


async def settle(s: Session, label: str) -> None:
    """等页面真正画完，再截图。"""
    if not await s.wait_for(SETTLED, timeout=120, label=f"{label} 画完"):
        print(f"    [!] {label} 仍有未填充的骨架屏，截图可能不全")
    await asyncio.sleep(1.5)


async def shot(s: Session, name: str) -> None:
    """普通视口截图。不要用 captureBeyondViewport —— 它会让 Chrome 重设布局视口，
    position:fixed 的侧边栏在位图里被压窄（实测 300px -> 233px），看着像被切。"""
    res = await s.call("Page.captureScreenshot", format="png")
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
            ready = await s.wait_for(
                f"{BODY_TEXT}.includes('开始分析') && {BODY_TEXT}.includes('对抗演示')",
                timeout=180,
                label="首页",
            )
            if not ready:
                print("    [!] 首页未就绪，当前按钮:", await s.js(BUTTONS))
                return 1
            print("    首页就绪，按钮:", (await s.js(BUTTONS))[:200])

            print(f"[C1] 点对抗场景 A「{SCENE_A}」...")
            if not await click_text(s, SCENE_A):
                print("    [!] 场景 A 没点中")
                return 1
            await s.wait_for(DUEL_READY, timeout=600, label="版本对照结论")
            await settle(s, "场景 A")
            print("    右栏高亮节点数:", await s.js(HIGHLIGHTED))
            await shot(s, "08-对抗A-版本口径冲突.png")

            print(f"[C2] 点对抗场景 B「{SCENE_B}」...")
            if not await click_text(s, SCENE_B):
                print("    [!] 没点中场景 B")
                return 1
            await s.wait_for(GUARD_READY, timeout=600, label="403 回执")
            await settle(s, "场景 B")
            print("    右栏高亮节点数:", await s.js(HIGHLIGHTED))
            await shot(s, "09-对抗B-写操作拦截.png")

            (OUT_DIR / "页面文本3.txt").write_text(await s.js(BODY_TEXT), encoding="utf-8")
            print("\n两张对抗演示截图完成。")
    finally:
        chrome.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))