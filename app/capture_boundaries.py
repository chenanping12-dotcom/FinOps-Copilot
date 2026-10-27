"""第四轮截图：三栏下方的「🛡️ 架构边界与设计权衡说明」折叠面板。

这一页是主动自证用的，展开态才有信息量，所以单独存一张图而不是塞进总览缩略。
面板最初挂在右栏（409px），三列中文被横向截断，后来改到通栏（1124px）。

用法（先把 Streamlit 跑在 8501，见 docs/Demo设计说明.md §6）：
    .venv/Scripts/python.exe app/capture_boundaries.py
输出目录可用环境变量 ``DEMO_OUT`` 覆盖，默认 FinOps-Copilot/docs/demo。
"""

import asyncio
import base64
import math
import os
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).parent))
from capture_demo import BODY_TEXT, Chrome, Session  # noqa: E402

OUT_DIR = Path(os.environ.get("DEMO_OUT", Path(__file__).resolve().parents[1] / "docs" / "demo"))

PANEL_TITLE = "架构边界与设计权衡"
SHOT_NAME = "11-架构边界与设计权衡.png"

# 面板本体（Streamlit 的 expander 是 <details>，不是 button）。
_FIND = (
    "[...document.querySelectorAll('details')]"
    f".find(n => (n.querySelector('summary')?.innerText || '').includes('{PANEL_TITLE}'))"
)

HAS_PANEL = f"(() => Boolean({_FIND}))()"

# 点开它。返回 'open' 表示本来就已经展开，避免重复点击把它又合上。
OPEN_PANEL = f"""
(() => {{
  const d = {_FIND};
  if (!d) return 'not-found';
  if (d.open) return 'open';
  const sum = d.querySelector('summary');
  if (sum) sum.click();
  return d.open ? 'open' : 'clicked';
}})()
"""

# 展开后里面的东西是否真的画完：三节都出齐，卡片正文到位，且没有骨架屏。
# 注意：第 1、2 节现在是**卡片**（markdown），只有第 3 节（护栏分层表）还是 data frame，
# 所以这里不能再要求 >= 3 个 table —— 上一版就是卡在这个条件上等到超时的。
READY = f"""
(() => {{
  const d = {_FIND};
  if (!d || !d.open) return false;
  if (document.querySelectorAll('[data-testid="stSkeleton"]').length) return false;
  const t = d.innerText || '';
  return t.includes('底层真实组件') && t.includes('演示替身')
      && t.includes('只读护栏的真实分层') && t.includes('自我纠正')
      && t.includes('可核对位置') && t.includes('RRF 倒数融合')
      && d.querySelectorAll('table, [data-testid="stDataFrame"]').length >= 1;
}})()
"""

RECT = f"""
(() => {{
  const d = {_FIND};
  if (!d) return null;
  const r = d.getBoundingClientRect();
  return {{top: Math.round(r.top + window.scrollY), left: Math.round(r.left + window.scrollX),
           width: Math.round(r.width), height: Math.round(r.height)}};
}})()
"""

SET_VIEWPORT = """
(() => { window.scrollTo(0, 0); return [window.innerWidth, window.innerHeight]; })()
"""


async def shot(s: Session, name: str) -> None:
    """普通视口截图（不要 captureBeyondViewport：它会让 fixed 侧边栏在位图里被压窄）。"""
    res = await s.call("Page.captureScreenshot", format="png")
    path = OUT_DIR / name
    path.write_bytes(base64.b64decode(res["data"]))
    print(f"    已截图 {name}  ({path.stat().st_size / 1024:.0f} KB)")


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chrome = Chrome()
    try:
        ws_url = chrome.wait_ready()
        async with websockets.connect(
            ws_url, max_size=64 * 1024 * 1024, ping_interval=None
        ) as ws:
            s = Session(ws)
            await s.call("Page.enable")
            await s.call("Runtime.enable")

            if not await s.wait_for(
                f"{BODY_TEXT}.includes('开始分析')", timeout=240, label="工作台首页"
            ):
                print("    [!] 首页未就绪")
                return 1

            if not await s.wait_for(HAS_PANEL, timeout=120, label="架构边界面板存在"):
                print("    [!] 页面上找不到「架构边界与设计权衡」面板")
                return 1

            print("[B1] 展开架构边界面板 ...")
            print("    点击结果:", await s.js(OPEN_PANEL))

            if not await s.wait_for(READY, timeout=180, label="面板内容画完"):
                print("    [!] 面板未画完，仍截图作为证据")
            await asyncio.sleep(1.0)

            rect = await s.js(RECT)
            print("    面板尺寸:", rect)
            if not rect:
                return 1

            # 面板很长，把视口撑到能装下它再截图。注意 setDeviceMetricsOverride 之后
            # 整页会重排，**旧的 rect 立刻失效**；而且 Streamlit 的主区是内部滚动容器，
            # window.scrollTo 根本不动它——上一版就是这么拍出一张「没滚动的首页」的。
            # 用 scrollIntoView 让它自己找滚动容器，并重新量一次高度。
            await s.call(
                "Emulation.setDeviceMetricsOverride",
                width=1584,
                height=int(math.ceil(rect["height"])) + 130,
                deviceScaleFactor=1,
                mobile=False,
            )
            await asyncio.sleep(1.2)
            await s.js(f"""
            (() => {{
              const d = {_FIND};
              if (!d) return false;
              d.scrollIntoView({{block: 'start'}});
              return true;
            }})()
            """)
            await asyncio.sleep(1.0)
            settled = await s.js(RECT)
            print("    滚动后面板尺寸:", settled)

            await shot(s, SHOT_NAME)
            text = await s.js(BODY_TEXT)
            print(f"    页面文本 {len(text)} 字")
    finally:
        chrome.close()
    print("\n架构边界面板截图完成。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))