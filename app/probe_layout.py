"""一次性诊断：量侧边栏的真实 DOM 几何，判断截图里「左侧文字被切」是不是真问题。

用法（先另开终端把 Streamlit 跑在 8501）：
    .venv/Scripts/python.exe app/probe_layout.py
"""

import asyncio
import json
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).parent))
from capture_demo import Chrome, Session  # noqa: E402

PROBE = """
(() => {
  const sb = document.querySelector('section[data-testid="stSidebar"]');
  const r = (el) => { if (!el) return null; const b = el.getBoundingClientRect();
    return {left: Math.round(b.left), right: Math.round(b.right), width: Math.round(b.width),
            top: Math.round(b.top), height: Math.round(b.height)}; };
  const firstLabel = sb ? sb.querySelector('label') : null;
  const firstHeading = sb ? [...sb.querySelectorAll('*')].find(e => (e.innerText||'').trim().startsWith('检索')) : null;
  return JSON.stringify({
    innerWidth: window.innerWidth,
    innerHeight: window.innerHeight,
    docScrollWidth: document.documentElement.scrollWidth,
    docClientWidth: document.documentElement.clientWidth,
    bodyScrollLeft: document.body.scrollLeft,
    docScrollLeft: document.documentElement.scrollLeft,
    sidebar: r(sb),
    sidebar_scrollWidth: sb ? sb.scrollWidth : null,
    sidebar_clientWidth: sb ? sb.clientWidth : null,
    sidebar_scrollLeft: sb ? sb.scrollLeft : null,
    sidebar_inner: sb ? r(sb.firstElementChild) : null,
    firstLabel: r(firstLabel),
    firstLabelText: firstLabel ? firstLabel.innerText.trim().slice(0, 40) : null,
    heading: r(firstHeading),
    headingText: firstHeading ? firstHeading.innerText.trim().slice(0, 40) : null,
    main: r(document.querySelector('section.main') || document.querySelector('[data-testid="stMain"]')),
  }, null, 2);
})()
"""


async def main() -> int:
    chrome = Chrome()
    try:
        ws_url = chrome.wait_ready()
        async with websockets.connect(ws_url, max_size=16 * 1024 * 1024, ping_interval=None) as ws:
            s = Session(ws)
            await s.call("Page.enable")
            await s.call("Runtime.enable")
            await s.wait_for("document.body.innerText.includes('开始分析')", timeout=120, label="首页")
            await asyncio.sleep(3)
            print("--- 未覆盖设备指标 ---")
            print(await s.js(PROBE))

            # 钉死视口尺寸与 DSF，让布局确定化（无头截图歪掉的头号嫌疑）。
            await s.call(
                "Emulation.setDeviceMetricsOverride",
                width=1584,
                height=1505,
                deviceScaleFactor=1,
                mobile=False,
            )
            await s.js("window.scrollTo(0,0)")
            await asyncio.sleep(3)
            print("--- 覆盖设备指标后 ---")
            print(await s.js(PROBE))
            # 同一次会话里再截一张，做到「DOM 与像素」同源对比。
            import base64

            shots = {
                "_probe_beyond.png": {"captureBeyondViewport": True},
                "_probe_nosurface.png": {"captureBeyondViewport": True, "fromSurface": False},
                "_probe_viewport2.png": {},
                "_probe_viewport_ns.png": {"fromSurface": False},
                "_probe_clip_full.png": {
                    "captureBeyondViewport": True,
                    "clip": {"x": 0, "y": 0, "width": 1584, "height": 1505, "scale": 1},
                },
            }
            for tag, extra in shots.items():
                res = await s.call("Page.captureScreenshot", format="png", **extra)
                out = Path(__file__).resolve().parents[1] / "docs" / "demo" / tag
                out.write_bytes(base64.b64decode(res["data"]))
                print(f"已截图 {out} ({out.stat().st_size} bytes)")
    finally:
        chrome.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))