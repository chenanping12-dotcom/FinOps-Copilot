"""采集「📈 评测基准与鲁棒性」Tab 的整页截图。

窗口高度故意开得很大：截图走的是普通视口（不能开 captureBeyondViewport，否则
fixed 侧边栏会被横向压窄，见 ``capture_demo.Session.shot``），而看板是整页长图，
视口不够高底部就会被裁。

用法（需先把 Streamlit 起在 8501）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe app/capture_benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import websockets  # noqa: E402

from capture_demo import BODY_TEXT, OUT_DIR, Chrome, Session  # noqa: E402

TAB_LABEL = "评测基准与鲁棒性"
# 视口高度按内容实测来定（脚本会打印实测值）。注意不能用 scrollHeight 当内容高度：
# 窗口比内容高时 scrollHeight 就等于视口高，会误判成「页面很高」而截出一大块空白。
WINDOW = os.environ.get("BENCH_WINDOW", "1600,2850")

CLICK_TAB = """
(() => {
  const hit = [...document.querySelectorAll('[role="tab"]')]
    .find(t => (t.innerText || '').includes(%s));
  if (!hit) return false;
  hit.click();
  return true;
})()
"""

TAB_SELECTED = """
[...document.querySelectorAll('[role="tab"]')]
  .some(t => (t.innerText || '').includes(%s) && t.getAttribute('aria-selected') === 'true')
"""

BENCH_READY = """
(() => {
  const txt = document.body.innerText;
  return txt.includes('离线检索基准')
      && txt.includes('真实公开数据鲁棒性')
      && document.querySelectorAll('[data-testid="stDataFrame"]').length >= 3;
})()
"""

PAGE_HEIGHT = "Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"

# 真实内容底边 = 所有元素 rect.bottom 的最大值。scrollHeight 在「窗口比内容高」时
# 会退化成视口高，用它当内容高度就会截出一大块纯色空白。
CONTENT_BOTTOM = """
Math.max(...[...document.querySelectorAll('.stApp *')]
  .map(e => e.getBoundingClientRect().bottom + window.scrollY))
"""

VIEWPORT_HEIGHT = "window.innerHeight"


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chrome = Chrome(window=WINDOW)
    try:
        ws_url = chrome.wait_ready()
        async with websockets.connect(
            ws_url, max_size=64 * 1024 * 1024, ping_interval=None
        ) as ws:
            s = Session(ws)
            await s.call("Page.enable")
            await s.call("Runtime.enable")

            print("[1/4] 等待首页渲染 ...")
            await s.wait_for(f"{BODY_TEXT}.includes('开始分析')", timeout=180, label="首页")
            await asyncio.sleep(3)

            print(f"[2/4] 切到 Tab「{TAB_LABEL}」...")
            print("    已点 Tab:", await s.js(CLICK_TAB % json.dumps(TAB_LABEL)))
            await s.wait_for(TAB_SELECTED % json.dumps(TAB_LABEL), timeout=60, label="Tab 选中")

            print("[3/4] 等待看板四个小节全部渲染 ...")
            ready = await s.wait_for(BENCH_READY, timeout=180, label="看板就绪")
            print("    看板就绪:", ready)
            await asyncio.sleep(4)

            height = await s.js(PAGE_HEIGHT)
            bottom = await s.js(CONTENT_BOTTOM)
            viewport = await s.js(VIEWPORT_HEIGHT)
            print(
                f"    内容底边 {float(bottom):.0f}px ｜ 视口高 {viewport}px ｜ "
                f"scrollHeight {height}px ｜ 窗口 {WINDOW}"
            )
            if isinstance(bottom, (int, float)) and isinstance(viewport, (int, float)):
                if bottom > viewport:
                    print(f"    [!] 内容比视口高 {bottom - viewport:.0f}px，底部会被裁")
                    print("        → 调 BENCH_WINDOW 把窗口开高一点再跑")
                elif viewport - bottom > 260:
                    print(f"    [!] 视口比内容高 {viewport - bottom:.0f}px，底部会留一大块空白")

            print("[4/4] 截图 ...")
            path = await s.shot("10-评测基准看板.png")
            text = await s.js(BODY_TEXT)
            (OUT_DIR / "页面文本-评测看板.txt").write_text(text, encoding="utf-8")
            print(f"    页面文本 {len(text)} 字")
            print("    产出:", path)
    finally:
        chrome.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))