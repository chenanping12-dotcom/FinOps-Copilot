"""用 CDP 驱动无头 Chrome，抓取 Streamlit Demo 的真实运行截图。

为什么不用 `chrome --screenshot`：那个只是截「首次绘制」，Streamlit 靠 WebSocket
拉渲染指令，--virtual-time-budget 等不到内容出来，截到的只有骨架屏。
这里改成连 CDP：navigate -> 轮询 DOM 直到真内容出现 -> 点按钮 -> 再截。
"""

import asyncio
import os
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
import websockets

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
APP_URL = "http://127.0.0.1:8501/"
CDP_PORT = 9222
OUT_DIR = Path(os.environ.get("DEMO_OUT", Path(__file__).resolve().parents[1] / "docs" / "demo"))


class Chrome:
    def __init__(self) -> None:
        self.profile = tempfile.mkdtemp(prefix="cdp-profile-")
        self.proc = subprocess.Popen(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={CDP_PORT}",
                "--window-size=1500,1500",
                "--hide-scrollbars",
                f"--user-data-dir={self.profile}",
                APP_URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 40.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                tabs = requests.get(f"http://127.0.0.1:{CDP_PORT}/json", timeout=2).json()
                for tab in tabs:
                    if tab.get("type") == "page" and tab.get("webSocketDebuggerUrl"):
                        return tab["webSocketDebuggerUrl"]
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("CDP 未就绪")

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)


class Session:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.n = 0

    async def call(self, method: str, **params):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method} -> {msg['error']}")
                return msg.get("result", {})

    async def js(self, expression: str):
        res = await self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        return res.get("result", {}).get("value")

    async def wait_for(self, expression: str, timeout: float = 180.0, label: str = "") -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if await self.js(expression):
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
        print(f"    [!] 等待超时: {label or expression[:60]}")
        return False

    async def shot(self, name: str) -> Path:
        res = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        path = OUT_DIR / name
        path.write_bytes(base64.b64decode(res["data"]))
        print(f"    已截图 {name}  ({path.stat().st_size / 1024:.0f} KB)")
        return path


CLICK_BY_TEXT = """
(() => {
  const want = %s;
  const nodes = [...document.querySelectorAll('button, [role="tab"]')];
  const hit = nodes.find(b => (b.innerText || '').replace(/\\s+/g, ' ').trim().includes(want));
  if (!hit) return false;
  hit.scrollIntoView({block: 'center'});
  hit.click();
  return true;
})()
"""

QUESTION_VALUE = """
(() => {
  const el = [...document.querySelectorAll('input')].find(i => (i.value || '').includes('口径'));
  return el ? el.value : '';
})()
"""

HAS_TABLE = "document.querySelectorAll('[data-testid=\"stDataFrame\"]').length"
BODY_TEXT = "document.body.innerText"

# Streamlit 的 expander 是 <details>/<summary>，不是 button，
# 所以 CLICK_BY_TEXT 点不开它，必须单独处理（否则「溯源明细展开」这张图是假的）。
EXPAND_HIT_1 = """
(() => {
  const dets = [...document.querySelectorAll('details')];
  const d = dets.find(x => (x.innerText || '').includes('#1 ·'));
  if (!d || !d.querySelector('summary')) return false;
  d.querySelector('summary').scrollIntoView({block: 'center'});
  d.querySelector('summary').click();
  return true;
})()
"""
HIT_1_OPEN = """
[...document.querySelectorAll('details')].some(d => (d.innerText || '').includes('#1 ·') && d.open)
"""


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chrome = Chrome()
    try:
        ws_url = chrome.wait_ready()
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
            s = Session(ws)
            await s.call("Page.enable")
            await s.call("Runtime.enable")

            print("[1/6] 等待首页渲染 ...")
            await s.wait_for(
                f"{BODY_TEXT}.includes('开始分析') && {BODY_TEXT}.includes('知识库预设问题')",
                timeout=120,
                label="首页",
            )
            await asyncio.sleep(3)
            await s.shot("01-首页.png")

            print("[2/6] 点击知识库预设问题 EVAL-001 ...")
            await s.wait_for(
                f"[...document.querySelectorAll('button')].some(b => b.innerText.includes('EVAL-001'))",
                timeout=60,
                label="预设题按钮",
            )
            print("    已点:", await s.js(CLICK_BY_TEXT % json.dumps("EVAL-001")))
            # 等输入框里真的填上问题，否则「开始分析」会因为 question 为空而不触发
            await s.wait_for(f"{QUESTION_VALUE}.includes('口径')", timeout=60, label="问题已填入")
            print("    输入框内容:", (await s.js(QUESTION_VALUE))[:50])

            print("[3/6] 点击「开始分析」（首次加载 BGE + CrossEncoder，约 1-3 分钟）...")
            print("    已点:", await s.js(CLICK_BY_TEXT % json.dumps("开始分析")))
            ok = await s.wait_for(
                f"{HAS_TABLE} > 0 && {BODY_TEXT}.includes('检索器已就绪')",
                timeout=480,
                label="RAG 表格",
            )
            print("    RAG 结果就绪:", ok, "｜表格数:", await s.js(HAS_TABLE))
            await asyncio.sleep(4)
            await s.shot("02-RAG检索结果.png")

            print("[4/6] 展开第 1 条溯源明细 ...")
            print("    已点:", await s.js(EXPAND_HIT_1))
            print("    已展开:", await s.wait_for(HIT_1_OPEN, timeout=30, label="第1条展开"))
            await asyncio.sleep(3)
            await s.shot("03-溯源明细展开.png")

            print("[5/6] 切到经营指标（Agent）Tab ...")
            print("    已点:", await s.js(CLICK_BY_TEXT % json.dumps("经营指标")))
            await asyncio.sleep(4)
            await s.shot("04-Agent指标路由.png")

            print("[6/6] 存档页面文本 ...")
            full = await s.js(BODY_TEXT)
            (OUT_DIR / "页面文本.txt").write_text(full, encoding="utf-8")
            print(f"    页面文本 {len(full)} 字；表格数 {await s.js(HAS_TABLE)}")
    finally:
        chrome.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))