"""用 CDP 驱动无头 Chrome，抓取 Streamlit Demo 的真实运行截图。

为什么不用 `chrome --screenshot`：那个只是截「首次绘制」，Streamlit 靠 WebSocket
拉渲染指令，--virtual-time-budget 等不到内容出来，截到的只有骨架屏。
这里改成连 CDP：navigate -> 轮询 DOM 直到真内容出现 -> 点按钮 -> 再截。

用法（先另开终端把 Streamlit 跑在 8501）：
    .venv/Scripts/python.exe app/capture_demo.py
输出目录由 DEMO_OUT 环境变量控制，默认 `FinOps-Copilot/docs/demo/`。
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

# 场景快选胶囊的可见文案（取自 app/streamlit_app.py 的 scenario_catalog）。
SCENARIO_RAG = "退款确认口径核验"
SCENARIO_RETURN = "数码退货窗口规则"
SCENARIO_METRIC = "品类净营收汇总"
SCENARIO_GUARD = "只读 SQL 护栏拦截测试"


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
                # 本机显示缩放不是 100% 时，无头 Chrome 的位图与 CSS 像素会不一致，
                # 截出来的侧边栏会被横向压窄（文字看着像被切）。钉死 DSF=1。
                "--force-device-scale-factor=1",
                "--window-size=1600,1700",
                "--hide-scrollbars",
                f"--user-data-dir={self.profile}",
                APP_URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 40.0) -> str:
        """返回**应用页面**的 CDP WebSocket。

        坑：Chrome 启动时除 APP_URL 外还会带一个 ``about:blank`` 标签页，``/json``
        里它常常排在最前面。早先直接取第一个 ``type == "page"`` 会连到那个空白页，
        于是所有 DOM 查询都返回空、每次点击都报「找不到元素」。这里改成按 URL 认
        页面，空白页只作兜底。
        """
        deadline = time.time() + timeout
        fallback: str | None = None
        while time.time() < deadline:
            try:
                tabs = requests.get(f"http://127.0.0.1:{CDP_PORT}/json", timeout=2).json()
                for tab in tabs:
                    if tab.get("type") != "page" or not tab.get("webSocketDebuggerUrl"):
                        continue
                    url = tab.get("url", "")
                    if url.startswith(APP_URL.rstrip("/")):
                        return tab["webSocketDebuggerUrl"]
                    if url and url != "about:blank" and fallback is None:
                        fallback = tab["webSocketDebuggerUrl"]
            except Exception:
                pass
            time.sleep(0.5)
        if fallback:
            return fallback
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
        # 不要加 captureBeyondViewport=True：那会让 Chrome 在截图前把布局视口
        # 重设成整页尺寸，fixed 定位的侧边栏会被横向压窄（DOM 量到 300px，
        # 位图里只剩 233px，看截图就像左侧文字被切）。实测量过：
        # 普通视口截图得到的侧边栏边界 = 299px，与 DOM 完全一致。
        # 代价是必须把窗口开得比整页高，否则底部会被裁——见 Chrome 的 --window-size。
        res = await self.call("Page.captureScreenshot", format="png")
        path = OUT_DIR / name
        path.write_bytes(base64.b64decode(res["data"]))
        print(f"    已截图 {name}  ({path.stat().st_size / 1024:.0f} KB)")
        return path


CLICK_BY_TEXT = """
(() => {
  const want = %s;
  const nodes = [...document.querySelectorAll('button, [role="tab"], [data-testid="stBaseButton-pills"]')];
  const hit = nodes.find(b => (b.innerText || '').replace(/\\s+/g, ' ').trim().includes(want));
  if (!hit) return false;
  hit.scrollIntoView({block: 'center'});
  hit.click();
  return true;
})()
"""

CLICK_DIAGNOSTIC = """
(() => {
  const nodes = [...document.querySelectorAll('button')];
  return nodes.map(b => (b.innerText || '').replace(/\\s+/g, ' ').trim()).filter(Boolean).join(' | ');
})()
"""

QUESTION_VALUE = """
(() => {
  const el = [...document.querySelectorAll('input')].find(i => (i.value || '').includes('口径') || (i.value || '').includes('营收'));
  return el ? el.value : '';
})()
"""

HAS_TABLE = "document.querySelectorAll('[data-testid=\"stDataFrame\"]').length"
BODY_TEXT = "document.body.innerText"

# Streamlit 的 expander 是 <details>/<summary>，不是 button，
# 所以 CLICK_BY_TEXT 点不开它，必须单独处理（否则「溯源明细展开」这张图是假的）。
# 注意：`#1 · ...` 这层 expander 嵌在「其余命中」里，父层收起时它根本不在 DOM 中，
# 必须先展开父层再点它——第一次跑就是因为漏了这一步，拍到的还是收起状态。
EXPAND_MORE_HITS = """
(() => {
  const d = [...document.querySelectorAll('details')]
    .find(x => (x.querySelector('summary')?.innerText || '').includes('其余命中'));
  if (!d) return false;
  if (!d.open) d.querySelector('summary').click();
  return true;
})()
"""
MORE_HITS_OPEN = """
[...document.querySelectorAll('details')].some(d => (d.querySelector('summary')?.innerText || '').includes('其余命中') && d.open)
"""
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


async def click_text(s: "Session", text: str, label: str = "") -> bool:
    """点一个可见文案含 ``text`` 的按钮/胶囊；失败时打印页面上的按钮做诊断。"""
    ok = await s.js(CLICK_BY_TEXT % json.dumps(text))
    if not ok:
        print(f"    [!] 未找到「{text}」；页面上的按钮: {await s.js(CLICK_DIAGNOSTIC)}")
    return bool(ok)


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
                f"{BODY_TEXT}.includes('开始分析') && {BODY_TEXT}.includes('业务场景')",
                timeout=120,
                label="首页",
            )
            await asyncio.sleep(3)
            await s.shot("01-首页.png")

            print(f"[2/6] 点业务场景胶囊「{SCENARIO_RAG}」...")
            await s.wait_for(
                f"{BODY_TEXT}.includes({json.dumps(SCENARIO_RAG)})", timeout=60, label="场景胶囊"
            )
            print("    已点:", await click_text(s, SCENARIO_RAG))
            # 点胶囊 = 填问句 + 立即执行，等输入框真的填上，再补点一次「开始分析」
            await s.wait_for(f"{QUESTION_VALUE}.includes('口径')", timeout=120, label="问题已填入")
            print("    输入框内容:", (await s.js(QUESTION_VALUE))[:50])

            print("[3/6] 点「开始分析」（首次加载 BGE + CrossEncoder，约 1-3 分钟）...")
            print("    已点:", await click_text(s, "开始分析"))
            ok = await s.wait_for(
                f"{HAS_TABLE} > 0 && {BODY_TEXT}.includes('[DOC:')",
                timeout=600,
                label="RAG 证据链 + 命中表格",
            )
            print("    RAG 结果就绪:", ok, "｜表格数:", await s.js(HAS_TABLE))
            await asyncio.sleep(4)
            await s.shot("02-RAG检索结果.png")

            print("[4/6] 展开「其余命中」->「#1 ·」溯源明细 ...")
            print("    展开其余命中:", await s.js(EXPAND_MORE_HITS))
            print("    其余命中已展开:", await s.wait_for(MORE_HITS_OPEN, timeout=30, label="其余命中展开"))
            await asyncio.sleep(2)
            print("    已点 #1:", await s.js(EXPAND_HIT_1))
            print("    #1 已展开:", await s.wait_for(HIT_1_OPEN, timeout=30, label="第1条展开"))
            await asyncio.sleep(3)
            await s.shot("03-溯源明细展开.png")

            print(f"[5/6] 点经营指标场景「{SCENARIO_METRIC}」...")
            print("    已点:", await click_text(s, SCENARIO_METRIC))
            await s.wait_for(f"{BODY_TEXT}.includes('核心结论')", timeout=120, label="Agent 结论")
            await asyncio.sleep(4)
            await s.shot("04-Agent业务决策.png")

            print(f"[6/6] 点安全护栏场景「{SCENARIO_GUARD}」...")
            print("    已点:", await click_text(s, SCENARIO_GUARD))
            await s.wait_for(
                f"{BODY_TEXT}.includes('Security Guardrail Intercepted')", timeout=120, label="护栏横幅"
            )
            await asyncio.sleep(4)
            await s.shot("05-安全护栏拦截.png")

            print("[存档] 页面文本 ...")
            full = await s.js(BODY_TEXT)
            (OUT_DIR / "页面文本.txt").write_text(full, encoding="utf-8")
            print(f"    页面文本 {len(full)} 字；表格数 {await s.js(HAS_TABLE)}")
    finally:
        chrome.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))