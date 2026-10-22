# FinOps-Copilot 运行镜像：一个镜像同时承载 API（FastAPI/uvicorn，默认 CMD）和
# UI（Streamlit，由 docker-compose.yml 覆盖 command 启动）。
#
# ⚠️ 诚实声明：本机没有安装 Docker（实测 `docker --version` -> command not found），
# 所以这个文件**从未执行过 `docker build`**。里面每一条都做过人工自查，逻辑可验证的
# 部分（依赖抽取、serve_gateway.py 的生成与运行）已在宿主 venv 上实测，详见 DEPLOY.md
# 的「验证边界」与「未验证项」。
#
# 构建：docker build -t finops-copilot:0.1.0 .
# 运行 API：docker run --rm -p 8000:8000 -v "$PWD/models/huggingface:/app/models/huggingface" finops-copilot:0.1.0

FROM python:3.13-slim-bookworm AS runtime

# PYTHONUNBUFFERED: 容器日志实时可见（uvicorn/streamlit 都是长驻进程）
# PYTHONDONTWRITEBYTECODE: 不往挂载卷里写 __pycache__
# HF_HUB_*: 只影响 HuggingFace 自己的缓存位置；真正的模型缓存由 config/rag.yaml 的
#           retrieval.model_cache 决定（见下方 ENV 注释），不要用 HF_HOME 代替。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8000

# 系统依赖：
#   fonts-noto-cjk —— 镜像内做 matplotlib/PDF 出图时中文不乱码（前端图表走浏览器渲染，不依赖它）
#   curl           —— HEALTHCHECK 用；也方便进容器排查
# 注意：python:3.13-slim 已自带 ca-certificates，httpx 访问 HTTPS 网关不需要额外装。
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        curl \
 && rm -rf /var/lib/apt/lists/*

# 非 root 运行：uid/gid 固定 1001，方便在宿主机上 chown 挂载卷（见 DEPLOY.md）。
RUN groupadd --system --gid 1001 app \
 && useradd --system --uid 1001 --gid app --create-home --home-dir /home/app app

WORKDIR /app

# ---------------------------------------------------------------------------
# 第 1 层：只依赖声明。源码改动不会让这一层缓存失效。
# 这里从 pyproject.toml 里把 data/rag/agent/gateway 四个 extra 的依赖抽出来安装，
# 而不是直接 `pip install -e .`：后者要求源码已在镜像里，会把依赖层和代码绑死。
# setuptools/wheel 是 [build-system] 的 requires，这里显式装掉，下一步就能用
# --no-build-isolation，构建更确定、更快。
# ---------------------------------------------------------------------------
COPY --chown=app:app pyproject.toml requirements.txt README.md ./

RUN python -c 'import tomllib; d = tomllib.load(open("pyproject.toml", "rb")); e = d["project"]["optional-dependencies"]; reqs = list(d["project"]["dependencies"]) + e["data"] + e["rag"] + e["agent"] + e["gateway"]; open("/tmp/runtime-requirements.txt", "w").write(chr(10).join(reqs) + chr(10))' \
 && echo "=== 镜像内将安装的运行时依赖 ===" \
 && cat /tmp/runtime-requirements.txt \
 && pip install --no-cache-dir -r /tmp/runtime-requirements.txt \
 && pip install --no-cache-dir "setuptools>=75" wheel "streamlit>=1.40,<2"

# ---------------------------------------------------------------------------
# 第 2 层：源码与配置。改代码只会重建这一层及之后。
# app/ 是展示层（Streamlit），eval/ 只要数据集（报告已在 .dockerignore 里排除）。
# data/ 与 models/ 故意不 COPY：见 DEPLOY.md「数据与模型：为什么用 volume」。
# ---------------------------------------------------------------------------
COPY --chown=app:app src/ ./src/
COPY --chown=app:app app/ ./app/
COPY --chown=app:app config/ ./config/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app eval/ ./eval/

# 让项目本身可安装（无 console_scripts，纯属打包正确性 + 万一有人用 `gateway.app`
# 这种已安装布局的 import）。--no-deps 避免重新解析依赖，--no-build-isolation 用上一层装好的 setuptools。
RUN pip install --no-cache-dir --no-deps --no-build-isolation .

# ---------------------------------------------------------------------------
# 容器内入口引导。
# 为什么需要它：src/gateway/app.py 只提供工厂函数 create_app(client, cache)，
# 仓库里没有 module-level 的 ASGI `app`，也没有任何读取环境变量的配置层；
# 而 src/ 属于别人的写范围，本任务不动它，所以在镜像构建时生成这个引导文件。
# 上游正确修法是加 src/gateway/server.py（全文见 DEPLOY.md「建议的上游修复」），
# 之后这里换成 CMD ["uvicorn", "src.gateway.server:app", "--host", "0.0.0.0", "--port", "8000"]。
# 该文件的生成命令与运行结果已在宿主 venv 上实测（py_compile + /health 200）。
# ---------------------------------------------------------------------------
RUN printf '%s\n' \
'"""FinOps-Copilot gateway API 入口（由 Dockerfile 在构建时生成）。' \
'' \
'仓库只提供工厂函数 create_app(client, cache)，没有 module-level app，所以在镜像内' \
'做这层引导：从环境变量装配 LLMClient（HttpTransport，OpenAI 兼容）与 SemanticCache。' \
'"""' \
'' \
'import os' \
'import sys' \
'' \
'import uvicorn' \
'' \
'ROOT = os.path.dirname(os.path.abspath(__file__))' \
'if ROOT not in sys.path:' \
'    sys.path.insert(0, ROOT)' \
'' \
'try:  # 仓库布局：/app/src/...（与 app/streamlit_app.py 的 import 约定一致）' \
'    from src.gateway.app import create_app' \
'    from src.gateway.client import HttpTransport, LLMClient, SemanticCache' \
'except ModuleNotFoundError:  # 已安装布局：pip install .' \
'    from gateway.app import create_app' \
'    from gateway.client import HttpTransport, LLMClient, SemanticCache' \
'' \
'' \
'def build_app():' \
'    providers = [p.strip() for p in os.environ.get("LLM_PROVIDERS", "primary").split(",") if p.strip()]' \
'    base_urls = {' \
'        name: os.environ.get("LLM_BASE_URL_" + name.upper()) or os.environ.get("LLM_BASE_URL", "")' \
'        for name in providers' \
'    }' \
'    transport = HttpTransport(' \
'        base_urls,' \
'        api_key=os.environ.get("LLM_API_KEY", ""),' \
'        timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),' \
'    )' \
'    cache = SemanticCache(threshold=float(os.environ.get("CACHE_THRESHOLD", "0.95")))' \
'    client = LLMClient(providers, transport=transport, cache=cache)' \
'    return create_app(client, cache)' \
'' \
'' \
'app = build_app()' \
'' \
'if __name__ == "__main__":' \
'    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))' \
 > /app/serve_gateway.py \
 && chown app:app /app/serve_gateway.py \
 && python -m py_compile /app/serve_gateway.py

# 挂载点：模型缓存、数据、评测报告（compose 会把宿主机目录挂到这里）。
# 目录先建好并 chown 到 1001，容器以非 root 启动时才能写入。
RUN mkdir -p /app/models/huggingface /app/data /app/eval/reports /app/reports \
 && chown -R app:app /app/models /app/data /app/eval /app/reports \
 && chown app:app /app

USER app

EXPOSE 8000 8501

# 打 /health（app.py:36）。start-period 给 20s，避开冷启动窗口。
# 没有配 LLM_BASE_URL 时 /health 依然 200（它不调用 provider），这一点已实测。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 默认跑 API；UI 由 docker-compose.yml 覆盖 command 启动 Streamlit。
CMD ["python", "serve_gateway.py"]