# FinOps-Copilot 部署指南

给陈安平自己看的操作手册。目标只有两个：

1. **一周内能把一条公网链接发给面试官**（不需要面试官装任何东西）；
2. 把「部署痕迹」补进简历（本文件 + `Dockerfile` + `docker-compose.yml` 就是痕迹）。

> 本仓库 4 个新增文件：`Dockerfile`、`docker-compose.yml`、`.dockerignore`、`DEPLOY.md`。
> 只新增文件，没有改动 `src/`、`app/`、`config/`、`scripts/`、`eval/` 里的任何一行。

---

## 0. 先读这一节：什么验证过，什么没验证

这是本指南最重要的一节。**没验证的一律写「未验证」，不要拿去当作事实。**

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 本机是否装了 Docker | **没有**。实测 `docker --version` → `docker: command not found` | 因此 `docker build` / `docker compose up` **从未执行过** |
| `docker-compose.yml` 是合法 YAML | ✅ 已验证 | `yaml.safe_load(..., encoding="utf-8")` 通过，含结构断言（见 §4.4） |
| `docker-compose.yml` 能被 Docker Compose 接受 | ❌ **未验证** | 需要 Docker；`docker compose config -q` 无法运行 |
| Dockerfile 里抽依赖的那条 python 命令 | ✅ 已验证 | 在宿主上等价执行，抽出 14 条运行时依赖，无语法错误 |
| Dockerfile 生成的 `serve_gateway.py`（容器入口） | ✅ 已验证 | 从 Dockerfile 里把这段 `printf` 抽出来真跑过：生成 1584 字节，`py_compile` 通过，用它起服务 `GET /health` → 200 `{"status":"ok",...}` |
| 裸跑 API（uvicorn + `create_app`） | ✅ 已验证 | 本机 8000/8123/8124/8126 端口实测，`/health` 200、`/cache/stats` 200 |
| 裸跑 UI（Streamlit） | ✅ 已验证 | `python -m streamlit run app/streamlit_app.py` 启动成功，`/_stcore/health` → `ok`，首页 200 |
| compose 里两个 healthcheck 打的路由真实存在 | ✅ 已验证 | API `/health`（`src/gateway/app.py:36`）、UI `/_stcore/health`（Streamlit 内置）都实测返回正常 |
| `docker build` 能否成功 | ❌ **未验证** | 本机无 Docker。见 §5.3「Dockerfile 自查清单」 |
| 镜像体积、构建耗时 | ❌ **未验证** | 估算 2.5–3.5 GB / 5–15 分钟（torch CPU + sentence-transformers 是大头），仅为估算 |
| cloudflared / frp 公网映射 | ❌ **未验证** | 本机未安装这两个工具，命令按官方文档写，未实跑 |
| 云服务器上的 Docker 安装命令 | ❌ **未验证** | Ubuntu/Debian 通用命令，未在真机跑过 |
| UI 首次启动会下载哪些模型 | ⚠️ 未实测 | 按 `config/rag.yaml` 推断：dense `BAAI/bge-small-zh-v1.5` 约 92 MB 必下；reranker `BAAI/bge-reranker-base` 约 1.1 GB 只有真正走精排时才需要 |

**结论**：本机没 Docker 不影响发链接——走 §1 的裸跑路径即可；Docker 那条路（§2）等装了 Docker Desktop 或租了云服务器再跑，第一次跑请按 §5 的清单逐条核对。

---

## 1. 路径 A：不装 Docker，本机裸跑 + 公网映射（推荐，一周内发链接就走这条）

本机已经是可运行状态：venv 在 `FinOps-Copilot/.venv`（Python 3.13.3），依赖已装好，模型和数据也都在。

### 1.1 起 API（网关，占 8000 端口）

`src/gateway/app.py` 只提供工厂函数 `create_app(client, cache)`，仓库里**没有**可以直接 `uvicorn` 的模块级 `app`，也没有读环境变量的配置层（`src/` 不在本次改动范围内）。所以先做一层引导。

**做法 1（推荐，一次写好长期用）**：把下面这段存成 `FinOps-Copilot/serve_gateway.py`。
它与容器里用的入口**逐字一致**，就是 `Dockerfile:80-126` 生成的那个文件，已经实测能跑。

```python
"""FinOps-Copilot gateway API 入口（由 Dockerfile 在构建时生成）。

仓库只提供工厂函数 create_app(client, cache)，没有 module-level app，所以在镜像内
做这层引导：从环境变量装配 LLMClient（HttpTransport，OpenAI 兼容）与 SemanticCache。
"""

import os
import sys

import uvicorn

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:  # 仓库布局：/app/src/...（与 app/streamlit_app.py 的 import 约定一致）
    from src.gateway.app import create_app
    from src.gateway.client import HttpTransport, LLMClient, SemanticCache
except ModuleNotFoundError:  # 已安装布局：pip install .
    from gateway.app import create_app
    from gateway.client import HttpTransport, LLMClient, SemanticCache


def build_app():
    providers = [p.strip() for p in os.environ.get("LLM_PROVIDERS", "primary").split(",") if p.strip()]
    base_urls = {
        name: os.environ.get("LLM_BASE_URL_" + name.upper()) or os.environ.get("LLM_BASE_URL", "")
        for name in providers
    }
    transport = HttpTransport(
        base_urls,
        api_key=os.environ.get("LLM_API_KEY", ""),
        timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
    )
    cache = SemanticCache(threshold=float(os.environ.get("CACHE_THRESHOLD", "0.95")))
    client = LLMClient(providers, transport=transport, cache=cache)
    return create_app(client, cache)


app = build_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
```

然后（PowerShell，在 `FinOps-Copilot` 目录下）：

```powershell
.\.venv\Scripts\python.exe serve_gateway.py
# 验证
curl.exe http://127.0.0.1:8000/health
# 期望：{"status":"ok","providers":["primary"],"cache_entries":0,"cache_hit_rate":0.0}
```

**做法 2（不想新建文件，一行命令）**：实测可用的写法（注意引号，见下面的坑）：

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); import uvicorn; from gateway.app import create_app; from gateway.client import LLMClient,HttpTransport,SemanticCache; cache=SemanticCache(); app=create_app(LLMClient(['primary'],transport=HttpTransport({'primary':'http://127.0.0.1:9/v1'}),cache=cache),cache); uvicorn.run(app,host='127.0.0.1',port=8000)"
```

> ⚠️ **PowerShell 引号坑（实测踩到）**：如果你把 Python 代码里的字符串写成**双引号**，PowerShell 会把双引号吃掉再传给 exe，结果是
> `SyntaxError: invalid syntax`（例如 `sys.path.insert(0,src)`）。
> 规则：**外层用双引号，Python 内部只用单引号**。上面这条就是这么写的，已实测 `/health` 200。

**离网演示（没有真 LLM Key 也能点）**：把 `LLM_BASE_URL` 指到一个不存在的地址，`/health` 照样 200，`/chat` 会返回 502 `all_providers_failed`；UI 侧有内置的 `LocalDemoProvider` 兜底。所以**面试演示不需要任何 API Key**。

### 1.2 起 UI（Streamlit，占 8501 端口）

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless=true --browser.gatherUsageStats=false
```

实测输出：

```
  You can now view your Streamlit app in your browser.
  URL: http://127.0.0.1:8501
```

UI 默认去 `http://127.0.0.1:8000` 找网关（`app/streamlit_app.py:516-518`，可用 `FINOPS_GATEWAY_URL` 覆盖）。**API 没起也不会白屏**：UI 会自己拉一个本地 demo 网关（`start_demo_gateway`，默认 8010 端口）。

两个服务要同时开着 → 开两个 PowerShell 窗口，别关。

### 1.3 把本机 8501 映射到公网（面试官点链接就能看）

#### 方案 1：cloudflared 快速隧道（最快，免费，不用域名、不用备案、不用登录）

```powershell
# 安装（二选一）
winget install --id Cloudflare.cloudflared
# 或直接下载 exe：https://github.com/cloudflare/cloudflared/releases/latest 里的 cloudflared-windows-amd64.exe

# 开隧道（保持这个窗口不要关）
cloudflared tunnel --url http://localhost:8501
```

终端会打印一条 `https://<随机单词>.trycloudflare.com` 的公网地址，**直接发给面试官**即可。

- 优点：3 分钟搞定，全 HTTPS，本机不用开路由器端口，不用服务器。
- 缺点：地址是随机的、每次重启都会变；免费快速隧道没有 SLA，Cloudflare 官方定位是「测试用途」。
- 建议：`cloudflared` 也支持把 `http://localhost:8000` 同时开一条隧道，这样面试官能同时看 UI 和 `/health`（后端有部署痕迹的直观证据）。

#### 方案 2：frp（有一台公网服务器的话，地址固定、更可控）

服务器端（`frps.toml`）：

```toml
bindPort = 7000
auth.token = "<自己设一个长随机串>"
```

```bash
./frps -c frps.toml
```

本机（`frpc.toml`，把本机 8501 映射到服务器 18080）：

```toml
serverAddr = "<服务器公网 IP>"
serverPort = 7000
auth.token = "<和上面一致>"

[[proxies]]
name = "finops-ui"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8501
remotePort = 18080
```

```powershell
.\frpc.exe -c frpc.toml
# 面试官访问：http://<服务器公网 IP>:18080
```

- 优点：地址固定，可以长期挂；服务器还能顺便用来跑 §2 的 Docker 版本。
- 缺点：需要一台服务器 + 在云厂商安全组放行 7000/18080；裸 HTTP 没有证书（要 HTTPS 得配 nginx + Let's Encrypt）。
- `auth.token` 一定要设，否则任何人都能把你的 frps 当跳板。

**这两个工具我都没有在本机验证过（未安装）**，命令按官方文档写。第一次用请先在自己浏览器里打开那个公网地址确认能用，再把链接发出去。

### 1.4 发链接前的安全检查（务必看一眼）

- **别把 `.env` 或任何真 Key 传上去**：`.gitignore` 已忽略 `.env`，演示用 demo provider 就够了。
- **Streamlit 没有内置登录**：拿到链接的人都能点。缓解办法：
  - 只在面试前后几个小时内开着隧道，演示完 `Ctrl+C` 关掉；
  - trycloudflare 的随机域名不可猜，但不等于安全；
  - 不要在演示环境里放真实数据——本项目的订单/规则文档全部是合成样例；
  - 真要长期的加密码，得改 `app/`（不在本次改动范围，交给前端那一层做）。
- **`/chat` 的 SafeSQLTool 是只读的**，但公网暴露任何能触达数据库的接口都值得再确认一遍只读约束仍然生效（`reports/stage3_agent_gateway_findings.md` 里有四层防护的实测表）。

---

## 2. 路径 B：Docker Compose（本机 Docker Desktop 或云服务器）

### 2.1 前置

- **本机 Docker Desktop**：Windows 需要 WSL2 后端（家庭版也能装），首次拉 `python:3.13-slim` 和 torch 大约 1 GB，国内网络建议给 Docker Desktop 配镜像加速。
- **云服务器**：Ubuntu 22.04 / Debian 12，建议 **2 核 4 GB 起**（torch + reranker 常驻吃内存，2 GB 的机器很容易 OOM，这是我基于组件体积的推断、未实测）。

```bash
# 云服务器装 Docker（未在本机/真机验证，Ubuntu/Debian 通用）
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"     # 重新登录后生效
docker compose version              # 需要 Compose v2（插件形式）
```

### 2.2 起服务

```bash
cd FinOps-Copilot
docker compose up -d --build
docker compose ps                    # 两个服务都应是 healthy
curl -fsS http://127.0.0.1:8000/health
```

- UI：`http://<主机>:8501`，API：`http://<主机>:8000/health`。
- **云服务器要在安全组放行的端口**：`8501`（面试官访问 UI 用）。`8000` 建议**不要**对公网开放——容器里的 UI 是走 compose 内网 `http://api:8000` 找 API 的，只有你自己调试时才需要直连；真要开就临时开或只放行自己的 IP。走 frp 那条路还要放行 `7000`（frps 控制端口）和 `18080`（对外映射端口），见 §1.3。
- `ui` 用 `depends_on: condition: service_healthy` 等 `api` 健康后再起，避免 UI 抢跑找不到网关。
- 常用命令：

```bash
docker compose logs -f api        # 看 API 日志（PYTHONUNBUFFERED=1，实时）
docker compose logs -f ui
docker compose restart api
docker compose down               # 停止并删除容器（volume 里的模型/数据保留）
docker compose up -d --build      # 改了代码后重建
```

### 2.3 数据与模型：为什么用 volume，不用 COPY

这是最容易踩的地方，先说清楚三个目录的现状：

| 目录 | 现状 | 容器怎么办 |
| --- | --- | --- |
| `models/huggingface` | 本机已有 **1.2 GB**（dense 92 MB + reranker 1.1 GB）。`config/rag.yaml` 的 `retrieval.model_cache` 指向这里 | **绝不 COPY 进镜像**（镜像会大到没法推、还违反模型许可的再分发惯例）。用 volume 挂载 |
| `data/generated/` | 本机 12 MB（`orders.csv` 6.0 MiB + `chunks.jsonl` + 250 份规则文档）。**被 `.gitignore` 忽略** → 新克隆的机器上不存在 | 挂载卷，或进容器重新生成（见下） |
| `data/processed/orders.db` | 10 MB，`*.db` 被 `.gitignore` 忽略 → 新克隆的机器上也不存在 | 需要生成，命令见下 |

`docker-compose.yml` 里 `api` 服务的挂载：

```yaml
volumes:
  - ./models/huggingface:/app/models/huggingface   # 模型缓存（只读挂也行：:ro）
  - ./data:/app/data                               # 订单 CSV / 切片 / sqlite
  - ./config:/app/config:ro                        # 配置只读挂载
  - ./eval/reports:/app/eval/reports               # 评测报告落回宿主机
```

#### A. 本机已有数据/模型 → 直接挂载（最省事）

上面就是。注意 `Dockerfile` 里 uid/gid 固定为 `1001`，在 Linux 宿主机上如果挂载目录写不进去：

```bash
sudo chown -R 1001:1001 models data eval/reports
```

Windows + Docker Desktop 的 bind mount 一般不用管权限。

#### B. Linux 云服务器（没有本机那 1.2 GB 模型和数据）→ 上传或重建

**上传**（推荐，保证和本机演示完全一致）：

```bash
# 在 Windows 本机执行（PowerShell），传到服务器
scp -r .\models\huggingface user@<server>:/opt/finops-copilot/models/
scp -r .\data user@<server>:/opt/finops-copilot/
# 或者用 rsync -avz --progress 更稳，可断点续传
```

**重建**（数据本来就有固定随机种子，重建出来的是一致的；README:41 明确写了「生成器遇到已有输出会停止，不会覆盖」，所以空目录上跑是安全的）：

```bash
docker compose run --rm api python scripts/generate_dataset.py --config config/data_spec.yaml
docker compose run --rm api python scripts/generate_eval_set.py --config config/data_spec.yaml
docker compose run --rm api python scripts/build_orders_db.py    # 产出 data/processed/orders.db
```

**模型**：不挂载也会自己下——`sentence-transformers` 会按 `config/rag.yaml` 的 `retrieval.model_cache` 把权重下载到容器内 `/app/models/huggingface`（该路径是 build 时建好的可写目录）。但容器一删就没了，所以长期跑**还是挂一个卷**，否则每次重建镜像/容器都要重下 1.1 GB。

> ⚠️ 我没有实跑过「容器内首次下载模型」这条路径，上面是按 `config/rag.yaml` 的配置推断的。首次 `docker compose up` 后请 `docker compose logs -f api` 盯一下有没有下载日志。

### 2.4 环境变量

`serve_gateway.py`（以及容器）认识这些变量，全部有默认值，**不配也能起来**：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `LLM_PROVIDERS` | `primary` | 逗号分隔的 provider 名列表，主备故障转移顺序 |
| `LLM_BASE_URL` | 空 | 所有 provider 共用的 OpenAI 兼容 base url |
| `LLM_BASE_URL_<NAME>` | 空 | 单独覆盖某个 provider（如 `LLM_BASE_URL_PRIMARY`） |
| `LLM_API_KEY` | 空 | 网关 Key |
| `LLM_TIMEOUT_SECONDS` | `30` | 单次请求超时 |
| `CACHE_THRESHOLD` | `0.95` | 语义缓存相似度阈值 |
| `PORT` | `8000` | API 监听端口 |
| `FINOPS_GATEWAY_URL` | `http://127.0.0.1:8000` | **UI 用**，容器里已由 compose 设成 `http://api:8000` |

没有配置真 Key 时：`/health` 依然 200（它不调用 provider，已实测），`/chat` 会返回 502 `all_providers_failed` 并触发 UI 的本地 demo 兜底。

---

## 3. 建议的上游修复（不在本次改动范围，但值得做）

1. **加 `src/gateway/server.py`**：把 `serve_gateway.py` 的内容搬进 `src/`，提供模块级 `app` + 环境变量配置。之后容器里可以直接
   `CMD ["uvicorn", "src.gateway.server:app", "--host", "0.0.0.0", "--port", "8000"]`，不用构建时生成文件。
2. **给 UI 依赖一个正式出口**：`streamlit` / `requests` / `pandas` 目前不在 `pyproject.toml` 里，前端同事用了 `requirements-app.txt` 单独声明（该文件当前**未纳入 git**）。建议在 `pyproject.toml` 加一个 `ui` extra，或把 `requirements-app.txt` 提交进去。现在 `Dockerfile` 里是硬编码 `streamlit>=1.40,<2` 兜底，就是为了不依赖一个可能不存在的文件。
3. **`.dockerignore` 里不要把 `models/` 和 `data/` 漏掉**：现在两处都排除了。如果哪天有人改成 `COPY . .`，1.2 GB 模型会直接进镜像层。

---

## 4. 校验记录（可复现）

### 4.1 环境事实

```
docker --version            → docker: command not found     # 本机没有 Docker
.venv python                → Python 3.13.3                 # 与镜像基础镜像 python:3.13-slim 同小版本
pyyaml 6.0.3 / fastapi 0.156.8 / uvicorn 0.54.0 / streamlit 1.66.1 / requests 2.34.2
```

### 4.2 裸跑实测

```
GET http://127.0.0.1:8124/health  → 200 {"status":"ok","providers":["primary"],"cache_entries":0,"cache_hit_rate":0.0}
GET http://127.0.0.1:8124/cache/stats → 200
GET http://127.0.0.1:8502/_stcore/health → ok        # 与 compose 的 ui healthcheck 打的是同一个路由
GET http://127.0.0.1:8502/               → 200
# 无 Key 情况下打 /chat（provider 指向死端口）：HTTP 502，{"error":"all_providers_failed",...}，约 5.6s 后返回，SSE 走 event: error 帧；/health 不受影响
```

### 4.3 容器入口实测

把 `Dockerfile:80-126` 的 `printf` 段原样抽出来在宿主 bash 里执行 → 生成 `serve_gateway.py` 1584 字节 → `python -m py_compile` 通过 → 用它起服务，`/health` 200。也就是说**入口脚本本身是验证过的**，没验证的只是「Docker 能不能把镜像构建出来」。

另外做了一次交叉校验：本文件 §1.1 贴出来的那段 `serve_gateway.py` 与 Dockerfile 生成物**逐字一致**（都是 1584 字节 / 43 行，两份都 `py_compile` 通过）。所以从这份文档复制出来裸跑，和容器里跑的是同一段代码。

### 4.4 compose YAML 校验

任务是这么要求的（本机没 Docker，至少要证明 compose 是合法 YAML）：

```bash
python -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"
```

⚠️ **这条命令在本机会失败**，原因不是 YAML 有问题，而是 Windows 的 Python 默认用 GBK 打开文件，而 compose 里有中文注释（UTF-8）：

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0xad in position 58: illegal multibyte sequence
```

正确写法（均已实测通过）：

方式 1——显式指定 utf-8，跨平台通用：

```bash
python -c "import yaml; d=yaml.safe_load(open('docker-compose.yml', encoding='utf-8')); print(sorted(d['services']))"
```

方式 2——把 Python 切到 UTF-8 模式，等价于任务里那条原始命令，注意各 shell 写法不同：

```powershell
# PowerShell（实测 OK）
$env:PYTHONUTF8=1; .\.venv\Scripts\python.exe -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('OK')"
```

```bat
:: cmd.exe
set PYTHONUTF8=1 && python -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"
```

```bash
# bash / Linux
PYTHONUTF8=1 python -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"
```

> ⚠️ 别在 PowerShell 里抄 cmd 那行：`set` 在 PowerShell 是 `Set-Variable` 的别名，`set PYTHONUTF8=1` 不会设成环境变量（`&&` 也要 PowerShell 7 才支持）。

> Docker Compose 自己按 UTF-8 读文件，不受这个 Windows locale 问题影响。

结构断言（实测结果）：

```
services             → ['api', 'ui']
api.ports            → ['8000:8000']
ui.ports             → ['8501:8501']
ui.depends_on        → {'api': {'condition': 'service_healthy'}}
api.healthcheck.test → ['CMD', 'curl', '-fsS', 'http://127.0.0.1:8000/health']
api.volumes          → 含 ./models/huggingface、./data、./config:ro、./eval/reports
ui.environment       → FINOPS_GATEWAY_URL=http://api:8000
```

真正想 100% 确认 compose 语义合法，还是得在装了 Docker 的机器上跑：

```bash
docker compose config -q     # 语法/引用校验，未验证
```

---

## 5. Dockerfile 自查清单

### 5.1 设计要点

- **一个镜像两个角色**：默认 `CMD ["python", "serve_gateway.py"]` 跑 API；compose 用 `command:` 覆盖成 Streamlit 跑 UI。避免维护两份构建。
- **依赖层和代码层分开**：第 1 层只 COPY `pyproject.toml` / `requirements.txt` / `README.md` 并装依赖，改代码不会让这一层 cache 失效。
- **不 `pip install -e .`**：从 `pyproject.toml` 的 `dependencies` + `data/rag/agent/gateway` 四个 extra 里抽依赖装（14 条），避免把依赖层和源码绑死；`setuptools/wheel` 用 `[build-system].requires` 显式装掉，后面就能 `--no-build-isolation`。
- **非 root 运行**：`uid/gid=1001`，`/app` 与所有挂载点提前 `chown` 好。
- **`PYTHONUNBUFFERED=1`**：uvicorn/streamlit 的长驻日志实时可见；`PYTHONDONTWRITEBYTECODE=1` 避免往挂载卷写 `__pycache__`。
- **healthcheck 打 `/health`**，`start-period=20s` 避开冷启动；`/health` 不依赖 provider，无 Key 也是 200（已实测）。
- **`init: true`**（compose）：Streamlit/uvicorn 都不是 PID 1 友好型，docker 的 init 负责收尸和转发信号，`docker compose down` 才能干净退出。
- **镜像里生成入口脚本**：因为 `src/` 不在改动范围（见 §3.1 的上游修法）。

### 5.2 逐条自查（能验证的 / 不能验证的）

| # | 自查项 | 结论 |
| --- | --- | --- |
| 1 | 基础镜像 `python:3.13-slim-bookworm`，与本机 3.13.3 同小版本 | ✅ 版本对齐 |
| 2 | `pip install` 的依赖列表来自 `pyproject.toml` 实跑抽取（14 条） | ✅ 命令实测 |
| 3 | `streamlit>=1.40,<2` 硬编码兜底（`pyproject.toml` 里没有 UI 依赖） | ✅ 有意识的设计，见 §3.2 |
| 4 | `serve_gateway.py` 生成逻辑 | ✅ 实测 py_compile + `/health` 200 |
| 5 | `HEALTHCHECK` 的 URL 与 `src/gateway/app.py:36` 一致 | ✅ 路由源码已核对 |
| 6 | `models/`、`data/`、`.venv/` 不进镜像 | ✅ `.dockerignore` 已排除；`COPY` 也根本没写这两个目录 |
| 7 | `apt-get` 装完清 `lists`；`--no-install-recommends` | ✅ 自查 |
| 8 | `EXPOSE 8000 8501` 与实际监听端口一致（`PORT=8000`、Streamlit `8501`） | ✅ 自查 |
| 9 | `fonts-noto-cjk` 只为容器内出图不乱码；前端图表走浏览器渲染不依赖它 | ⚠️ 未实测（容器内没跑过 matplotlib/PDF） |
| 10 | `docker build` 整体 | ❌ **未验证（本机无 Docker）** |
| 11 | 镜像体积 2.5–3.5 GB、构建 5–15 分钟 | ❌ 未验证，纯估算 |
| 12 | `pip install .` 在第 2 层用 `--no-deps --no-build-isolation` | ⚠️ 逻辑自查通过，未在容器里实跑 |

### 5.3 第一次 `docker build` 时重点盯什么

1. 第 1 层 `=== 镜像内将安装的运行时依赖 ===` 打印出来的清单是否符合预期（14 条）。
2. torch/sentence-transformers 下载是否超时（国内网络 → 配镜像加速或给 pip 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`）。
3. `pip install --no-deps --no-build-isolation .` 是否报缺 setuptools（第 1 层已装，理论上不会）。
4. 构建完 `docker run --rm finops-copilot:0.1.0 python -c "import serve_gateway"` 应无输出（即导入成功）。

---

## 6. 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| `docker: command not found` | 本机没装 Docker。走 §1 裸跑路径，或装 Docker Desktop（需 WSL2） |
| `yaml.safe_load(open(...))` 报 `UnicodeDecodeError: 'gbk' codec` | Windows Python 默认 GBK 读 UTF-8 文件。加 `encoding="utf-8"`，或按 §4.4 用 `PYTHONUTF8=1` |
| PowerShell 里 `python -c` 报 `SyntaxError: invalid syntax` 且代码里的双引号不见了 | PowerShell 吃掉了内层双引号。改成**外层双引号 + Python 内部只用单引号**，或直接用 `serve_gateway.py`（§1.1） |
| UI 起来了但一直提示连不上网关 | UI 默认找 `127.0.0.1:8000`；容器里必须是 `http://api:8000`（compose 已设 `FINOPS_GATEWAY_URL`）。本机裸跑就先把 API 起起来 |
| `ui` 容器一直不 healthy，`/health` 却正常 | UI 的 healthcheck 打的是 `/_stcore/health`；如果 Streamlit 还没监听完，`start_period` 内属正常。持续不健康就 `docker compose logs ui` |
| `docker compose up` 后 `api` 立刻退出 | 看 `logs api`。最常见是端口占用（8000 被本机裸跑的 uvicorn 占了）或模型路径不存在 |
| 挂载目录 Permission denied | 容器跑 uid 1001：`sudo chown -R 1001:1001 models data eval/reports` |
| 首次启动卡很久 | 在下载模型（dense 92 MB / reranker 1.1 GB）。挂载 `models/huggingface` 可避免重复下载 |
| 页面能开但图表是方框 | 容器内缺中文字体（已装 `fonts-noto-cjk`）；如果确实出现，说明字体没生效或走的是服务端出图路径 |
| `docker compose` 报 unknown field / version 相关错误 | 本文件用的是 Compose v2 规范（顶级 `name:`，无 `version:`）。老版本 docker-compose v1 请升级到 v2 插件 |

---

## 7. 简历里怎么写这段「部署痕迹」

不要写「熟悉 Docker」这种空话，写**做过什么、遇到过什么问题、怎么解决的**：

> **容器化与部署**：为 FinOps-Copilot 编写 `Dockerfile` + `docker-compose`（单镜像双角色：FastAPI 网关 8000 / Streamlit 展示层 8501），依赖层与代码层分离以复用构建缓存，非 root(uid 1001) 运行，服务级 healthcheck + `depends_on: service_healthy` 编排启动顺序；模型权重（1.2 GB）与数据不入镜像，通过 bind mount 注入；**在无 Docker 环境（本机 `docker: command not found`）下用 uvicorn + Streamlit 裸跑等价复现了镜像内入口脚本，并以 cloudflared 隧道把本机 8501 映射为公网 HTTPS 链接做在线演示**。

面试官大概率会追问的三个点，先想好答案：

1. **为什么模型不 COPY 进镜像？** → 体积（1.2 GB×每次构建）、层缓存失效、模型许可的再分发问题；用 volume 让「镜像=代码，卷=资产」。
2. **没装 Docker 怎么保证 Dockerfile 是对的？** → 把 Dockerfile 里可独立验证的部分（依赖抽取命令、生成入口脚本、healthcheck 打的路由）抽出来在宿主 venv 上实测，并**明确标注未经 `docker build` 验证**；这比声称「已验证」可信。
3. **两个服务怎么保证启动顺序？** → `healthcheck`（`/health` 不依赖外部 LLM，无 Key 也 200）+ `depends_on: condition: service_healthy` + `init: true` 收信号。

---

## 附：一页速查

```powershell
# ===== 路径 A：不装 Docker，发链接 =====
cd FinOps-Copilot
# 窗口 1：API
.\.venv\Scripts\python.exe serve_gateway.py                 # 或 §1.1 的一行命令
curl.exe http://127.0.0.1:8000/health                       # 期望 {"status":"ok",...}
# 窗口 2：UI
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.port 8501 --server.headless=true
# 窗口 3：公网
cloudflared tunnel --url http://localhost:8501              # 打印 https://xxx.trycloudflare.com 发出去

# ===== 路径 B：Docker =====
docker compose up -d --build
docker compose ps                                            # 两个都 healthy
curl.exe http://127.0.0.1:8000/health
docker compose logs -f api
docker compose down                                          # 收工
```