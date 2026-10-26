# FinOps-Copilot 交互式 Web Demo · 设计说明

> 代码入口：`app/streamlit_app.py`（748 行）
> 运行方式见 `app/README.md`；部署方式见 `DEPLOY.md`；截图见 `docs/demo/`。

---

## 0. 这份 Demo 想证明什么

一句话：**它是一个「不调用任何外部大模型也能完整跑通」的全栈 AI 系统演示壳** —— 检索层和 Agent 层是 `src/` 里的真实实现，只有「生成自然语言」这一层被替换成了本地确定性规则。

| 它要证明的 | 它**不**证明的 |
|---|---|
| 混合检索（BM25 + Dense + RRF + Cross-Encoder）真能跑，且**每次检索都带 `[Doc_ID, Page, Line]` 溯源** | 不证明我能拿到大模型 API key |
| LangGraph 状态机（route → metric/sql/refuse → reflect → answer）真的按分支走 | 不证明微调能力 |
| 只读 SQL 护栏、写操作拒绝、PII 拒绝真的生效 | 不证明线上高并发（单机 Streamlit） |
| 语义缓存与主备故障转移的网关逻辑真实可用 | 不证明真实平台政策（数据与规则均为本地合成） |

**为什么刻意不接大模型**：规则文档切成的 5,000 个切片里，`POL-013-V1` 和 `POL-013-V2` 是同一业务的两版口径（V1 只认 `completed`，V2 认 `completed` + `shipped`）。如果让大模型"把原文改写成通顺的话"，它会顺手把 `completed、shipped` 概括成"已完成订单" —— **口径就在这里被抹掉了**。所以 Demo 直接展示规则原文，宁可读起来不像人话，也不缩口径。

想换成自然语言也可以：把 `src/gateway/` 的 `LLMClient` 接上任意一家 API 即可，**检索层与 Agent 层一行都不用改**。这是刻意的分层。

---

## 1. 界面结构

```
┌─ 侧边栏 ─────────────┬─ 主区 ───────────────────────────────────┐
│ 检索模式              │ 标题 + 一句话定位                          │
│  ○ 混合检索+精排(默认)│ 检索配置 caption（模型/超参直接来自 rag.yaml）│
│  ○ 混合检索(BM25+向量)│                                          │
│  ○ BM25（词法基线）   │ ┌ 提问 ────────────────────────────────┐ │
│  ○ 向量（BGE 语义）   │ │ 输入框（key="question"）              │ │
│                       │ │ 知识库预设题 ×3（来自真实评测集）      │ │
│ 返回条数 top_k [1-10] │ │ 经营指标预设题 ×4                     │ │
│                       │ └──────────────────────────────────────┘ │
│ 语义缓存              │ [开始分析（同时跑 RAG 与 Agent）]          │
│  /cache/stats 面板    │                                          │
│ [启动本地演示网关]    │ ┌ 📚 知识库问答(RAG) ┬ 📊 经营指标(Agent)┐│
│ [发2次/chat预热缓存]  │ │                                    ││
│ 网关地址 [127.0.0.1:8000] │ 各自的「仅运行…」按钮 + 结果区  ││
└───────────────────────┴──────────────────────────────────────────┘
```

两处细节：

- **`st.dataframe(width="stretch")` + `hide_index=True`**：命中表把 `排名/溯源引用/方法/分数/Doc_ID/版本` 六列一次列全，不需要展开就能看到"这条来自哪份文档、第几页、哪一版"。
- **`_as_json_object()` 包装**：Streamlit 1.66 的 `st.json()` 底层是 `reactJsonViewCompat`，**只接受 JSON 对象**；传 `None` 会被渲染成 `{"ERROR": "Object ... is not valid JSON"}`。Agent 的 `decision` / `trace` 在 refuse 路由下可能是 `None`，所以统一走这个包装函数。

---

## 2. 三条链路

### 2.1 RAG 链路（左 Tab）

```
query
  └─ run_retrieval(retriever, query, mode, top_k)
       └─ HybridRetriever.search()      ← src/rag/ 的真实实现
            ├─ BM25Retriever.search()     词法召回
            ├─ DenseRetriever.search()    BGE 向量召回（bge-small-zh-v1.5）
            ├─ RRF 融合                    k=60，候选 candidate_k=100
            └─ CrossEncoderReranker        bge-reranker-base，取前 20 精排
  └─ _render_rag_result(hits, elapsed, mode, query)
       ├─ st.success  ← 「回答依据」：top-1 的 citation_label + 原文
       ├─ st.dataframe ← 其余命中（排名/引用/方法/分数/Doc_ID/版本）
       └─ st.expander ×N  ← 每条的四项元数据 + 命中片段正文 + 同页父块原文
```

**Parent-Child 分块在 UI 上被显式暴露**：每个 expander 里既有命中的**子块**正文，也有它所属的**同页父块**原文 —— 这是为了让"检索命中子块、但阅读需要父块"这个设计可被肉眼验证。

`top_k` 滑杆是**实时传进检索器**的，不是前端截断 —— 改滑杆重跑，Cross-Encoder 的候选窗口会真的跟着变。

### 2.2 Agent 链路（右 Tab）

```
question
  └─ run_agent(agent, question)
       └─ FinOpsAgent.invoke()          ← src/agent/graph.py 的 LangGraph 状态机
            ├─ route        意图分流：metric / sql / refuse
            ├─ MetricTool   固定 schema，防口径漂移（order_count / revenue /
            │               avg_order_value / cancel_rate × category/merchant/status/month）
            ├─ SafeSQLTool  只读长尾查询 + AST 校验（拒绝写操作）
            ├─ Reflector    报错与异常值自我纠错，最多重试 2 次
            └─ answer       只用工具真实返回的数字组织回答
  └─ _render_agent_result(state, elapsed)
       ├─ st.success            ← 最终答案
       ├─ st.metric ×5          ← 路由 / 工具 / 返回行数 / 重试次数 / 耗时
       ├─ [条件] st.info        ← 「口径类问题」提示（见 §3.8）
       ├─ st.json(state.decision)    ← 工具调用参数（可核对选了哪个 metric / dimension）
       ├─ 实际执行的 SQL（只读）
       ├─ st.dataframe(result.rows)  ← 返回数据原文
       └─ st.expander ×2        ← state.trace（每节点轨迹）+ 完整 AgentState.as_dict()
```

`state.trace` 展开后能看到 `route → plan → execute → answer` 每个节点的输入输出，**面试时可以直接用它证明"路由不是写死的 if-else"**。

### 2.3 网关链路（侧边栏）

```
[启动本地演示网关]
  └─ start_demo_gateway(port=8010)   ← src/gateway/app.py 的 create_app
       └─ EchoTransport              ← 同契约的本地 transport（见 §3.5）

[发 2 次 /chat 预热缓存（看命中率）]
  └─ warm_gateway_cache(url)
       └─ POST /chat ×2 → 读 /cache/stats 的 hit_rate
```

**语义缓存的 `hit_rate` 是真实可观察的**：同一句话发两次，第二次命中，`/cache/stats` 的 `hits` 从 0 变 1、`hit_rate` 变 0.5。这一层跑的是 `src/gateway/` 的真实 `LLMClient` 逻辑（缓存键、主备切换、SSE 首帧降级），只有"真正调模型"那一下被 `EchoTransport` 换掉了。

---

## 3. 九个刻意的设计决策

| # | 决策 | 原因 |
|---|---|---|
| 1 | **不调外部大模型**，直接展示规则原文 | 生成式改写会缩口径（§0）。宁可"不像人话"，也不让 `completed、shipped` 变成"已完成订单" |
| 2 | top-1 单独拎出来做「**回答依据**」块 | 原先只有一张命中表，用户跑完以为"没返回回答"。见 §3.9 |
| 3 | 拒绝路由**保留英文原文**，不做翻译 | `src/agent/graph.py:222-227`（`_refuse_node`）硬编码英文句子、不走 `ANSWER_SYSTEM`。这是 `src/` 的真实行为，demo **原样展示、不粉饰**，并在 `app/README.md` 里列为已知问题 |
| 4 | 「纯 BM25」实现为 `mode="bm25"`，而不是 `hybrid(rerank=False)` | `src/rag/hybrid.py:49-53` 的 `search_hybrid(rerank=False)` **仍会做 RRF 融合**（转交 `search_hybrid_from_candidates`，见 `src/rag/hybrid.py:55`），只是跳过精排。如果拿它当"纯 BM25"基线，对照实验就是错的 |
| 5 | 缓存粒度是**整个 `HybridRetriever`**，不是模型 | 首屏慢的主因不是 Cross-Encoder，而是 `src/rag/dense.py:143` 在 `__init__` 里对全部 5,000 切片做一次 `embedder.encode()` 且无落盘缓存。缓存必须从这一层开始 |
| 6 | 用 `EchoTransport` 而不是 `ScriptedTransport` | `ScriptedTransport` 队列耗尽会抛 503（`src/gateway/client.py:339-344`），不适合可反复点击的页面 |
| 7 | 三个「预设问题」按钮**点了直接跑** | 原先只填输入框不触发运行，用户看到有字了以为在跑，实际什么都没发生 |
| 8 | 经营指标 Tab 加「口径类问题」提示 | 该 Tab 是固定口径 Agent，规则类问题它给的是**数字**不是**规则原文**，容易被误读成答非所问 |
| 9 | 侧边栏 `selectbox`/`slider` 显式给 `key` | 不给 key 时 Streamlit 用 label 生成 key，`_set_question` 回调读不到当前档位，会用错默认值重跑 |

### 3.9 为什么加了「回答依据」块（真实 bug 复盘）

用户反馈："跑完问题之后 RAG 和 Agent 没有返回回答。" 复现后确认是**三个独立原因叠加**：

1. **RAG Tab 原本根本没有"答案块"** —— 只有命中表格。设计上就对（不调大模型），但**这层意图在 UI 上一个字都没写**，看起来像 bug。→ 加 `st.success` 展示 top-1 的 `citation_label` + 原文，并注明"这是规则原文、未经生成式改写"。
2. **预设题按钮只填字不运行**（决策 7）。
3. **输入框按回车不触发运行**，且页面没提示。→ 两个 Tab 的"还没跑过"提示改写清楚三种开始方式，并显式标注"回车不触发，必须点按钮"。

修完后的效果可以直接对比 `docs/demo/` 里的一组对照截图：

| 截图 | 模式 | top-1 | 那一版的口径 |
|---|---|---|---|
| `06-纯BM25排序.png` | `bm25` | `[POL-013-V1]`，bm25 分 52.091 | "仅将状态为 **completed** 的订单纳入营收统计" ← **旧口径** |
| `02-RAG检索结果.png` | `hybrid_rerank` | `[POL-013-V2]`，rerank 分 0.9993 | "completed、**shipped**" ← **生效口径** |

同一句问题："当前生效口径下，退款核算/家居生活中哪些订单状态可以计入营收？" —— **词法检索被 V1 的表面词面骗了，混合+精排抓到了 V2。** 这两张图是简历那条"混合检索修好 V1/V2 版本混排"的视觉证据。

---

## 4. 哪些是 `src/` 的真逻辑、哪些是 demo 的壳

这是**最重要的诚实边界**，面试被追问"这是不是真的"时照着这张表答：

| 组件 | 归属 | 说明 |
|---|---|---|
| `HybridRetriever` / BM25 / Dense / RRF / Cross-Encoder | **真实** | `src/rag/`，一行未改 |
| `FinOpsAgent` LangGraph 状态机 | **真实** | `src/agent/graph.py` |
| `MetricTool` / `SafeSQLTool` / `Reflector` | **真实** | `src/agent/` |
| `create_app` / 语义缓存 / 主备故障转移 / SSE 降级 | **真实** | `src/gateway/` |
| 订单库 `orders.db`（50,000 单） | **真实** | `scripts/` 合成，含 6 类业务脏数据 |
| 规则语料 `chunks.jsonl`（5,000 片 / 250 文档） | **真实** | Stage 1 合成 |
| 评测集 `stage1_eval.jsonl`（120 题） | **真实** | 预设题直接取自这里，不是手编的 |
| **`LocalDemoProvider`** | **demo 壳** | `app/streamlit_app.py:156`。规则版 provider，按 system prompt 签名分流（`"You are the router"` → 路由、`"self-correction node"` → 纠错、`"answer node"` → 组织回答、`"read-only SQLite SELECT"` → 计划 SQL）。**只替换"模型吐字"这一步，工具与状态机都是真的** |
| **`EchoTransport`** | **demo 壳** | `app/streamlit_app.py:325`。与 `ScriptedTransport` 同契约，但不断流 |
| Streamlit 界面本身 | **demo 壳** | 这一层就是给人看的 |

`LocalDemoProvider` 的决策依据是**显式关键词表**，全部写在 `app/streamlit_app.py:120-153`：

- `_UNSAFE_KEYWORDS`（删掉/修改/insert/update/drop…）→ 路由到 `refuse`
- `_PII_KEYWORDS`（手机号/收货地址/身份证…）→ 拒绝
- `_SQL_KEYWORDS`（支付方式/渠道/分组/占比…）→ 路由到长尾 `sql`
- `_METRIC_KEYWORDS`（取消率/客单价/营收/订单数，**先特化再泛化**）→ 路由到 `metric`
- `_DIMENSION_KEYWORDS`（品类/商家/状态/月份）→ 决定 metric 的 dimension

**面试官问"你不用大模型，那路由不还是规则吗"——答案就是这张表。** 区别在于：规则在这里是**确定性替身**，换回真 provider（`lambda system, user: llm.chat(...)`，契约见 `src/agent/graph.py:101`）时，状态机、工具、护栏全都不用动。这也是为什么 `tests/unit/` 里用 `lambda system, user: next(queue)` 就能测整个图。

---

## 5. 已知限制（不要对外声称）

| 限制 | 事实 |
|---|---|
| **Docker 端到端未验证** | 仓库有 `Dockerfile`（143 行）与 `docker-compose.yml`（101 行，暴露 8000/8501），但**开发机没装 Docker**，镜像从未构建过。可以说"写了容器化编排"，**不能说"部署过"** |
| 单机 Streamlit，无并发压测 | 只验证了单用户交互 |
| 拒绝路由回答是英文 | `src/` 的真实行为，见决策 3 |
| `DenseRetriever.search_many` 重复编码 query | `src/rag/dense.py:154` 每次调用都对 query 重新 `encode`，单条交互无感，批量评测时有重复开销 |
| 规则语料是模板化合成的 | `EVAL-116` 那类"重叠近似副本"问题源于语料模板化，**治本要重跑 Stage 1**，目前如实记录在 `reports/stage2_retrieval_findings.md` |
| 真实公开数据只做了**旁证** | `data/raw/` 接入的 UCI Online Retail II 只用于验证"接数管线能跑通 + 真实脏数据长什么样"，**主口径仍是合成数据**；结论见 `reports/real_data_findings.md` |

---

## 6. 怎么复现验证（三层，从便宜到贵）

```bash
cd FinOps-Copilot

# ① 单元测试（~8s，不打 UI）
.venv/Scripts/python.exe -m pytest tests/unit/ -q          # → 57 passed

# ② 命令行语义自检（不启 UI，直接跑 RAG 溯源 + Agent 三路由）
.venv/Scripts/python.exe app/smoke_check.py

# ③ AppTest 全流程（模拟点击，~1-3 分钟含首次模型加载）
.venv/Scripts/python.exe app/apptest_check.py              # → APPTEST OK
.venv/Scripts/python.exe app/probe_combined.py             # 逐预设题 dump 两个 Tab 的返回
```

启动界面：

```bash
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
# http://127.0.0.1:8501
```

Windows 上一键（含自动找浏览器）：

```bat
..\启动Demo.bat
```

**断网也能跑**：模型已在 `models/huggingface/` 时，`src/rag/hf_cache.py` 会自动把进程切成离线模式。此前断网会报 `ValueError: Unrecognized processing class in BAAI/bge-small-zh-v1.5` —— `transformers` 5.x 请求 huggingface.co 失败后抛 `ValueError`，不在 `build_embedder()` 的降级 `except` 列表里，**模型明明就在本地也会把整个检索器构造打挂**。回归测试见 `tests/unit/test_hf_cache.py`。

---

## 7. 文件地图

| 路径 | 作用 |
|---|---|
| `app/streamlit_app.py` | 界面全部代码（748 行） |
| `app/README.md` | 运行方式 + 与 `src/` 对不上的 5 处已知问题 |
| `app/smoke_check.py` | 命令行自检：RAG 溯源 + Agent 三路由 |
| `app/apptest_check.py` | AppTest 全流程检查 |
| `app/probe_combined.py` | 逐预设题 dump 两个 Tab 的返回（排查"没返回回答"用的） |
| `app/capture_demo.py` | CDP 驱动真实浏览器截图（01–04） |
| `app/capture_demo2.py` | 同上，采写操作拒绝 + 纯 BM25 对照（05–07） |
| `docs/demo/` | 7 张截图 + `00-Demo总览.png`（1 页看全） |
| `docs/Demo设计说明.md` | 本文件 |
| `docs/真实数据接入.md` | 真实公开数据接入说明 |
| `reports/stage2_retrieval_findings.md` | EVAL-116 检索回退归因（含 tie-break 修复） |
| `reports/stage3_agent_gateway_findings.md` | Agent + 网关实测记录 |
| `DEPLOY.md` | 部署指南（容器化部分**未经端到端验证**） |

---

## 附：一句话总结（面试可直接背）

> 这是一个**不依赖大模型 API 也能完整跑通**的电商经营分析与合规审计系统。检索层是 BM25 + BGE 向量双路召回、RRF 融合、Cross-Encoder 精排，**每次命中都带 `[Doc_ID, Page, Line]` 溯源**；Agent 层是 LangGraph 状态机，把意图分成固定口径指标、只读长尾 SQL 和拒绝三类，带自我纠错重试。Demo 刻意**不接大模型**，因为规则语料里存在 V1/V2 两版口径，生成式改写会把口径抹平 —— 所以直接展示规则原文，把"生成自然语言"做成可替换的一层。