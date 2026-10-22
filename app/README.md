# FinOps-Copilot 交互式 Web Demo（Streamlit）

```bash
cd FinOps-Copilot
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
# 打开 http://127.0.0.1:8501（首次加载检索器约 1-3 分钟，之后走 @st.cache_resource 秒开）
```

命令行自检（不启 UI，直接跑 RAG 溯源 + Agent 三路由）：`.venv/Scripts/python.exe app/smoke_check.py`

## 与 src/ 真实 API 对不上的地方（只记录，未修改 src/）

1. **首屏慢的主因不是 Cross-Encoder，而是 DenseRetriever 构造**：`src/rag/dense.py:139`
   在 `__init__` 里对全部 5,000 切片做一次 `embedder.encode(...)`，且没有落盘缓存；
   `BAAI/bge-reranker-base` 的加载只是第二段开销。因此缓存粒度必须是整个 `HybridRetriever`。
2. **refuse 分支的回答是硬编码英文**：`src/agent/graph.py:222-226` 直接写死英文句子，
   不经过 `ANSWER_SYSTEM`，所以「拒绝」路由在中文界面里会冒出一段英文；本 demo 原样展示，未改 src。
3. **`hybrid`（rerank=False）不是"纯 BM25"**：`src/rag/hybrid.py:49-97` 里 `rerank=False`
   仍会做 RRF 融合，只是跳过精排。所以侧边栏把「纯 BM25」实现为 `mode="bm25"`
   （走 `search_bm25`），另设 `hybrid` 档位对比"有无精排"。
4. **`DenseRetriever.search_many` 每查一次就重新编码 query**（`src/rag/dense.py:150`），
   单条交互没问题，批量评测时是重复开销。
5. **`ScriptedTransport` 队列耗尽会抛 503**（`src/gateway/client.py:339-344`），
   不适合做可反复点击的页面网关；demo 用同契约的 `EchoTransport` 起 `create_app`，
   `LLMClient` 的 failover / 语义缓存逻辑仍是 src 的真实现。