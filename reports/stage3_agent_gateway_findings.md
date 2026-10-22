# Stage 3 实测记录：LangGraph 智能体 + 高可用网关

本文件记录 Stage 3 的**本机实测**结果。全部数字由下方脚本现场跑出，非估算。

## 1. LangGraph 状态机四条路径

图结构：`route -> (metric | sql | refuse)`，`metric/sql -> _should_retry -> reflect | answer`，
`reflect -> answer -> END`。

| 路径 | route | retries | trace | 结果 |
| --- | --- | ---: | --- | --- |
| 固定指标 | `metric` | 0 | route → metric → answer | revenue = 345,398,219.59 |
| 长尾 SQL | `sql` | 0 | route → sql → answer | 正常返回 |
| **SQL 自纠错闭环** | `sql` | **1** | route → sql(False) → **reflect(True)** → answer | `[{'net': 348656558.4}]` |
| 危险请求 | `refuse` | 0 | route → refuse | 正确拒答"删掉所有订单" |

自纠错闭环的实测细节：第一次 SQL 故意写错列名（`total_revenue`），
`SafeSQLTool` 返回 `ok=False` + 错误文本 + hint；
`reflect` 节点读入失败 SQL、错误、hint 后重新生成正确 SQL；
第二次执行 `ok=True`，`state.retries=1`，随后进入 `answer`。
**这条链路证明了"报错 → 反馈 → 纠正"是真跑通的，不是设计文档里的设想。**

## 2. 网关高可用行为（`src/gateway/client.py`）

| 场景 | 实测结果 |
| --- | --- |
| 主 Provider 连续 503 → 备 Provider | 3 次 attempt：primary(503) / primary(503) / **backup(200)**，`served_by=backup` |
| 非可重试 400 | **快速失败**，`transport_calls=['primary']`，**未调用 backup** |
| 全部 Provider 失败 | 抛 `AllProvidersFailed`，4 次 attempt 全部记录 |
| 客户端断开 | 探针返回 True → 抛 `ClientDisconnected`，backup **0 次调用**（不再浪费重试） |
| 语义缓存 | 重复同一问题 → `cache_hit=True`，`similarity=1.0`，transport 仅 1 次 |
| 语义缓存防误命中 | 换一个不同问题 → `cache_hit=False`，**不会返回别的问题的答案** |

`client.py` 的设计约束（写在模块 docstring 里）：只在首个 token 之前故障转移；
客户端断开立即停止重试；4xx（非 429）快速失败不重试；语义缓存必须带相似度阈值。

## 3. 只读 SQL 的四层防护（`src/agent/tools.py`）

任何单层都可被绕过，因此叠了四层：`mode=ro` 连接 / `set_authorizer` 回调 /
表列白名单 / 行数上限 + VM 指令预算。

实测拦截结果：

| 攻击载荷 | 结果 |
| --- | --- |
| `DROP TABLE orders` | 拒绝：only a single read-only SELECT statement is permitted |
| `PRAGMA table_info(orders)` | 拒绝（同上） |
| 多语句拼接 | 拒绝（同上） |
| `SELECT * FROM sqlite_master` | 拒绝：access to sqlite_master.type is prohibited |
| `SELECT * FROM customers` | 拒绝：no such table: customers |
| `SELECT load_extension(1)` | 拒绝：not authorized to use function: load_extension |

## 4. 复现方式

```bash
./.venv/Scripts/python.exe -m pytest tests/unit/ -q     # 38 passed
```

`tests/unit/test_agent_gateway.py` 覆盖上述四条路由、fallback 循环、
自纠错闭环、以及四层安全防护的回归。

---

*语料与数据库均为合成数据，延迟为本机单次调用墙钟时间，不代表生产性能。*