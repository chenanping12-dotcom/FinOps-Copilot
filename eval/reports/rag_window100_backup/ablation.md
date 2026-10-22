# Stage 2 检索消融对比

- 生成时间: 2026-10-21T13:07:27.255281+00:00
- 语料切片: 5000   评测题: 120   k=5
- Embedding: `BAAI/bge-small-zh-v1.5` (sentence_transformers)
- Reranker: `BAAI/bge-reranker-base` (sentence_transformers_cross_encoder)

| 检索方案 | Recall@5 | MRR@5 | 平均延迟(ms) |
| --- | ---: | ---: | ---: |
| 纯 BM25（词法） | 1.0000 | 0.5000 | 13.63 |
| 纯向量（BGE 双塔） | 0.9667 | 0.4646 | 20.51 |
| Hybrid + Rerank（BM25+Dense→RRF→CrossEncoder） | 0.9889 | 0.6106 | 6564.20 |

计入题数 90，排除不可回答题 30。

> 合成语料的离线检索指标，不等于端到端回答准确率，也不代表生产性能。

## 已知回退与归因（必读）

Hybrid+Rerank 的 Recall@5 低于纯 BM25。这不是数值噪声，已定位到机制：

- `src/rag/rerank.py` 的 tie-break 用 chunk_id 字典序；本语料中
  Cross-Encoder 打分饱和（EVAL-116 的 100 条候选分数极差仅 0.0034），
  打平后按 ID 排序，使 gold 被字典序更靠前的近似副本挤出 top-5。
- 窗口扫描：窗口 5 / 20 的 Recall@5 均为 1.0000，窗口 50 / 100 为 0.9889，
  与上述机制一致（窗口越大，进入饱和区的近似副本越多）。
- 完整证据链：`reports/stage2_retrieval_findings.md`；
  复现脚本：`eval/diagnose_hybrid_window.py`、`eval/probe_eval116_rerank_scores.py`。
