# Stage 2 检索消融对比

- 生成时间: 2026-10-21T13:57:15.823248+00:00
- 语料切片: 5000   评测题: 120   k=5
- Embedding: `BAAI/bge-small-zh-v1.5` (sentence_transformers)
- Reranker: `BAAI/bge-reranker-base` (sentence_transformers_cross_encoder)

| 检索方案 | Recall@5 | MRR@5 | 平均延迟(ms) |
| --- | ---: | ---: | ---: |
| 纯 BM25（词法） | 1.0000 | 0.5000 | 7.92 |
| 纯向量（BGE 双塔） | 0.9667 | 0.4646 | 13.51 |
| Hybrid + Rerank（BM25+Dense→RRF→CrossEncoder） | 1.0000 | 0.6133 | 771.15 |

计入题数 90，排除不可回答题 30。

> 合成语料的离线检索指标，不等于端到端回答准确率，也不代表生产性能。

## 指标读法与既往回退的处置（必读）

本次运行 Hybrid+Rerank 的 Recall@5 与纯 BM25 持平（1.0000），
MRR@5 由 0.5000 提升到 0.6133（+0.1133）。

两点必须同时讲清楚，否则这张表会被误读：

1. **Recall 持平不等于重排无效。** Recall@5 已触顶（1.0000），
   天花板效应下它无法再涨；重排的收益体现在 MRR——把答案文档排得更靠前。
   全量 8 道提升题全部属于 `version_conflict` 类型：BM25 与 Dense 都把
   gold 排在后面，Cross-Encoder 把它提到第 1。这正是 RRF 融合后再精排该起的作用。
2. **`bad_cases.json` 的 `hybrid_fixed` 为空是必然的，不是脚本坏了。** BM25 的
   Recall@5 已经是 1.0000，不存在「BM25 漏掉、Hybrid 救回」的题。
   真实可讲的案例在 `rerank_promoted_gold`（8 个）。

### 此前出现过的回退及处置（历史，已修复）

修复前的配置（`reranker.candidate_k: 100`）下，Hybrid 的 Recall@5 曾是 0.9889，
低于纯 BM25 的 1.0000。归因结论：Cross-Encoder 打分饱和——EVAL-116 的 100 条候选
分数极差仅 0.0034，打平后按 `chunk_id` 字典序回退，把 gold 挤出了 top-5。
处置（已经人工拍板）：

- 窗口 `candidate_k: 100 -> 20`（`config/rag.yaml`）：窗口越小，进入饱和区的近似副本越少。
- tie-break 由 `chunk_id` 改为 `rrf_score`（`src/rag/rerank.py: rerank_sort_key`）：
  分数饱和时让词法/向量共识来定序，而不是字母序。

效果：EVAL-116 的 gold 名次由「掉出 top-5」回到第 4（仍在 top-5 内），
全量 Recall@5 恢复 1.0000，单题延迟由 4586 ms 降到约 1070 ms。

修复前的完整证据链保留在 `reports/stage2_retrieval_findings.md`，
复现脚本：`eval/diagnose_hybrid_window.py`、`eval/probe_eval116_rerank_scores.py`。
