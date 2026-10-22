# FinOps-Copilot

电商经营分析与规则审计智能体系统。当前完成 Stage 1：可复现的合成订单、带版本与行号溯源的规则文档库，以及分层检索评测集。文档和数据均为离线合成样例，不构成真实平台政策或法律合规意见。

## Stage 1 范围

- 固定随机种子生成 50,000 条订单，其中 6% 为单一异常；异常标签保存在独立 sidecar。
- 生成 250 份 V1/V2 规则文档和 5,000 个带 `doc_id/version/page/line` 来源信息的切片。
- 基于实际切片生成 120 组 QA，覆盖版本冲突、边界例外、否定词面干扰、无答案四类陷阱，并按 60/30/30 划分。
- 当前生效营收口径：只统计 `completed` / `shipped`，净额为 `paid_amount_minor - refunded_amount_minor`；金额以分存储，展示时除以 100 转元。

品类、渠道、退款窗口、对账容差及异常类型等未由用户指定的值均为合成默认值，配置内以 `TODO(HUMAN)` 标注；不可将它们当作真实业务口径。

## 环境

需要 Python 3.11 或更高版本。

    py -3.13 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements-stage1.txt

全项目依赖入口为 requirements.txt；仅准备 Stage 1 时使用 requirements-stage1.txt。

## 生成数据

从项目根目录依次运行：

    python scripts/generate_dataset.py --config config/data_spec.yaml
    python scripts/generate_eval_set.py --config config/data_spec.yaml

主要产物：

- `data/generated/orders/orders.csv`：50,000 条合成订单。
- `data/generated/orders/anomaly_labels.jsonl`：仅供离线分析的异常真值标签，不应提供给 Agent。
- `data/generated/rule_docs/documents/`：250 份 Markdown 规则文档。
- `data/generated/rule_docs/chunks.jsonl`：5,000 个规则切片。
- `eval/datasets/stage1_eval.jsonl`：120 组 QA 与标准答案、gold 引用、hard negatives 和分集。
- `reports/stage1_manifest.json`：配置快照、记录摘要及产物 SHA-256。

生成器遇到已有输出会停止，不会覆盖。需要重新生成时，请使用独立项目副本或新的输出路径，并保留现有数据作为可复现证据。

## 目录

    config/                 数据规格及运行配置
    data/raw/               外部输入；不得放入真实个人敏感数据
    data/generated/         Stage 1 合成订单与规则文档
    data/processed/         规范化文档、索引构建中间件
    docs/                   架构说明与经人工确认的规则原文
    eval/                   QA、标注集、评测配置与报告
    reports/                可复现运行报告
    scripts/                各 Stage 的命令行入口
    src/rag/                混合检索、引用溯源与重排
    src/agent/              LangGraph 工作流和业务工具
    src/gateway/            异步模型网关、容错与语义缓存
    src/common/             配置、日志和共享类型
    tests/                  单元、集成和固定样例

Stage 1 自动化测试可用 `python -m unittest tests.unit.test_stage1_generation -v` 运行。
