# data/raw — 真实公开数据接入目录

本目录存放 **外部真实数据集** 的原始下载文件。它与 `data/generated/` 无任何关系：
`data/generated/` 是 Stage 1 合成产物（`config/data_spec.yaml` 里
`synthetic_only: true`），`data/processed/orders.db` 是它的派生物。
**本目录的接入链路不会读写、也不会覆盖 Stage 1 的任何产物。**

## 1. 数据来源

| 项 | 值 |
| --- | --- |
| 数据集 | Online Retail II / Online Retail（英国在线零售真实交易流水，UCI Machine Learning Repository） |
| 提供方主页 | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| 直链 | https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip |
| 备用镜像（同一零售商的 2010-2011 切片） | https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv |
| 备用镜像对应主页 | https://archive.ics.uci.edu/dataset/352/online+retail |
| 许可 | CC BY 4.0（UCI 数据集默认许可；引用时请标注 UCI Machine Learning Repository） |
| 抓取日期 | 见 `reports/real_data_stats.json` 的 `download.downloaded_at` 字段（脚本自动写入） |
| 原始行数 / 清洗后行数 / SHA-256 | 见 `reports/real_data_stats.json`（脚本自动写入，勿手工编辑） |

原始数据的字段（真实列名，与项目契约完全不同）：

```
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
```

- `InvoiceNo`：发票号。以 `C` 开头 = 取消单（cancellation）。
- `StockCode`：商品/货号。含 `85123A` 这类子码，以及 `POST`/`M`/`D` 这类人工调整码。
- `Description`：商品描述。**同一 StockCode 可能对应多种写法**（真实观测，见报告 3.3）。
- `Quantity`：数量。**负数 = 退货/取消腿**，不是单独的退款字段。
- `InvoiceDate`：无时区的时间戳，本项目按 `+00:00` 解释。
- `UnitPrice`：英镑浮点单价，存在 0 与负值。
- `CustomerID`：客户号，**约四分之一的行是空值**。
- `Country`：收货国家。**同一客户可能出现在多个国家**。

## 2. 是否应纳入 git

**建议不纳入本目录下的大文件。** 理由：

- zip / csv 单体在 4 MB（备用 CSV 约 43 MB）量级，属于可复现的外部数据，
  不应污染仓库历史；
- 许可允许再分发，但仓库最佳实践是只提交**来源说明 + 校验和 + 复跑脚本**。

因此本目录只提交 `README.md` 与 `.gitkeep`；数据文件本身由
`scripts/ingest_real_data.py` 按需下载。

> 需要 `.gitignore` 加规则（例如 `FinOps-Copilot/data/raw/*.zip`、
> `FinOps-Copilot/data/raw/*.csv`、`!FinOps-Copilot/data/raw/README.md`）——
> **该改动由 lead 决策，本任务未修改 `.gitignore`。**

## 3. 复跑

```bash
cd FinOps-Copilot
# 下载 + 解析 + 规范化 + 建库 + 跑项目口径 + 写统计
.venv/Scripts/python.exe scripts/ingest_real_data.py
# 渲染对照报告
.venv/Scripts/python.exe scripts/real_data_report.py
```

需要额外安装一个包才能解析 Online Retail II 的 `.xlsx` 成员：

```bash
.venv/Scripts/python.exe -m pip install openpyxl
```

该依赖**刻意没有**写进 `requirements.txt` / `pyproject.toml`：它只服务这条离线
入库链路，与运行时服务无关。

详细说明见 [`docs/真实数据接入.md`](../../docs/真实数据接入.md)。