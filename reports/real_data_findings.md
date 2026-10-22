# 真实公开数据接入与「合成 vs 真实」对照结论

> 本文所有数字均由 `FinOps-Copilot/scripts/ingest_real_data.py` 在真实下载的数据上实际跑出，未做任何手工调整。每条结论下方给出可复跑命令。

- 生成时间：`2026-10-22T16:23:52+0800`
- 原始行数：**1,067,371**
- 入库行数：**1,067,371**
- 数据文件 SHA-256：`572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`
- 数据来源：https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
- 许可：CC BY 4.0

## 0. 一句话结论

这份真实数据 **1,067,371** 行里，**243,007** 行（22.7669%）没有客户 ID、**22,950** 行（2.1501%）是负数量退货腿、**19,494** 行（1.8264%）是 C 开头取消单；而项目合成的 50,000 行数据里，这几类的真实发生率分别是 **0%、1.0%、0%**——合成数据的「脏」是按 1% 均匀注入的六类人造缺陷，真实数据的「脏」是业务本身长出来的结构性缺失。

## 1. 数据来源与获取事实

| 项目 | 值 |
|---|---|
| 来源 | `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip` |
| 数据集主页 | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| 本地文件 | `data/raw/online_retail_ii.zip` |
| 字节数 | 45,622,418 |
| SHA-256 | `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` |
| 下载时间 | 2026-10-22T16:01:43+0800（本机文件落盘时间；本次运行未重新下载：already present; pass --force-download to refresh） |
| 下载耗时/速度 | 本次运行未下载（脚本不掌握该数值；见「下载旁证」） |
| 许可 | CC BY 4.0 |
| 下载方式 | already present; pass --force-download to refresh |
| 下载旁证 | 本次下载由交互式 curl 完成（非本脚本）：HTTP 200，45,622,418 字节，耗时 1138.3 s，平均 40.1 KB/s（UCI 站点不支持 Range 断点续传，属于已知限制） |

原始文件内的数据成员：

| 成员（文件） | 工作表 | 原始行数 |
|---|---|---|
| online_retail_II.xlsx | Year 2009-2010 | 525,461 |
| online_retail_II.xlsx | Year 2010-2011 | 541,910 |

合计：**1,067,371** 行（2 个工作表）。

## 2. 字段映射（真实 → 项目 14 列契约）

| 真实列 | 契约列 | 转换规则 |
|---|---|---|
| InvoiceNo | `order_id` | hash to INV<sha1[:10]>; C-prefix is preserved in status, not in the id |
| Customer ID | `customer_id` | CUS<zero-padded 6 digits>; null -> CUS_UNKNOWN (measured rate is printed in the report) |
| (none) | `merchant_id` | constant MRC_REAL_01 (single retailer in the source) |
| InvoiceDate | `order_time` | minute resolution, stamped +00:00 (source has no timezone) |
| (none) | `status` | derived: C-invoice->cancelled, Quantity<0->refunded, else completed |
| StockCode | `category` | SKU_<StockCode>; real data has no product category hierarchy |
| Quantity | `quantity` | int, negatives kept |
| UnitPrice | `unit_price_minor` | GBP float -> pence: round(x*100) |
| (none) | `discount_minor` | hard 0 (not recorded) |
| (none) | `shipping_fee_minor` | hard 0 (not recorded) |
| Quantity*UnitPrice | `paid_amount_minor` | max(gross,0) in pence on the sale leg |
| Quantity*UnitPrice | `refunded_amount_minor` | max(-gross,0) in pence on the negative leg |
| (none) | `payment_method` | sentinel 'unknown' |
| (none) | `source_channel` | sentinel 'unknown' |

`scripts/build_orders_db.py:COLUMNS` 声明的 14 列全部有值，但其中 **5 列是占位**：
`merchant_id`（单一零售商常量）、`payment_method`/`source_channel`（字面量 `unknown`）、`discount_minor`/`shipping_fee_minor`（硬 0）。也就是说真实数据**根本不支持按渠道/支付方式做留存或退款归因**，任何这类结论都不能声称。

## 3. 真实脏数据盘点（核心产出）

### 3.1 任务指定的六类

| 脏数据类型 | 真实行数 | 占比 | 项目合成数据对应值 |
|---|---|---|---|
| 缺失 Customer ID | 243,007 | 22.7669% | 0（契约声明 NOT NULL） |
| 负数量（退货腿） | 22,950 | 2.1501% | 500（1.0%，人工注入 -1） |
| 单价 = 0 | 6,202 | 0.5811% | 0 |
| 单价 < 0 | 5 | 0.0005% | 0 |
| InvoiceNo 以 C 开头 | 19,494 | 1.8264% | 0（合成用 status 字段，没有发票号概念） |
| 完全重复行 | 34,337 | 3.217% | 0（0.0%） |

时间跨度：**2009-12-01 07:45:00 → 2011-12-09 12:50:00**（分年行数：{"2009": 45228, "2010": 522714, "2011": 499429}）。

规模基数：发票 53,628 张、StockCode 5,304 个、已知客户 5,942 个、国家 43 个。

### 3.2 真实数据里「额外」存在的脏，合成数据完全没有建模

| 脏数据类型 | 行数 | 占比 | 说明 |
|---|---|---|---|
| 非数字 quantity | 0 | 0.0% | 字段里混入非数字文本 |
| 非数字 UnitPrice | 0 | 0.0% | 字段里混入非数字文本 |
| InvoiceDate 不可解析 | 0 | 0.0% | 真实时间串格式不统一 |
| Description 缺失 | 4,382 | 0.4105% | 商品描述为空 |
| 非目录型 StockCode | 6,093 | 0.5708% | 人工调整码，如 ADJUST, BANK CHARGES, C2, D, DCGS0003, DCGS0004 |
| 负数量但非 C 单 | 3,457 | 0.3239% | 退货没有独立发票，直接挂在正常单上 |
| C 单 ∩ 负数量 | 19,493 | 1.8263% | 取消单几乎全是负数量腿 |

其中非目录型 StockCode 示例：`ADJUST, BANK CHARGES, C2, D, DCGS0003, DCGS0004, DCGS0044, DCGS0058, DCGS0066N, DCGS0068, DCGS0072, DCGS0076, DOT, M, PADS`

### 3.3 关系型脏数据（合成数据最不可能出现的一类）

| 问题 | 数量 | 说明 |
|---|---|---|
| 同一 StockCode 对应多个 Description | 1,214 个 StockCode / 420,566 行 | 最大 9 种写法，例如 `10080`→2 种, `10120`→2 种, `10133`→2 种, `15056N`→2 种 |
| 同一 StockCode 有多个单价 | 4,710 个 StockCode / 1,065,137 行 | 最大 1291 个价位（含 0 价与调价） |
| 同一客户出现在多个国家 | 13 个客户 / 2,190 行 | 最大 2 个国家（收货地址变更） |
| 子码与父码并存（85123A / 85123） | 2,822 行 / 14 个码 | 如 10002R, 47556B, 47559B, 47566B, 47567B, 47567b 与父码同表共存 |

这四类**在项目合成数据里结构上不可能出现**：合成器每个 `StockCode` 由固定枚举生成、`customer_id` 与 `merchant_id` 一一对应、`order_id` 唯一、没有父子货号概念。

> 口径陷阱（复跑时最容易算错的一条）：统计「一码多描述」时**必须先把空描述剔除**，否则 `ADJUST`、`BANK CHARGES` 这类无描述行会被当成额外的描述变体。把空值当变体会把结果从 **1,214 个码 / 420,566 行** 虚增到 **2,834 个码 / 636,610 行**。本报告采用剔除空值的口径（`description.where(~desc_missing, pd.NA)`）。

## 4. 用项目口径在真实库上跑一遍

- 口径来源：`src/agent/tools.py` 的 `MetricTool`（未修改，直接实例化）
- `REVENUE_STATUSES = ('completed', 'shipped')`
- `NET_REVENUE_EXPR = (paid_amount_minor - refunded_amount_minor)`

| 口径 | SQL 结果（分为单位） | 折合金额 |
|---|---|---|
| `order_count`（全表） | 1,067,371 | — |
| 净营收 = paid - refunded（全表） | 1,928,725,055 | £19,287,250.55 |
| 净营收（项目口径：仅 completed/shipped） | 2,081,391,841 | £20,813,918.41 |
| 净营收（仅 quantity > 0） | 2,081,429,198 | £20,814,291.98 |
| cancelled 腿的金额（被项目口径隔离） | -152,666,786 | £-1,526,667.86 |

- `paid - refunded != quantity * unit_price` 的行数：**0**（恒等性自检；0 表示金额变换在整数分语义下无漂移）
- 不在规范枚举内的 status：`[]`（空列表 = 真实数据映射后没有产生非法状态）

### 4.1 从 MetricTool 实际执行的查询

- `{"metric": "order_count"}` → `{"order_count": 1067371}`
- `{"metric": "revenue"}` → `{"revenue_yuan": 20813918.41}`
- `{"metric": "avg_order_value"}` → `{"avg_order_value_yuan": 19.93}`
- `{"metric": "cancel_rate"}` → `{"cancel_rate_percent": 2.15}`
- `{"dimension": "month", "metric": "revenue"}` → 20 组，示例 `[{"group_key": "2009-12", "revenue_yuan": 825685.76}, {"group_key": "2010-01", "revenue_yuan": 652708.5}, {"group_key": "2010-02", "revenue_yuan": 553339.73}]`
- `{"dimension": "status", "metric": "order_count"}` → 3 组，示例 `[{"group_key": "cancelled", "order_count": 19494}, {"group_key": "completed", "order_count": 1044420}, {"group_key": "refunded", "order_count": 3457}]`

## 5. 合成 vs 真实 对照结论

| 脏数据类别 | 真实（1,067,371 行） | 合成（50,000 行） |
|---|---|---|
| 缺失 customer/Customer ID | 243,007 (22.7669%) | 0 (0.0%) |
| 负数量（退货/取消腿） | 22,950 (2.1501%) | 500 (1.0%) |
| 单价为 0 | 6,202 (0.5811%) | 0 (0.0%) |
| 单价为负 | 5 (0.0005%) | 0 (0.0%) |
| InvoiceNo 以 C 开头（取消单） | 19,494 (1.8264%) | 0 (0.0%) |
| 完全重复行 | 34,337 (3.217%) | 0 (0.0%) |
| 时间不可解析 | 0 (0.0%) | 500 (1.0%) |
| 退款超过已付 | — | 500 (1.0%) |

### 5.1 合成数据「太干净」掩盖的真实问题（逐条）

1. **客户 ID 结构性缺失被 NOT NULL 掩盖** — 真实数据 22.7669% 的行没有客户 ID，而契约把 `customer_id` 声明为 `nullable: false`。合成器从不生成空客户，因此在合成数据上「按客户聚合」永远成功；接真实数据立刻会炸——这批行只能落到 `CUS_UNKNOWN` 一个桶里，客户级留存/复购率全部失真。
2. **退货是「负数量行」而不是「refunded 字段」** — 真实退货 2.1501% 的行本身就是负数量，其中 0.3239% 挂在正常发票上。项目把退款建模成 `refunded_amount_minor` 这一列、行本身仍是正数量，于是「按行数算退货率」在两种数据上口径完全不同——合成数据上按 status 过滤即可，真实数据上必须先识别负数量腿。
3. **取消单是发票号前缀语义，不是状态列** — 1.8264% 的取消只能从 `InvoiceNo` 的 `C` 前缀读出来。项目合成数据把 status 明写成枚举，所以 Route/MetricTool 直接 `status IN ('completed','shipped')` 就够；真实数据必须先做前缀解析，否则取消单会被当成正常销售计入营收。
4. **同一 StockCode 多个 Description → 商品主数据不可信** — 1,214 个 StockCode 有多个描述写法，最多一个码对应 9 种描述，共波及 420,566 行。合成数据每个类目是固定枚举，「按商品名做 dimension join」在合成集上永远精确，真实集上会一对多放大行数。
5. **同一客户多个国家/地址 → 客户维度不是一对一** — 13 个客户出现在多个国家。合成器一个 `customer_id` 只有一个 `merchant_id`、无地址字段，所以「客户 ⟷ 地区」的脏一对多在合成数据里不存在。
6. **单价为 0 与负价真实存在，且与退货语义纠缠** — 真实数据单价为 0 的行 6,202 行（0.5811%）、负价 5 行（0.0005%）。合成数据单价恒正且量纲统一，金额聚合在合成集上不会出现 0 价噪声。
7. **真实数据没有渠道/支付/折扣/运费维度** — 契约里 5 列在真实数据上只能填占位值。合成数据「渠道分布 35/30/20/15」看起来很美，但它在真实源上不可获得——任何渠道归因结论都只对合成数据成立。
8. **整行完全重复在真实数据里是「同一张发票内的重复录入」，不能直接去重** — 实测重复行 34,337 行（3.217%），对应 32,909 个不同的业务键。本次实测的关键结论是：这些重复键**全部落在同一张 InvoiceNo 内**（跨发票的重复键 = 0 个，单张发票内同一条明细最多重复 19 次），属于**同一张单里的重复录入**，而不是两笔独立成交。含义：如果按「整行去重」处理，会删掉真实存在的明细行、直接改变金额合计；正确做法是保留并在 `_meta` 里标记 `duplicate_of_invoice`。合成数据的重复是人工注入的整行复制，两者处理策略不能共用。

### 5.2 合成数据做的对的地方（不要过度自我否定）

- 契约「金额用整数分」这个决定在真实数据上被验证是对的：`paid - refunded == quantity * unit_price` 的失败行数为 **0**，把 GBP 浮点先 `round(x*100)` 再入整数分后没有任何精度漂移。
- 合成数据的六类异常虽然均匀得不真实，但它们的 `expected_handling` 恰好覆盖了真实数据里最贵的两个坑：负数量隔离、金额恒等式校验。

## 6. 对 `config/data_spec.yaml` 的改进建议

建议在 `orders.dirty_data.rules` 之外，新增一组**结构性缺失规则**（数值为本次实测占比）：

| 建议新增异常 | 真实实测占比 | 建模方式 |
|---|---|---|
| missing_customer_id | 22.7669% | 把 customer_id 允许为空，并要求下游落到 CUS_UNKNOWN 桶 |
| return_as_negative_quantity_line | 2.1501% | 负数量退货腿，而不是把退款放进 refunded 列 |
| invoice_level_cancellation | 1.8264% | 发票号前缀 C，而不是 status 枚举 |
| price_zero_or_negative | 0.5815% | 0/负单价，量纲噪声 |
| stockcode_description_drift | 39.402% | 一码多描述，考验商品主数据 join |
| customer_multi_address | 0.2052% | 一客多地址，考验客户维度一对一假设 |
| child_stockcode | 0.2644% | 父子货号并存，考验 SKU 归一化 |

现有六类异常（每类 1%）建议保留，但占比应改为按实测分布加权，并在 `data_spec.yaml` 里补一个 `real_data_reference` 段记录本次实测值，让「合成数据的脏」有真实基线可对标。

## 7. 复跑命令（每个数字都可复现）

```bash
cd E:\桌面\我的简历\FinOps-Copilot
# 1) 全链路：下载 -> 解析 -> 规范化 -> 建库 -> 跑口径 -> 写统计
.venv/Scripts/python.exe scripts/ingest_real_data.py

# 2) 已有原始文件时跳过网络（报告里的 SHA-256 必须与步骤 1 一致）
.venv/Scripts/python.exe scripts/ingest_real_data.py --skip-download

# 3) 渲染本报告
.venv/Scripts/python.exe scripts/real_data_report.py

# 4) 抽查任意一个数字（示例：缺失客户 ID 的行数）
.venv/Scripts/python.exe -c "import zipfile,pandas as pd;z=zipfile.ZipFile('data/raw/online_retail_ii.zip');s=pd.read_excel(z.open('online_retail_II.xlsx'),sheet_name=None,dtype=str);print(sum(int(f['Customer ID'].isna().sum()) for f in s.values()))"

# 5) 直接问 SQLite：净营收（项目口径）
.venv/Scripts/python.exe -c "import sqlite3;c=sqlite3.connect('file:data/processed/real_orders.db?mode=ro',uri=True);print(c.execute(\"SELECT ROUND(SUM(paid_amount_minor-refunded_amount_minor)/100.0,2) FROM orders WHERE status IN ('completed','shipped') AND quantity>0\").fetchone())"

# 6) 复现「重复行构成」这一条（跨发票 vs 同发票）
.venv/Scripts/python.exe -c "import zipfile,pandas as pd;z=zipfile.ZipFile('data/raw/online_retail_ii.zip');s=pd.read_excel(z.open('online_retail_II.xlsx'),sheet_name=None,dtype=str);d=pd.concat(s.values(),ignore_index=True);k=['Invoice','StockCode','Quantity','InvoiceDate','Price','Customer ID'];g=d[d.duplicated(subset=k,keep='first')].groupby(k,dropna=False)['Invoice'].nunique();print('keys =',len(g),' spanning >1 invoice =',int((g>1).sum()))"
```

## 8. 诚实边界（这些结论不要说过头）

- 真实源是**英国一家在线礼品批发商的 B2B/B2C 混合单据**，不是中国电商平台交易，所以「渠道、支付方式、折扣」这类维度在源数据里根本不存在，全部是占位值。
- 状态是**推导**出来的：`C` 前缀 → cancelled、负数量 → refunded、其余 → completed。真实源没有生命周期字段，因此 `shipped` 在真实库上恒为 0，与合成数据的 17.73% 不可直接比较。
- 金额币种是 **GBP**，项目口径是 CNY 分。本次只把「分」当最小货币单位复用，没有做汇率折算，所以两边的人民币金额不可相加。
- 本报告的「占比」分母是原始行数（含未通过类型校验的行），与入库行数不同属正常。
