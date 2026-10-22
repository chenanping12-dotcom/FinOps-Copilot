"""Generate reproducible synthetic orders and versioned policy documents."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.stage1_common import artifact_entry, load_spec, resolve_path, validate_weights, write_jsonl
except ModuleNotFoundError:  # Direct execution places scripts/, not the project root, on sys.path.
    from stage1_common import artifact_entry, load_spec, resolve_path, validate_weights, write_jsonl


FAMILY_TITLES = {
    "order_lifecycle": "订单生命周期管理",
    "revenue_recognition": "营收确认",
    "refund_accounting": "退款核算",
    "discount_allocation": "折扣分摊",
    "shipping_fee_settlement": "运费结算",
    "merchant_settlement": "商家结算",
    "channel_attribution": "渠道归因",
    "payment_capture": "支付入账",
    "fulfillment_confirmation": "履约确认",
    "cancellation_handling": "取消订单处理",
    "duplicate_order_handling": "重复订单处理",
    "reporting_timezone": "报表时区口径",
    "category_eligibility": "类目适用规则",
    "return_window": "退货受理时限",
    "partial_refund": "部分退款核算",
    "chargeback_handling": "拒付处理",
    "merchant_fee": "商家费用核算",
    "promotion_stacking": "促销叠加规则",
    "split_shipment": "拆单履约处理",
    "reconciliation_tolerance": "对账差异容忍度",
    "order_date_basis": "订单日期口径",
    "tax_documentation": "税务凭证管理",
    "coupon_eligibility": "优惠券适用规则",
    "inventory_reservation": "库存预占处理",
    "data_quality_review": "数据质量复核",
}

SCOPE_TITLES = {
    "global": "全品类",
    "electronics": "数码电子",
    "home_living": "家居生活",
    "apparel": "服饰鞋包",
    "beauty_personal_care": "美妆个护",
}

ORDER_FIELDS = [
    "order_id",
    "customer_id",
    "merchant_id",
    "order_time",
    "status",
    "category",
    "quantity",
    "unit_price_minor",
    "discount_minor",
    "shipping_fee_minor",
    "paid_amount_minor",
    "refunded_amount_minor",
    "payment_method",
    "source_channel",
]


def _weighted_id(rng: random.Random, entries: list[dict[str, Any]]) -> str:
    return rng.choices(
        [entry["id"] for entry in entries],
        weights=[float(entry["share"]) for entry in entries],
        k=1,
    )[0]


def _date_pools(date_spec: dict[str, Any]) -> tuple[list[date], list[date]]:
    start = date.fromisoformat(date_spec["start"])
    end = date.fromisoformat(date_spec["end"])
    if not date_spec.get("end_inclusive", True):
        end -= timedelta(days=1)
    if end < start:
        raise ValueError("orders.date_range.end must not precede start")

    all_days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    campaign_days: set[date] = set()
    radius = int(date_spec["campaign_window_radius_days"])
    for campaign_date in date_spec.get("campaign_dates", []):
        center = date.fromisoformat(campaign_date)
        for delta in range(-radius, radius + 1):
            day = center + timedelta(days=delta)
            if start <= day <= end:
                campaign_days.add(day)
    baseline_days = [day for day in all_days if day not in campaign_days]
    if not baseline_days:
        raise ValueError("The campaign windows must not cover the entire date range")
    return baseline_days, sorted(campaign_days)


def _order_timestamp(rng: random.Random, date_spec: dict[str, Any], pools: tuple[list[date], list[date]]) -> tuple[str, bool]:
    baseline_days, campaign_days = pools
    is_campaign = bool(campaign_days) and rng.random() < float(date_spec["campaign_probability"])
    day = rng.choice(campaign_days if is_campaign else baseline_days)
    naive_time = datetime.combine(day, time()) + timedelta(seconds=rng.randrange(24 * 60 * 60))
    local_time = naive_time.replace(tzinfo=timezone(timedelta(hours=8)))
    return local_time.isoformat(timespec="seconds"), is_campaign


def _normal_orders(spec: dict[str, Any]) -> list[dict[str, Any]]:
    order_spec = spec["orders"]
    dimensions = order_spec["dimensions"]
    generation = order_spec["generation_defaults"]
    rng = random.Random(int(spec["project"]["random_seed"]))
    date_spec = order_spec["date_range"]
    date_pools = _date_pools(date_spec)

    quantity_min, quantity_max = map(int, generation["quantity_range_inclusive"])
    merchants = [f"{dimensions['merchants']['id_prefix']}{i:04d}" for i in range(1, int(dimensions["merchants"]["count"]) + 1)]
    customer_count = int(dimensions["customer_count"])
    methods = dimensions["payment_methods"]
    statuses = dimensions["statuses"]
    categories = dimensions["categories"]
    channels = dimensions["source_channels"]
    price_ranges = generation["unit_price_minor_by_category"]

    rows: list[dict[str, Any]] = []
    for sequence in range(1, int(order_spec["target_rows"]) + 1):
        category = _weighted_id(rng, categories)
        status = _weighted_id(rng, statuses)
        source_channel = _weighted_id(rng, channels)
        timestamp, is_campaign = _order_timestamp(rng, date_spec, date_pools)
        quantity = rng.randint(quantity_min, quantity_max)
        price_low, price_high = map(int, price_ranges[category])
        unit_price = rng.randint(price_low, price_high)
        gross_amount = quantity * unit_price

        discount_range = (
            generation["campaign_discount_rate_range"]
            if is_campaign
            else generation["normal_discount_rate_range"]
        )
        discount_rate = rng.uniform(float(discount_range[0]), float(discount_range[1]))
        discount = int(gross_amount * discount_rate)
        shipping_fee = 0
        if rng.random() < float(generation["shipping_fee_probability"]):
            fee_low, fee_high = map(int, generation["shipping_fee_minor_range"])
            shipping_fee = rng.randint(fee_low, fee_high)
        paid_amount = gross_amount - discount + shipping_fee

        refunded_amount = 0
        if status == "refunded":
            refund_low, refund_high = map(float, generation["refunded_status_refund_rate_range"])
            refunded_amount = int(paid_amount * rng.uniform(refund_low, refund_high))
        elif status in {"completed", "shipped"} and rng.random() < float(
            generation["completed_or_shipped_partial_refund_probability"]
        ):
            refund_low, refund_high = map(float, generation["partial_refund_rate_range"])
            refunded_amount = max(1, int(paid_amount * rng.uniform(refund_low, refund_high)))

        rows.append(
            {
                "order_id": f"ORD-{sequence:08d}",
                "customer_id": f"CUS{rng.randint(1, customer_count):06d}",
                "merchant_id": rng.choice(merchants),
                "order_time": timestamp,
                "status": status,
                "category": category,
                "quantity": quantity,
                "unit_price_minor": unit_price,
                "discount_minor": discount,
                "shipping_fee_minor": shipping_fee,
                "paid_amount_minor": paid_amount,
                "refunded_amount_minor": refunded_amount,
                "payment_method": rng.choice(methods),
                "source_channel": source_channel,
            }
        )
    return rows


def _inject_anomalies(rows: list[dict[str, Any]], dirty_spec: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    target_rate = float(dirty_spec["target_row_rate"])
    target_dirty = int(round(len(rows) * target_rate))
    if not math.isclose(len(rows) * target_rate, target_dirty, abs_tol=1e-9):
        raise ValueError("target_rows * dirty_data.target_row_rate must be an integer")

    rules = dirty_spec["rules"]
    rule_counts: list[int] = []
    for rule in rules:
        exact_count = len(rows) * float(rule["rate_of_all_rows"])
        count = int(round(exact_count))
        if not math.isclose(exact_count, count, abs_tol=1e-9):
            raise ValueError(f"Anomaly rate for {rule['id']} does not produce an integer row count")
        rule_counts.append(count)
    if sum(rule_counts) != target_dirty:
        raise ValueError("Per-rule anomaly rates must sum to target_row_rate exactly")

    dirty_indices = rng.sample(range(len(rows)), target_dirty)
    assignments = [rule for rule, count in zip(rules, rule_counts, strict=True) for _ in range(count)]
    rng.shuffle(assignments)
    labels: list[dict[str, Any]] = []
    for row_index, rule in zip(dirty_indices, assignments, strict=True):
        row = rows[row_index]
        anomaly_id = rule["id"]
        if anomaly_id == "negative_quantity":
            row["quantity"] = -1
        elif anomaly_id == "unsupported_status":
            row["status"] = "legacy_in_transit"
        elif anomaly_id == "refund_exceeds_paid":
            row["refunded_amount_minor"] = row["paid_amount_minor"] + 1
        elif anomaly_id == "paid_amount_mismatch":
            row["paid_amount_minor"] += 137
        elif anomaly_id == "malformed_order_time":
            row["order_time"] = "not-a-timestamp"
        elif anomaly_id == "unknown_channel":
            row["source_channel"] = "partner_feed_x"
        else:
            raise ValueError(f"No anomaly implementation exists for {anomaly_id}")
        labels.append(
            {
                "order_id": row["order_id"],
                "row_index": row_index + 1,
                "anomaly_id": anomaly_id,
                "affected_fields": [rule["field"]],
                "expected_handling": rule["expected_handling"],
            }
        )
    return sorted(labels, key=lambda label: label["row_index"])


def _category_label(scope: str) -> str:
    return SCOPE_TITLES.get(scope, scope.replace("_", " "))


def _refund_window_days(scope: str, spec: dict[str, Any]) -> int:
    windows = spec["rule_documents"]["synthetic_parameters"]["refund_window_days_by_scope"]
    return int(windows.get(scope, windows["global"]))


def _clauses(topic: dict[str, Any], version: str, spec: dict[str, Any]) -> list[dict[str, str]]:
    scope_text = _category_label(topic["scope"])
    family_text = FAMILY_TITLES.get(topic["family"], topic["family"])
    scope_context = f"本条适用于{scope_text}范围内的{family_text}业务。"
    revenue_statuses = "completed" if version == "V1" else "completed、shipped"
    days = _refund_window_days(topic["scope"], spec)
    tolerance = int(spec["rule_documents"]["synthetic_parameters"]["reconciliation_tolerance_minor"])
    return [
        {
            "rule_type": "revenue_status",
            "title": "营收纳入状态",
            "text": f"{scope_context}当前版本仅将状态为 {revenue_statuses} 的订单纳入营收统计；其他状态不因已创建或已支付而自动计入。",
        },
        {
            "rule_type": "net_revenue",
            "title": "净营收计算",
            "text": f"{scope_context}单笔净额按 paid_amount_minor 减 refunded_amount_minor 计算；汇总时先按营收状态过滤，再对净额求和。金额字段以人民币分存储。",
        },
        {
            "rule_type": "paid_amount_formula",
            "title": "实付金额构成",
            "text": f"{scope_context}paid_amount_minor = quantity × unit_price_minor - discount_minor + shipping_fee_minor。平台补贴不属于买家实付，不得重复加进实付金额。",
        },
        {
            "rule_type": "refund_boundary",
            "title": "退款申请边界",
            "text": f"{scope_context}常规退货申请须在签收后 {days} 个自然日内提交，第 {days} 日当日仍在受理窗口内；超过该日须进入人工例外审批，不得按常规窗口自动受理。",
        },
        {
            "rule_type": "refund_exception",
            "title": "质量问题例外",
            "text": f"{scope_context}若有可核验的质量问题证据，常规退货时限可由售后复核后例外处理；仅有主观描述不足以自动覆盖常规时限。",
        },
        {
            "rule_type": "partial_refund",
            "title": "部分退款处理",
            "text": f"{scope_context}部分退款按实际退款金额冲减订单净额，不得把部分退款误标为全额退款；同一笔退款仅在退款成功后计入 refunded_amount_minor。",
        },
        {
            "rule_type": "promotion_stacking",
            "title": "优惠叠加顺序",
            "text": f"{scope_context}订单级优惠先于支付完成后计算；同一商品默认只接受一个互斥主优惠，只有活动规则明确允许时才可叠加。",
        },
        {
            "rule_type": "buyer_shipping_fee",
            "title": "买家实付运费",
            "text": f"{scope_context}买家实际支付的运费只计入 paid_amount_minor 一次；商品销售额报表可单列运费，但不得在已含运费的实付金额上再次累加。",
        },
        {
            "rule_type": "platform_subsidy_exclusion",
            "title": "平台运费补贴排除",
            "text": f"{scope_context}平台承担的运费补贴不是买家支付款，不计入商家销售净额，也不得写入 paid_amount_minor；报表中应与买家实付运费区分。",
        },
        {
            "rule_type": "merchant_shipping_income",
            "title": "商家收取的运费",
            "text": f"{scope_context}商家向买家实际收取的运费属于订单实付组成，退款后按净额口径保留；该规则仅适用于买家支付项，不适用于平台运费补贴。",
        },
        {
            "rule_type": "non_revenue_statuses",
            "title": "未完成及关闭订单",
            "text": f"{scope_context}pending、cancelled、refunded 不属于本规则的营收纳入状态；即使存在支付流水，也必须依照当前版本的营收状态条款过滤。",
        },
        {
            "rule_type": "duplicate_order",
            "title": "重复记录去重",
            "text": f"{scope_context}订单明细以 order_id 作为订单级主键；支付重试产生的多条流水应先按支付流水标识去重，再汇总到唯一订单。",
        },
        {
            "rule_type": "channel_attribution",
            "title": "渠道归因字段",
            "text": f"{scope_context}渠道分析使用订单创建时记录的 source_channel；未知渠道保留原始值并进入映射队列，不得静默并入 app 或 web。",
        },
        {
            "rule_type": "reporting_timezone",
            "title": "经营报表时区",
            "text": f"{scope_context}订单日期按 Asia/Shanghai 时区解释；跨日统计先转换到该时区，再按自然日聚合。",
        },
        {
            "rule_type": "refund_timing",
            "title": "退款入账时间",
            "text": f"{scope_context}退款按退款成功时间进入退款明细；订单净额报表仍按订单事实关联已成功退款，不以退款申请时间替代成功时间。",
        },
        {
            "rule_type": "reconciliation_tolerance",
            "title": "对账差异阈值",
            "text": f"{scope_context}订单与结算明细差异绝对值不超过 {tolerance} 分时标记为可自动核对；超过阈值须保留差异并转人工复核。",
        },
        {
            "rule_type": "amount_display",
            "title": "金额展示单位",
            "text": f"{scope_context}数据库和交换文件中的金额以整数分保存；面向用户展示时除以 100 转成人民币元，并固定保留两位小数。",
        },
        {
            "rule_type": "invalid_quantity",
            "title": "数量异常拦截",
            "text": f"{scope_context}quantity 必须为正整数；零值或负值记录应进入数据质量隔离区，不参与金额汇总，也不得在清洗时静默改成 1。",
        },
        {
            "rule_type": "version_authority",
            "title": "版本生效与追溯",
            "text": f"{scope_context}{version} 的生效日期为 {spec['rule_documents']['version_policy'][version.lower() + '_effective_from']}；历史查询按业务发生日选择当时有效版本，当前口径按最新已生效版本执行。",
        },
        {
            "rule_type": "evidence_precedence",
            "title": "证据与冲突处理",
            "text": f"{scope_context}同一主题出现多个版本时，先核对生效日期，再引用适用于查询日期的条款；无法找到有效条款时应明确说明证据不足。",
        },
    ]


def _build_documents(spec: dict[str, Any], staging_rule_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    doc_spec = spec["rule_documents"]
    families = doc_spec["document_families"]
    scopes = doc_spec["scopes"]
    versions = doc_spec["version_policy"]["versions"]
    expected_docs = len(families) * len(scopes) * len(versions)
    expected_chunks = expected_docs * int(doc_spec["chunks_per_document_target"])
    if expected_docs != int(doc_spec["document_count"]):
        raise ValueError(f"Expected {doc_spec['document_count']} documents but taxonomy creates {expected_docs}")
    if expected_chunks != int(doc_spec["target_chunks"]):
        raise ValueError(f"Expected {doc_spec['target_chunks']} chunks but configuration creates {expected_chunks}")
    if not all(family in FAMILY_TITLES for family in families):
        raise ValueError("Every document family must have a human-readable title mapping")
    if not all(scope in SCOPE_TITLES for scope in scopes):
        raise ValueError("Every document scope must have a human-readable title mapping")

    documents_dir = staging_rule_dir / doc_spec["output"]["documents_subdirectory"]
    documents_dir.mkdir(parents=True)
    chunks: list[dict[str, Any]] = []
    markdown_paths: list[Path] = []
    topics = [(family, scope) for family in families for scope in scopes]
    chunk_target = int(doc_spec["chunks_per_document_target"])

    for topic_number, (family, scope) in enumerate(topics, start=1):
        topic = {"family": family, "scope": scope, "policy_key": f"{family}:{scope}"}
        topic_label = f"{FAMILY_TITLES[family]} / {SCOPE_TITLES[scope]}"
        for version in versions:
            doc_id = f"POL-{topic_number:03d}-{version}"
            effective_from = doc_spec["version_policy"][version.lower() + "_effective_from"]
            lines = [
                f"# 电商经营规则：{topic_label}（{version}）",
                "",
                f"- 文档编号：{doc_id}",
                f"- 版本：{version}",
                f"- 生效日期：{effective_from}",
                f"- 适用范围：{SCOPE_TITLES[scope]}",
                "- 文档性质：合成示例规则，仅用于离线数据与检索评测。",
                "",
            ]
            clauses = _clauses(topic, version, spec)
            if len(clauses) != chunk_target:
                raise ValueError(f"{doc_id} produced {len(clauses)} clauses; expected {chunk_target}")
            for clause_number, clause in enumerate(clauses, start=1):
                lines.append(f"## 条款 {clause_number:02d}：{clause['title']}")
                line_start = len(lines) + 1
                chunk_text = f"{clause['title']}。{clause['text']}"
                lines.extend([chunk_text, ""])
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}-C{clause_number:02d}",
                        "doc_id": doc_id,
                        "version": version,
                        "effective_from": effective_from,
                        "page": (clause_number - 1) // 5 + 1,
                        "line_start": line_start,
                        "line_end": line_start,
                        "policy_key": topic["policy_key"],
                        "policy_family": family,
                        "scope": scope,
                        "rule_type": clause["rule_type"],
                        "text": chunk_text,
                    }
                )
            markdown_path = documents_dir / f"{doc_id}.md"
            markdown_path.write_text("\n".join(lines), encoding=doc_spec["output"]["encoding"])
            markdown_paths.append(markdown_path)

    if len(chunks) != expected_chunks:
        raise AssertionError(f"Generated {len(chunks)} chunks; expected {expected_chunks}")
    return chunks, markdown_paths


def _write_orders(path: Path, rows: list[dict[str, Any]], encoding: str) -> None:
    with path.open("w", encoding=encoding, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ORDER_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_payload(
    spec: dict[str, Any], raw_config: str, root: Path, artifact_paths: list[Path], summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "project": spec["project"]["name"],
        "stage": spec["project"]["stage"],
        "synthetic_only": bool(spec["project"]["synthetic_only"]),
        "random_seed": int(spec["project"]["random_seed"]),
        "summary": summary,
        "artifacts": [artifact_entry(path, root) for path in sorted(artifact_paths)],
        "config_snapshot": raw_config if spec["artifacts"].get("include_config_snapshot", True) else None,
    }


def generate_dataset(config_path: Path) -> dict[str, Any]:
    spec, root, raw_config = load_spec(Path(config_path))
    order_spec = spec["orders"]
    dimensions = order_spec["dimensions"]
    validate_weights(dimensions["categories"], "orders.dimensions.categories")
    validate_weights(dimensions["statuses"], "orders.dimensions.statuses")
    validate_weights(dimensions["source_channels"], "orders.dimensions.source_channels")

    order_dir = resolve_path(root, order_spec["output"]["directory"])
    order_csv = order_dir / order_spec["output"]["filename"]
    label_path = order_dir / order_spec["output"]["anomaly_labels_filename"]
    rule_dir = resolve_path(root, spec["rule_documents"]["output"]["directory"])
    chunks_path = rule_dir / spec["rule_documents"]["output"]["chunks_filename"]
    manifest_path = resolve_path(root, spec["artifacts"]["manifest_path"])
    if spec["artifacts"].get("fail_if_output_exists", True):
        existing = [path for path in (order_dir, rule_dir, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing Stage 1 output: " + ", ".join(map(str, existing)))

    target_rows = int(order_spec["target_rows"])
    rows = _normal_orders(spec)
    if len(rows) != target_rows:
        raise AssertionError(f"Generated {len(rows)} orders; expected {target_rows}")
    rng = random.Random(int(spec["project"]["random_seed"]) + 1)
    labels = _inject_anomalies(rows, order_spec["dirty_data"], rng)

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage1-", dir=root) as temp_name:
        staging_root = Path(temp_name)
        staging_order_dir = staging_root / "orders"
        staging_order_dir.mkdir()
        staged_order_csv = staging_order_dir / order_csv.name
        staged_label_path = staging_order_dir / label_path.name
        _write_orders(staged_order_csv, rows, order_spec["output"]["encoding"])
        write_jsonl(staged_label_path, labels)

        staging_rule_dir = staging_root / "rule_docs"
        staging_rule_dir.mkdir()
        chunks, markdown_paths = _build_documents(spec, staging_rule_dir)
        staged_chunks_path = staging_rule_dir / chunks_path.name
        write_jsonl(staged_chunks_path, chunks)

        if order_dir.exists() or rule_dir.exists() or manifest_path.exists():
            raise FileExistsError("Stage 1 output appeared during generation; refusing to overwrite it")
        order_dir.parent.mkdir(parents=True, exist_ok=True)
        rule_dir.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_order_dir), str(order_dir))
        shutil.move(str(staging_rule_dir), str(rule_dir))

    final_markdown_paths = [rule_dir / path.relative_to(staging_rule_dir) for path in markdown_paths]
    artifact_paths = [order_csv, label_path, chunks_path, *final_markdown_paths]
    anomaly_counts: dict[str, int] = {}
    for label in labels:
        anomaly_counts[label["anomaly_id"]] = anomaly_counts.get(label["anomaly_id"], 0) + 1
    summary = {
        "orders": len(rows),
        "dirty_orders": len(labels),
        "dirty_order_rate": len(labels) / len(rows),
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
        "rule_documents": len(final_markdown_paths),
        "rule_chunks": len(chunks),
    }
    manifest = _manifest_payload(spec, raw_config, root, artifact_paths, summary)
    try:
        from scripts.stage1_common import atomic_write_json
    except ModuleNotFoundError:
        from stage1_common import atomic_write_json

    atomic_write_json(manifest_path, manifest)
    return {**summary, "orders_path": str(order_csv), "chunks_path": str(chunks_path), "manifest_path": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/data_spec.yaml"))
    args = parser.parse_args()
    print(json.dumps(generate_dataset(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
