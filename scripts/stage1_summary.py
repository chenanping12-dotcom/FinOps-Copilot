"""Verify the frozen Stage 1 artifacts and print a data statistics summary.

Stage 1 outputs are treated as immutable evidence: this script never regenerates
or overwrites them. It (1) re-computes every artifact SHA-256 recorded in the
manifest, (2) recomputes the headline counts from the raw files, and (3) prints a
summary so the numbers quoted downstream can be traced to the bytes on disk.

Usage:
    python scripts/stage1_summary.py --config config/data_spec.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# The console on this machine defaults to GBK; force UTF-8 so the Chinese
# summary is readable and never raises UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage1_common import (  # noqa: E402
    jsonl_rows,
    load_spec,
    resolve_path,
    sha256_file,
)


def verify_artifacts(manifest: dict, root: Path) -> tuple[int, int, list[str]]:
    """Re-hash every manifest artifact; return (ok, checked, problems)."""
    ok = 0
    problems: list[str] = []
    artifacts = manifest.get("artifacts", [])
    for artifact in artifacts:
        path = resolve_path(root, artifact["path"])
        if not path.is_file():
            problems.append(f"MISSING {artifact['path']}")
            continue
        actual = sha256_file(path)
        if actual != artifact.get("sha256"):
            problems.append(f"HASH MISMATCH {artifact['path']}")
            continue
        ok += 1
    return ok, len(artifacts), problems


def collect_stats(root: Path, spec: dict) -> dict:
    orders_path = root / spec["orders"]["output"]["directory"] / spec["orders"]["output"]["filename"]
    labels_path = (
        root
        / spec["orders"]["output"]["directory"]
        / spec["orders"]["output"]["anomaly_labels_filename"]
    )
    chunks_path = (
        root
        / spec["rule_documents"]["output"]["directory"]
        / spec["rule_documents"]["output"]["chunks_filename"]
    )
    docs_dir = (
        root
        / spec["rule_documents"]["output"]["directory"]
        / spec["rule_documents"]["output"]["documents_subdirectory"]
    )
    eval_path = (
        root / spec["evaluation"]["output"]["directory"] / spec["evaluation"]["output"]["filename"]
    )

    with orders_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    status_counts = Counter(row["status"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    channel_counts = Counter(row["source_channel"] for row in rows)
    payment_counts = Counter(row["payment_method"] for row in rows)

    anomalies = jsonl_rows(labels_path)
    anomaly_types = Counter(a.get("anomaly_id", "?") for a in anomalies)

    chunks = jsonl_rows(chunks_path)
    chunk_versions = Counter(c.get("version", "?") for c in chunks)
    chunk_doc_ids = {c.get("doc_id") for c in chunks}

    questions = jsonl_rows(eval_path)

    return {
        "orders_path": orders_path,
        "rows": rows,
        "status_counts": status_counts,
        "category_counts": category_counts,
        "channel_counts": channel_counts,
        "payment_counts": payment_counts,
        "anomalies": anomalies,
        "anomaly_types": anomaly_types,
        "chunks": chunks,
        "chunk_versions": chunk_versions,
        "chunk_doc_ids": chunk_doc_ids,
        "questions": questions,
        "document_files": sorted(p.name for p in docs_dir.glob("*.md")),
    }


def compute_revenue(rows: list[dict], spec: dict) -> dict:
    """Recompute the revenue KPI straight from the CSV, applying the stated caliber."""
    semantics = spec["orders"]["business_semantics"]
    statuses = set(semantics["revenue_statuses"])
    gross = 0
    net = 0
    counted = 0
    for row in rows:
        if row["status"] not in statuses:
            continue
        try:
            paid = int(row["paid_amount_minor"])
            refunded = int(row["refunded_amount_minor"])
            qty = int(row["quantity"])
        except (TypeError, ValueError):
            continue  # malformed rows are excluded from the caliber
        if qty <= 0:
            continue
        counted += 1
        gross += paid
        net += paid - refunded
    return {
        "counted_orders": counted,
        "gross_yuan": gross / 100,
        "net_yuan": net / 100,
        "refund_yuan": (gross - net) / 100,
    }


def print_summary(spec: dict, stats: dict, revenue: dict, ok: int, total: int, problems: list[str]):
    line = "=" * 68
    print(line)
    print("FinOps-Copilot  Stage 1  数据统计摘要 (synthetic, frozen)")
    print(line)
    print(f"random_seed              : {spec['project']['random_seed']}")
    print(f"synthetic_only           : {spec['project']['synthetic_only']}")
    print()

    print("[订单]")
    print(f"  行数 (含表头)          : {len(stats['rows']) + 1}")
    print(f"  订单数                 : {len(stats['rows']):,}")
    print(f"  金额口径               : {spec['orders']['business_semantics']['revenue_statuses']}")
    print(f"  净额公式               : {spec['orders']['business_semantics']['net_revenue_formula']}")
    print("  状态分布:")
    for key, val in stats["status_counts"].most_common():
        print(f"    {key:<22}: {val:>6,}  ({val / len(stats['rows']):6.2%})")
    print("  品类分布:")
    for key, val in stats["category_counts"].most_common():
        print(f"    {key:<22}: {val:>6,}  ({val / len(stats['rows']):6.2%})")
    print("  渠道分布:")
    for key, val in stats["channel_counts"].most_common():
        print(f"    {key:<22}: {val:>6,}  ({val / len(stats['rows']):6.2%})")
    print()

    print("[脏数据 / 异常真值]")
    dirty = len(stats["anomalies"])
    print(f"  异常行总数             : {dirty:,}  ({dirty / len(stats['rows']):.2%})")
    for key, val in sorted(stats["anomaly_types"].items()):
        print(f"    {key:<22}: {val:>6,}")
    print()

    print("[规则文档库]")
    print(f"  文档数                 : {len(stats['document_files']):,}")
    print(f"  切片数                 : {len(stats['chunks']):,}")
    print(f"  覆盖 doc_id 数          : {len(stats['chunk_doc_ids']):,}")
    print("  版本分布:")
    for key, val in sorted(stats["chunk_versions"].items()):
        print(f"    {key:<22}: {val:>6,}")
    sample = stats["chunks"][0]
    print(f"  切片溯源字段           : doc_id={sample.get('doc_id')} page={sample.get('page')} "
          f"line={sample.get('line_start')}-{sample.get('line_end')}")
    print()

    print("[评测集]")
    questions = stats["questions"]
    split_counts = Counter(q["split"] for q in questions)
    trap_counts = Counter(q["trap_type"] for q in questions)
    print(f"  QA 对数                : {len(questions):,}")
    print("  分集:")
    for key, val in sorted(split_counts.items()):
        print(f"    {key:<22}: {val:>6,}")
    print("  陷阱类型:")
    for key, val in sorted(trap_counts.items()):
        print(f"    {key:<22}: {val:>6,}")
    print(f"  应拒答(must_abstain)   : {sum(1 for q in questions if q['must_abstain']):,}")
    print()

    print("[营收口径复算 — 直接读 CSV]")
    print(f"  纳入订单数             : {revenue['counted_orders']:,}")
    print(f"  毛额 (paid)            : {revenue['gross_yuan']:,.2f} 元")
    print(f"  退款                   : {revenue['refund_yuan']:,.2f} 元")
    print(f"  净营收 (net)           : {revenue['net_yuan']:,.2f} 元")
    print()

    print("[产物完整性校验]")
    print(f"  SHA-256 校验           : {ok}/{total} 通过")
    if problems:
        for problem in problems:
            print(f"    !! {problem}")
    else:
        print("    所有 manifest 产物字节级一致")
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/data_spec.yaml")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    spec, root, _ = load_spec(Path(args.config))
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else resolve_path(root, spec["artifacts"]["manifest_path"])
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ok, total, problems = verify_artifacts(manifest, root)
    stats = collect_stats(root, spec)
    revenue = compute_revenue(stats["rows"], spec)
    print_summary(spec, stats, revenue, ok, total, problems)

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())