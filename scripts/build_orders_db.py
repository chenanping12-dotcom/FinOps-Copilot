"""Load the synthesized order CSV into a read-only SQLite database.

Stage 1 emits ``data/generated/orders/orders.csv``; the agent layer needs SQL. This
module is the single bridge between them, and it is deliberately strict:

* the schema is declared here rather than inferred, so a silent dtype change in the
  generator fails loudly instead of corrupting money calibers;
* money is stored as integer minor units (cents), never float, because
  ``SUM`` over 50k float rows accumulates visible drift;
* the build is verified against ``reports/stage1_manifest.json`` — if the CSV hash
  changed, the derived DB is stale and must be rebuilt.

Usage:
    python scripts/build_orders_db.py --config config/data_spec.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Declared column contract: name -> (sqlite type, not-null).
COLUMNS: dict[str, tuple[str, bool]] = {
    "order_id": ("TEXT", True),
    "customer_id": ("TEXT", True),
    "merchant_id": ("TEXT", True),
    "order_time": ("TEXT", True),
    "status": ("TEXT", True),
    "category": ("TEXT", True),
    "quantity": ("INTEGER", True),
    "unit_price_minor": ("INTEGER", True),
    "discount_minor": ("INTEGER", True),
    "shipping_fee_minor": ("INTEGER", True),
    "paid_amount_minor": ("INTEGER", True),
    "refunded_amount_minor": ("INTEGER", True),
    "payment_method": ("TEXT", True),
    "source_channel": ("TEXT", True),
}

INTEGER_COLUMNS = {
    name for name, (sqltype, _) in COLUMNS.items() if sqltype == "INTEGER"
}

INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
    "CREATE INDEX IF NOT EXISTS idx_orders_merchant ON orders(merchant_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_category ON orders(category)",
    "CREATE INDEX IF NOT EXISTS idx_orders_time ON orders(order_time)",
)


@dataclass
class LoadReport:
    csv_path: str
    db_path: str
    rows_loaded: int
    rows_rejected: int
    csv_sha256: str
    manifest_match: bool | None

    def as_dict(self) -> dict[str, object]:
        return {
            "csv_path": self.csv_path,
            "db_path": self.db_path,
            "rows_loaded": self.rows_loaded,
            "rows_rejected": self.rows_rejected,
            "csv_sha256": self.csv_sha256,
            "manifest_match": self.manifest_match,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_hash_for(manifest_path: Path, relative: str) -> str | None:
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in payload.get("artifacts", []):
        if artifact.get("path") == relative:
            return str(artifact["sha256"])
    return None


def coerce(row: dict[str, str], line_number: int) -> tuple[tuple | None, str | None]:
    """Convert a CSV row to typed values, rejecting dirty rows explicitly."""
    values: list[object] = []
    for name, (sqltype, _) in COLUMNS.items():
        raw = row.get(name)
        if raw is None or raw == "":
            return None, f"line {line_number}: missing column {name}"
        if sqltype == "INTEGER":
            try:
                values.append(int(raw))
            except ValueError:
                return None, f"line {line_number}: column {name} is not an integer: {raw!r}"
        else:
            values.append(raw)
    return tuple(values), None


def build(csv_path: Path, db_path: Path, manifest_path: Path, *, strict: bool = True) -> LoadReport:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing order CSV: {csv_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)
    try:
        column_sql = ", ".join(
            f"{name} {sqltype}{' NOT NULL' if notnull else ''}"
            for name, (sqltype, notnull) in COLUMNS.items()
        )
        connection.execute(f"CREATE TABLE orders ({column_sql})")

        placeholders = ", ".join("?" for _ in COLUMNS)
        insert_sql = (
            f"INSERT INTO orders ({', '.join(COLUMNS)}) VALUES ({placeholders})"
        )

        loaded = 0
        rejected = 0
        rejects: list[str] = []
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
            batch: list[tuple] = []
            for line_number, row in enumerate(reader, start=2):
                values, error = coerce(row, line_number)
                if values is None:
                    rejected += 1
                    if len(rejects) < 10:
                        rejects.append(error or "unknown")
                    continue
                batch.append(values)
                if len(batch) >= 5000:
                    connection.executemany(insert_sql, batch)
                    loaded += len(batch)
                    batch.clear()
            if batch:
                connection.executemany(insert_sql, batch)
                loaded += len(batch)

        for statement in INDEXES:
            connection.execute(statement)
        connection.commit()

        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"sqlite integrity_check failed: {integrity}")

        if strict and rejected:
            raise RuntimeError(
                f"rejected {rejected} dirty rows in strict mode: {rejects}"
            )
    finally:
        connection.close()

    csv_hash = sha256_file(csv_path)
    expected = manifest_hash_for(manifest_path, "data/generated/orders/orders.csv")
    return LoadReport(
        csv_path=str(csv_path.relative_to(PROJECT_ROOT)),
        db_path=str(db_path.relative_to(PROJECT_ROOT)),
        rows_loaded=loaded,
        rows_rejected=rejected,
        csv_sha256=csv_hash,
        manifest_match=None if expected is None else expected == csv_hash,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "data_spec.yaml")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "reports" / "stage1_manifest.json")
    parser.add_argument("--allow-dirty", action="store_true", help="keep rows that fail typing")
    args = parser.parse_args()

    csv_path = args.csv or PROJECT_ROOT / "data" / "generated" / "orders" / "orders.csv"
    db_path = args.db or PROJECT_ROOT / "data" / "processed" / "orders.db"

    report = build(csv_path, db_path, args.manifest, strict=not args.allow_dirty)
    payload = report.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    connection = sqlite3.connect(db_path)
    try:
        print("status counts:", connection.execute(
            "SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY 2 DESC"
        ).fetchall())
        print("net revenue:", connection.execute(
            "SELECT ROUND(SUM(paid_amount_minor - refunded_amount_minor)/100.0, 2) "
            "FROM orders WHERE status IN ('completed','shipped') AND quantity > 0"
        ).fetchone())
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())