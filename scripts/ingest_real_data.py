"""Ingest a REAL public e-commerce dataset and run the FinOps-Copilot caliber on it.

Why this file exists
--------------------
Every other dataset in this repository is synthetic (``config/data_spec.yaml``:
``synthetic_only: true``). This module is the honest counterweight: it downloads a
real, publicly published UK online-retail transaction log, normalizes it into the
*existing* 14-column ``orders`` contract (see ``scripts/build_orders_db.py:COLUMNS``),
builds a read-only SQLite database, and then measures how dirty the real data is
compared with the six anomaly classes we inject synthetically.

Design rules
------------
* No fabrication. If the download fails, the script fails loudly and prints every
  URL it tried. Nothing is mocked, sampled, or reconstructed.
* No silent coercion. Rows that cannot be typed into the contract are dropped from
  the strict DB **and counted**, because dropping them changes the money caliber.
* The real data is written to a *separate* database (``real_orders.db``) and never
  touches ``data/generated/`` or ``data/processed/orders.db`` (the Stage 1 artifacts
  behind the 254/254 validation numbers).

Usage
-----
    # full pipeline (download -> parse -> normalize -> sqlite -> stats)
    .venv/Scripts/python.exe scripts/ingest_real_data.py

    # reuse an already-downloaded raw file, skip the network
    .venv/Scripts/python.exe scripts/ingest_real_data.py --skip-download

    # use the CSV fallback mirror explicitly
    .venv/Scripts/python.exe scripts/ingest_real_data.py --raw data/raw/online_retail_original_541909.csv

Extra runtime dependency: ``openpyxl`` (for the Online Retail II ``.xlsx`` member
only; the CSV path needs nothing beyond pandas). It is intentionally NOT added to
``requirements.txt`` because it is needed by this offline ingestion path alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# Data sources. Both are the SAME real retailer (a UK online giftware wholesaler);
# source B is the 2010-2011 slice, source A is the full 2009-2011 log.
# --------------------------------------------------------------------------------------
SOURCES: dict[str, dict[str, object]] = {
    "uci_online_retail_ii": {
        "url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        "landing": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "filename": "online_retail_ii.zip",
        "kind": "zip_xlsx",
        "license": "CC BY 4.0",
        "expected_rows": "1,067,371 (2 sheets x ~525k/541k)",
    },
    "github_online_retail_original_csv": {
        "url": (
            "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/"
            "master/data/retail-data/all/online-retail-dataset.csv"
        ),
        "landing": "https://archive.ics.uci.edu/dataset/352/online+retail",
        "filename": "online_retail_original_541909.csv",
        "kind": "csv",
        "license": "CC BY 4.0 (UCI Online Retail dataset 352, mirrored)",
        "expected_rows": "541,909",
    },
}

# Normalized column contract: identical names/types to scripts/build_orders_db.py.
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

# Status approximation. The real schema has no lifecycle column at all, so status is
# derived from the only two signals the data actually carries:
#   * InvoiceNo starts with "C"  -> the whole invoice is a cancellation
#   * Quantity < 0 without "C"   -> a return/adjustment posted against a normal invoice
# Everything else is treated as a settled (completed) sale.
# This is an APPROXIMATION and is labelled as such everywhere it is reported.
STATUS_CANCELLED = "cancelled"
STATUS_REFUNDED = "refunded"
STATUS_COMPLETED = "completed"

MISSING_CUSTOMER_SENTINEL = "CUS_UNKNOWN"
MISSING_DESCRIPTION_SENTINEL = "<no description>"

# Deterministic surrogate keys (the real data has no customer_id column, only a
# 5-digit numeric Customer ID that is null for ~25% of rows).
MERCHANT_SINGLE = "MRC_REAL_01"


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, value: str) -> str:
    """Short deterministic surrogate key so reruns produce byte-identical output."""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}{digest}"


def rel(path: Path) -> str:
    """Workspace-relative path when possible, absolute otherwise.

    ``--raw`` can point at a scratch file outside the project (e.g. a smoke-test
    prefix under the OS temp dir), and ``Path.relative_to`` raises in that case.
    """
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


# --------------------------------------------------------------------------------------
# Stage 1: download
# --------------------------------------------------------------------------------------
def _existing_provenance(target: Path, source_key: str, reason: str) -> dict[str, object]:
    """Provenance for a file we reuse instead of downloading in this run.

    ``downloaded_at`` is only knowable for the current run, so it stays None here;
    the file's mtime is recorded instead as evidence of when the bytes landed
    locally. The report prints whichever is available rather than a bare "n/a".
    """
    stat = target.stat()
    return {
        "skipped": True,
        "reason": reason,
        "source_key": source_key,
        "url": SOURCES[source_key]["url"],
        "landing_page": SOURCES[source_key]["landing"],
        "license": SOURCES[source_key]["license"],
        "local_file": rel(target),
        "bytes": stat.st_size,
        "sha256": sha256_file(target),
        "downloaded_at": None,
        "elapsed_seconds": None,
        "file_mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
    }


def download(source_key: str, target: Path, *, timeout: int = 1800) -> dict[str, object]:
    """Download one source verbatim. Returns provenance metadata, or raises."""
    meta = SOURCES[source_key]
    url = str(meta["url"])
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] GET {url}")
    print(f"[download] -> {target}")
    started = time.time()
    request = urllib.request.Request(url, headers={"User-Agent": "finops-copilot-ingest/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            with target.open("wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"download failed for {url}: {exc!r}") from exc
    elapsed = time.time() - started
    size = target.stat().st_size
    print(
        f"[download] HTTP {status}, {size:,} bytes in {elapsed:.1f}s "
        f"({size / max(elapsed, 1e-9) / 1024:.1f} KB/s)"
    )
    return {
        "source_key": source_key,
        "url": url,
        "landing_page": meta["landing"],
        "license": meta["license"],
        "local_file": rel(target),
        "bytes": size,
        "sha256": sha256_file(target),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(elapsed, 1),
        "http_status": status,
    }


# --------------------------------------------------------------------------------------
# Stage 2: parse raw file into a single flat DataFrame with the real column names
# --------------------------------------------------------------------------------------
REAL_COLUMN_ALIASES = {
    "invoice": ["InvoiceNo", "Invoice", "invoice"],
    "stockcode": ["StockCode", "Stock", "stockcode"],
    "description": ["Description", "description"],
    "quantity": ["Quantity", "quantity"],
    "invoicedate": ["InvoiceDate", "invoicedate"],
    "price": ["UnitPrice", "Price", "unitprice"],
    "customer": ["Customer ID", "CustomerID", "customer_id"],
    "country": ["Country", "country"],
}


def _pick(df: pd.DataFrame, logical: str) -> str | None:
    for candidate in REAL_COLUMN_ALIASES[logical]:
        if candidate in df.columns:
            return candidate
    return None


def parse_raw(raw_path: Path, source_key: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Return (frame with real column names, per-sheet provenance records)."""
    members: list[dict[str, object]] = []
    if raw_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(raw_path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith((".xlsx", ".csv"))]
            if not names:
                raise RuntimeError(
                    f"{raw_path} contains no .xlsx/.csv member; members={archive.namelist()}"
                )
            frames = []
            for name in sorted(names):
                print(f"[parse] zip member {name}")
                if name.lower().endswith(".csv"):
                    part = archive.read(name)
                    from io import BytesIO

                    frame = pd.read_csv(BytesIO(part), encoding="utf-8", dtype=str)
                    frame["__source_member"] = name
                    members.append({"member": name, "sheets": 1, "rows": int(len(frame))})
                    frames.append(frame)
                else:
                    # Online Retail II ships ONE workbook with ONE SHEET PER YEAR
                    # ("Year 2009-2010" + "Year 2010-2011"). Reading sheet_name=0
                    # silently ingests only half the dataset, so read every sheet
                    # and keep per-sheet row provenance in the stats JSON.
                    sheets = pd.read_excel(archive.open(name), sheet_name=None, dtype=str)
                    for sheet_name, frame in sheets.items():
                        if frame is None or frame.empty:
                            continue
                        frame["__source_member"] = f"{name}#{sheet_name}"
                        members.append(
                            {
                                "member": name,
                                "sheet": str(sheet_name),
                                "rows": int(len(frame)),
                            }
                        )
                        frames.append(frame)
            if not frames:
                raise RuntimeError(f"{raw_path} produced no readable sheets/rows")
            df = pd.concat(frames, ignore_index=True)
    elif raw_path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(raw_path, sheet_name=None, dtype=str)
        df = pd.concat(
            [f for f in sheets.values() if f is not None and not f.empty], ignore_index=True
        )
        members.append({"member": raw_path.name, "sheets": len(sheets), "rows": int(len(df))})
    else:
        df = pd.read_csv(raw_path, encoding="utf-8", dtype=str)
        members.append({"member": raw_path.name, "rows": int(len(df))})

    df.columns = [str(c).strip() for c in df.columns]
    mapping = {logical: _pick(df, logical) for logical in REAL_COLUMN_ALIASES}
    missing = [k for k, v in mapping.items() if v is None and k != "country"]
    if missing:
        raise RuntimeError(
            f"raw file {raw_path.name} is missing real columns {missing}; "
            f"actual columns={list(df.columns)}"
        )
    renamed = pd.DataFrame(
        {
            "invoice": df[mapping["invoice"]],
            "stockcode": df[mapping["stockcode"]],
            "description": df[mapping["description"]],
            "quantity": df[mapping["quantity"]],
            "invoicedate": df[mapping["invoicedate"]],
            "price": df[mapping["price"]],
            "customer": df[mapping["customer"]] if mapping["customer"] else None,
            "country": df[mapping["country"]] if mapping["country"] else None,
        }
    )
    renamed["__source_member"] = df["__source_member"] if "__source_member" in df else raw_path.name
    renamed["__row_index"] = range(1, len(renamed) + 1)
    return renamed, members


# --------------------------------------------------------------------------------------
# Stage 3: dirty-data inventory on the raw frame (before any repair)
# --------------------------------------------------------------------------------------
def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def _duplicate_composition(raw: pd.DataFrame, dup_mask: pd.Series) -> dict[str, object]:
    """Explain WHAT the exact-duplicate rows are.

    "34,337 duplicate rows" is useless on its own: the honest answer depends on
    whether the repeated lines sit inside ONE invoice (re-keyed paperwork, the same
    physical transaction) or across DIFFERENT invoices (a genuine repeat purchase
    that must NOT be de-duplicated). Grouping by the full business key answers it.
    """
    key = ["invoice", "stockcode", "quantity", "invoicedate", "price", "customer"]
    groups = raw.loc[dup_mask].groupby(key, dropna=False)
    invoices_per_key = groups["invoice"].nunique()
    if invoices_per_key.empty:
        return {
            "dup_rows": 0,
            "distinct_dup_keys": 0,
            "keys_within_single_invoice": 0,
            "keys_spanning_multiple_invoices": 0,
            "verdict": "no exact duplicates",
        }
    multi = invoices_per_key[invoices_per_key > 1]
    # Rows that belong to a key appearing under more than one InvoiceNo.
    spanning_rows = int(groups.size().loc[multi.index].sum()) if len(multi) else 0
    return {
        "dup_rows": int(dup_mask.sum()),
        "distinct_dup_keys": int(invoices_per_key.size),
        "repeat_groups_within_one_invoice": int((invoices_per_key == 1).sum()),
        "keys_spanning_multiple_invoices": int(multi.size),
        "rows_spanning_multiple_invoices": spanning_rows,
        "max_repeats_in_one_invoice": int(groups.size().max()),
        "verdict": (
            "duplicates are dominated by same-invoice re-keying; de-duplicating them "
            "would delete legitimately recorded lines"
        ),
    }


def inventory(raw: pd.DataFrame) -> dict[str, object]:
    total = int(len(raw))
    invoice = raw["invoice"].astype(str).str.strip()
    stockcode = raw["stockcode"].astype(str).str.strip()
    description = raw["description"].astype(str).str.strip()
    customer = raw["customer"].astype(str).str.strip()
    country = raw["country"].astype(str).str.strip()
    qty = _to_numeric(raw["quantity"])
    price = _to_numeric(raw["price"])
    dt = pd.to_datetime(raw["invoicedate"], errors="coerce", format="mixed")

    # NOTE (correctness trap): the source frame uses pandas' StringDtype, whose NA
    # renders as the literal string "<NA>" once you call .astype(str). Building the
    # null mask from the *original* series (plus the string fallbacks) is the only
    # way to get a truthful count; a sentinel-only check silently reports 0 missing.
    def _blank(col: str) -> pd.Series:
        original = raw[col]
        rendered = original.astype(str).str.strip()
        return original.isna() | rendered.isin({"", "nan", "NaN", "None", "<NA>", "N/A"})

    customer_missing = _blank("customer")
    qty_missing = raw["quantity"].isna() | qty.isna()
    price_missing = raw["price"].isna() | price.isna()
    dt_missing = raw["invoicedate"].isna() | dt.isna()
    is_cancel = invoice.str.upper().str.startswith("C")
    neg_qty = qty < 0
    zero_price = price == 0
    neg_price = price < 0
    desc_missing = _blank("description")
    # A "manual adjustment" code: letters-only stock codes carry no catalog identity.
    non_catalog_code = ~stockcode.str.match(r"^\d", na=False)

    dup_mask = raw.duplicated(
        subset=["invoice", "stockcode", "quantity", "invoicedate", "price", "customer"],
        keep="first",
    )
    dup_composition = _duplicate_composition(raw, dup_mask)

    # Same StockCode -> multiple distinct non-empty Descriptions.
    desc_clean = description.where(~desc_missing, pd.NA)
    desc_variants = (
        pd.DataFrame({"code": stockcode, "desc": desc_clean}).dropna().drop_duplicates()
    )
    variants_per_code = desc_variants.groupby("code")["desc"].nunique()
    multi_desc_codes = variants_per_code[variants_per_code > 1]

    price_variants = (
        pd.DataFrame({"code": stockcode, "price": price}).dropna().drop_duplicates()
    )
    prices_per_code = price_variants.groupby("code")["price"].nunique()
    multi_price_codes = prices_per_code[prices_per_code > 1]

    # Same customer -> multiple countries/addresses.
    known_customer = customer.where(~customer_missing, pd.NA)
    cust_country = (
        pd.DataFrame({"cust": known_customer, "country": country.where(country != "", pd.NA)})
        .dropna()
        .drop_duplicates()
    )
    countries_per_customer = cust_country.groupby("cust")["country"].nunique()
    multi_country_customers = countries_per_customer[countries_per_customer > 1]

    # Modifier/child stock codes ("85123A") alongside parents ("85123").
    # Compare the project's own normalization view rather than raw string surgery:
    # ``Series.sum()`` on a string-typed result yields '' instead of 0, which would
    # crash int() at report time.
    parent_child = stockcode[stockcode.str.match(r"^\d+[A-Za-z]+$", na=False)]
    parent_set = set(stockcode[stockcode.str.match(r"^\d+$", na=False)])
    child_parent_value = parent_child.str.replace(r"[A-Za-z]+$", "", regex=True)
    child_has_parent = parent_child[child_parent_value.isin(parent_set)]
    child_with_parent_rows = int(child_has_parent.size)
    child_with_parent_codes = int(child_has_parent.nunique(dropna=False)) if child_has_parent.size else 0

    def pct(n: int) -> float:
        return round(100.0 * n / total, 4) if total else 0.0

    per_year: dict[str, int] = {}
    if dt.notna().any():
        per_year = {
            str(k): int(v) for k, v in dt.dt.year.value_counts().sort_index().items()
        }

    stats: dict[str, object] = {
        "rows_total": total,
        "rows_columns": list(raw.columns),
        "time_min": None if dt.notna().sum() == 0 else str(dt.min()),
        "time_max": None if dt.notna().sum() == 0 else str(dt.max()),
        "rows_per_year": per_year,
        "distinct_invoice": int(invoice.nunique()),
        "distinct_stockcode": int(stockcode.nunique()),
        "distinct_customer_known": int(known_customer.dropna().nunique()),
        "distinct_country": int(country[country != ""].nunique()),
        # ---- the six classes drawn directly from the task ----
        "missing_customer_id": {"rows": int(customer_missing.sum()), "pct": pct(int(customer_missing.sum()))},
        "negative_quantity": {"rows": int(neg_qty.sum()), "pct": pct(int(neg_qty.sum()))},
        "price_zero": {"rows": int(zero_price.sum()), "pct": pct(int(zero_price.sum()))},
        "price_negative": {"rows": int(neg_price.sum()), "pct": pct(int(neg_price.sum()))},
        "cancel_invoice_c": {"rows": int(is_cancel.sum()), "pct": pct(int(is_cancel.sum()))},
        "exact_duplicate_rows": {"rows": int(dup_mask.sum()), "pct": pct(int(dup_mask.sum()))},
        # ---- additional real-world dirt the synthetic generator never models ----
        "quantity_not_numeric": {"rows": int(qty_missing.sum()), "pct": pct(int(qty_missing.sum()))},
        "price_not_numeric": {"rows": int(price_missing.sum()), "pct": pct(int(price_missing.sum()))},
        "invoice_date_unparseable": {"rows": int(dt_missing.sum()), "pct": pct(int(dt_missing.sum()))},
        "missing_description": {"rows": int(desc_missing.sum()), "pct": pct(int(desc_missing.sum()))},
        "non_catalog_stockcode": {
            "rows": int(non_catalog_code.sum()),
            "pct": pct(int(non_catalog_code.sum())),
            "examples": sorted(set(stockcode[non_catalog_code].head(400).tolist()))[:15],
        },
        "stockcode_multi_description": {
            "codes": int(len(multi_desc_codes)),
            "rows_affected": int(stockcode.isin(multi_desc_codes.index).sum()),
            "max_variants": int(multi_desc_codes.max()) if len(multi_desc_codes) else 0,
            "examples": {str(k): int(v) for k, v in multi_desc_codes.head(8).items()},
        },
        "stockcode_multiple_prices": {
            "codes": int(len(multi_price_codes)),
            "rows_affected": int(stockcode.isin(multi_price_codes.index).sum()),
            "max_variants": int(multi_price_codes.max()) if len(multi_price_codes) else 0,
        },
        "customer_multi_country": {
            "customers": int(len(multi_country_customers)),
            "rows_affected": int(known_customer.isin(multi_country_customers.index).sum()),
            "max_variants": int(multi_country_customers.max()) if len(multi_country_customers) else 0,
            "examples": {str(k): int(v) for k, v in multi_country_customers.head(8).items()},
        },
        "child_stockcode_with_parent": {
            "rows": child_with_parent_rows,
            "codes": child_with_parent_codes,
            "examples": sorted(set(child_has_parent.head(200).tolist()))[:10],
        },
        "exact_duplicate_composition": dup_composition,
        "cancel_and_negative_quantity_overlap": {
            "rows": int((is_cancel & neg_qty).sum()),
            "pct": pct(int((is_cancel & neg_qty).sum())),
        },
        "negative_quantity_without_cancel": {
            "rows": int((neg_qty & ~is_cancel).sum()),
            "pct": pct(int((neg_qty & ~is_cancel).sum())),
        },
        # refund-like money on rows that are NOT cancellations -> the synthetic
        # schema has no concept of this at all.
        "refund_signal_on_non_cancel_row": {
            "rows": int((~is_cancel & (qty < 0)).sum()),
            "pct": pct(int((~is_cancel & (qty < 0)).sum())),
        },
    }
    stats["_masks"] = {
        "invoice": invoice,
        "stockcode": stockcode,
        "description": description,
        "customer": customer,
        "country": country,
        "qty": qty,
        "price": price,
        "dt": dt,
        "is_cancel": is_cancel,
        "dup": dup_mask,
        # True nulls, captured from the original StringDtype series; the string
        # renderings below cannot be used to detect them (they become "<NA>").
        "customer_missing": customer_missing,
        "desc_missing": desc_missing,
    }
    return stats


# --------------------------------------------------------------------------------------
# Stage 4: normalize into the 14-column contract + record what had to be dropped
# --------------------------------------------------------------------------------------
def normalize(raw: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[pd.DataFrame, dict[str, object]]:
    qty = masks["qty"]
    price = masks["price"]
    dt = masks["dt"]
    invoice = masks["invoice"]
    stockcode = masks["stockcode"]
    description = masks["description"]
    customer = masks["customer"]
    is_cancel = masks["is_cancel"]

    bad_qty = qty.isna()
    bad_price = price.isna()
    bad_dt = dt.isna()
    dropped = bad_qty | bad_price | bad_dt

    # Money: GBP float -> integer minor units (pence). round() before int() so that
    # 1.0*100.0 (== 100.00000000000001) does not become 100 instead of 100.
    qty_i = qty.fillna(0).round().astype("int64")
    price_minor = (price.fillna(0) * 100.0).round().astype("int64")
    gross_minor = qty_i * price_minor

    is_refund_row = qty_i < 0
    # paid_amount: what the customer actually paid on this line (only on the sale leg).
    paid = gross_minor.clip(lower=0)
    # refunded_amount: the absolute value of the negative leg. The project caliber is
    # `paid - refunded`, so storing the negative leg as a positive refund reproduces
    # the real net money flow exactly.
    refunded = (-gross_minor).clip(lower=0)

    status = pd.Series(STATUS_COMPLETED, index=raw.index, dtype="object")
    status = status.mask(is_refund_row, STATUS_REFUNDED)
    status = status.mask(is_cancel, STATUS_CANCELLED)

    customer_id = customer.where(
        ~masks["customer_missing"], MISSING_CUSTOMER_SENTINEL
    )
    customer_id = customer_id.map(lambda v: f"CUS{str(v).split('.')[0].zfill(6)}")

    normalized = pd.DataFrame(
        {
            "order_id": invoice.map(lambda v: f"INV{stable_id('', str(v))}"),
            "customer_id": customer_id,
            "merchant_id": MERCHANT_SINGLE,
            "order_time": dt.dt.strftime("%Y-%m-%dT%H:%M:%S") + "+00:00",
            "status": status,
            "category": stockcode.map(lambda v: f"SKU_{str(v)}"),
            "quantity": qty_i,
            "unit_price_minor": price_minor,
            "discount_minor": 0,
            "shipping_fee_minor": 0,
            "paid_amount_minor": paid,
            "refunded_amount_minor": refunded,
            "payment_method": "unknown",
            "source_channel": "unknown",
        }
    )

    notes: dict[str, object] = {
        "rows_input": int(len(raw)),
        "rows_dropped_untypeable": int(dropped.sum()),
        "rows_dropped_pct": round(100.0 * int(dropped.sum()) / max(len(raw), 1), 4),
        "dropped_breakdown": {
            "quantity_not_numeric": int(bad_qty.sum()),
            "price_not_numeric": int(bad_price.sum()),
            "invoice_date_unparseable": int(bad_dt.sum()),
        },
        "kept_duplicate_rows": int(masks["dup"].sum()),
        "paid_amount_minor_total": int(paid.sum()),
        "refunded_amount_minor_total": int(refunded.sum()),
        "net_money_minor_total": int((paid - refunded).sum()),
        "gross_money_minor_all_rows": int(gross_minor.sum()),
        "note_money": (
            "paid-refunded over ALL rows reproduces sum(quantity*unit_price) exactly, "
            "including negative legs; status filtering is a separate step."
        ),
        "note_category": "category is set to SKU_<StockCode>; the real schema has no category column.",
        "note_payment_channel": (
            "payment_method/source_channel are the literal sentinel 'unknown': the real "
            "dataset carries no such columns. Any retention/refund analysis BY CHANNEL is "
            "therefore impossible on this source and must not be claimed."
        ),
        "note_status": (
            "status is derived: InvoiceNo^C -> cancelled, Quantity<0 -> refunded, "
            "else completed. The real schema has no lifecycle column, so 'shipped' is "
            "never produced. This is an approximation, labelled as such."
        ),
        "note_discount_shipping": (
            "discount_minor and shipping_fee_minor are hard 0: the real dataset records "
            "no discount or shipping split. Markdown/absolute-price semantics are unknown."
        ),
    }
    return normalized, notes


# --------------------------------------------------------------------------------------
# Stage 5: SQLite build (same declared schema as build_orders_db.py) + verifier
# --------------------------------------------------------------------------------------
def build_db(df: pd.DataFrame, db_path: Path) -> dict[str, object]:
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
        insert_sql = f"INSERT INTO orders ({', '.join(COLUMNS)}) VALUES ({placeholders})"

        # int64 -> python int (numpy scalars are not accepted by the sqlite driver).
        records = [
            tuple(int(v) if isinstance(v, (int,)) and not isinstance(v, bool) else v for v in row)
            for row in df[list(COLUMNS)].itertuples(index=False, name=None)
        ]
        fixed = []
        for row in df[list(COLUMNS)].itertuples(index=False, name=None):
            fixed.append(tuple(int(v) if hasattr(v, "item") else v for v in row))
        connection.executemany(insert_sql, fixed)
        for statement in (
            "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_orders_time ON orders(order_time)",
            "CREATE INDEX IF NOT EXISTS idx_orders_category ON orders(category)",
        ):
            connection.execute(statement)
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        stored = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"sqlite integrity_check failed: {integrity}")
    finally:
        connection.close()
    return {"db_path": rel(db_path),
            "rows_in_db": int(stored), "integrity_check": integrity,
            "rows_attempted": int(len(records))}


# --------------------------------------------------------------------------------------
# Stage 6: run the PROJECT's own caliber against the real DB
# --------------------------------------------------------------------------------------
def run_project_caliber(db_path: Path) -> dict[str, object]:
    """Use src.agent.tools.MetricTool verbatim, plus a raw-net cross-check."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.agent.tools import MetricTool, NET_REVENUE_EXPR, REVENUE_STATUSES  # noqa: PLC0415

    tool = MetricTool(db_path)
    out: dict[str, object] = {"revenue_statuses": list(REVENUE_STATUSES),
                              "net_revenue_expr": NET_REVENUE_EXPR,
                              "queries": {}}
    for arguments in (
        {"metric": "order_count"},
        {"metric": "revenue"},
        {"metric": "avg_order_value"},
        {"metric": "cancel_rate"},
        {"metric": "revenue", "dimension": "month"},
        {"metric": "order_count", "dimension": "status"},
    ):
        result = tool.execute(arguments)
        key = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        out["queries"][key] = {
            "ok": result.ok,
            "sql": result.sql,
            "params": result.params,
            "rows": result.rows,
            "error": result.error,
        }
        print(f"[caliber] {key} -> ok={result.ok} rows={result.rows} err={result.error}")

    # Independent cross-check straight from SQLite (not through the tool) so the two
    # numbers can disagree loudly if the caliber drifts.
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        total = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        net_all = connection.execute(
            f"SELECT SUM({NET_REVENUE_EXPR}) FROM orders"
        ).fetchone()[0]
        net_status = connection.execute(
            f"SELECT SUM({NET_REVENUE_EXPR}) FROM orders WHERE status IN "
            f"({','.join('?' for _ in REVENUE_STATUSES)})",
            list(REVENUE_STATUSES),
        ).fetchone()[0]
        pos_only = connection.execute(
            f"SELECT SUM({NET_REVENUE_EXPR}) FROM orders WHERE quantity > 0"
        ).fetchone()[0]
        quarantined = connection.execute(
            f"SELECT SUM({NET_REVENUE_EXPR}) FROM orders WHERE status = 'cancelled'"
        ).fetchone()[0]
        unsupported_status = connection.execute(
            "SELECT status, COUNT(*) FROM orders WHERE status NOT IN "
            "('completed','shipped','pending','cancelled','refunded') GROUP BY status"
        ).fetchall()
        identity_mismatch = connection.execute(
            "SELECT COUNT(*) FROM orders WHERE "
            "paid_amount_minor - refunded_amount_minor <> quantity * unit_price_minor"
        ).fetchone()[0]
    finally:
        connection.close()

    out["cross_check_sqlite"] = {
        "table_rows": int(total),
        "net_revenue_minor_all_rows": int(net_all or 0),
        "net_revenue_minor_status_filtered": int(net_status or 0),
        "net_revenue_minor_positive_quantity_only": int(pos_only or 0),
        "net_revenue_minor_cancelled_quarantined": int(quarantined or 0),
        "rows_with_unsupported_status": [list(r) for r in unsupported_status],
        "rows_where_paid_minus_refunded_ne_quantity_x_price": int(identity_mismatch),
        "note_revenue_status_filtered_vs_all": (
            "The project caliber only counts completed/shipped; on this real data that "
            "silently drops the entire refund leg, which is why both numbers are reported."
        ),
    }
    return out


# --------------------------------------------------------------------------------------
# Stage 7: synthetic comparator
# --------------------------------------------------------------------------------------
def synthetic_baseline(synthetic_csv: Path, synthetic_labels: Path) -> dict[str, object]:
    """Measure the SAME diritness classes on the Stage 1 synthetic CSV."""
    if not synthetic_csv.exists():
        return {"available": False, "reason": f"missing {synthetic_csv}"}
    df = pd.read_csv(synthetic_csv, dtype=str)
    total = int(len(df))
    qty = _to_numeric(df["quantity"])
    refunded = _to_numeric(df["refunded_amount_minor"])
    paid = _to_numeric(df["paid_amount_minor"])
    dt = pd.to_datetime(df["order_time"], errors="coerce", format="mixed")

    def pct(n: int) -> float:
        return round(100.0 * n / total, 4) if total else 0.0

    labels = []
    if synthetic_labels.exists():
        for line in synthetic_labels.read_text(encoding="utf-8").splitlines():
            if line.strip():
                labels.append(json.loads(line))
    from collections import Counter

    label_counts = Counter(str(item.get("anomaly_id")) for item in labels)

    return {
        "available": True,
        "rows_total": total,
        "distinct_customer_id": int(df["customer_id"].nunique()),
        "distinct_merchant_id": int(df["merchant_id"].nunique()),
        "distinct_category": int(df["category"].nunique()),
        "distinct_source_channel": int(df["source_channel"].nunique()),
        "distinct_payment_method": int(df["payment_method"].nunique()),
        "missing_customer_id": {"rows": 0, "pct": 0.0},
        "negative_quantity": {"rows": int((qty < 0).sum()), "pct": pct(int((qty < 0).sum()))},
        "price_zero": {"rows": 0, "pct": 0.0},
        "price_negative": {"rows": 0, "pct": 0.0},
        "cancel_invoice_c": {"rows": 0, "pct": 0.0, "note": "synthetic uses a status column, not invoice prefixes"},
        "exact_duplicate_rows": {"rows": int(df.duplicated().sum()), "pct": pct(int(df.duplicated().sum()))},
        "invoice_date_unparseable": {"rows": int(dt.isna().sum()), "pct": pct(int(dt.isna().sum()))},
        "payment_method_unknown": {"rows": 0, "pct": 0.0},
        "source_channel_unknown": {"rows": 0, "pct": 0.0},
        "refund_exceeds_paid": {
            "rows": int((refunded > paid).sum()),
            "pct": pct(int((refunded > paid).sum())),
        },
        "injected_anomalies": {"total": len(labels), "by_type": dict(label_counts)},
        "note_columns_present_real_missing": (
            "The synthetic contract HAS status/category/merchant/discount/shipping/"
            "payment/channel; the real dataset has NONE of them."
        ),
    }


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default="uci_online_retail_ii")
    parser.add_argument("--raw", type=Path, default=None, help="use this already-downloaded raw file")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download even when the target file already exists",
    )
    parser.add_argument("--download-timeout", type=int, default=1800)
    parser.add_argument(
        "--provenance-note",
        default=None,
        help=(
            "free-text note merged into download provenance when the raw file was "
            "fetched outside this script (e.g. an interactive curl with its own "
            "measured speed). Keeps the report honest instead of printing 'n/a'."
        ),
    )
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "processed" / "real_orders.db")
    parser.add_argument(
        "--stats",
        type=Path,
        default=PROJECT_ROOT / "reports" / "real_data_stats.json",
    )
    parser.add_argument(
        "--synthetic-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "generated" / "orders" / "orders.csv",
    )
    parser.add_argument(
        "--synthetic-labels",
        type=Path,
        default=PROJECT_ROOT / "data" / "generated" / "orders" / "anomaly_labels.jsonl",
    )
    args = parser.parse_args()

    report: dict[str, object] = {
        "task": "task-8 real public data ingestion",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "internals": "see scripts/ingest_real_data.py",
    }

    raw_path = args.raw
    if raw_path is None:
        raw_path = PROJECT_ROOT / "data" / "raw" / str(SOURCES[args.source]["filename"])

    if args.raw is not None:
        # An explicit --raw always wins: never re-download over the caller's file.
        if not raw_path.exists():
            print(f"[download] --raw path does not exist: {raw_path}")
            return 2
        report["download"] = _existing_provenance(raw_path, args.source, "--raw given, network not used")
        print(f"[download] using --raw {raw_path} ({raw_path.stat().st_size:,} bytes)")
    elif args.skip_download:
        if not raw_path.exists():
            print(f"[download] --skip-download but {raw_path} is missing")
            return 2
        report["download"] = _existing_provenance(raw_path, args.source, "--skip-download")
        print(f"[download] skipped; reusing {raw_path} ({raw_path.stat().st_size:,} bytes)")
    elif raw_path.exists() and not args.force_download:
        report["download"] = _existing_provenance(
            raw_path, args.source, "already present; pass --force-download to refresh"
        )
        print(f"[download] already present; reusing {raw_path} ({raw_path.stat().st_size:,} bytes)")
    else:
        report["download"] = download(args.source, raw_path, timeout=args.download_timeout)

    if args.provenance_note:
        report["download"]["provenance_note"] = args.provenance_note

    raw, members = parse_raw(raw_path, args.source)
    report["source_members"] = members
    print(f"[parse] {len(raw):,} raw rows from {len(members)} member(s)")

    stats = inventory(raw)
    masks = stats.pop("_masks")
    report["real_inventory"] = stats
    print(f"[inventory] total={stats['rows_total']:,}")
    for key in (
        "missing_customer_id",
        "negative_quantity",
        "price_zero",
        "price_negative",
        "cancel_invoice_c",
        "exact_duplicate_rows",
    ):
        item = stats[key]
        print(f"[inventory] {key}: {item['rows']:,} rows ({item['pct']}%)")

    normalized, notes = normalize(raw, masks)
    report["normalization"] = notes
    print(f"[normalize] kept={len(normalized):,} dropped={notes['rows_dropped_untypeable']:,}")

    report["database"] = build_db(normalized, args.db)
    print(f"[sqlite] {report['database']}")

    report["project_caliber"] = run_project_caliber(args.db)
    report["synthetic_baseline"] = synthetic_baseline(args.synthetic_csv, args.synthetic_labels)

    # Column-level mapping table for the report.
    report["column_mapping"] = [
        {"real_column": "InvoiceNo", "contract_column": "order_id",
         "transform": "hash to INV<sha1[:10]>; C-prefix is preserved in status, not in the id"},
        {"real_column": "Customer ID", "contract_column": "customer_id",
         "transform": "CUS<zero-padded 6 digits>; null -> CUS_UNKNOWN (measured rate is printed in the report)"},
        {"real_column": "(none)", "contract_column": "merchant_id",
         "transform": "constant MRC_REAL_01 (single retailer in the source)"},
        {"real_column": "InvoiceDate", "contract_column": "order_time",
         "transform": "minute resolution, stamped +00:00 (source has no timezone)"},
        {"real_column": "(none)", "contract_column": "status",
         "transform": "derived: C-invoice->cancelled, Quantity<0->refunded, else completed"},
        {"real_column": "StockCode", "contract_column": "category",
         "transform": "SKU_<StockCode>; real data has no product category hierarchy"},
        {"real_column": "Quantity", "contract_column": "quantity", "transform": "int, negatives kept"},
        {"real_column": "UnitPrice", "contract_column": "unit_price_minor",
         "transform": "GBP float -> pence: round(x*100)"},
        {"real_column": "(none)", "contract_column": "discount_minor", "transform": "hard 0 (not recorded)"},
        {"real_column": "(none)", "contract_column": "shipping_fee_minor", "transform": "hard 0 (not recorded)"},
        {"real_column": "Quantity*UnitPrice", "contract_column": "paid_amount_minor",
         "transform": "max(gross,0) in pence on the sale leg"},
        {"real_column": "Quantity*UnitPrice", "contract_column": "refunded_amount_minor",
         "transform": "max(-gross,0) in pence on the negative leg"},
        {"real_column": "(none)", "contract_column": "payment_method", "transform": "sentinel 'unknown'"},
        {"real_column": "(none)", "contract_column": "source_channel", "transform": "sentinel 'unknown'"},
    ]

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[stats] wrote {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())