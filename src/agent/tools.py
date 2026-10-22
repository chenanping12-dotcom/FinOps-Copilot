"""Business tools for the FinOps agent, kept deliberately narrow and auditable.

The tools exist to remove a class of failure that a free-form text-to-SQL agent
cannot avoid: the model inventing a metric caliber. ``metric_query`` therefore
accepts a *fixed* argument schema (metric + dimension + filters) and compiles the
SQL itself, so "revenue" always means ``paid - refunded`` over
``completed``/``shipped`` regardless of how the question is phrased.

``sql_query`` is the escape hatch for long-tail questions. It is read-only at four
independent layers, because any single layer is bypassable:

  1. connection opened with SQLite ``mode=ro``
  2. a ``set_authorizer`` callback denying anything but SELECT/READ
  3. a table/function allow-list
  4. a row cap plus a VM-instruction budget, so a cartesian join cannot hang the box
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

REVENUE_STATUSES = ("completed", "shipped")
NET_REVENUE_EXPR = "(paid_amount_minor - refunded_amount_minor)"

ALLOWED_TABLES = frozenset({"orders"})
ALLOWED_COLUMNS = frozenset(
    {
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
    }
)

# SQLite authorizer action codes that are read-only.
_SQLITE_READ_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
}

# Scalar/aggregate functions the compiled metrics use. Everything else is denied,
# so a long-tail query cannot reach ``load_extension`` or filesystem helpers.
ALLOWED_FUNCTIONS = frozenset(
    {
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "round",
        "abs",
        "strftime",
        "substr",
        "length",
        "lower",
        "upper",
        "coalesce",
        "ifnull",
        "cast",
        "date",
        "julianday",
        "typeof",
    }
)

_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


class Metric(str, Enum):
    ORDER_COUNT = "order_count"
    REVENUE = "revenue"
    AVG_ORDER_VALUE = "avg_order_value"
    CANCEL_RATE = "cancel_rate"


class Dimension(str, Enum):
    NONE = "none"
    MERCHANT = "merchant"
    CATEGORY = "category"
    STATUS = "status"
    MONTH = "month"


class MetricQueryError(ValueError):
    """Raised when metric arguments are outside the fixed schema."""


@dataclass
class MetricArguments:
    """The only argument shape the model may use for business metrics."""

    metric: Metric
    dimension: Dimension = Dimension.NONE
    merchant_id: str | None = None
    category: str | None = None
    month: str | None = None
    limit: int = 20

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MetricArguments":
        if not isinstance(raw, dict):
            raise MetricQueryError("arguments must be an object")
        unknown = set(raw) - {
            "metric",
            "dimension",
            "merchant_id",
            "category",
            "month",
            "limit",
        }
        if unknown:
            raise MetricQueryError(f"unsupported argument(s): {sorted(unknown)}")
        try:
            metric = Metric(str(raw["metric"]))
        except (KeyError, ValueError) as exc:
            raise MetricQueryError(
                f"metric must be one of {[m.value for m in Metric]}"
            ) from exc
        try:
            dimension = Dimension(str(raw.get("dimension", "none")))
        except ValueError as exc:
            raise MetricQueryError(
                f"dimension must be one of {[d.value for d in Dimension]}"
            ) from exc
        month = raw.get("month")
        if month is not None and not re.fullmatch(r"\d{4}-\d{2}", str(month)):
            raise MetricQueryError("month must be formatted YYYY-MM")
        limit = int(raw.get("limit", 20))
        if not 1 <= limit <= 100:
            raise MetricQueryError("limit must be between 1 and 100")
        return cls(
            metric=metric,
            dimension=dimension,
            merchant_id=raw.get("merchant_id"),
            category=raw.get("category"),
            month=str(month) if month else None,
            limit=limit,
        )

    def compile(self) -> tuple[str, list[Any]]:
        """Compile to parameterized SQL. Values are always bound, never interpolated."""
        where: list[str] = []
        params: list[Any] = []

        if self.metric in {Metric.REVENUE, Metric.AVG_ORDER_VALUE}:
            placeholders = ",".join("?" for _ in REVENUE_STATUSES)
            where.append(f"status IN ({placeholders})")
            params.extend(REVENUE_STATUSES)
            # Exclude rows with impossible quantities from money calibers.
            where.append("quantity > 0")
        # NOTE: cancel_rate deliberately adds NO status filter. Its denominator is
        # every order in scope, so filtering to cancelled/refunded here would make
        # the ratio structurally 100% for any input.

        if self.merchant_id:
            where.append("merchant_id = ?")
            params.append(self.merchant_id)
        if self.category:
            where.append("category = ?")
            params.append(self.category)
        if self.month:
            where.append("strftime('%Y-%m', order_time) = ?")
            params.append(self.month)

        where_sql = f" WHERE {' AND '.join(where)}" if where else ""

        group_expr = {
            Dimension.NONE: None,
            Dimension.MERCHANT: "merchant_id",
            Dimension.CATEGORY: "category",
            Dimension.STATUS: "status",
            Dimension.MONTH: "strftime('%Y-%m', order_time)",
        }[self.dimension]

        if self.metric is Metric.ORDER_COUNT:
            select = "COUNT(*) AS order_count"
        elif self.metric is Metric.REVENUE:
            select = f"ROUND(SUM({NET_REVENUE_EXPR}) / 100.0, 2) AS revenue_yuan"
        elif self.metric is Metric.AVG_ORDER_VALUE:
            select = f"ROUND(AVG({NET_REVENUE_EXPR}) / 100.0, 2) AS avg_order_value_yuan"
        else:  # CANCEL_RATE
            select = (
                "ROUND(100.0 * SUM(CASE WHEN status = 'cancelled' OR status = 'refunded' "
                "THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancel_rate_percent"
            )

        if group_expr:
            sql = (
                f"SELECT {group_expr} AS group_key, {select} FROM orders"
                f"{where_sql} GROUP BY group_key ORDER BY group_key LIMIT ?"
            )
            # The LIMIT placeholder is only appended when the statement has one;
            # appending it unconditionally desynchronizes bindings.
            params.append(self.limit)
        else:
            sql = f"SELECT {select} FROM orders{where_sql}"
        return sql, params


@dataclass
class ToolResult:
    ok: bool
    tool: str
    sql: str | None = None
    params: list[Any] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    hint: str | None = None
    attempts: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "sql": self.sql,
            "params": self.params,
            "rows": self.rows,
            "row_count": len(self.rows),
            "error": self.error,
            "hint": self.hint,
            "attempts": self.attempts,
        }


class ReadOnlyGuard:
    """Four independent read-only layers around a SQLite connection."""

    def __init__(self, database: Path, instruction_budget: int = 50_000_000, max_rows: int = 100):
        self.database = Path(database)
        self.instruction_budget = instruction_budget
        self.max_rows = max_rows

    def connect(self) -> sqlite3.Connection:
        uri = f"file:{self.database.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.set_authorizer(self._authorizer)
        # ``set_progress_handler`` fires every N VM steps, so one callback is N
        # instructions. The counter is per-connection: keeping it on ``self``
        # leaked across queries, so a second query inherited an exhausted budget
        # and aborted with a spurious "interrupted".
        steps_per_callback = 10_000
        budget = {"calls": 0, "limit": max(1, self.instruction_budget // steps_per_callback)}

        def progress() -> int:
            budget["calls"] += 1
            return 1 if budget["calls"] > budget["limit"] else 0

        connection.set_progress_handler(progress, steps_per_callback)
        return connection

    @staticmethod
    def _authorizer(action: int, arg1: Any, arg2: Any, _db: Any, _trigger: Any) -> int:
        if action not in _SQLITE_READ_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and arg1 is not None:
            if arg1 not in ALLOWED_TABLES:
                return sqlite3.SQLITE_DENY
            # SQLite reports a whole-table read with an EMPTY-STRING column name
            # (not None), so compare against "" as well or every scan is denied.
            column = "" if arg2 is None else str(arg2)
            if column and column not in ALLOWED_COLUMNS:
                return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and arg2 is not None:
            # Only the functions our own compiled metrics rely on. Without this,
            # ``count`` reaching the authorizer would be denied.
            name = str(arg2).lower()
            if name not in ALLOWED_FUNCTIONS:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    @staticmethod
    def validate_sql(sql: str) -> None:
        if _FORBIDDEN_SQL.search(sql):
            raise MetricQueryError("only a single read-only SELECT statement is permitted")
        stripped = sql.strip().rstrip(";")
        if ";" in stripped:
            raise MetricQueryError("multiple statements are not permitted")
        if not stripped.lower().startswith(("select", "with")):
            raise MetricQueryError("statement must start with SELECT")

    def execute(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        self.validate_sql(sql)
        connection = self.connect()
        try:
            cursor = connection.execute(sql, params or [])
            rows = cursor.fetchmany(self.max_rows)
            return [dict(row) for row in rows]
        finally:
            connection.close()


class MetricTool:
    """Fixed-schema metric entry point."""

    name = "metric_query"

    def __init__(self, database: Path) -> None:
        self.guard = ReadOnlyGuard(database)

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            parsed = MetricArguments.from_dict(arguments)
            sql, params = parsed.compile()
        except MetricQueryError as exc:
            return ToolResult(
                ok=False,
                tool=self.name,
                error=str(exc),
                hint="metric must be one of order_count/revenue/avg_order_value/cancel_rate",
            )
        try:
            rows = self.guard.execute(sql, params)
        except sqlite3.Error as exc:
            return ToolResult(ok=False, tool=self.name, sql=sql, params=params, error=str(exc))
        return ToolResult(ok=True, tool=self.name, sql=sql, params=params, rows=rows)


class SafeSQLTool:
    """Read-only long-tail query tool with an error-feedback histogram."""

    name = "sql_query"

    def __init__(self, database: Path) -> None:
        self.guard = ReadOnlyGuard(database)

    def execute(self, sql: str) -> ToolResult:
        try:
            rows = self.guard.execute(sql)
        except (sqlite3.Error, MetricQueryError) as exc:
            return ToolResult(
                ok=False,
                tool=self.name,
                sql=sql,
                error=str(exc),
                hint=self._hint_for(str(exc)),
            )
        return ToolResult(ok=True, tool=self.name, sql=sql, rows=rows)

    @staticmethod
    def _hint_for(error: str) -> str:
        lowered = error.lower()
        if "no such column" in lowered:
            return f"available columns: {sorted(ALLOWED_COLUMNS)}"
        if "no such table" in lowered:
            return f"available tables: {sorted(ALLOWED_TABLES)}"
        if "not authorized" in lowered:
            return "only SELECT over the orders table is permitted"
        if "syntax error" in lowered:
            return "check SQLite syntax; one SELECT statement only"
        return "fix the query and retry"


TOOL_SCHEMA: dict[str, Any] = {
    "name": "metric_query",
    "description": (
        "Query a fixed business metric. Revenue is always net (paid minus refunded) "
        "over completed/shipped orders, so the caliber cannot drift."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": [m.value for m in Metric],
            },
            "dimension": {
                "type": "string",
                "enum": [d.value for d in Dimension],
            },
            "merchant_id": {"type": "string"},
            "category": {"type": "string"},
            "month": {"type": "string", "pattern": r"^\d{4}-\d{2}$"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": ["metric"],
        "additionalProperties": False,
    },
}


def schema_json() -> str:
    return json.dumps(TOOL_SCHEMA, ensure_ascii=False, sort_keys=True)