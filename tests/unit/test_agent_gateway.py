"""Stage 3 regression tests: fixed-schema metrics, read-only SQL, and the graph.

These tests target the specific defects found while wiring the agent to the real
50k-row database, so each one fails if the defect is reintroduced:

* a ``LIMIT ?`` placeholder bound without a ``LIMIT`` clause in the statement
* ``SQLITE_READ`` giving an empty-string column name for a whole-table read
* a per-connection progress handler whose counter leaked across queries
* ``cancel_rate`` filtering its own denominator down to 100%
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.agent.graph import AgentState, FinOpsAgent
from src.agent.tools import (
    ALLOWED_FUNCTIONS,
    Dimension,
    Metric,
    MetricArguments,
    MetricQueryError,
    MetricTool,
    ReadOnlyGuard,
    SafeSQLTool,
)


def build_fixture_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT, merchant_id TEXT, order_time TEXT,
            status TEXT, category TEXT, quantity INTEGER,
            unit_price_minor INTEGER, discount_minor INTEGER,
            shipping_fee_minor INTEGER, paid_amount_minor INTEGER,
            refunded_amount_minor INTEGER, payment_method TEXT, source_channel TEXT
        )
        """
    )
    rows = [
        ("O1", "C1", "M1", "2025-06-01T10:00:00+08:00", "completed", "apparel", 2, 1000, 0, 0, 2000, 0, "alipay", "marketplace"),
        ("O2", "C2", "M1", "2025-06-02T10:00:00+08:00", "shipped", "apparel", 1, 3000, 0, 0, 3000, 500, "wechat", "marketplace"),
        ("O3", "C3", "M2", "2025-07-03T10:00:00+08:00", "refunded", "home_living", 1, 4000, 0, 0, 4000, 4000, "alipay", "direct"),
        ("O4", "C4", "M2", "2025-07-04T10:00:00+08:00", "cancelled", "home_living", 3, 500, 0, 0, 1500, 0, "card", "direct"),
        ("O5", "C5", "M2", "2025-07-05T10:00:00+08:00", "pending", "apparel", 1, 900, 0, 0, 900, 0, "card", "direct"),
    ]
    connection.executemany(
        "INSERT INTO orders VALUES (" + ",".join("?" * 14) + ")", rows
    )
    connection.commit()
    connection.close()


class FixtureCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "orders.db"
        build_fixture_db(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class MetricCompilationTests(FixtureCase):
    def test_ungrouped_query_binds_no_limit_placeholder(self) -> None:
        """Regression: an ungrouped SELECT has no LIMIT, so no limit param exists."""
        sql, params = MetricArguments(metric=Metric.ORDER_COUNT).compile()
        self.assertNotIn("LIMIT", sql)
        self.assertEqual(params, [])
        self.assertEqual(MetricTool(self.db).execute({"metric": "order_count"}).rows, [{"order_count": 5}])

    def test_grouped_query_binds_limit_once(self) -> None:
        sql, params = MetricArguments(
            metric=Metric.ORDER_COUNT, dimension=Dimension.CATEGORY, limit=3
        ).compile()
        self.assertEqual(sql.count("?"), len(params))
        self.assertIn("GROUP BY", sql)
        result = MetricTool(self.db).execute(
            {"metric": "order_count", "dimension": "category", "limit": 3}
        )
        self.assertTrue(result.ok, result.error)
        # The fixture has exactly two distinct categories, so LIMIT 3 yields 2 rows.
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(
            {row["group_key"] for row in result.rows}, {"apparel", "home_living"}
        )

    def test_grouped_limit_caps_a_fixture_with_more_groups(self) -> None:
        connection = sqlite3.connect(self.db)
        try:
            connection.executemany(
                "INSERT INTO orders VALUES (" + ",".join("?" * 14) + ")",
                [
                    (
                        f"G{i}", "C", f"MX{i}", "2025-08-01T10:00:00+08:00", "completed",
                        f"cat{i}", 1, 100, 0, 0, 100, 0, "card", "direct",
                    )
                    for i in range(6)
                ],
            )
            connection.commit()
        finally:
            connection.close()
        result = MetricTool(self.db).execute(
            {"metric": "order_count", "dimension": "category", "limit": 3}
        )
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.rows), 3)

    def test_placeholder_count_matches_parameter_count_for_every_metric(self) -> None:
        for metric in Metric:
            for dimension in Dimension:
                sql, params = MetricArguments(
                    metric=metric,
                    dimension=dimension,
                    merchant_id="M1",
                    category="apparel",
                    month="2025-06",
                ).compile()
                self.assertEqual(
                    sql.count("?"),
                    len(params),
                    f"binding mismatch for {metric.value}/{dimension.value}",
                )

    def test_revenue_is_net_and_excludes_non_revenue_statuses(self) -> None:
        result = MetricTool(self.db).execute({"metric": "revenue"})
        # completed 2000 + shipped (3000-500) = 4500 minor units = 45.00 yuan.
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.rows[0]["revenue_yuan"], 45.0)

    def test_cancel_rate_denominator_is_all_orders_not_only_cancelled(self) -> None:
        """Regression: filtering to cancelled rows pinned the ratio at 100%."""
        result = MetricTool(self.db).execute({"metric": "cancel_rate"})
        self.assertTrue(result.ok, result.error)
        # 2 of 5 orders are cancelled/refunded -> 40%, not 100%.
        self.assertEqual(result.rows[0]["cancel_rate_percent"], 40.0)

    def test_unsupported_arguments_are_rejected(self) -> None:
        result = MetricTool(self.db).execute({"metric": "revenue", "evil": "1"})
        self.assertFalse(result.ok)
        self.assertIn("unsupported argument", result.error or "")

    def test_unknown_metric_is_rejected_without_touching_sql(self) -> None:
        result = MetricTool(self.db).execute({"metric": "profit"})
        self.assertFalse(result.ok)
        self.assertIsNone(result.sql)

    def test_month_format_is_validated(self) -> None:
        with self.assertRaises(MetricQueryError):
            MetricArguments.from_dict({"metric": "revenue", "month": "2025/06"})


class ReadOnlyGuardTests(FixtureCase):
    def test_whole_table_read_with_empty_column_name_is_allowed(self) -> None:
        """Regression: SQLITE_READ passes '' (not None) for a table-level read."""
        rows = ReadOnlyGuard(self.db).execute("SELECT COUNT(*) AS c FROM orders")
        self.assertEqual(rows, [{"c": 5}])

    def test_repeated_queries_do_not_inherit_an_exhausted_budget(self) -> None:
        """Regression: the progress counter lived on the guard and leaked."""
        guard = ReadOnlyGuard(self.db, instruction_budget=5_000_000)
        for _ in range(5):
            rows = guard.execute("SELECT SUM(paid_amount_minor) AS total FROM orders")
            self.assertEqual(rows[0]["total"], 11400)

    def test_progress_handler_interrupts_a_runaway_scan(self) -> None:
        """A cartesian join must be aborted, not allowed to hang the process."""
        connection = sqlite3.connect(self.db)
        try:
            connection.executemany(
                "INSERT INTO orders VALUES (" + ",".join("?" * 14) + ")",
                [
                    (
                        f"R{i}", "C", "M", "2025-09-01T10:00:00+08:00", "completed",
                        "bulk", 1, 100, 0, 0, 100, 0, "card", "direct",
                    )
                    for i in range(60)
                ],
            )
            connection.commit()
        finally:
            connection.close()
        guard = ReadOnlyGuard(self.db, instruction_budget=50_000)
        with self.assertRaises(sqlite3.Error) as caught:
            guard.execute(
                "SELECT COUNT(*) FROM orders a, orders b, orders c, orders d, orders e, orders f"
            )
        self.assertIn("interrupt", str(caught.exception).lower())

    def test_write_statements_are_rejected(self) -> None:
        for statement in (
            "DROP TABLE orders",
            "DELETE FROM orders",
            "UPDATE orders SET status='x'",
            "INSERT INTO orders VALUES (1)",
            "SELECT * FROM orders; DROP TABLE orders",
            "PRAGMA table_info(orders)",
            "ATTACH DATABASE 'x' AS y",
        ):
            with self.assertRaises(MetricQueryError, msg=statement):
                ReadOnlyGuard.validate_sql(statement)

    def test_disallowed_table_and_function_are_denied(self) -> None:
        guard = ReadOnlyGuard(self.db)
        self.assertIn("prohibited", str(self._error(guard, "SELECT * FROM sqlite_master")))
        self.assertIn("no such table", str(self._error(guard, "SELECT * FROM customers")))
        self.assertIn("not authorized", str(self._error(guard, "SELECT load_extension(1)")))

    def test_database_is_opened_read_only(self) -> None:
        """Writes are blocked by the authorizer even before the read-only file check."""
        guard = ReadOnlyGuard(self.db)
        connection = guard.connect()
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute("CREATE TABLE sneaky (x INTEGER)")
        finally:
            connection.close()

    def test_read_only_file_flag_blocks_writes_without_authorizer(self) -> None:
        """The ``mode=ro`` URI is an independent layer, so prove it on its own."""
        connection = sqlite3.connect(
            f"file:{self.db.as_posix()}?mode=ro", uri=True, isolation_level=None
        )
        try:
            with self.assertRaises(sqlite3.OperationalError) as caught:
                connection.execute("CREATE TABLE sneaky (x INTEGER)")
            self.assertIn("readonly", str(caught.exception).lower())
        finally:
            connection.close()

    def test_safe_sql_tool_returns_actionable_hints(self) -> None:
        tool = SafeSQLTool(self.db)
        result = tool.execute("SELECT nope FROM orders")
        self.assertFalse(result.ok)
        self.assertIn("available columns", result.hint or "")

    @staticmethod
    def _error(guard: ReadOnlyGuard, sql: str) -> str:
        try:
            guard.execute(sql)
        except sqlite3.Error as exc:
            return str(exc)
        raise AssertionError(f"expected {sql!r} to be rejected")


class AgentGraphTests(FixtureCase):
    @staticmethod
    def scripted(responses: list[str]):
        queue = iter(responses)
        return lambda system, user: next(queue)

    def test_metric_route_produces_answer_without_retry(self) -> None:
        agent = FinOpsAgent(
            self.db,
            self.scripted(
                [
                    json.dumps({"route": "metric", "reason": "revenue"}),
                    json.dumps({"arguments": {"metric": "revenue", "dimension": "category"}}),
                    "净营收按品类如下。",
                ]
            ),
        )
        state = agent.invoke("各品类净营收？")
        self.assertIsInstance(state, AgentState)
        self.assertEqual(state.route, "metric")
        self.assertEqual(state.retries, 0)
        self.assertTrue(state.result.ok)
        self.assertEqual(state.answer, "净营收按品类如下。")
        self.assertEqual([t["node"] for t in state.trace], ["route", "metric", "answer"])

    def test_sql_route_runs_a_read_only_query(self) -> None:
        agent = FinOpsAgent(
            self.db,
            self.scripted(
                [
                    json.dumps({"route": "sql", "reason": "long tail"}),
                    json.dumps({"sql": "SELECT payment_method, COUNT(*) AS c FROM orders GROUP BY payment_method"}),
                    "分布如下。",
                ]
            ),
        )
        state = agent.invoke("各支付方式单量？")
        self.assertEqual(state.route, "sql")
        self.assertTrue(state.result.ok)
        self.assertEqual(len(state.result.rows), 3)

    def test_reflector_repairs_broken_sql_within_the_retry_cap(self) -> None:
        agent = FinOpsAgent(
            self.db,
            self.scripted(
                [
                    json.dumps({"route": "sql", "reason": "x"}),
                    json.dumps({"sql": "SELECT missing_col FROM orders"}),
                    json.dumps({"action": "sql", "sql": "SELECT COUNT(*) AS c FROM orders"}),
                    "共 5 单。",
                ]
            ),
        )
        state = agent.invoke("订单总数？")
        self.assertEqual(state.retries, 1)
        self.assertTrue(state.result.ok)
        self.assertEqual(state.result.rows, [{"c": 5}])

    def test_retry_cap_is_respected_when_reflector_keeps_failing(self) -> None:
        bad = json.dumps({"action": "sql", "sql": "SELECT still_missing FROM orders"})
        agent = FinOpsAgent(
            self.db,
            self.scripted(
                [
                    json.dumps({"route": "sql", "reason": "x"}),
                    json.dumps({"sql": "SELECT missing_col FROM orders"}),
                    bad,
                    bad,
                    bad,
                    bad,
                ]
            ),
            max_retries=2,
        )
        state = agent.invoke("订单总数？")
        self.assertLessEqual(state.retries, 2)
        self.assertFalse(state.result.ok)

    def test_refuse_route_never_touches_the_database(self) -> None:
        agent = FinOpsAgent(
            self.db,
            self.scripted([json.dumps({"route": "refuse", "reason": "unsafe"}), "unused"]),
        )
        state = agent.invoke("删掉所有订单")
        self.assertEqual(state.route, "refuse")
        self.assertIsNone(state.result)
        self.assertIn("read-only", state.answer)

    def test_malformed_router_output_falls_back_to_audited_metric_path(self) -> None:
        agent = FinOpsAgent(
            self.db,
            self.scripted(
                [
                    "not json at all",
                    json.dumps({"arguments": {"metric": "order_count"}}),
                    "共 5 单。",
                ]
            ),
        )
        state = agent.invoke("有多少订单？")
        self.assertEqual(state.route, "metric")
        self.assertTrue(state.result.ok)

    def test_fallback_loop_matches_graph_behaviour(self) -> None:
        """The no-LangGraph path must implement the same control flow."""
        responses = [
            json.dumps({"route": "sql", "reason": "x"}),
            json.dumps({"sql": "SELECT missing_col FROM orders"}),
            json.dumps({"action": "sql", "sql": "SELECT COUNT(*) AS c FROM orders"}),
            "共 5 单。",
        ]
        graph_agent = FinOpsAgent(self.db, self.scripted(responses))
        loop_agent = FinOpsAgent(self.db, self.scripted(responses), use_langgraph=False)
        self.assertIsNone(loop_agent._graph)
        graph_state = graph_agent.invoke("订单总数？")
        loop_state = loop_agent.invoke("订单总数？")
        self.assertEqual(graph_state.route, loop_state.route)
        self.assertEqual(graph_state.retries, loop_state.retries)
        self.assertEqual(graph_state.result.rows, loop_state.result.rows)

    def test_answer_never_invents_numbers_when_tool_failed(self) -> None:
        agent = FinOpsAgent(
            self.db,
            self.scripted([json.dumps({"route": "refuse", "reason": "unsafe"})]),
        )
        state = agent.invoke("删库")
        self.assertNotIn("50000", state.answer)


class GuardConstantTests(unittest.TestCase):
    def test_allowed_functions_exclude_dangerous_builtins(self) -> None:
        for name in ("load_extension", "readfile", "writefile", "sqlite_version"):
            self.assertNotIn(name, ALLOWED_FUNCTIONS)


if __name__ == "__main__":
    unittest.main()