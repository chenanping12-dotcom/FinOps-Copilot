"""LangGraph state machine for the FinOps agent.

Shape of the graph::

    route -> (metric | sql | refuse) -> reflect -> answer
                   ^                     |
                   +------ retry --------+

Design notes worth defending in an interview:

* ``reflect`` is the only node allowed to rewrite the query, and it is capped at
  ``max_retries`` so a confused model cannot loop forever costing tokens.
* Router and reflector use a *low* temperature and a strict JSON contract; the
  final answer uses a higher temperature for natural phrasing.
* Every tool call is recorded in ``trace`` so the evaluation harness can score
  tool choice independently from answer wording.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .tools import (
    MetricTool,
    SafeSQLTool,
    ToolResult,
    schema_json,
)

Route = Literal["metric", "sql", "refuse"]

ROUTER_SYSTEM = """You are the router of a FinOps analytics agent.
Classify the user's question into exactly one route.

- "metric": the question is about order count, net revenue, average order value,
  or cancel/refund rate. These have a fixed, audited definition.
- "sql": the question needs a long-tail breakdown that the fixed metrics cannot
  express (custom filters, unusual groupings, per-customer statistics).
- "refuse": the question asks to modify data, requests PII beyond customer_id,
  or is unrelated to order analytics.

Reply with JSON only: {"route": "metric|sql|refuse", "reason": "<short reason>"}"""

REFLECT_SYSTEM = """You are the self-correction node of a FinOps analytics agent.
A previous attempt failed. You receive the failure, the schema, and the original
question. Produce a corrected action.

Reply with JSON only.
If the fix is a fixed-schema metric call:
  {"action": "metric", "arguments": {"metric": "...", "dimension": "...", ...}}
If the fix is a read-only SQL statement:
  {"action": "sql", "sql": "SELECT ..."}
If the failure is not fixable (question unanswerable with this data, or unsafe):
  {"action": "give_up", "reason": "<why>"}"""

ANSWER_SYSTEM = """You are the answer node of a FinOps analytics agent.
You are given the user's question and the verified tool result. Write a concise
answer in the user's language.

Hard rules:
- Use ONLY numbers present in the tool result. Never estimate or invent a figure.
- State the caliber you used (net revenue = paid - refunded, completed/shipped only).
- If the result set is empty, say the data contains no matching rows.
"""


@dataclass
class AgentState:
    question: str
    route: Route | None = None
    route_reason: str = ""
    decision: dict[str, Any] | None = None
    result: ToolResult | None = None
    error: str | None = None
    retries: int = 0
    answer: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "route": self.route,
            "route_reason": self.route_reason,
            "decision": self.decision,
            "result": self.result.as_dict() if self.result else None,
            "error": self.error,
            "retries": self.retries,
            "answer": self.answer,
            "trace": self.trace,
        }


class LLMError(RuntimeError):
    pass


Provider = Callable[[str, str], str]


def _extract_json(text: str) -> dict[str, Any]:
    """LLMs wrap JSON in prose and fences; recover the first balanced object."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        if start == -1:
            raise LLMError(f"no JSON object in model output: {text[:200]!r}")
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    break
        if candidate is None:
            raise LLMError(f"unbalanced JSON in model output: {text[:200]!r}")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("model returned JSON that is not an object")
    return parsed


class FinOpsAgent:
    """LangGraph-backed agent; falls back to a deterministic loop if unavailable."""

    def __init__(
        self,
        database: Path,
        provider: Provider,
        *,
        max_retries: int = 2,
        use_langgraph: bool = True,
    ) -> None:
        self.metric_tool = MetricTool(database)
        self.sql_tool = SafeSQLTool(database)
        self.provider = provider
        self.max_retries = max_retries
        self.use_langgraph = use_langgraph
        self._graph = self._build_graph() if use_langgraph else None

    # ---------------------------------------------------------------- routing
    def _route(self, state: AgentState) -> AgentState:
        raw = self.provider(ROUTER_SYSTEM, state.question)
        try:
            payload = _extract_json(raw)
            route = str(payload.get("route", "")).strip().lower()
            if route not in ("metric", "sql", "refuse"):
                raise LLMError(f"invalid route {route!r}")
        except LLMError as exc:
            # A broken router must not silently answer; default to the audited path.
            route = "metric"
            payload = {"reason": f"router fallback after malformed output: {exc}"}
        state.route = route  # type: ignore[assignment]
        state.route_reason = str(payload.get("reason", ""))
        state.trace.append({"node": "route", "route": route, "raw": raw[:400]})
        return state

    def _plan_route(self, state: AgentState) -> str:
        return state.route or "refuse"

    # ------------------------------------------------------------------ nodes
    def _metric_node(self, state: AgentState) -> AgentState:
        arguments = (state.decision or {}).get("arguments")
        if arguments is None:
            raw = self.provider(
                self._planner_prompt(), f"{state.question}\n\nTool schema:\n{schema_json()}"
            )
            try:
                payload = _extract_json(raw)
                arguments = payload.get("arguments", payload)
            except LLMError as exc:
                state.error = f"planner failed: {exc}"
                state.result = ToolResult(ok=False, tool="metric_query", error=state.error)
                return state
        state.decision = {"action": "metric", "arguments": arguments}
        state.result = self.metric_tool.execute(arguments)
        state.error = state.result.error
        state.trace.append(
            {
                "node": "metric",
                "arguments": arguments,
                "ok": state.result.ok,
                "rows": len(state.result.rows),
            }
        )
        return state

    def _sql_node(self, state: AgentState) -> AgentState:
        sql = (state.decision or {}).get("sql")
        if sql is None:
            raw = self.provider(self._planner_prompt(sql_route=True), state.question)
            try:
                payload = _extract_json(raw)
                sql = payload.get("sql")
            except LLMError as exc:
                state.error = f"planner failed: {exc}"
                state.result = ToolResult(ok=False, tool="sql_query", error=state.error)
                return state
        state.decision = {"action": "sql", "sql": sql}
        state.result = self.sql_tool.execute(str(sql))
        state.error = state.result.error
        state.trace.append(
            {
                "node": "sql",
                "sql": sql,
                "ok": state.result.ok,
                "rows": len(state.result.rows),
            }
        )
        return state

    def _refuse_node(self, state: AgentState) -> AgentState:
        state.answer = (
            "I can only answer read-only questions about order analytics "
            "(order count, net revenue, average order value, cancel rate, or a "
            "read-only SQL breakdown). I cannot modify data or answer unrelated questions."
        )
        state.trace.append({"node": "refuse", "reason": state.route_reason})
        return state

    def _reflect_node(self, state: AgentState) -> AgentState:
        if state.result is not None and state.result.ok:
            return state
        if state.retries >= self.max_retries:
            return state
        failure = state.error or "unknown failure"
        # An empty-but-valid result is not an error worth retrying.
        if state.result is not None and state.result.ok is False and "no matching rows" in failure:
            return state
        raw = self.provider(
            REFLECT_SYSTEM,
            json.dumps(
                {
                    "question": state.question,
                    "failed_sql": (state.decision or {}).get("sql"),
                    "failed_arguments": (state.decision or {}).get("arguments"),
                    "error": failure,
                    "hint": state.result.hint if state.result else None,
                },
                ensure_ascii=False,
            ),
        )
        try:
            payload = _extract_json(raw)
        except LLMError as exc:
            state.error = f"reflector failed: {exc}"
            state.trace.append({"node": "reflect", "ok": False, "error": state.error})
            return state

        action = str(payload.get("action", "give_up")).lower()
        if action == "metric":
            arguments = payload.get("arguments", {})
            state.decision = {"action": "metric", "arguments": arguments}
            state.result = self.metric_tool.execute(arguments)
        elif action == "sql":
            state.decision = {"action": "sql", "sql": payload.get("sql")}
            state.result = self.sql_tool.execute(str(payload.get("sql", "")))
        else:
            state.answer = (
                "I could not answer this within the read-only data I have. "
                f"Last error: {failure}"
            )
            state.trace.append({"node": "reflect", "action": "give_up", "error": failure})
            return state

        state.retries += 1
        state.error = state.result.error
        state.trace.append(
            {
                "node": "reflect",
                "action": action,
                "attempt": state.retries,
                "ok": state.result.ok,
            }
        )
        return state

    def _should_retry(self, state: AgentState) -> str:
        if state.result is not None and state.result.ok:
            return "answer"
        if state.retries >= self.max_retries:
            return "answer"
        return "reflect"

    def _answer_node(self, state: AgentState) -> AgentState:
        if state.answer:
            return state
        if state.result is None or not state.result.ok:
            state.answer = (
                "I could not complete this query. "
                f"Last error: {state.error or 'unknown'}"
            )
            return state
        payload = {
            "question": state.question,
            "tool": state.result.tool,
            "sql": state.result.sql,
            "rows": state.result.rows[:50],
            "row_count": len(state.result.rows),
        }
        state.answer = self.provider(
            ANSWER_SYSTEM, json.dumps(payload, ensure_ascii=False)
        )
        state.trace.append({"node": "answer", "chars": len(state.answer)})
        return state

    # ------------------------------------------------------------------ graph
    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            self.use_langgraph = False
            return None

        graph = StateGraph(AgentState)
        graph.add_node("route", self._route)
        graph.add_node("metric", self._metric_node)
        graph.add_node("sql", self._sql_node)
        graph.add_node("refuse", self._refuse_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("answer", self._answer_node)

        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            self._plan_route,
            {"metric": "metric", "sql": "sql", "refuse": "refuse"},
        )
        graph.add_conditional_edges(
            "metric", self._should_retry, {"reflect": "reflect", "answer": "answer"}
        )
        graph.add_conditional_edges(
            "sql", self._should_retry, {"reflect": "reflect", "answer": "answer"}
        )
        graph.add_edge("reflect", "answer")
        graph.add_edge("refuse", END)
        graph.add_edge("answer", END)
        return graph.compile()

    def invoke(self, question: str, decision: dict[str, Any] | None = None) -> AgentState:
        state = AgentState(question=question, decision=decision)
        if self._graph is None:
            return self._invoke_fallback(state)
        raw = self._graph.invoke(state)
        # LangGraph hands back its internal channel mapping (a dict), not the
        # dataclass. Normalize so callers and the eval harness see one shape.
        if isinstance(raw, AgentState):
            return raw
        return AgentState(
            question=raw.get("question", question),
            route=raw.get("route"),
            route_reason=raw.get("route_reason", ""),
            decision=raw.get("decision"),
            result=raw.get("result"),
            error=raw.get("error"),
            retries=raw.get("retries", 0),
            answer=raw.get("answer", ""),
            trace=list(raw.get("trace", [])),
        )

    def _invoke_fallback(self, state: AgentState) -> AgentState:
        """Same control flow without LangGraph, so the agent is testable offline."""
        state = self._route(state)
        if state.route == "refuse":
            return self._refuse_node(state)
        state = self._metric_node(state) if state.route == "metric" else self._sql_node(state)
        while self._should_retry(state) == "reflect":
            state = self._reflect_node(state)
        return self._answer_node(state)

    # ---------------------------------------------------------------- prompts
    @staticmethod
    def _planner_prompt(sql_route: bool = False) -> str:
        if sql_route:
            return (
                "Write one read-only SQLite SELECT over the `orders` table to answer "
                "the question. Reply with JSON only: {\"sql\": \"SELECT ...\"}. "
                "Never write, never use multiple statements."
            )
        return (
            "Choose arguments for the fixed-schema metric tool. Reply with JSON only: "
            '{"arguments": {...}}. Do not invent metrics outside the schema.'
        )