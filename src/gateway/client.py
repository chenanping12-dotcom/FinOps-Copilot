"""Async LLM gateway: provider failover, SSE degradation, and semantic caching.

Design constraints this module is built around:

* **Failover only before the first streamed token.** Once bytes have reached the
  client, switching providers would duplicate or reorder output, so a mid-stream
  failure is surfaced instead of silently retried.
* **Client disconnect stops retries.** If the caller has gone away, retrying the
  next provider burns quota for a response nobody will read.
* **Retryable = transport error, 429, or 5xx.** A 4xx (other than 429) is a
  request-shape problem; another provider will reject it identically, so it fails
  fast instead of spending the whole provider budget.
* **Semantic cache never returns a different question's answer silently.** It
  gates on a similarity threshold and records the score on the hit, so a
  near-miss is visible in the trace rather than passed off as an exact match.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterable, Sequence

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def is_retryable_status(status: int) -> bool:
    """True when another provider could plausibly succeed with the same request."""
    return status in RETRYABLE_STATUS or status >= 500


@dataclass
class ProviderAttempt:
    provider: str
    status: int | None
    ok: bool
    error: str | None
    latency_ms: float
    retryable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "status": self.status,
            "ok": self.ok,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 3),
            "retryable": self.retryable,
        }


@dataclass
class CompletionResult:
    text: str
    provider: str
    attempts: list[ProviderAttempt] = field(default_factory=list)
    cache_hit: bool = False
    cache_similarity: float | None = None
    degraded_before_first_token: bool = False

    @property
    def total_latency_ms(self) -> float:
        return sum(attempt.latency_ms for attempt in self.attempts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "cache_hit": self.cache_hit,
            "cache_similarity": self.cache_similarity,
            "degraded_before_first_token": self.degraded_before_first_token,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "total_latency_ms": round(self.total_latency_ms, 3),
        }


class ProviderError(RuntimeError):
    def __init__(self, provider: str, status: int | None, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status = status


class AllProvidersFailed(RuntimeError):
    def __init__(self, attempts: Sequence[ProviderAttempt]) -> None:
        detail = "; ".join(
            f"{a.provider}:{a.status if a.status is not None else a.error}" for a in attempts
        )
        super().__init__(f"all providers failed ({detail})")
        self.attempts = list(attempts)


class ClientDisconnected(RuntimeError):
    """Raised when the caller disappeared; retries stop immediately."""


class SemanticCache:
    """Bounded in-memory cache keyed by prompt embedding with a similarity gate."""

    def __init__(
        self,
        threshold: float = 0.95,
        max_entries: int = 512,
        embed: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.max_entries = int(max_entries)
        self._embed = embed or self._default_embed
        self._entries: OrderedDict[str, tuple[list[float], str, str]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _default_embed(text: str) -> list[float]:
        """Deterministic character-trigram hashing; dependency-free and offline."""
        vector = [0.0] * 512
        compact = "".join(text.split()).lower()
        for index in range(max(0, len(compact) - 2)):
            trigram = compact[index : index + 3]
            bucket = int.from_bytes(
                hashlib.blake2b(trigram.encode("utf-8"), digest_size=4).digest(), "big"
            ) % len(vector)
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def get(self, prompt: str) -> tuple[str, str, float] | None:
        query = self._embed(prompt)
        best: tuple[str, str, float] | None = None
        best_score = -1.0
        for key, (vector, provider, text) in self._entries.items():
            score = self._cosine(query, vector)
            if score > best_score:
                best_score = score
                best = (key, provider, text, score)  # type: ignore[assignment]
        if best is not None and best_score >= self.threshold:
            self.hits += 1
            self._entries.move_to_end(best[0])
            return best[1], best[2], best_score
        self.misses += 1
        return None

    def put(self, prompt: str, provider: str, text: str) -> None:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        self._entries[key] = (self._embed(prompt), provider, text)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class LLMClient:
    """Ordered multi-provider client with failover, SSE degradation and caching."""

    def __init__(
        self,
        providers: Iterable[str],
        *,
        transport: Any = None,
        max_retries_per_provider: int = 1,
        cache: SemanticCache | None = None,
        cache_enabled: bool = True,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self.providers = list(providers)
        if not self.providers:
            raise ValueError("at least one provider is required")
        self.transport = transport
        self.max_retries_per_provider = int(max_retries_per_provider)
        self.cache = cache
        self.cache_enabled = cache_enabled
        self._sleep = sleep or asyncio.sleep
        self.attempt_log: list[ProviderAttempt] = []

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        disconnected: Callable[[], bool] | None = None,
    ) -> CompletionResult:
        if self.cache_enabled and self.cache is not None:
            cached = self.cache.get(prompt)
            if cached is not None:
                provider, text, score = cached
                return CompletionResult(
                    text=text,
                    provider=provider,
                    cache_hit=True,
                    cache_similarity=score,
                    attempts=[],
                )

        attempts: list[ProviderAttempt] = []
        for provider in self.providers:
            for retry in range(self.max_retries_per_provider + 1):
                if disconnected is not None and disconnected():
                    self.attempt_log.extend(attempts)
                    raise ClientDisconnected(
                        f"client disconnected after {len(attempts)} attempt(s); stopped retrying"
                    )
                started = time.perf_counter()
                try:
                    text = await self._invoke(provider, prompt, model)
                except ProviderError as exc:
                    retryable = exc.status is None or is_retryable_status(exc.status)
                    attempts.append(
                        ProviderAttempt(
                            provider=provider,
                            status=exc.status,
                            ok=False,
                            error=str(exc),
                            latency_ms=(time.perf_counter() - started) * 1000,
                            retryable=retryable,
                        )
                    )
                    if not retryable:
                        self.attempt_log.extend(attempts)
                        raise
                    if retry < self.max_retries_per_provider:
                        await self._sleep(0.01 * (2**retry))
                    continue
                attempts.append(
                    ProviderAttempt(
                        provider=provider,
                        status=200,
                        ok=True,
                        error=None,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        retryable=False,
                    )
                )
                self.attempt_log.extend(attempts)
                if self.cache_enabled and self.cache is not None:
                    self.cache.put(prompt, provider, text)
                return CompletionResult(text=text, provider=provider, attempts=attempts)

        self.attempt_log.extend(attempts)
        raise AllProvidersFailed(attempts)

    async def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        disconnected: Callable[[], bool] | None = None,
    ) -> AsyncIterator[tuple[str, int]]:
        """Yield ``(delta, provider_index)``; fail over only before the first token.

        Returns the index of the provider that served the stream as the second
        element so a caller can tell whether degradation happened. Because
        ``async def`` + ``yield`` cannot also return a value, the serving provider
        is exposed afterward on ``last_stream_provider``.
        """
        attempts: list[ProviderAttempt] = []
        for index, provider in enumerate(self.providers):
            if disconnected is not None and disconnected():
                self.attempt_log.extend(attempts)
                raise ClientDisconnected("client disconnected before stream start")
            started = time.perf_counter()
            emitted = False
            try:
                async for delta in self._stream_provider(provider, prompt, model):
                    if disconnected is not None and disconnected():
                        self.attempt_log.extend(attempts)
                        raise ClientDisconnected("client disconnected mid-stream")
                    emitted = True
                    yield delta, index
            except ProviderError as exc:
                attempts.append(
                    ProviderAttempt(
                        provider=provider,
                        status=exc.status,
                        ok=False,
                        error=str(exc),
                        latency_ms=(time.perf_counter() - started) * 1000,
                        retryable=exc.status is None or is_retryable_status(exc.status),
                    )
                )
                if emitted:
                    # Mid-stream failure: the client already holds partial output.
                    self.attempt_log.extend(attempts)
                    raise
                if index == len(self.providers) - 1:
                    self.attempt_log.extend(attempts)
                    raise AllProvidersFailed(attempts) from exc
                continue
            attempts.append(
                ProviderAttempt(
                    provider=provider,
                    status=200,
                    ok=True,
                    error=None,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    retryable=False,
                )
            )
            self.attempt_log.extend(attempts)
            self.last_stream_provider = provider
            self.last_stream_degraded = index > 0
            return
        self.attempt_log.extend(attempts)
        raise AllProvidersFailed(attempts)

    async def _invoke(self, provider: str, prompt: str, model: str | None) -> str:
        if self.transport is None:
            raise ProviderError(provider, None, "no transport configured")
        return await self.transport.complete(provider, prompt, model)

    def _stream_provider(self, provider: str, prompt: str, model: str | None):
        if self.transport is None:
            raise ProviderError(provider, None, "no transport configured")
        return self.transport.stream(provider, prompt, model)


class ScriptedTransport:
    """Deterministic test transport: per-provider scripted outcomes."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self.script = {name: list(items) for name, items in script.items()}
        self.calls: list[str] = []

    def _next(self, provider: str) -> Any:
        self.calls.append(provider)
        queue = self.script.get(provider)
        if not queue:
            raise ProviderError(provider, 503, "scripted queue exhausted")
        return queue.pop(0)

    async def complete(self, provider: str, prompt: str, model: str | None) -> str:
        outcome = self._next(provider)
        if isinstance(outcome, ProviderError):
            raise outcome
        if isinstance(outcome, Exception):
            raise ProviderError(provider, None, str(outcome))
        return str(outcome)

    async def stream(self, provider: str, prompt: str, model: str | None):
        outcome = self._next(provider)
        if isinstance(outcome, Exception):
            raise outcome if isinstance(outcome, ProviderError) else ProviderError(
                provider, None, str(outcome)
            )
        for chunk in outcome:
            if isinstance(chunk, Exception):
                raise chunk if isinstance(chunk, ProviderError) else ProviderError(
                    provider, None, str(chunk)
                )
            yield str(chunk)


class HttpTransport:
    """Minimal OpenAI-compatible transport; imported httpx lazily."""

    def __init__(self, base_urls: dict[str, str], api_key: str = "", timeout: float = 30.0) -> None:
        self.base_urls = dict(base_urls)
        self.api_key = api_key
        self.timeout = timeout

    async def complete(self, provider: str, prompt: str, model: str | None) -> str:
        import httpx

        base = self.base_urls[provider]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{base.rstrip('/')}/chat/completions",
                    headers=headers,
                    json={
                        "model": model or "default",
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
        except Exception as exc:  # transport-level failure is retryable
            raise ProviderError(provider, None, str(exc)) from exc
        if response.status_code >= 400:
            raise ProviderError(provider, response.status_code, response.text[:200])
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    async def stream(self, provider: str, prompt: str, model: str | None):
        import httpx

        base = self.base_urls[provider]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{base.rstrip('/')}/chat/completions",
                    headers=headers,
                    json={
                        "model": model or "default",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                    },
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise ProviderError(
                            provider, response.status_code, body.decode("utf-8", "replace")[:200]
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            yield delta
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(provider, None, str(exc)) from exc