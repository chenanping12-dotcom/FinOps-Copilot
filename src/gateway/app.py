"""FastAPI surface for the gateway: chat, SSE streaming, cache stats and health."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, AsyncIterator

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - surfaced only when extras are missing
    raise RuntimeError(
        "FastAPI and pydantic are required for the gateway app; install the [gateway] extra"
    ) from exc

from .client import (
    AllProvidersFailed,
    ClientDisconnected,
    LLMClient,
    SemanticCache,
)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str | None = None
    stream: bool = False


def create_app(client: LLMClient, cache: SemanticCache | None = None) -> FastAPI:
    app = FastAPI(title="FinOps-Copilot Gateway", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "providers": client.providers,
            "cache_entries": len(cache) if cache is not None else 0,
            "cache_hit_rate": round(cache.hit_rate, 4) if cache is not None else None,
        }

    @app.get("/cache/stats")
    async def cache_stats() -> dict[str, Any]:
        if cache is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "entries": len(cache),
            "hits": cache.hits,
            "misses": cache.misses,
            "hit_rate": round(cache.hit_rate, 4),
            "threshold": cache.threshold,
        }

    @app.post("/chat")
    async def chat(request: ChatRequest) -> Any:
        disconnected = _disconnect_probe(request)
        if request.stream:
            return StreamingResponse(
                _sse(client, request.prompt, request.model, disconnected),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            result = await client.complete(
                request.prompt, model=request.model, disconnected=disconnected
            )
        except ClientDisconnected as exc:
            return JSONResponse({"error": "client_disconnected", "detail": str(exc)}, status_code=499)
        except AllProvidersFailed as exc:
            return JSONResponse(
                {
                    "error": "all_providers_failed",
                    "attempts": [attempt.as_dict() for attempt in exc.attempts],
                },
                status_code=502,
            )
        return result.as_dict()

    return app


def _disconnect_probe(request: Request):
    """Return a sync callable reporting whether the client has gone away.

    Forwarding a disconnect into the retry loop has to happen *synchronously*:
    `LLMClient.complete` takes a `Callable[[], bool]` and is already inside the
    event loop, so there is no await point available between attempts. That rules
    out simply awaiting `Request.is_disconnected()` at the call site.

    It also rules out every way of driving that coroutine from sync code, which
    is worth spelling out because each one fails differently and two of them fail
    *silently*:

    * `loop.run_until_complete(...)` raises `RuntimeError: This event loop is
      already running` (measured). Catching that and calling `asyncio.run()`
      instead cannot work either -- there is still a loop running.
    * `asyncio.run_coroutine_threadsafe(...)` + `future.result()` deadlocks when
      called from the loop thread: the thread is parked in `result()`, so it can
      never run the coroutine it is waiting on. Measured at 1007 ms per request
      -- the full timeout -- before falling back to False.

    Both of those end up returning a constant `False`, which is exactly the bug
    this function exists to fix, so neither is used here.

    Instead we read the state we need without awaiting anything. The single fact
    that makes this possible, verified against Starlette 1.9.0: `stream()` sets
    `self._is_disconnected = True` the moment it observes `http.disconnect`, and
    `is_disconnected()` reads that same cached attribute. So if the body read
    already saw the disconnect, the flag is set and we can read it synchronously.

    Reading the flag is a snapshot, not a poll: if the client disconnects after we
    look, we miss it and simply retry once more. That is the deliberate failure
    direction -- a false negative costs a wasted retry, while a false positive
    would abort a response for a client that is still listening.

    There is no message buffer to fall back on: Starlette 1.9.0 has no
    `_CachedRequest` and `Request` keeps no record of received messages, so
    scanning for a stored `http.disconnect` finds nothing and silently reports
    "connected". Only the flag is trustworthy.
    """
    if not hasattr(request, "is_disconnected"):
        return lambda: False

    def disconnected() -> bool:
        return bool(getattr(request, "_is_disconnected", False))

    return disconnected


async def _sse(
    client: LLMClient,
    prompt: str,
    model: str | None,
    disconnected,
) -> AsyncIterator[str]:
    """Emit SSE frames; a pre-first-token failure degrades to the next provider."""
    try:
        first_index: int | None = None
        async for delta, provider_index in client.stream(
            prompt, model=model, disconnected=disconnected
        ):
            if first_index is None:
                first_index = provider_index
                if provider_index > 0:
                    yield _frame(
                        "degraded",
                        {
                            "served_by": client.providers[provider_index],
                            "skipped": client.providers[:provider_index],
                        },
                    )
            yield _frame("delta", {"text": delta})
        yield _frame(
            "done",
            {
                "provider": client.last_stream_provider,
                "degraded": client.last_stream_degraded,
            },
        )
    except ClientDisconnected as exc:
        yield _frame("error", {"error": "client_disconnected", "detail": str(exc)})
    except AllProvidersFailed as exc:
        yield _frame(
            "error",
            {
                "error": "all_providers_failed",
                "attempts": [attempt.as_dict() for attempt in exc.attempts],
            },
        )


def _frame(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"