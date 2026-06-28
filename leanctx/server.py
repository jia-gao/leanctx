"""leanctx HTTP compression service.

A minimal long-lived HTTP sidecar that exposes leanctx's compression
``Middleware`` over ``POST /compress``. It exists so non-Python
consumers — a TypeScript proxy (e.g. ClawRouter), a LiteLLM callback,
any cross-language caller — can use leanctx's LLMLingua-2 / SelfLLM
compression without embedding a Python runtime.

The service is opt-in (``pip install 'leanctx[server]'``) and never
imported by ``import leanctx`` — so the core library's cold-import
budget is unaffected.

Run it::

    pip install 'leanctx[server,lingua]'
    leanctx-serve --host 127.0.0.1 --port 8459        # console script
    # or:
    uvicorn leanctx.server:app --host 127.0.0.1 --port 8459 --workers 1

Then::

    curl -s localhost:8459/compress -H 'content-type: application/json' \
      -d '{"messages":[{"role":"user","content":"<long prose>"}]}'

Design notes (the safe defaults matter):

- ``mode`` defaults to ``"on"`` — unlike the library default of
  ``"off"`` — because a compression service that doesn't compress is
  useless. Override with ``LEANCTX_SERVER_MODE=off`` for a passthrough
  smoke test (no model load).
- ``trigger.threshold_tokens`` defaults to **1500, not 0**. Compressing
  very short turns (system prompts, terse instructions) risks dropping
  load-bearing tokens; the threshold protects them. The HOST should
  also gate at the request level (only call this for large requests).
- ``strategies.dedup`` defaults **off**. A fronting proxy (ClawRouter's
  Layer 1, LiteLLM, etc.) usually owns deduplication; running it twice
  is wasted work. Re-enable via config if leanctx is the only layer.
- The service **does not rewrite** ``role == "tool"`` messages or any
  non-string (multimodal ``content`` list) message — it forwards them
  verbatim. Tool results and image parts must not be lossily rewritten;
  callers that want tool-result compression should route those
  explicitly. This mirrors the message-shape contract an OpenAI-format
  proxy expects (``role`` ∈ {system,user,assistant,tool}).
- A startup warmup pays the one-time ~1.2 GB LLMLingua-2 model load
  before the first real request, so request latency is steady-state.
"""

from __future__ import annotations

import json
import os
from typing import Any

from leanctx.middleware import Middleware

# Roles whose string content is eligible for compression. ``tool``
# messages (tool results) are forwarded verbatim — they often carry
# structured payloads a lossy prose pass must not touch.
_COMPRESSIBLE_ROLES = frozenset({"system", "user", "assistant"})


def _default_config() -> dict[str, Any]:
    """Build the Middleware config from environment, with safe defaults.

    Env overrides:
        LEANCTX_SERVER_MODE          on | off            (default: on)
        LEANCTX_SERVER_THRESHOLD     int tokens          (default: 1500)
        LEANCTX_SERVER_ROUTING       json object         (default: {"prose":"lingua"})
        LEANCTX_SERVER_LINGUA_RATIO  float 0..1           (default: 0.5)
        LEANCTX_SERVER_STRUCTURED_RATIO float 0..1        (default: 0.8, gentler)
        LEANCTX_SERVER_LINGUA_DEVICE cpu|cuda|mps|''      (default: auto)
        LEANCTX_SERVER_DEDUP         on | off            (default: off)
        LEANCTX_SERVER_CONFIG        full json config (wins over the above)
    """
    raw_full = os.environ.get("LEANCTX_SERVER_CONFIG")
    if raw_full:
        parsed = json.loads(raw_full)
        if not isinstance(parsed, dict):
            raise ValueError("LEANCTX_SERVER_CONFIG must be a JSON object")
        return parsed

    mode = os.environ.get("LEANCTX_SERVER_MODE", "on")
    threshold = int(os.environ.get("LEANCTX_SERVER_THRESHOLD", "1500"))
    routing_raw = os.environ.get("LEANCTX_SERVER_ROUTING")
    routing = json.loads(routing_raw) if routing_raw else {"prose": "lingua"}
    ratio = float(os.environ.get("LEANCTX_SERVER_LINGUA_RATIO", "0.5"))
    structured_ratio = float(os.environ.get("LEANCTX_SERVER_STRUCTURED_RATIO", "0.8"))
    device_env = os.environ.get("LEANCTX_SERVER_LINGUA_DEVICE", "")
    device = device_env or None
    dedup = os.environ.get("LEANCTX_SERVER_DEDUP", "off").lower() in ("on", "true", "1")

    return {
        "mode": mode,
        "trigger": {"threshold_tokens": threshold},
        "routing": routing,
        "lingua": {"ratio": ratio, "structured_ratio": structured_ratio, "device": device},
        "strategies": {"dedup": dedup, "purge_errors": False},
    }


def _split_compressible(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Return (compressible_subset, original_indices).

    Only messages whose ``role`` is compressible AND whose ``content``
    is a plain string are eligible. Everything else (tool results,
    multimodal content lists, null content) is left untouched.
    """
    subset: list[dict[str, Any]] = []
    indices: list[int] = []
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        if role in _COMPRESSIBLE_ROLES and isinstance(content, str):
            subset.append(msg)
            indices.append(i)
    return subset, indices


class CompressionService:
    """Holds a single warm Middleware and applies it over HTTP requests."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config if config is not None else _default_config()
        self._middleware = Middleware(self._config)

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    async def warmup(self) -> None:
        """Force the one-time model load (LLMLingua-2 weights are lazy).

        Sends a synthetic over-threshold prose message so the Lingua
        path actually loads its model, rather than short-circuiting on
        the token gate.
        """
        await self._middleware.compress_messages_async(
            [{"role": "user", "content": "warmup " * 3000}]
        )

    async def compress(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Compress the eligible subset; splice results back by index."""
        subset, indices = _split_compressible(messages)
        if not subset:
            return {
                "messages": messages,
                "stats": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "ratio": 1.0,
                    "method": "passthrough",
                    "cost_usd": 0.0,
                },
                "compressed_message_count": 0,
            }

        compressed, stats = await self._middleware.compress_messages_async(subset)

        # The middleware preserves message count (one-in-one-out per
        # message), so we can splice by index. Guard against a length
        # change and fall back to the originals if the invariant breaks.
        out = list(messages)
        if len(compressed) == len(indices):
            for idx, new_msg in zip(indices, compressed, strict=True):
                out[idx] = new_msg

        return {
            "messages": out,
            "stats": {
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "ratio": stats.ratio,
                "method": stats.method,
                "cost_usd": stats.cost_usd,
            },
            "compressed_message_count": len(indices),
        }


def create_app(config: dict[str, Any] | None = None) -> Any:
    """Build the FastAPI app. Imported lazily so the [server] extra is
    only required when actually serving."""
    try:
        import contextlib
        from contextlib import asynccontextmanager

        from fastapi import Body, FastAPI
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "the [server] extra is required to run the leanctx HTTP service. "
            "Install with: pip install 'leanctx[server]'"
        ) from exc

    service = CompressionService(config)

    @asynccontextmanager
    async def _lifespan(_app: Any) -> Any:  # pragma: no cover - runtime only
        # Pay the one-time model load before serving, but only when a
        # real compressor is configured; mode=off needs no model. A
        # warmup failure (e.g. [lingua] not installed) must not block
        # startup — the first real call surfaces the error instead.
        if str(service.config.get("mode", "off")).lower() in ("on", "true", "1"):
            with contextlib.suppress(Exception):
                await service.warmup()
        yield

    app = FastAPI(title="leanctx compression service", version="1", lifespan=_lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "mode": service.config.get("mode")}

    @app.post("/compress")
    async def compress(body: dict[str, Any] = Body(...)) -> dict[str, Any]:  # noqa: B008
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        return await service.compress(messages)

    return app


# Module-level app for ``uvicorn leanctx.server:app``. Built lazily-ish:
# constructing it builds a Middleware (cheap; no model load until a
# request or warmup).
try:
    app = create_app()
except RuntimeError:  # pragma: no cover - [server] extra absent
    app = None


def main() -> int:
    """Console-script entry point: ``leanctx-serve``."""
    import argparse

    parser = argparse.ArgumentParser(prog="leanctx-serve", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8459)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "the [server] extra is required: pip install 'leanctx[server]'",
            flush=True,
        )
        return 1

    # Surface leanctx INFO logs (e.g. the resolved Lingua device) alongside
    # uvicorn's own output; without this the leanctx.* loggers stay silent.
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    uvicorn.run(
        "leanctx.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
