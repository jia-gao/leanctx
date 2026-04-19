"""Middleware — provider-agnostic compression orchestrator.

The Middleware is what the SDK wrappers call. It reads config, decides
whether to compress, and runs the pipeline:

    1. Check mode (off → passthrough)
    2. Check token budget (below threshold → passthrough)
    3. For each message:
         a. Drop it if RepeatTracker has seen this content before
         b. classify() to pick a ContentType
         c. Router picks a Compressor
         d. Compressor.compress([msg]) returns compressed form
    4. Aggregate stats and return

In v0.0.x the only Compressor implementation is :class:`Verbatim`, so
even with ``mode: "on"`` the messages come through unchanged — but
through the real pipeline, with real stats. v0.1 plugs Lingua / SelfLLM
in via config without touching this file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from leanctx.classifier import RepeatTracker, classify
from leanctx.compressors import Compressor, ContentType, Lingua, SelfLLM, Verbatim
from leanctx.router import Router
from leanctx.stats import CompressionStats
from leanctx.tokens import count_message_tokens

__all__ = ["CompressionStats", "Middleware"]

_log = logging.getLogger(__name__)

_DEFAULT_THRESHOLD_TOKENS = 2000

# Factories for Compressor names that appear in routing config. Each
# factory accepts a per-compressor config dict (e.g. config["lingua"])
# and returns an instance. Missing keys fall back to each compressor's
# own defaults.
_COMPRESSOR_FACTORIES: dict[str, Callable[[dict[str, Any]], Compressor]] = {
    "verbatim": lambda _cfg: Verbatim(),
    "lingua": lambda cfg: Lingua(
        model=cfg.get("model", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"),
        ratio=cfg.get("ratio", 0.5),
        device=cfg.get("device"),
    ),
    "selfllm": lambda cfg: SelfLLM(
        provider=cfg.get("provider", "anthropic"),
        model=cfg.get("model", "claude-haiku-4-5"),
        api_key=cfg.get("api_key"),
        ratio=cfg.get("ratio", 0.3),
        max_summary_tokens=cfg.get("max_summary_tokens", 500),
    ),
}


class Middleware:
    """Orchestrates compression across a single SDK call."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        mode = str(config.get("mode", "off")).lower()
        self._active = mode not in ("off", "passthrough", "disabled")

        trigger_cfg = config.get("trigger") or {}
        self._threshold = int(trigger_cfg.get("threshold_tokens", _DEFAULT_THRESHOLD_TOKENS))

        self._router = self._build_router(config)

        # One tracker per Middleware instance ≈ one tracker per client. v0.1
        # may move this to a session-scoped store so repeat dedup works
        # across multiple requests in the same logical session.
        self._repeat: RepeatTracker | None = RepeatTracker() if self._active else None

    # ----------------------------------------------------------------- #
    # Public sync / async entry points
    # ----------------------------------------------------------------- #

    def compress_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        if not self._active or not messages:
            return messages, CompressionStats()

        input_tokens = count_message_tokens(messages)
        if input_tokens < self._threshold:
            return messages, CompressionStats(
                input_tokens=input_tokens,
                output_tokens=input_tokens,
                ratio=1.0,
                method="below-threshold",
            )

        out_msgs: list[dict[str, Any]] = []
        total_in = 0
        total_out = 0
        methods: set[str] = set()

        for msg in messages:
            if self._repeat is not None and self._repeat.is_repeat(msg):
                # Drop the repeat entirely. Token savings come from the
                # simple fact that the duplicate never reaches the LLM.
                continue
            ctype = classify(msg)
            compressor = self._router.route(ctype)
            compressed, stats = compressor.compress([msg])
            out_msgs.extend(compressed)
            total_in += stats.input_tokens
            total_out += stats.output_tokens
            methods.add(stats.method)

        return out_msgs, _aggregate(total_in, total_out, methods)

    async def compress_messages_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        if not self._active or not messages:
            return messages, CompressionStats()

        input_tokens = count_message_tokens(messages)
        if input_tokens < self._threshold:
            return messages, CompressionStats(
                input_tokens=input_tokens,
                output_tokens=input_tokens,
                ratio=1.0,
                method="below-threshold",
            )

        out_msgs: list[dict[str, Any]] = []
        total_in = 0
        total_out = 0
        methods: set[str] = set()

        for msg in messages:
            if self._repeat is not None and self._repeat.is_repeat(msg):
                continue
            ctype = classify(msg)
            compressor = self._router.route(ctype)
            compressed, stats = await compressor.compress_async([msg])
            out_msgs.extend(compressed)
            total_in += stats.input_tokens
            total_out += stats.output_tokens
            methods.add(stats.method)

        return out_msgs, _aggregate(total_in, total_out, methods)

    # ----------------------------------------------------------------- #
    # Internal wiring
    # ----------------------------------------------------------------- #

    def _build_router(self, config: dict[str, Any]) -> Router:
        router = Router(default=Verbatim())
        routing = config.get("routing") or {}
        for ctype_str, compressor_name in routing.items():
            try:
                ctype = ContentType(ctype_str)
            except ValueError:
                _log.warning("Unknown content type %r in routing config; ignored", ctype_str)
                continue
            factory = _COMPRESSOR_FACTORIES.get(compressor_name)
            if factory is None:
                # Forward-compatible: a config referencing a compressor name
                # we haven't implemented yet (e.g. 'semantic' in the future)
                # falls back to default Verbatim silently with a log warning.
                _log.warning(
                    "Compressor %r not available in this leanctx version; "
                    "falling back to default (verbatim) for %s",
                    compressor_name,
                    ctype_str,
                )
                continue
            # Per-compressor config lives under its own key, e.g.
            # {"lingua": {...}, "selfllm": {...}}. Missing key -> {}.
            compressor_cfg = config.get(compressor_name) or {}
            router.register(ctype, factory(compressor_cfg))
        return router


def _aggregate(total_in: int, total_out: int, methods: set[str]) -> CompressionStats:
    if not methods:
        method = "empty"
    elif len(methods) == 1:
        method = next(iter(methods))
    else:
        method = "hybrid"
    return CompressionStats(
        input_tokens=total_in,
        output_tokens=total_out,
        ratio=(total_out / total_in) if total_in else 1.0,
        method=method,
    )
