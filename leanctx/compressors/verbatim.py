"""Verbatim — the no-op compressor.

Used when content must not be altered: code, diffs, stack traces, tool
schemas, short messages that wouldn't compress well. Also the safe
default when the classifier emits ``UNKNOWN``.

Returns input unchanged with accurate input/output token counts so
telemetry still reports meaningful totals (ratio is always 1.0).
"""

from __future__ import annotations

from typing import Any

from leanctx.observability.compressor_hooks import compressor_span
from leanctx.observability.config import ObservabilityConfig
from leanctx.stats import CompressionStats
from leanctx.tokens import count_message_tokens


class Verbatim:
    """A pass-through Compressor that preserves messages exactly."""

    name = "verbatim"

    def __init__(
        self,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self.observability = observability or ObservabilityConfig()

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        with compressor_span(self.observability, name=self.name) as span:
            tokens = count_message_tokens(messages)
            stats = CompressionStats(
                input_tokens=tokens,
                output_tokens=tokens,
                ratio=1.0,
                method="verbatim",
            )
            span.set_stats(stats)
            return messages, stats

    async def compress_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return self.compress(messages)
