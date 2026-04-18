"""Verbatim — the no-op compressor.

Used when content must not be altered: code, diffs, stack traces, tool
schemas, short messages that wouldn't compress well. Also the safe
default when the classifier emits ``UNKNOWN``.

Returns input unchanged with accurate input/output token counts so
telemetry still reports meaningful totals (ratio is always 1.0).
"""

from __future__ import annotations

from typing import Any

from leanctx.stats import CompressionStats
from leanctx.tokens import count_message_tokens


class Verbatim:
    """A pass-through Compressor that preserves messages exactly."""

    name = "verbatim"

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        tokens = count_message_tokens(messages)
        return messages, CompressionStats(
            input_tokens=tokens,
            output_tokens=tokens,
            ratio=1.0,
            method="verbatim",
        )

    async def compress_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return self.compress(messages)
