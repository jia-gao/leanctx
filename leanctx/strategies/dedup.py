"""DedupStrategy — drop duplicate messages within a session.

Repeated tool calls with identical arguments produce identical outputs.
Dedup detects this via a hash of each message's text content and drops
all but the first occurrence within the Strategy's lifetime.

The tracker is persistent per-instance: the Middleware constructs a
single DedupStrategy and reuses it across every ``compress_messages``
call, so dedup works across requests within a single Middleware.
"""

from __future__ import annotations

from typing import Any

from leanctx.classifier import RepeatTracker


class DedupStrategy:
    """Drop messages whose text content was seen before."""

    name = "dedup"

    def __init__(self) -> None:
        self._tracker = RepeatTracker()

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [m for m in messages if not self._tracker.is_repeat(m)]

    def reset(self) -> None:
        """Clear seen-content state (e.g. on session boundary)."""
        self._tracker.reset()
