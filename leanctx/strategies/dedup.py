"""DedupStrategy — drop duplicate messages *within a single request*.

Agent conversation histories often accumulate identical tool outputs —
the same ``grep`` query, the same status check, the same retrieved doc
surfaced by two different retrieval paths. DedupStrategy walks each
request's message list and keeps only the first occurrence of any
given text content.

**Scope is per-call, not per-client.** Earlier iterations held a
persistent tracker on the Strategy instance, which caused identical
user prompts across independent requests to be silently dropped — a
correctness bug. Each call to :meth:`apply` now uses a fresh tracker,
so this strategy is always safe to enable as a default.

Users who want across-request deduplication (e.g. to drop recomputed
embeddings that are genuinely redundant across a whole session) should
implement that at their session layer where the session boundary is
explicit.
"""

from __future__ import annotations

from typing import Any

from leanctx.classifier import RepeatTracker


class DedupStrategy:
    """Drop messages whose text content already appeared earlier in the same list."""

    name = "dedup"

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tracker = RepeatTracker()
        return [m for m in messages if not tracker.is_repeat(m)]

    def reset(self) -> None:
        """No-op. State is per-call; preserved for backward compat."""
        return None
