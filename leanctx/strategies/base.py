"""Strategy protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Strategy(Protocol):
    """A deterministic message-span filter.

    Strategies run as a pre-compression pipeline: each takes a list of
    messages and returns a filtered/mutated list. They're ordered, so
    ``[Dedup, PurgeErrors]`` means dedup first. All strategies must be
    cheap (no LLM calls, no model loads) — they're invoked on every
    request, possibly at high QPS.
    """

    name: str

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ...
