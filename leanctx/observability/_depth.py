"""Shared depth counter for compression / compressor spans.

A single ``ContextVar[int]`` tracks how many leanctx span frames are
currently open in the current async/thread context. Both
``compression_span`` and ``compressor_span`` increment on enter and
decrement on exit; their **emission rules** differ (see the AC-6
asymmetric-emission rule in plan-otel.md), but they share the counter.
"""

from __future__ import annotations

from contextvars import ContextVar

DEPTH: ContextVar[int] = ContextVar("leanctx_span_depth", default=0)
