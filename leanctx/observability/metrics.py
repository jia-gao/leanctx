"""Lazy registration of the four counters + one histogram.

Metric names follow the documented schema in ``docs/observability.md``:

    leanctx.compress.calls          (counter, requests)
    leanctx.compress.input_tokens   (counter, tokens)
    leanctx.compress.output_tokens  (counter, tokens)
    leanctx.compress.cost_usd       (counter, USD)
    leanctx.compress.duration_ms    (histogram, milliseconds)

All five carry ``provider``, ``method``, and ``status`` attributes when
recorded. The instruments are created on first access using the
application's globally-configured ``MeterProvider``; if no MeterProvider
is configured, the OTel API returns no-op instruments and recording is
a fast no-op.

When ``opentelemetry-api`` is not installed, every accessor returns
``None`` and callers should branch on the return value.
"""

from __future__ import annotations

from typing import Any

from leanctx.observability import api

_calls: Any = None
_input_tokens: Any = None
_output_tokens: Any = None
_cost_usd: Any = None
_duration_ms: Any = None
_initialized: bool = False


def _init() -> bool:
    """Create the five instruments on first call. Idempotent."""
    global _calls, _input_tokens, _output_tokens, _cost_usd, _duration_ms, _initialized
    if _initialized:
        return _calls is not None
    meter = api.get_meter()
    if meter is None:
        _initialized = True
        return False
    _calls = meter.create_counter(
        "leanctx.compress.calls",
        unit="1",
        description="Compression calls handled by leanctx.",
    )
    _input_tokens = meter.create_counter(
        "leanctx.compress.input_tokens",
        unit="tokens",
        description="Input tokens observed by leanctx before compression.",
    )
    _output_tokens = meter.create_counter(
        "leanctx.compress.output_tokens",
        unit="tokens",
        description="Output tokens emitted by leanctx after compression.",
    )
    _cost_usd = meter.create_counter(
        "leanctx.compress.cost_usd",
        unit="USD",
        description="Cumulative compression cost in USD (best-effort; "
        "depends on Compressor reporting cost_usd).",
    )
    _duration_ms = meter.create_histogram(
        "leanctx.compress.duration_ms",
        unit="ms",
        description="End-to-end leanctx compression duration in milliseconds.",
    )
    _initialized = True
    return True


def record(
    *,
    provider: str,
    method: str,
    status: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    duration_ms: float,
) -> None:
    """Record one compression call across all five instruments.

    No-op when ``opentelemetry-api`` is not installed. Cost is recorded
    only when ``cost_usd > 0`` so a string of Verbatim/Lingua calls
    leaves the cost counter's running total unchanged (per AC-5).
    """
    if not _init():
        return
    attrs = {"provider": provider, "method": method, "status": status}
    _calls.add(1, attrs)
    if input_tokens:
        _input_tokens.add(input_tokens, attrs)
    if output_tokens:
        _output_tokens.add(output_tokens, attrs)
    if cost_usd > 0.0:
        _cost_usd.add(cost_usd, attrs)
    _duration_ms.record(duration_ms, attrs)


def _reset_for_tests() -> None:
    """Test-only hook: clear cached instrument handles."""
    global _calls, _input_tokens, _output_tokens, _cost_usd, _duration_ms, _initialized
    _calls = None
    _input_tokens = None
    _output_tokens = None
    _cost_usd = None
    _duration_ms = None
    _initialized = False
