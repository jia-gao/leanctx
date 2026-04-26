"""AC-2 / AC-5 / AC-6: Middleware-level span emission, cost contract, parentage."""

from __future__ import annotations

import asyncio

from leanctx.middleware import Middleware
from leanctx.observability.config import ObservabilityConfig
from leanctx.stats import CompressionStats
from tests.observability.conftest import otel_required


@otel_required
def test_direct_middleware_emits_one_root_compress_span(spans: object) -> None:
    """AC-6 row 2: Middleware.compress_messages with no wrapper opens a root span."""
    obs = ObservabilityConfig(otel=True)
    mw = Middleware({"mode": "off"}, observability=obs)

    mw.compress_messages([{"role": "user", "content": "hi"}])

    finished = spans.get_finished_spans()  # type: ignore[attr-defined]
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes is not None
    assert roots[0].attributes["leanctx.provider"] == "none"


@otel_required
def test_middleware_passthrough_method_when_off(spans: object) -> None:
    """AC-3: mode=off produces leanctx.method=passthrough on the span."""
    obs = ObservabilityConfig(otel=True)
    mw = Middleware({"mode": "off"}, observability=obs)

    mw.compress_messages([{"role": "user", "content": "x"}])
    finished = spans.get_finished_spans()  # type: ignore[attr-defined]
    span = next(s for s in finished if s.name == "leanctx.compress")
    assert span.attributes is not None
    assert span.attributes["leanctx.method"] == "passthrough"


@otel_required
def test_middleware_empty_method_with_no_messages(spans: object) -> None:
    obs = ObservabilityConfig(otel=True)
    mw = Middleware({"mode": "on"}, observability=obs)

    mw.compress_messages([])
    finished = spans.get_finished_spans()  # type: ignore[attr-defined]
    span = next(s for s in finished if s.name == "leanctx.compress")
    assert span.attributes is not None
    assert span.attributes["leanctx.method"] == "empty"


@otel_required
def test_middleware_async_emits_root(spans: object) -> None:
    obs = ObservabilityConfig(otel=True)
    mw = Middleware({"mode": "off"}, observability=obs)
    asyncio.run(mw.compress_messages_async([{"role": "user", "content": "y"}]))
    finished = spans.get_finished_spans()  # type: ignore[attr-defined]
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1


@otel_required
def test_aggregate_threads_cost_through_for_hybrid(spans: object) -> None:
    """AC-5: hybrid call sums constituent costs via _aggregate.

    Regression guard for the v0.2 cost-loss bug. We verify the
    behavior at the unit level by calling _aggregate directly.
    """
    from leanctx.middleware import _aggregate

    stats = _aggregate(
        total_in=100,
        total_out=50,
        total_cost=0.42,
        methods={"selfllm", "verbatim"},
    )
    assert stats.method == "hybrid"
    assert stats.cost_usd == 0.42


@otel_required
def test_aggregate_preserves_single_compressor_cost() -> None:
    from leanctx.middleware import _aggregate

    stats = _aggregate(
        total_in=200,
        total_out=80,
        total_cost=0.0123,
        methods={"selfllm"},
    )
    assert stats.method == "selfllm"
    assert stats.cost_usd == 0.0123


@otel_required
def test_aggregate_zero_cost_for_lingua_or_verbatim() -> None:
    """AC-5: Verbatim/Lingua-only calls produce cost_usd=0.0."""
    from leanctx.middleware import _aggregate

    s_verbatim = _aggregate(100, 100, 0.0, {"verbatim"})
    s_lingua = _aggregate(100, 50, 0.0, {"lingua"})
    assert s_verbatim.cost_usd == 0.0
    assert s_lingua.cost_usd == 0.0


@otel_required
def test_compression_stats_carries_cost_usd_field() -> None:
    """AC-5: CompressionStats has cost_usd; _aggregate threads it."""
    s = CompressionStats(input_tokens=10, output_tokens=5, cost_usd=0.001)
    assert s.cost_usd == 0.001
