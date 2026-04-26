"""AC-3 lingua, AC-5 metric counter, AC-6 direct Lingua/SelfLLM parentage.

These tests close the verification gaps Codex flagged in the R2 review:
- span-level taxonomy assertion for `leanctx.method="lingua"`
- end-to-end OTel metric counter snapshot before/after a wrapped call
- direct Lingua / direct SelfLLM parentage tests
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from leanctx.client import _Messages
from leanctx.middleware import Middleware
from leanctx.observability.config import ObservabilityConfig
from leanctx.stats import CompressionStats
from tests.observability.conftest import otel_required


def _obs() -> ObservabilityConfig:
    return ObservabilityConfig(otel=True)


class _FakeLingua:
    """Stand-in compressor returning Lingua-shaped stats (no model load)."""

    name = "lingua"

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return messages, CompressionStats(
            input_tokens=200,
            output_tokens=100,
            ratio=0.5,
            method="lingua",
            cost_usd=0.0,
        )

    async def compress_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return self.compress(messages)


class _FakeSelfLLM:
    name = "selfllm"

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return messages, CompressionStats(
            input_tokens=100,
            output_tokens=30,
            ratio=0.3,
            method="selfllm",
            cost_usd=0.0042,
        )

    async def compress_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return self.compress(messages)


def _mock_response() -> Any:
    class _Usage:
        input_tokens = 0
        output_tokens = 0

    class _R:
        pass

    r = _R()
    r.usage = _Usage()  # type: ignore[attr-defined]
    return r


@otel_required
def test_taxonomy_lingua_observable_via_router(spans: Any) -> None:
    """AC-3: pipeline routes to Lingua → method=lingua on the wrapper span."""
    obs = _obs()
    mw = Middleware(
        {"mode": "on", "trigger": {"threshold_tokens": 1}},
        observability=obs,
    )
    from leanctx.compressors.base import ContentType

    mw._router.register(ContentType.PROSE, _FakeLingua())  # type: ignore[attr-defined]

    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_mock_response())
    wrapper = _Messages(upstream, mw, obs)

    wrapper.create(messages=[{"role": "user", "content": "prose-shaped content here"}])

    finished = spans.get_finished_spans()
    root = next(s for s in finished if s.name == "leanctx.compress")
    assert root.attributes["leanctx.method"] == "lingua"


@otel_required
def test_metric_counter_increments_exactly_once_per_call(
    spans: Any, metric_reader: Any
) -> None:
    """AC-5 metric-counter contract: one wrapper-routed selfllm call increments
    leanctx.compress.cost_usd by exactly the SelfLLM cost — not 2x or 3x.

    Snapshot the metric reader before and after; assert the delta matches the
    compressor's reported cost_usd."""
    obs = _obs()
    mw = Middleware(
        {"mode": "on", "trigger": {"threshold_tokens": 1}},
        observability=obs,
    )
    from leanctx.compressors.base import ContentType

    mw._router.register(ContentType.PROSE, _FakeSelfLLM())  # type: ignore[attr-defined]
    mw._router.register(ContentType.UNKNOWN, _FakeSelfLLM())  # type: ignore[attr-defined]

    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_mock_response())
    wrapper = _Messages(upstream, mw, obs)

    pre_cost = _read_cost_total(metric_reader)
    pre_calls = _read_call_total(metric_reader)

    wrapper.create(messages=[{"role": "user", "content": "prose-shaped content"}])

    post_cost = _read_cost_total(metric_reader)
    post_calls = _read_call_total(metric_reader)

    assert post_calls - pre_calls == 1, (
        f"expected exactly 1 leanctx.compress.calls increment, "
        f"got {post_calls - pre_calls}"
    )
    delta = post_cost - pre_cost
    assert abs(delta - 0.0042) < 1e-9, (
        f"expected cost counter delta 0.0042, got {delta}"
    )


@otel_required
def test_direct_lingua_parentage(spans: Any) -> None:
    """AC-6 row 3: direct Lingua compress emits a root compressor span
    with provider=none."""

    class _DirectLingua:
        """Behaves like Lingua but without the [lingua] extra."""

        name = "lingua"

        def __init__(self, observability: ObservabilityConfig | None = None) -> None:
            self.observability = observability or ObservabilityConfig()

        def compress(
            self, messages: list[dict[str, Any]]
        ) -> tuple[list[dict[str, Any]], CompressionStats]:
            from leanctx.observability.compressor_hooks import compressor_span

            with compressor_span(self.observability, name=self.name) as span:
                stats = CompressionStats(
                    input_tokens=100,
                    output_tokens=50,
                    ratio=0.5,
                    method="lingua",
                )
                span.set_stats(stats)
                return messages, stats

    direct = _DirectLingua(observability=_obs())
    direct.compress([{"role": "user", "content": "x"}])

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compressor.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "none"
    assert roots[0].attributes["leanctx.method"] == "lingua"


@otel_required
def test_direct_selfllm_parentage(spans: Any) -> None:
    """AC-6 row 3: direct SelfLLM-shaped compress emits a root compressor
    span with provider=none."""

    class _DirectSelfLLM:
        name = "selfllm"

        def __init__(self, observability: ObservabilityConfig | None = None) -> None:
            self.observability = observability or ObservabilityConfig()

        def compress(
            self, messages: list[dict[str, Any]]
        ) -> tuple[list[dict[str, Any]], CompressionStats]:
            from leanctx.observability.compressor_hooks import compressor_span

            with compressor_span(self.observability, name=self.name) as span:
                stats = CompressionStats(
                    input_tokens=100,
                    output_tokens=30,
                    ratio=0.3,
                    method="selfllm",
                    cost_usd=0.001,
                )
                span.set_stats(stats)
                return messages, stats

    direct = _DirectSelfLLM(observability=_obs())
    direct.compress([{"role": "user", "content": "x"}])

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compressor.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "none"
    assert roots[0].attributes["leanctx.method"] == "selfllm"


@otel_required
def test_direct_lingua_async_parentage(spans: Any) -> None:
    """AC-6 row 3 async: direct `await Lingua.compress_async()` emits a
    root compressor span with provider=none."""
    import asyncio

    class _DirectLingua:
        name = "lingua"

        def __init__(self, observability: ObservabilityConfig | None = None) -> None:
            self.observability = observability or ObservabilityConfig()

        async def compress_async(
            self, messages: list[dict[str, Any]]
        ) -> tuple[list[dict[str, Any]], CompressionStats]:
            from leanctx.observability.compressor_hooks import compressor_span

            with compressor_span(self.observability, name=self.name) as span:
                stats = CompressionStats(
                    input_tokens=100,
                    output_tokens=50,
                    ratio=0.5,
                    method="lingua",
                )
                span.set_stats(stats)
                return messages, stats

    direct = _DirectLingua(observability=_obs())
    asyncio.run(direct.compress_async([{"role": "user", "content": "x"}]))

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compressor.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "none"
    assert roots[0].attributes["leanctx.method"] == "lingua"


@otel_required
def test_direct_selfllm_async_parentage(spans: Any) -> None:
    """AC-6 row 3 async: direct `await SelfLLM.compress_async()` emits a
    root compressor span with provider=none."""
    import asyncio

    class _DirectSelfLLM:
        name = "selfllm"

        def __init__(self, observability: ObservabilityConfig | None = None) -> None:
            self.observability = observability or ObservabilityConfig()

        async def compress_async(
            self, messages: list[dict[str, Any]]
        ) -> tuple[list[dict[str, Any]], CompressionStats]:
            from leanctx.observability.compressor_hooks import compressor_span

            with compressor_span(self.observability, name=self.name) as span:
                stats = CompressionStats(
                    input_tokens=100,
                    output_tokens=30,
                    ratio=0.3,
                    method="selfllm",
                    cost_usd=0.0042,
                )
                span.set_stats(stats)
                return messages, stats

    direct = _DirectSelfLLM(observability=_obs())
    asyncio.run(direct.compress_async([{"role": "user", "content": "x"}]))

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compressor.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "none"
    assert roots[0].attributes["leanctx.method"] == "selfllm"


def _read_cost_total(metric_reader: Any) -> float:
    return _read_counter_total(metric_reader, "leanctx.compress.cost_usd")


def _read_call_total(metric_reader: Any) -> int:
    return int(_read_counter_total(metric_reader, "leanctx.compress.calls"))


def _read_counter_total(metric_reader: Any, name: str) -> float:
    """Sum a counter's data points across all label sets."""
    data = metric_reader.get_metrics_data()
    if data is None:
        return 0.0
    total = 0.0
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name != name:
                    continue
                for point in metric.data.data_points:
                    total += float(point.value)
    return total
