"""Shared fixtures for observability tests using OTel SDK in-memory exporters.

Tests in this directory verify span shape and metric values without
requiring a real OTLP collector. We register a process-global
TracerProvider + MeterProvider on session start (just like an
application would) and provide span-list / metric-snapshot fixtures
per test.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

try:
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        InMemoryMetricReader,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


otel_required = pytest.mark.skipif(
    not _OTEL_AVAILABLE,
    reason="opentelemetry-sdk not installed; run `pip install 'leanctx[otel]'`",
)


@pytest.fixture(scope="session", autouse=False)
def otel_setup() -> Iterator[dict[str, Any]]:
    """Configure global OTel providers once per session.

    Yields the in-memory span exporter and metric reader so tests can
    snapshot state. Tests should call ``span_exporter.clear()`` at
    the start to isolate themselves from previous tests.
    """
    if not _OTEL_AVAILABLE:
        yield {}
        return

    from leanctx.observability import api as leanctx_api
    from leanctx.observability import metrics as leanctx_metrics

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Force leanctx to re-probe so it picks up the providers.
    leanctx_api._reset_for_tests()
    leanctx_metrics._reset_for_tests()

    yield {
        "span_exporter": span_exporter,
        "metric_reader": metric_reader,
        "tracer_provider": tracer_provider,
        "meter_provider": meter_provider,
    }


@pytest.fixture
def spans(otel_setup: dict[str, Any]) -> Iterator[Any]:
    """Yield the span exporter; clears it at fixture entry."""
    if not otel_setup:
        pytest.skip("opentelemetry-sdk not installed")
    span_exporter = otel_setup["span_exporter"]
    span_exporter.clear()
    yield span_exporter


@pytest.fixture
def metric_reader(otel_setup: dict[str, Any]) -> Any:
    if not otel_setup:
        pytest.skip("opentelemetry-sdk not installed")
    return otel_setup["metric_reader"]
