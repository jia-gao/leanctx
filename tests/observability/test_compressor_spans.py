"""AC-3 / AC-4 / AC-6: direct compressor calls emit span trees correctly."""

from __future__ import annotations

import asyncio

from leanctx.compressors import Verbatim
from leanctx.observability.config import ObservabilityConfig
from tests.observability.conftest import otel_required


@otel_required
def test_direct_verbatim_emits_root_span_with_provider_none(spans: object) -> None:
    """AC-6 row 3: direct compressor call → one root span with provider=none."""
    obs = ObservabilityConfig(otel=True)
    v = Verbatim(observability=obs)

    out, stats = v.compress([{"role": "user", "content": "hello world"}])

    assert stats.method == "verbatim"

    finished = spans.get_finished_spans()  # type: ignore[attr-defined]
    assert len(finished) == 1, f"expected 1 root span, got {len(finished)}"
    span = finished[0]
    assert span.name == "leanctx.compressor.compress"
    assert span.attributes is not None
    assert span.attributes["leanctx.provider"] == "none"
    assert span.attributes["leanctx.compressor.name"] == "verbatim"
    assert span.attributes["leanctx.method"] == "verbatim"


@otel_required
def test_direct_verbatim_async_emits_root_span(spans: object) -> None:
    """AC-6 row 3 async path."""
    obs = ObservabilityConfig(otel=True)
    v = Verbatim(observability=obs)

    asyncio.run(v.compress_async([{"role": "user", "content": "hi"}]))

    finished = spans.get_finished_spans()  # type: ignore[attr-defined]
    assert len(finished) == 1
    assert finished[0].name == "leanctx.compressor.compress"


@otel_required
def test_observability_disabled_emits_no_spans(spans: object) -> None:
    """When ObservabilityConfig.otel is False, no spans are emitted."""
    obs = ObservabilityConfig(otel=False)
    v = Verbatim(observability=obs)

    v.compress([{"role": "user", "content": "hello"}])

    finished = spans.get_finished_spans()  # type: ignore[attr-defined]
    assert len(finished) == 0
