"""compressor_span — always-emit span for direct compressor calls.

Unlike :func:`leanctx.observability.middleware_hooks.compression_span`,
``compressor_span`` emits a ``leanctx.compressor.compress`` span on
every entry: at depth 0 it becomes a root, at depth > 0 it becomes a
child of the current OTel span context. This is what gives users
per-compressor granularity inside wrapper / middleware traces while
keeping direct-compressor calls observable.

Inner compressor spans carry only per-compressor attributes (`name`,
`input_tokens`, `output_tokens`, `ratio`, `cost_usd`); they do NOT
re-emit ``leanctx.provider``. The outermost span owns provider /
method / cost (see AC-6 attribute ownership rule).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from leanctx.observability import api, metrics
from leanctx.observability._depth import DEPTH
from leanctx.observability.config import ObservabilityConfig
from leanctx.stats import CompressionStats


class _NoopCompressorSpan:
    is_recording = False

    def set_stats(self, stats: CompressionStats) -> None:
        return None

    def set_error(self, exc: BaseException) -> None:
        return None


class _CompressorSpan:
    """Per-compressor span. Always emits when observability is enabled."""

    is_recording = True

    def __init__(
        self,
        otel_span: Any,
        *,
        name: str,
        is_root: bool,
        config: ObservabilityConfig,
        start_ns: int,
    ) -> None:
        self._span = otel_span
        self._name = name
        self._is_root = is_root
        self._config = config
        self._start_ns = start_ns
        self._method: str = name
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cost_usd: float = 0.0
        self._set("leanctx.compressor.name", name)
        if is_root:
            # Direct-compressor (root) spans carry provider=none per AC-4.
            self._set("leanctx.provider", "none")
            for k, v in (config.extra_attributes or {}).items():
                self._set(k, v)

    @property
    def is_root(self) -> bool:
        return self._is_root

    def _set(self, key: str, value: Any) -> None:
        setter = getattr(self._span, "set_attribute", None)
        if setter is not None:
            setter(key, value)

    def set_stats(self, stats: CompressionStats) -> None:
        self._method = stats.method or self._name
        self._input_tokens = int(stats.input_tokens)
        self._output_tokens = int(stats.output_tokens)
        self._cost_usd = float(stats.cost_usd)
        self._set("leanctx.method", self._method)
        self._set("leanctx.input_tokens", self._input_tokens)
        self._set("leanctx.output_tokens", self._output_tokens)
        self._set("leanctx.ratio", float(stats.ratio))
        self._set("leanctx.cost_usd", self._cost_usd)

    def set_error(self, exc: BaseException) -> None:
        self._set("leanctx.error", True)
        self._set("error.type", type(exc).__name__)
        self._set("error.message", str(exc))
        try:
            from opentelemetry.trace import Status, StatusCode  # noqa: PLC0415

            setter = getattr(self._span, "set_status", None)
            if setter is not None:
                setter(Status(StatusCode.ERROR, str(exc)))
        except Exception:
            pass

    def record_metrics(self) -> None:
        if not self._is_root:
            return
        duration_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000.0
        self._set("leanctx.duration_ms", duration_ms)
        metrics.record(
            provider="none",
            method=self._method,
            status="success",
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=self._cost_usd,
            duration_ms=duration_ms,
        )


@contextmanager
def compressor_span(
    observability: ObservabilityConfig | None,
    *,
    name: str,
) -> Iterator[_CompressorSpan | _NoopCompressorSpan]:
    """Enter a per-compressor span.

    Always emits a ``leanctx.compressor.compress`` span when
    observability is enabled. Becomes a child when nested under
    ``compression_span`` (or another ``compressor_span``); becomes a
    root when called directly by the user.
    """
    if observability is None or not observability.enabled:
        yield _NoopCompressorSpan()
        return

    parent_depth = DEPTH.get()
    token = DEPTH.set(parent_depth + 1)
    try:
        tracer = api.get_tracer()
        if tracer is None:
            yield _NoopCompressorSpan()
            return

        start_ns = time.perf_counter_ns()
        with tracer.start_as_current_span("leanctx.compressor.compress") as otel_span:
            span = _CompressorSpan(
                otel_span,
                name=name,
                is_root=(parent_depth == 0),
                config=observability,
                start_ns=start_ns,
            )
            try:
                yield span
            except BaseException as exc:
                span.set_error(exc)
                span.record_metrics()
                raise
            else:
                span.record_metrics()
    finally:
        DEPTH.reset(token)
