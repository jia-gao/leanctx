"""compression_span — root-or-passthrough span for wrapper / middleware paths.

Behavior is governed by the shared depth counter in :mod:`_depth`:

- ``parent_depth == 0``: open a real ``leanctx.compress`` span. The
  outermost frame owns provider / method / cost attributes and records
  the metric counters + duration histogram on close.
- ``parent_depth > 0``: yield a passthrough proxy. Stats fed to the
  proxy via :meth:`_LeanctxSpan.set_stats` flow up to the outermost
  open span (the wrapper's). No new span is emitted; no metric is
  re-recorded.

For streaming wrapper paths (OpenAI ``stream=True`` etc.), the wrapper
calls :meth:`compression_span.detach_span` while still inside the
``with`` block. ``__exit__`` then resets the depth counter and the
OTel current-span context but leaves the underlying OTel span open;
the caller (an iterator wrapper) takes ownership and calls
``span.close()`` on iterator exhaustion / ``.close()`` / ``__del__``.

When ``observability.enabled`` is False or ``opentelemetry-api`` is
absent, a no-op proxy is yielded; no OTel work happens at all.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any

from leanctx.observability import api, metrics
from leanctx.observability._depth import DEPTH
from leanctx.observability.config import ObservabilityConfig
from leanctx.observability.method_status import MethodStatus, coerce
from leanctx.stats import CompressionStats


class _NoopSpan:
    """Returned when observability is off or OTel API unavailable."""

    is_recording = False
    is_root = False

    def set_stats(self, stats: CompressionStats, *, status: str = "success") -> None:
        return None

    def set_method(self, method: str | MethodStatus) -> None:
        return None

    def set_error(self, exc: BaseException) -> None:
        return None

    def close(self) -> None:
        return None


class _PassthroughSpan:
    """Yielded when nested inside an outer compression_span.

    Forwards stats up to the outer ``_LeanctxSpan`` so the outermost
    frame has accurate provider/method/cost when it closes. Does not
    open its own OTel span and does not record metrics.
    """

    is_recording = False
    is_root = False

    def __init__(self, outer: _LeanctxSpan | _NoopSpan) -> None:
        self._outer = outer

    def set_stats(self, stats: CompressionStats, *, status: str = "success") -> None:
        if isinstance(self._outer, _LeanctxSpan):
            self._outer.set_stats(stats, status=status)

    def set_method(self, method: str | MethodStatus) -> None:
        if isinstance(self._outer, _LeanctxSpan):
            self._outer.set_method(method)

    def set_error(self, exc: BaseException) -> None:
        if isinstance(self._outer, _LeanctxSpan):
            self._outer.set_error(exc)

    def close(self) -> None:
        return None


class _LeanctxSpan:
    """The outermost wrapper / middleware span; owns provider/method/cost.

    Created with an OTel span that has been opened via ``tracer.start_span``
    (NOT ``start_as_current_span``). The compression_span context manager
    is responsible for activating the span as current and resetting that
    activation on ``__exit__``; this class is responsible for ending the
    underlying OTel span and recording metrics on :meth:`close`.
    """

    is_recording = True
    is_root = True

    def __init__(
        self,
        otel_span: Any,
        *,
        provider: str,
        config: ObservabilityConfig,
        start_ns: int,
    ) -> None:
        self._span = otel_span
        self._provider = provider
        self._config = config
        self._start_ns = start_ns
        self._method: str = MethodStatus.PASSTHROUGH.value
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cost_usd: float = 0.0
        self._ratio: float = 1.0
        self._status: str = "success"
        self._closed: bool = False
        self._set("leanctx.provider", provider)
        for k, v in (config.extra_attributes or {}).items():
            self._set(k, v)

    def _set(self, key: str, value: Any) -> None:
        setter = getattr(self._span, "set_attribute", None)
        if setter is not None:
            setter(key, value)

    def set_stats(self, stats: CompressionStats, *, status: str = "success") -> None:
        self._method = stats.method or MethodStatus.PASSTHROUGH.value
        self._input_tokens = int(stats.input_tokens)
        self._output_tokens = int(stats.output_tokens)
        self._cost_usd = float(stats.cost_usd)
        self._ratio = float(stats.ratio)
        self._status = status
        self._set("leanctx.method", self._method)
        self._set("leanctx.input_tokens", self._input_tokens)
        self._set("leanctx.output_tokens", self._output_tokens)
        self._set("leanctx.ratio", self._ratio)
        self._set("leanctx.cost_usd", self._cost_usd)

    def set_method(self, method: str | MethodStatus) -> None:
        self._method = coerce(method).value
        self._set("leanctx.method", self._method)

    def set_error(self, exc: BaseException) -> None:
        self._status = "error"
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

    def close(self) -> None:
        """End the OTel span and record metrics. Idempotent."""
        if self._closed:
            return
        self._closed = True
        duration_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000.0
        self._set("leanctx.duration_ms", duration_ms)
        try:
            metrics.record(
                provider=self._provider,
                method=self._method,
                status=self._status,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                cost_usd=self._cost_usd,
                duration_ms=duration_ms,
            )
        finally:
            ender = getattr(self._span, "end", None)
            if ender is not None:
                ender()


_CURRENT_OUTERMOST: ContextVar[_LeanctxSpan | None] = ContextVar(
    "leanctx_outermost_span", default=None
)


class compression_span:  # noqa: N801 — context-manager API, lowercase by convention
    """Class-based context manager for wrapper / middleware spans.

    Class-based (not ``@contextmanager``) so that streaming wrapper
    paths can call :meth:`detach_span` to transfer span ownership to a
    long-lived iterator. After ``detach_span`` is called, ``__exit__``
    resets the depth counter and OTel current-span context but does
    NOT end the underlying OTel span; the caller invokes
    ``span.close()`` later.
    """

    def __init__(
        self,
        observability: ObservabilityConfig | None,
        *,
        provider: str,
    ) -> None:
        self._obs = observability
        self._provider = provider
        self._span: _LeanctxSpan | _PassthroughSpan | _NoopSpan | None = None
        self._detached: bool = False
        self._depth_token: Any = None
        self._otel_ctx_token: Any = None
        self._outer_token: Any = None

    def __enter__(self) -> _LeanctxSpan | _PassthroughSpan | _NoopSpan:
        if self._obs is None or not self._obs.enabled:
            self._span = _NoopSpan()
            return self._span

        parent_depth = DEPTH.get()
        self._depth_token = DEPTH.set(parent_depth + 1)

        if parent_depth > 0:
            outer = _CURRENT_OUTERMOST.get()
            self._span = _PassthroughSpan(outer if outer is not None else _NoopSpan())
            return self._span

        tracer = api.get_tracer()
        if tracer is None:
            self._span = _NoopSpan()
            return self._span

        from opentelemetry import context as ot_context  # noqa: PLC0415
        from opentelemetry import trace as ot_trace  # noqa: PLC0415

        otel_span = tracer.start_span("leanctx.compress")
        self._otel_ctx_token = ot_context.attach(
            ot_trace.set_span_in_context(otel_span)
        )
        leanctx_span = _LeanctxSpan(
            otel_span,
            provider=self._provider,
            config=self._obs,
            start_ns=time.perf_counter_ns(),
        )
        self._outer_token = _CURRENT_OUTERMOST.set(leanctx_span)
        self._span = leanctx_span
        return leanctx_span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            span = self._span
            if isinstance(span, _LeanctxSpan):
                if exc_val is not None:
                    span.set_error(exc_val)
                if not self._detached:
                    span.close()
            if self._outer_token is not None:
                _CURRENT_OUTERMOST.reset(self._outer_token)
            if self._otel_ctx_token is not None:
                from opentelemetry import context as ot_context  # noqa: PLC0415

                ot_context.detach(self._otel_ctx_token)
        finally:
            if self._depth_token is not None:
                DEPTH.reset(self._depth_token)

    async def __aenter__(self) -> _LeanctxSpan | _PassthroughSpan | _NoopSpan:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    def detach_span(self) -> _LeanctxSpan | _NoopSpan:
        """Transfer span ownership to the caller.

        Must be called from within the ``with`` block. After this call,
        ``__exit__`` will not close the span; the caller is responsible
        for invoking ``span.close()`` exactly once. Useful for streaming
        wrapper paths where the span must outlive the wrapper's
        ``create()`` frame.

        Returns the live ``_LeanctxSpan`` (or a ``_NoopSpan`` if
        observability was disabled / API absent).
        """
        self._detached = True
        if isinstance(self._span, _LeanctxSpan):
            return self._span
        return _NoopSpan()
