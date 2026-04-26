"""API-only OpenTelemetry access.

leanctx never imports ``opentelemetry`` at module top-level — the
underlying packages are loaded lazily on first use, so cold-import of
``leanctx`` does not pay the OTel-API import cost when the application
isn't actively emitting telemetry.

When ``opentelemetry-api`` is not installed, accessor functions return
``None`` and callers should branch on the return value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — only for type-checkers
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Tracer

_INSTRUMENTATION_NAME = "leanctx"

_trace_module: Any = None
_metrics_module: Any = None
_otel_state: int = 0  # 0 = unprobed, 1 = available, -1 = unavailable


def _probe() -> int:
    """Lazy-import opentelemetry.trace + opentelemetry.metrics once.

    Returns 1 when both modules import cleanly, -1 when either is
    missing. Caches the result in :data:`_otel_state` so subsequent
    calls are O(1).
    """
    global _trace_module, _metrics_module, _otel_state
    if _otel_state != 0:
        return _otel_state
    try:
        from opentelemetry import metrics as _metrics  # noqa: PLC0415
        from opentelemetry import trace as _trace  # noqa: PLC0415
    except ImportError:
        _otel_state = -1
        return -1
    _trace_module = _trace
    _metrics_module = _metrics
    _otel_state = 1
    return 1


def is_available() -> bool:
    """Return True when ``opentelemetry-api`` is importable."""
    return _probe() == 1


def get_tracer() -> Tracer | None:
    """Return the leanctx tracer or None when OTel API is unavailable.

    leanctx never calls ``trace.set_tracer_provider``; this function
    asks the OTel API for a tracer, which proxies through whatever
    ``TracerProvider`` the application has configured (or the default
    no-op provider if the app hasn't configured one).
    """
    if _probe() != 1:
        return None
    tracer: Tracer = _trace_module.get_tracer(_INSTRUMENTATION_NAME)
    return tracer


def get_meter() -> Meter | None:
    """Return the leanctx meter or None when OTel API is unavailable.

    Same posture as :func:`get_tracer` — never registers a
    ``MeterProvider``; relies on the app's configuration.
    """
    if _probe() != 1:
        return None
    meter: Meter = _metrics_module.get_meter(_INSTRUMENTATION_NAME)
    return meter


def _reset_for_tests() -> None:
    """Test-only hook: clear the cached probe state.

    Used by tests that want to re-probe after manipulating
    ``sys.modules``.  Not part of the public API.
    """
    global _trace_module, _metrics_module, _otel_state
    _trace_module = None
    _metrics_module = None
    _otel_state = 0
