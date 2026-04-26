"""leanctx.observability — opt-in OpenTelemetry instrumentation.

This package provides API-only OTel access for leanctx. It never
constructs an OTel SDK, never registers a TracerProvider/MeterProvider,
and never instantiates an exporter. The application owns the OTel SDK
lifecycle; leanctx only emits.

When the ``opentelemetry-api`` package is not installed, every call in
this package becomes a fast no-op so downstream code can call into it
unconditionally.

Public surface
--------------
- :class:`ObservabilityConfig` — per-client config (off | on)
- :class:`MethodStatus` — taxonomy of compression outcomes
- :func:`api.is_available` — whether opentelemetry-api is importable
- :func:`api.get_tracer` / :func:`api.get_meter` — tracer/meter accessors

Telemetry-emitting context managers (``compression_span``,
``compressor_span``) live in submodules and are added in later phases.
"""

from __future__ import annotations

from leanctx.observability.config import ObservabilityConfig
from leanctx.observability.method_status import MethodStatus

__all__ = [
    "MethodStatus",
    "ObservabilityConfig",
]
