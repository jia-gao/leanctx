"""ObservabilityConfig — per-client observability settings.

Off by default. The application enables OTel emission by setting
``observability.otel = True`` in ``leanctx_config``; even then, leanctx
only emits via the OTel API. The SDK and exporter setup is owned by the
application.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservabilityConfig:
    """Observability settings on a leanctx client.

    Attributes:
        otel: When True, leanctx emits OTel spans and metrics for every
            compression call. When False (default), all telemetry is
            suppressed at the source — no span construction, no metric
            recording, no opentelemetry-api work happens beyond the
            initial check. Setting this to True with ``opentelemetry-api``
            not installed is silently downgraded to no-op behavior.
        service_name: Optional ``service.name`` resource attribute hint
            used for documentation. leanctx never overrides the user's
            app-side ``OTEL_SERVICE_NAME``; this field is informational
            and reserved for future use.
        extra_attributes: Static attributes added to every leanctx span
            (e.g. ``{"region": "us-west-2"}``). Cardinality is the
            user's responsibility — keep these bounded and stable.
    """

    otel: bool = False
    service_name: str | None = None
    extra_attributes: dict[str, str] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """True when OTel emission should be attempted."""
        return self.otel

    @classmethod
    def from_dict(cls, raw: dict[str, object] | None) -> ObservabilityConfig:
        """Build an ObservabilityConfig from a leanctx_config dict.

        Unknown keys are ignored (forward-compat). Missing values fall
        back to the dataclass defaults.
        """
        if not raw:
            return cls()
        otel = bool(raw.get("otel", False))
        raw_service = raw.get("service_name")
        service_name = raw_service if isinstance(raw_service, str) else None
        raw_extra = raw.get("extra_attributes")
        extra: dict[str, str]
        if isinstance(raw_extra, dict):
            extra = {str(k): str(v) for k, v in raw_extra.items()}
        else:
            extra = {}
        return cls(otel=otel, service_name=service_name, extra_attributes=extra)
