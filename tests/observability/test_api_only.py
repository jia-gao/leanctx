"""AC-1 — leanctx is an OTel-API-only library.

These tests pin the contract that leanctx never owns OTel SDK or
exporter configuration. The application is the sole owner of
TracerProvider, MeterProvider, and any exporter setup.

Tests run regardless of whether the ``[otel]`` extra is installed —
they assert leanctx never touches SDK construction either way.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from leanctx.observability import api, config, method_status, metrics


def test_no_opentelemetry_in_sys_modules_after_bare_import() -> None:
    """`import leanctx` (no observability use) must not import opentelemetry.

    Verified in a fresh Python subprocess so this test's own opentelemetry
    state cannot leak into the assertion.
    """
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import leanctx; "
            "print('|'.join(sorted(m for m in sys.modules if m.startswith('opentelemetry'))))",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = proc.stdout.strip()
    assert leaked == "", (
        f"importing leanctx leaked opentelemetry modules into sys.modules: {leaked!r}"
    )


def test_observability_package_does_not_import_opentelemetry_at_module_top() -> None:
    """The leanctx.observability package must defer all opentelemetry
    imports to first-call lazy paths."""
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import leanctx.observability; "
            "print('|'.join(sorted(m for m in sys.modules if m.startswith('opentelemetry'))))",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = proc.stdout.strip()
    assert leaked == "", (
        f"importing leanctx.observability leaked opentelemetry modules: {leaked!r}"
    )


def test_metrics_record_is_safe_when_otel_unavailable() -> None:
    """metrics.record must be a fast no-op when opentelemetry-api is absent
    OR when no MeterProvider is configured."""
    metrics._reset_for_tests()
    api._reset_for_tests()
    metrics.record(
        provider="anthropic",
        method="verbatim",
        status="success",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0,
        duration_ms=12.3,
    )


def test_get_tracer_and_meter_return_none_or_otel_objects() -> None:
    """Accessors return None when API absent, or real OTel objects when
    available (proxying through the app's provider)."""
    api._reset_for_tests()
    tracer = api.get_tracer()
    meter = api.get_meter()
    if api.is_available():
        assert tracer is not None
        assert meter is not None
    else:
        assert tracer is None
        assert meter is None


def test_observability_config_disabled_by_default() -> None:
    cfg = config.ObservabilityConfig()
    assert cfg.otel is False
    assert cfg.enabled is False
    assert cfg.extra_attributes == {}


def test_observability_config_from_dict_handles_none_and_unknowns() -> None:
    assert config.ObservabilityConfig.from_dict(None) == config.ObservabilityConfig()
    cfg = config.ObservabilityConfig.from_dict(
        {"otel": True, "service_name": "myapp", "unknown_key": "ignored"}
    )
    assert cfg.otel is True
    assert cfg.service_name == "myapp"


def test_method_status_taxonomy_has_eight_documented_values() -> None:
    """AC-3: the closed set is exactly the eight documented values."""
    expected = {
        "passthrough",
        "below-threshold",
        "empty",
        "opaque-bailout",
        "verbatim",
        "lingua",
        "selfllm",
        "hybrid",
    }
    actual = {member.value for member in method_status.MethodStatus}
    assert actual == expected


def test_method_status_coerce_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unknown leanctx.method value"):
        method_status.coerce("totally-made-up")


def test_method_status_coerce_accepts_known_values() -> None:
    assert (
        method_status.coerce("verbatim") is method_status.MethodStatus.VERBATIM
    )
    assert (
        method_status.coerce(method_status.MethodStatus.HYBRID)
        is method_status.MethodStatus.HYBRID
    )


def test_observability_module_round_trip_reload() -> None:
    """Re-importing the package must not trip lazy state."""
    importlib.reload(method_status)
    importlib.reload(config)
    importlib.reload(api)
    importlib.reload(metrics)
