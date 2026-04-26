"""BenchRecord — versioned JSON schema for bench output.

Every bench scenario emits one or more BenchRecord values per run.
The schema is versioned (``schema_version: "1"``); breaking schema
changes increment the version and old consumers can detect mismatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1"

REQUIRED_FIELDS = (
    "schema_version",
    "leanctx_version",
    "scenario",
    "workload",
    "status",
    "request_provider",
    "request_model",
    "compression_provider",
    "compression_model",
    "compressor",
    "input_tokens",
    "output_tokens",
    "tokens_saved",
    "ratio",
    "cost_usd",
    "duration_ms",
    "warmup",
    "timestamp",
)


@dataclass
class BenchRecord:
    """One scenario × workload × run output."""

    schema_version: str = SCHEMA_VERSION
    leanctx_version: str = ""
    scenario: str = ""
    workload: str = ""
    status: str = "success"  # success | failure
    request_provider: str | None = None
    request_model: str | None = None
    compression_provider: str | None = None
    compression_model: str | None = None
    compressor: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_saved: int = 0
    ratio: float = 1.0
    cost_usd: float = 0.0
    duration_ms: int = 0
    warmup: bool = False
    timestamp: str = ""
    lingua_model_revision: str | None = None
    invariants: dict[str, bool] | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "leanctx_version": self.leanctx_version,
            "scenario": self.scenario,
            "workload": self.workload,
            "status": self.status,
            "request_provider": self.request_provider,
            "request_model": self.request_model,
            "compression_provider": self.compression_provider,
            "compression_model": self.compression_model,
            "compressor": self.compressor,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_saved": self.tokens_saved,
            "ratio": self.ratio,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "warmup": self.warmup,
            "timestamp": self.timestamp,
        }
        if self.lingua_model_revision is not None:
            out["lingua_model_revision"] = self.lingua_model_revision
        if self.invariants is not None:
            out["invariants"] = self.invariants
        if self.error is not None:
            out["error"] = self.error
        if self.extra:
            out["extra"] = self.extra
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


def validate_record(raw: dict[str, Any]) -> list[str]:
    """Return a list of validation error messages for a record dict.

    Empty list = valid. Non-empty = invalid. The bench CLI itself uses
    this to fail-fast on malformed records (AC-8 negative test).
    """
    errors: list[str] = []
    for fld in REQUIRED_FIELDS:
        if fld not in raw:
            errors.append(f"missing required field {fld!r}")
    if "schema_version" in raw and raw["schema_version"] != SCHEMA_VERSION:
        errors.append(
            f"schema_version mismatch: got {raw['schema_version']!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )
    if raw.get("status") not in (None, "success", "failure"):
        errors.append(f"invalid status {raw.get('status')!r}; expected success | failure")
    return errors
