"""CompressionStats — telemetry shape shared by every compression call."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionStats:
    """Telemetry attached to every request that passed through the middleware.

    Lives in its own module (rather than in middleware) so that
    :mod:`leanctx.compressors` can depend on it without creating a cycle
    when the middleware in turn imports from compressors.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    ratio: float = 1.0
    method: str = "passthrough"
    cost_usd: float = 0.0
