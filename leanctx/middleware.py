"""Middleware — provider-agnostic compression orchestrator.

v0.0.x is a passthrough: compress_messages returns input unchanged with
zero-valued stats. v0.1 will plug in the Compressor router (local
LLMLingua-2 / self_llm / hybrid) and the content classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompressionStats:
    """Telemetry attached to every request that passed through the middleware."""

    input_tokens: int = 0
    output_tokens: int = 0
    ratio: float = 1.0
    method: str = "passthrough"
    cost_usd: float = 0.0


class Middleware:
    """Compression middleware. Passthrough in v0.0.x; real compression lands in v0.1."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def compress_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return messages, CompressionStats()

    async def compress_messages_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return messages, CompressionStats()
