"""Compressor protocol and content-type taxonomy."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from leanctx.middleware import CompressionStats


class ContentType(str, Enum):
    """Classification labels emitted by the classifier.

    The Router uses these to pick which Compressor handles a given span.
    Values are strings so config files can reference them directly, e.g.
    ``"routing": {"code": "verbatim", "prose": "lingua"}``.
    """

    UNKNOWN = "unknown"
    PROSE = "prose"
    CODE = "code"
    ERROR = "error"
    REPEAT = "repeat"
    LONG_IMPORTANT = "long_important"


@runtime_checkable
class Compressor(Protocol):
    """A compression strategy for a span of chat messages.

    Implementations must:

    * respect asyncio cancellation in ``compress_async``
    * be safe for concurrent use unless the docstring says otherwise
    * return messages in the same shape they received (list of dicts
      with ``role`` and ``content``)
    * never raise for empty inputs — return them unchanged with zero-
      valued stats
    """

    name: str

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        ...

    async def compress_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        ...
