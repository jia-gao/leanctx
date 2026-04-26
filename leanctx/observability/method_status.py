"""MethodStatus — exhaustive taxonomy of compression-call outcomes.

Every leanctx span carries a ``leanctx.method`` attribute set to one of
these values. The taxonomy is closed: callers must never emit a span
without one of these values, and span exporters can rely on the set
being stable across leanctx versions within a major release.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class MethodStatus(str, Enum):
    """Closed set of values for the ``leanctx.method`` span attribute."""

    PASSTHROUGH = "passthrough"
    """Mode is off; messages forwarded unchanged."""

    BELOW_THRESHOLD = "below-threshold"
    """Mode is on but message tokens did not exceed the trigger threshold."""

    EMPTY = "empty"
    """Empty message list; nothing to compress."""

    OPAQUE_BAILOUT = "opaque-bailout"
    """Gemini contents contained non-text parts (function_call / image /
    inline_data) that the adapter cannot safely compress."""

    VERBATIM = "verbatim"
    """Pipeline ran; all messages routed to the Verbatim compressor."""

    LINGUA = "lingua"
    """Pipeline ran; Lingua used for at least one message and no other
    compressor was used."""

    SELFLLM = "selfllm"
    """Pipeline ran; SelfLLM used for at least one message and no other
    compressor was used."""

    HYBRID = "hybrid"
    """Pipeline ran; two or more of {Verbatim, Lingua, SelfLLM} used."""


_VALUES: frozenset[str] = frozenset(m.value for m in MethodStatus)


def is_valid(value: str) -> bool:
    """True when ``value`` is one of the documented method-status strings."""
    return value in _VALUES


def coerce(value: str | MethodStatus) -> MethodStatus:
    """Coerce a string or MethodStatus into a MethodStatus.

    Raises ValueError for any string outside the documented set.
    """
    if isinstance(value, MethodStatus):
        return value
    try:
        return MethodStatus(value)
    except ValueError as exc:
        raise ValueError(
            f"unknown leanctx.method value {value!r}; "
            f"expected one of {sorted(_VALUES)}"
        ) from exc


def set_method(span: Any, value: str | MethodStatus) -> None:
    """Set ``leanctx.method`` on ``span`` to a validated taxonomy value.

    Accepts either a real OTel span or a no-op proxy; calls
    ``span.set_attribute`` if available, otherwise silently drops.
    """
    method = coerce(value)
    setter = getattr(span, "set_attribute", None)
    if setter is not None:
        setter("leanctx.method", method.value)
