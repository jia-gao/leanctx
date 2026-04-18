"""Drop-in LLM SDK wrappers.

Interface-compatible wrappers around the official anthropic SDK. Public
surface mirrors the upstream client; an additional ``leanctx_config``
kwarg controls compression behavior, and each response gains a
``usage.leanctx_tokens_saved`` attribute.

v0.0.x is a passthrough — the wrappers forward to the real SDK without
compressing. Compression lands in v0.1 via the Middleware.
"""

from __future__ import annotations

from typing import Any

from leanctx.middleware import CompressionStats, Middleware

_ANTHROPIC_INSTALL_HINT = (
    "The 'anthropic' package is required for leanctx.Anthropic. "
    "Install with: pip install 'leanctx[anthropic]'"
)


def _load_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(_ANTHROPIC_INSTALL_HINT) from e
    return anthropic


class Anthropic:
    """Drop-in replacement for ``anthropic.Anthropic``.

    Accepts the same positional / keyword args as the upstream SDK, plus a
    ``leanctx_config`` kwarg that configures the compression pipeline
    (no-op in v0.0.x).
    """

    def __init__(
        self,
        *args: Any,
        leanctx_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pkg = _load_anthropic()
        self._upstream = pkg.Anthropic(*args, **kwargs)
        self._middleware = Middleware(leanctx_config or {})
        self.messages = _Messages(self._upstream, self._middleware)


class AsyncAnthropic:
    """Async variant of :class:`Anthropic`."""

    def __init__(
        self,
        *args: Any,
        leanctx_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pkg = _load_anthropic()
        self._upstream = pkg.AsyncAnthropic(*args, **kwargs)
        self._middleware = Middleware(leanctx_config or {})
        self.messages = _AsyncMessages(self._upstream, self._middleware)


class _Messages:
    def __init__(self, upstream: Any, middleware: Middleware) -> None:
        self._upstream = upstream
        self._middleware = middleware

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        compressed, stats = self._middleware.compress_messages(messages)
        kwargs["messages"] = compressed
        response = self._upstream.messages.create(**kwargs)
        _attach_telemetry(response, stats)
        return response

    def stream(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        compressed, _ = self._middleware.compress_messages(messages)
        kwargs["messages"] = compressed
        return self._upstream.messages.stream(**kwargs)


class _AsyncMessages:
    def __init__(self, upstream: Any, middleware: Middleware) -> None:
        self._upstream = upstream
        self._middleware = middleware

    async def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        compressed, stats = await self._middleware.compress_messages_async(messages)
        kwargs["messages"] = compressed
        response = await self._upstream.messages.create(**kwargs)
        _attach_telemetry(response, stats)
        return response

    def stream(self, **kwargs: Any) -> Any:
        # The upstream returns a context manager for streaming, not a coroutine,
        # so we forward it directly rather than awaiting.
        messages = kwargs.get("messages", [])
        compressed, _ = self._middleware.compress_messages(messages)
        kwargs["messages"] = compressed
        return self._upstream.messages.stream(**kwargs)


def _attach_telemetry(response: Any, stats: CompressionStats) -> None:
    # Bypass Pydantic validation since `usage` is a frozen BaseModel field in
    # the anthropic SDK; object.__setattr__ sidesteps validators that would
    # reject an unknown attribute.
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    object.__setattr__(usage, "leanctx_tokens_saved", stats.input_tokens - stats.output_tokens)
    object.__setattr__(usage, "leanctx_ratio", stats.ratio)
    object.__setattr__(usage, "leanctx_method", stats.method)
