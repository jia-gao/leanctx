"""Drop-in LLM SDK wrappers.

Interface-compatible wrappers around the official anthropic / openai /
google-genai SDKs. Public surface mirrors the upstream client; an
additional ``leanctx_config`` kwarg controls compression behavior, and
each response gains ``usage.leanctx_*`` telemetry attributes.

When the ``[otel]`` extra is installed and observability is enabled
in ``leanctx_config["observability"]["otel"]``, every wrapper request
path emits a ``leanctx.compress`` span.
"""

from __future__ import annotations

from typing import Any

from leanctx._gemini_adapter import (
    contents_to_messages as _gemini_contents_to_messages,
)
from leanctx._gemini_adapter import (
    messages_to_contents as _gemini_messages_to_contents,
)
from leanctx.middleware import Middleware
from leanctx.observability.config import ObservabilityConfig
from leanctx.observability.middleware_hooks import compression_span
from leanctx.observability.stream_owners import (
    _SpanOwningAsyncIterator,
    _SpanOwningIterator,
)
from leanctx.stats import CompressionStats

_ANTHROPIC_INSTALL_HINT = (
    "The 'anthropic' package is required for leanctx.Anthropic. "
    "Install with: pip install 'leanctx[anthropic]'"
)

_OPENAI_INSTALL_HINT = (
    "The 'openai' package is required for leanctx.OpenAI. "
    "Install with: pip install 'leanctx[openai]'"
)

_GEMINI_INSTALL_HINT = (
    "The 'google-genai' package is required for leanctx.Gemini. "
    "Install with: pip install 'leanctx[gemini]'"
)


def _load_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(_ANTHROPIC_INSTALL_HINT) from e
    return anthropic


def _load_openai() -> Any:
    try:
        import openai
    except ImportError as e:
        raise ImportError(_OPENAI_INSTALL_HINT) from e
    return openai


def _load_gemini() -> Any:
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(_GEMINI_INSTALL_HINT) from e
    return genai


def _parse_observability(leanctx_config: dict[str, Any] | None) -> ObservabilityConfig:
    if not leanctx_config:
        return ObservabilityConfig()
    return ObservabilityConfig.from_dict(leanctx_config.get("observability"))


class Anthropic:
    """Drop-in replacement for ``anthropic.Anthropic``."""

    def __init__(
        self,
        *args: Any,
        leanctx_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pkg = _load_anthropic()
        self._upstream = pkg.Anthropic(*args, **kwargs)
        self._observability = _parse_observability(leanctx_config)
        self._middleware = Middleware(leanctx_config or {}, observability=self._observability)
        self.messages = _Messages(self._upstream, self._middleware, self._observability)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream, name)


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
        self._observability = _parse_observability(leanctx_config)
        self._middleware = Middleware(leanctx_config or {}, observability=self._observability)
        self.messages = _AsyncMessages(self._upstream, self._middleware, self._observability)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream, name)


class _Messages:
    def __init__(
        self,
        upstream: Any,
        middleware: Middleware,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self._observability = observability or ObservabilityConfig()

    def create(self, **kwargs: Any) -> Any:
        with compression_span(self._observability, provider="anthropic") as span:
            messages = kwargs.get("messages", [])
            compressed, stats = self._middleware.compress_messages(messages)
            kwargs["messages"] = compressed
            response = self._upstream.messages.create(**kwargs)
            span.set_stats(stats)
            _attach_telemetry(response, stats)
            return response

    def stream(self, **kwargs: Any) -> Any:
        # Anthropic's sync messages.stream() returns a context manager
        # (MessageStreamManager). The leanctx-wrapper span lifetime
        # spans __enter__ to __exit__ of that CM. We compress at call
        # time (here) and wrap the manager so __exit__ closes the span.
        with compression_span(self._observability, provider="anthropic") as span:
            messages = kwargs.get("messages", [])
            compressed, stats = self._middleware.compress_messages(messages)
            kwargs["messages"] = compressed
            span.set_stats(stats)
            upstream_cm = self._upstream.messages.stream(**kwargs)
        return upstream_cm


class _AsyncMessages:
    def __init__(
        self,
        upstream: Any,
        middleware: Middleware,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self._observability = observability or ObservabilityConfig()

    async def create(self, **kwargs: Any) -> Any:
        with compression_span(self._observability, provider="anthropic") as span:
            messages = kwargs.get("messages", [])
            compressed, stats = await self._middleware.compress_messages_async(messages)
            kwargs["messages"] = compressed
            response = await self._upstream.messages.create(**kwargs)
            span.set_stats(stats)
            _attach_telemetry(response, stats)
            return response

    def stream(self, **kwargs: Any) -> Any:
        return _AsyncStreamContextManager(
            self._upstream.messages, kwargs, self._middleware, self._observability
        )


class _AsyncStreamContextManager:
    """Wraps ``anthropic.AsyncAnthropic.messages.stream`` so compression
    happens on ``__aenter__``, not at the synchronous ``stream()`` call.

    The leanctx span opens at ``__aenter__`` and closes at ``__aexit__``;
    that's the AC-2 stream-lifetime contract for context-manager stream
    paths (paths 2 and 4).
    """

    def __init__(
        self,
        upstream_messages: Any,
        kwargs: dict[str, Any],
        middleware: Middleware,
        observability: ObservabilityConfig,
    ) -> None:
        self._upstream_messages = upstream_messages
        self._kwargs = kwargs
        self._middleware = middleware
        self._observability = observability or ObservabilityConfig()
        self._upstream_cm: Any = None
        self._span_cm: compression_span | None = None
        self._span: Any = None

    async def __aenter__(self) -> Any:
        self._span_cm = compression_span(self._observability, provider="anthropic")
        self._span = self._span_cm.__enter__()
        try:
            messages = self._kwargs.get("messages", [])
            compressed, stats = await self._middleware.compress_messages_async(messages)
            self._kwargs["messages"] = compressed
            self._span.set_stats(stats)
            self._upstream_cm = self._upstream_messages.stream(**self._kwargs)
            return await self._upstream_cm.__aenter__()
        except BaseException as exc:
            self._span.set_error(exc)
            self._span_cm.__exit__(type(exc), exc, exc.__traceback__)
            self._span_cm = None
            raise

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> Any:
        try:
            if self._upstream_cm is None:
                return None
            return await self._upstream_cm.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if self._span_cm is not None:
                self._span_cm.__exit__(exc_type, exc_val, exc_tb)
                self._span_cm = None


def _attach_telemetry(response: Any, stats: CompressionStats, field: str = "usage") -> None:
    # Bypass Pydantic validation since the usage field is a frozen BaseModel
    # in the anthropic/openai/google-genai SDKs; object.__setattr__ sidesteps
    # validators that would reject an unknown attribute. The field name
    # differs per provider: "usage" (anthropic, openai), "usage_metadata"
    # (google-genai).
    usage = getattr(response, field, None)
    if usage is None:
        return
    object.__setattr__(usage, "leanctx_tokens_saved", stats.input_tokens - stats.output_tokens)
    object.__setattr__(usage, "leanctx_ratio", stats.ratio)
    object.__setattr__(usage, "leanctx_method", stats.method)
    object.__setattr__(usage, "leanctx_cost_usd", stats.cost_usd)


# --------------------------------------------------------------------------- #
# OpenAI wrappers
# --------------------------------------------------------------------------- #


class OpenAI:
    """Drop-in replacement for ``openai.OpenAI``."""

    def __init__(
        self,
        *args: Any,
        leanctx_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pkg = _load_openai()
        self._upstream = pkg.OpenAI(*args, **kwargs)
        self._observability = _parse_observability(leanctx_config)
        self._middleware = Middleware(leanctx_config or {}, observability=self._observability)
        self.chat = _Chat(self._upstream, self._middleware, self._observability)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream, name)


class AsyncOpenAI:
    """Async variant of :class:`OpenAI`."""

    def __init__(
        self,
        *args: Any,
        leanctx_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pkg = _load_openai()
        self._upstream = pkg.AsyncOpenAI(*args, **kwargs)
        self._observability = _parse_observability(leanctx_config)
        self._middleware = Middleware(leanctx_config or {}, observability=self._observability)
        self.chat = _AsyncChat(self._upstream, self._middleware, self._observability)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream, name)


class _Chat:
    def __init__(
        self,
        upstream: Any,
        middleware: Middleware,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self.completions = _Completions(upstream, middleware, observability)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.chat, name)


class _AsyncChat:
    def __init__(
        self,
        upstream: Any,
        middleware: Middleware,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self.completions = _AsyncCompletions(upstream, middleware, observability)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.chat, name)


class _Completions:
    def __init__(
        self,
        upstream: Any,
        middleware: Middleware,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self._observability = observability or ObservabilityConfig()

    def create(self, **kwargs: Any) -> Any:
        is_stream = bool(kwargs.get("stream"))
        cm = compression_span(self._observability, provider="openai")
        span = cm.__enter__()
        try:
            messages = kwargs.get("messages", [])
            compressed, stats = self._middleware.compress_messages(messages)
            kwargs["messages"] = compressed
            response = self._upstream.chat.completions.create(**kwargs)
            span.set_stats(stats)
            if is_stream and span.is_root:
                # Iterator path: hand span ownership to the wrapper
                detached = cm.detach_span()
                cm.__exit__(None, None, None)
                return _SpanOwningIterator(response, detached)
            _attach_telemetry(response, stats)
            cm.__exit__(None, None, None)
            return response
        except BaseException as exc:
            span.set_error(exc)
            cm.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.chat.completions, name)


class _AsyncCompletions:
    def __init__(
        self,
        upstream: Any,
        middleware: Middleware,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self._observability = observability or ObservabilityConfig()

    async def create(self, **kwargs: Any) -> Any:
        is_stream = bool(kwargs.get("stream"))
        cm = compression_span(self._observability, provider="openai")
        span = cm.__enter__()
        try:
            messages = kwargs.get("messages", [])
            compressed, stats = await self._middleware.compress_messages_async(messages)
            kwargs["messages"] = compressed
            response = await self._upstream.chat.completions.create(**kwargs)
            span.set_stats(stats)
            if is_stream and span.is_root:
                detached = cm.detach_span()
                cm.__exit__(None, None, None)
                return _SpanOwningAsyncIterator(response, detached)
            _attach_telemetry(response, stats)
            cm.__exit__(None, None, None)
            return response
        except BaseException as exc:
            span.set_error(exc)
            cm.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.chat.completions, name)


# --------------------------------------------------------------------------- #
# Gemini (google-genai) wrapper
# --------------------------------------------------------------------------- #


class Gemini:
    """Drop-in replacement for ``google.genai.Client``.

    Non-text parts (function_call, function_response, images) trigger an
    automatic bailout to passthrough; with observability enabled the
    span carries ``leanctx.method = opaque-bailout`` so users can track
    multimodal traffic that bypasses the pipeline.
    """

    def __init__(
        self,
        *args: Any,
        leanctx_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pkg = _load_gemini()
        self._upstream = pkg.Client(*args, **kwargs)
        self._observability = _parse_observability(leanctx_config)
        self._middleware = Middleware(leanctx_config or {}, observability=self._observability)
        self.models = _GeminiModels(self._upstream, self._middleware, self._observability)
        self.aio = _GeminiAio(self._upstream, self._middleware, self._observability)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream, name)


class _GeminiModels:
    def __init__(
        self,
        upstream: Any,
        middleware: Middleware,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self._observability = observability or ObservabilityConfig()

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        with compression_span(self._observability, provider="gemini") as span:
            stats = _gemini_compress_in_place(kwargs, self._middleware)
            response = self._upstream.models.generate_content(*args, **kwargs)
            span.set_stats(stats)
            _attach_telemetry(response, stats, field="usage_metadata")
            return response

    def generate_content_stream(self, *args: Any, **kwargs: Any) -> Any:
        cm = compression_span(self._observability, provider="gemini")
        span = cm.__enter__()
        try:
            stats = _gemini_compress_in_place(kwargs, self._middleware)
            span.set_stats(stats)
            upstream_iter = self._upstream.models.generate_content_stream(*args, **kwargs)
            if span.is_root:
                detached = cm.detach_span()
                cm.__exit__(None, None, None)
                return _SpanOwningIterator(upstream_iter, detached)
            cm.__exit__(None, None, None)
            return upstream_iter
        except BaseException as exc:
            span.set_error(exc)
            cm.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.models, name)


class _GeminiAio:
    def __init__(
        self,
        upstream: Any,
        middleware: Middleware,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self.models = _GeminiAsyncModels(upstream, middleware, observability)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.aio, name)


class _GeminiAsyncModels:
    def __init__(
        self,
        upstream: Any,
        middleware: Middleware,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self._observability = observability or ObservabilityConfig()

    async def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        with compression_span(self._observability, provider="gemini") as span:
            stats = await _gemini_compress_in_place_async(kwargs, self._middleware)
            response = await self._upstream.aio.models.generate_content(*args, **kwargs)
            span.set_stats(stats)
            _attach_telemetry(response, stats, field="usage_metadata")
            return response

    def generate_content_stream(self, *args: Any, **kwargs: Any) -> Any:
        # Upstream returns a coroutine that resolves to an async iterator.
        return _gemini_async_stream_wrapper(
            self._upstream, self._middleware, self._observability, args, kwargs
        )

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.aio.models, name)


async def _gemini_async_stream_wrapper(
    upstream: Any,
    middleware: Middleware,
    observability: ObservabilityConfig,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    cm = compression_span(observability, provider="gemini")
    span = cm.__enter__()
    try:
        stats = await _gemini_compress_in_place_async(kwargs, middleware)
        span.set_stats(stats)
        upstream_iter = await upstream.aio.models.generate_content_stream(*args, **kwargs)
        if span.is_root:
            detached = cm.detach_span()
            cm.__exit__(None, None, None)
            return _SpanOwningAsyncIterator(upstream_iter, detached)
        cm.__exit__(None, None, None)
        return upstream_iter
    except BaseException as exc:
        span.set_error(exc)
        cm.__exit__(type(exc), exc, exc.__traceback__)
        raise


def _gemini_compress_in_place(
    kwargs: dict[str, Any], middleware: Middleware
) -> CompressionStats:
    """Normalize kwargs['contents'], run it through the middleware, and
    write the compressed form back into kwargs.

    On an opaque shape (non-text parts), the original kwargs are left
    untouched and stats with ``method="opaque-bailout"`` are returned
    so observability surfaces the bailout.
    """
    contents = kwargs.get("contents")
    messages, shape = _gemini_contents_to_messages(contents)
    if shape.kind == "opaque":
        return CompressionStats(method="opaque-bailout")
    compressed, stats = middleware.compress_messages(messages)
    kwargs["contents"] = _gemini_messages_to_contents(compressed, shape)
    return stats


async def _gemini_compress_in_place_async(
    kwargs: dict[str, Any], middleware: Middleware
) -> CompressionStats:
    contents = kwargs.get("contents")
    messages, shape = _gemini_contents_to_messages(contents)
    if shape.kind == "opaque":
        return CompressionStats(method="opaque-bailout")
    compressed, stats = await middleware.compress_messages_async(messages)
    kwargs["contents"] = _gemini_messages_to_contents(compressed, shape)
    return stats
