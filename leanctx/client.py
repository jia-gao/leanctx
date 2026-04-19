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

from leanctx.middleware import Middleware
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


# --------------------------------------------------------------------------- #
# OpenAI wrappers
# --------------------------------------------------------------------------- #


class OpenAI:
    """Drop-in replacement for ``openai.OpenAI``.

    Only ``chat.completions.create`` is intercepted for compression in v0.0.x
    (and that path is a passthrough). Other attributes — ``embeddings``,
    ``files``, ``completions`` (legacy), ``responses``, ``models``, etc. —
    forward directly to the upstream client.
    """

    def __init__(
        self,
        *args: Any,
        leanctx_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pkg = _load_openai()
        self._upstream = pkg.OpenAI(*args, **kwargs)
        self._middleware = Middleware(leanctx_config or {})
        self.chat = _Chat(self._upstream, self._middleware)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ fires only when normal lookup misses. Guard against the
        # pre-init state where _upstream isn't in __dict__ yet (debuggers,
        # pickling, autocomplete can poke at attrs before __init__ returns).
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
        self._middleware = Middleware(leanctx_config or {})
        self.chat = _AsyncChat(self._upstream, self._middleware)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream, name)


class _Chat:
    def __init__(self, upstream: Any, middleware: Middleware) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self.completions = _Completions(upstream, middleware)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.chat, name)


class _AsyncChat:
    def __init__(self, upstream: Any, middleware: Middleware) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self.completions = _AsyncCompletions(upstream, middleware)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.chat, name)


class _Completions:
    def __init__(self, upstream: Any, middleware: Middleware) -> None:
        self._upstream = upstream
        self._middleware = middleware

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        compressed, stats = self._middleware.compress_messages(messages)
        kwargs["messages"] = compressed
        # Returns the non-stream response or a Stream iterator depending on
        # kwargs["stream"]. Either way the upstream return value is correct;
        # telemetry attaches only when a concrete response with .usage exists.
        response = self._upstream.chat.completions.create(**kwargs)
        _attach_telemetry(response, stats)
        return response

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.chat.completions, name)


class _AsyncCompletions:
    def __init__(self, upstream: Any, middleware: Middleware) -> None:
        self._upstream = upstream
        self._middleware = middleware

    async def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        compressed, stats = await self._middleware.compress_messages_async(messages)
        kwargs["messages"] = compressed
        response = await self._upstream.chat.completions.create(**kwargs)
        _attach_telemetry(response, stats)
        return response

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.chat.completions, name)


# --------------------------------------------------------------------------- #
# Gemini (google-genai) wrapper
# --------------------------------------------------------------------------- #
#
# Unlike Anthropic/OpenAI, google-genai has a single Client class that exposes
# sync methods under `.models` and async variants under `.aio.models`. We match
# that shape — one Gemini class, no AsyncGemini.
#
# In v0.0.x the generate_content intercept skips the Middleware: the message
# shape differs (contents can be a string, list of strings, or Content objects
# with parts). v0.1 will add a Gemini-aware normalization step before
# dispatching to the shared Middleware.


class Gemini:
    """Drop-in replacement for ``google.genai.Client``.

    Exposes sync methods at ``client.models`` and async methods at
    ``client.aio.models``, matching the upstream SDK exactly.

    **v0.1 limitation:** Gemini's ``contents`` parameter can be a
    string, a list of strings, or a list of ``Content`` objects — none
    of which map directly to the leanctx middleware's OpenAI-style
    ``list[dict]`` message shape. Rather than ship a half-working
    normalization, v0.1 intercepts ``generate_content`` to attach
    zero-valued :class:`CompressionStats` telemetry and forwards the
    request unmodified. Users who need actual compression should route
    through :class:`Anthropic` or :class:`OpenAI` for now. Full
    normalization lands in v0.2.
    """

    def __init__(
        self,
        *args: Any,
        leanctx_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pkg = _load_gemini()
        self._upstream = pkg.Client(*args, **kwargs)
        self._middleware = Middleware(leanctx_config or {})
        self.models = _GeminiModels(self._upstream, self._middleware)
        self.aio = _GeminiAio(self._upstream, self._middleware)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream, name)


class _GeminiModels:
    def __init__(self, upstream: Any, middleware: Middleware) -> None:
        self._upstream = upstream
        self._middleware = middleware

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        # v0.1: middleware is skipped for Gemini pending contents-shape
        # normalization. See Gemini class docstring for details.
        response = self._upstream.models.generate_content(*args, **kwargs)
        _attach_telemetry(response, CompressionStats(), field="usage_metadata")
        return response

    def generate_content_stream(self, *args: Any, **kwargs: Any) -> Any:
        # Streaming returns an iterator of chunks; telemetry aggregation
        # across the stream is v0.2 work. Pass through untouched.
        return self._upstream.models.generate_content_stream(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.models, name)


class _GeminiAio:
    def __init__(self, upstream: Any, middleware: Middleware) -> None:
        self._upstream = upstream
        self._middleware = middleware
        self.models = _GeminiAsyncModels(upstream, middleware)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.aio, name)


class _GeminiAsyncModels:
    def __init__(self, upstream: Any, middleware: Middleware) -> None:
        self._upstream = upstream
        self._middleware = middleware

    async def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._upstream.aio.models.generate_content(*args, **kwargs)
        _attach_telemetry(response, CompressionStats(), field="usage_metadata")
        return response

    def generate_content_stream(self, *args: Any, **kwargs: Any) -> Any:
        # Upstream returns a coroutine that resolves to an async iterator.
        # Forwarding it directly preserves that semantics — the caller
        # awaits the result, then async-iterates. Not `async def` to avoid
        # adding an extra coroutine layer.
        return self._upstream.aio.models.generate_content_stream(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        upstream = self.__dict__.get("_upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream.aio.models, name)
