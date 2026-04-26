"""AC-2: stream-path span lifetime (paths 2, 4, 8, 12) + GC backstop."""

from __future__ import annotations

import asyncio
import gc
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from leanctx.client import (
    _AsyncCompletions,
    _AsyncStreamContextManager,
    _GeminiAsyncModels,
    _SyncStreamContextManager,
)
from leanctx.middleware import Middleware
from leanctx.observability.config import ObservabilityConfig
from tests.observability.conftest import otel_required


def _obs() -> ObservabilityConfig:
    return ObservabilityConfig(otel=True)


@otel_required
def test_anthropic_sync_stream_span_spans_enter_exit(spans: Any) -> None:
    """Path 2: Anthropic sync messages.stream(). Span opens on the
    leanctx wrapper's __enter__ and closes on __exit__ — NOT during
    the synchronous stream() call (which returns the wrapper)."""

    class _UpstreamCM:
        def __enter__(self) -> str:
            return "stream_iter"

        def __exit__(self, *_: Any) -> None:
            return None

    upstream = MagicMock()
    upstream.stream = MagicMock(return_value=_UpstreamCM())
    mw = Middleware({"mode": "off"}, observability=_obs())
    cm = _SyncStreamContextManager(
        upstream,
        {"messages": [{"role": "user", "content": "x"}]},
        mw,
        _obs(),
    )

    # Span is NOT yet emitted (no __enter__ yet).
    pre_finish = len(spans.get_finished_spans())

    with cm as stream_iter:
        # Span is open here; we mid-iteration check no spans have
        # finished yet.
        assert len(spans.get_finished_spans()) == pre_finish
        assert stream_iter == "stream_iter"

    # After __exit__, exactly one root span has been emitted.
    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "anthropic"


@otel_required
def test_anthropic_async_stream_span_spans_aenter_aexit(spans: Any) -> None:
    """Path 4: Anthropic async messages.stream() — span lifetime spans
    __aenter__ to __aexit__ of the leanctx wrapper."""

    class _UpstreamAsyncCM:
        async def __aenter__(self) -> str:
            return "async_stream_iter"

        async def __aexit__(self, *_: Any) -> None:
            return None

    upstream = MagicMock()
    upstream.stream = MagicMock(return_value=_UpstreamAsyncCM())
    mw = Middleware({"mode": "off"}, observability=_obs())

    async def go() -> None:
        cm = _AsyncStreamContextManager(
            upstream, {"messages": [{"role": "user", "content": "y"}]}, mw, _obs()
        )
        pre = len(spans.get_finished_spans())
        async with cm as it:
            assert it == "async_stream_iter"
            assert len(spans.get_finished_spans()) == pre

    asyncio.run(go())

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "anthropic"


@otel_required
def test_openai_async_stream_emits_one_root(spans: Any) -> None:
    """Path 8: OpenAI async stream=True returns a wrapped async iterator."""

    class _AsyncIter:
        def __init__(self) -> None:
            self._idx = 0

        def __aiter__(self) -> _AsyncIter:
            return self

        async def __anext__(self) -> int:
            self._idx += 1
            if self._idx > 2:
                raise StopAsyncIteration
            return self._idx

    upstream = MagicMock()
    upstream.chat.completions.create = AsyncMock(return_value=_AsyncIter())
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _AsyncCompletions(upstream, mw, _obs())

    async def go() -> list[int]:
        result = await wrapper.create(
            messages=[{"role": "user", "content": "hi"}], stream=True
        )
        return [c async for c in result]

    chunks = asyncio.run(go())
    assert chunks == [1, 2]

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "openai"


@otel_required
def test_gemini_async_stream_emits_one_root(spans: Any) -> None:
    """Path 12: Gemini async generate_content_stream returns a coroutine
    that resolves to a wrapped async iterator."""

    class _AsyncIter:
        def __init__(self) -> None:
            self._idx = 0

        def __aiter__(self) -> _AsyncIter:
            return self

        async def __anext__(self) -> int:
            self._idx += 1
            if self._idx > 2:
                raise StopAsyncIteration
            return self._idx

    upstream = MagicMock()
    upstream.aio.models.generate_content_stream = AsyncMock(return_value=_AsyncIter())
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _GeminiAsyncModels(upstream, mw, _obs())

    async def go() -> list[int]:
        coro = wrapper.generate_content_stream(model="gemini-test", contents="hi")
        result = await coro
        return [c async for c in result]

    chunks = asyncio.run(go())
    assert chunks == [1, 2]

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "gemini"


@otel_required
def test_openai_stream_span_open_during_iteration(spans: Any) -> None:
    """AC-2 negative test: a streaming wrapper that hasn't finished
    iterating must NOT have closed its span yet."""
    from leanctx.client import _Completions

    upstream = MagicMock()
    upstream.chat.completions.create = MagicMock(return_value=iter([1, 2, 3]))
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _Completions(upstream, mw, _obs())

    result = wrapper.create(messages=[{"role": "user", "content": "x"}], stream=True)

    # Pull one chunk; span must still be open.
    next(iter(result))
    finished = spans.get_finished_spans()
    assert not any(s.name == "leanctx.compress" for s in finished), (
        "leanctx.compress span closed mid-iteration"
    )

    # Drain the rest; span closes.
    list(result)
    finished = spans.get_finished_spans()
    assert any(s.name == "leanctx.compress" for s in finished)


@otel_required
def test_openai_stream_span_closes_on_gc_abandonment(spans: Any) -> None:
    """AC-2 negative test: a stream iterator dropped without close()
    must still close the span via __del__ GC backstop."""
    from leanctx.client import _Completions

    upstream = MagicMock()
    upstream.chat.completions.create = MagicMock(return_value=iter([1, 2]))
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _Completions(upstream, mw, _obs())

    pre = len(spans.get_finished_spans())

    # Start iteration but don't consume; then drop the reference.
    result = wrapper.create(messages=[{"role": "user", "content": "x"}], stream=True)
    next(iter(result))
    del result
    gc.collect()

    finished = spans.get_finished_spans()
    after = len(finished)
    assert after > pre, "expected the abandoned stream span to close on GC"
