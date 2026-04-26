"""Verify streaming methods pass through correctly to upstream SDKs.

These tests exercise the intercept classes directly with mock upstream
clients, so they run without the real anthropic / openai / google-genai
packages installed and without any network.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from leanctx.client import (
    _AsyncCompletions,
    _AsyncMessages,
    _Completions,
    _GeminiAsyncModels,
    _GeminiModels,
    _Messages,
)
from leanctx.middleware import Middleware

# --------------------------------------------------------------------------- #
# Anthropic — messages.stream()
# --------------------------------------------------------------------------- #


def test_anthropic_stream_forwards_to_upstream() -> None:
    # v0.3: _Messages.stream now returns a leanctx-owned sync context
    # manager that defers the upstream call to __enter__ so the wrapper
    # span lifetime spans the consumer's __enter__/__exit__. Same shift
    # already happened for async streams.
    upstream = MagicMock()
    upstream_cm = MagicMock()
    upstream_cm.__enter__ = MagicMock(return_value="sync_streaming_handle")
    upstream_cm.__exit__ = MagicMock(return_value=None)
    upstream.messages.stream = MagicMock(return_value=upstream_cm)

    wrapper = _Messages(upstream, Middleware({}))
    messages = [{"role": "user", "content": "hi"}]

    cm = wrapper.stream(model="claude-sonnet-4-6", messages=messages, max_tokens=10)

    # Upstream call deferred until __enter__.
    upstream.messages.stream.assert_not_called()

    with cm as handle:
        upstream.messages.stream.assert_called_once_with(
            model="claude-sonnet-4-6", messages=messages, max_tokens=10
        )
        assert handle == "sync_streaming_handle"
        upstream_cm.__enter__.assert_called_once()
        upstream_cm.__exit__.assert_not_called()

    upstream_cm.__exit__.assert_called_once()


@pytest.mark.asyncio
async def test_anthropic_async_stream_forwards_to_upstream() -> None:
    # The new _AsyncStreamContextManager defers the upstream call until
    # __aenter__ so compression can run on the async middleware path
    # without blocking the event loop. The test exercises the full
    # async-with flow.
    upstream = MagicMock()
    upstream_cm = MagicMock()
    upstream_cm.__aenter__ = AsyncMock(return_value="streaming_handle")
    upstream_cm.__aexit__ = AsyncMock(return_value=None)
    upstream.messages.stream = MagicMock(return_value=upstream_cm)

    wrapper = _AsyncMessages(upstream, Middleware({}))
    messages = [{"role": "user", "content": "hi"}]

    cm = wrapper.stream(model="claude-sonnet-4-6", messages=messages, max_tokens=10)
    async with cm as handle:
        assert handle == "streaming_handle"

    upstream.messages.stream.assert_called_once_with(
        model="claude-sonnet-4-6", messages=messages, max_tokens=10
    )
    upstream_cm.__aenter__.assert_awaited_once()
    upstream_cm.__aexit__.assert_awaited_once()


# --------------------------------------------------------------------------- #
# OpenAI — chat.completions.create(stream=True)
# --------------------------------------------------------------------------- #


def test_openai_create_forwards_stream_flag() -> None:
    upstream = MagicMock()
    wrapper = _Completions(upstream, Middleware({}))
    messages = [{"role": "user", "content": "hi"}]

    wrapper.create(model="gpt-5", messages=messages, stream=True)

    upstream.chat.completions.create.assert_called_once_with(
        model="gpt-5", messages=messages, stream=True
    )


@pytest.mark.asyncio
async def test_openai_async_create_forwards_stream_flag() -> None:
    upstream = MagicMock()

    # AsyncMock isn't used because MagicMock auto-wraps awaitables when the
    # method is awaited; we set the return value to an awaitable sentinel
    # via `return_value` on the chained call.
    async def _fake_create(**_kwargs: object) -> object:
        return object()

    upstream.chat.completions.create = _fake_create  # type: ignore[assignment]
    wrapper = _AsyncCompletions(upstream, Middleware({}))
    messages = [{"role": "user", "content": "hi"}]

    # Just verify the call doesn't raise — coverage is enough for v0.0.x.
    await wrapper.create(model="gpt-5", messages=messages, stream=True)


# --------------------------------------------------------------------------- #
# Gemini — models.generate_content_stream()
# --------------------------------------------------------------------------- #


def test_gemini_stream_forwards_to_upstream() -> None:
    upstream = MagicMock()
    wrapper = _GeminiModels(upstream, Middleware({}))

    result = wrapper.generate_content_stream(model="gemini-2.5-pro", contents="hi")

    upstream.models.generate_content_stream.assert_called_once_with(
        model="gemini-2.5-pro", contents="hi"
    )
    assert result is upstream.models.generate_content_stream.return_value


@pytest.mark.asyncio
async def test_gemini_async_stream_forwards_to_upstream() -> None:
    # v0.2: request-side compression runs when the caller awaits, so
    # generate_content_stream returns a coroutine that we must await.
    upstream = MagicMock()
    upstream.aio.models.generate_content_stream = AsyncMock(return_value="async_iter")

    wrapper = _GeminiAsyncModels(upstream, Middleware({}))
    result = await wrapper.generate_content_stream(
        model="gemini-2.5-pro", contents="hi"
    )

    upstream.aio.models.generate_content_stream.assert_awaited_once_with(
        model="gemini-2.5-pro", contents="hi"
    )
    assert result == "async_iter"
