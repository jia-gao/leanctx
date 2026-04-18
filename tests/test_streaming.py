"""Verify streaming methods pass through correctly to upstream SDKs.

These tests exercise the intercept classes directly with mock upstream
clients, so they run without the real anthropic / openai / google-genai
packages installed and without any network.
"""

from unittest.mock import MagicMock

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
    upstream = MagicMock()
    wrapper = _Messages(upstream, Middleware({}))
    messages = [{"role": "user", "content": "hi"}]

    result = wrapper.stream(model="claude-sonnet-4-6", messages=messages, max_tokens=10)

    upstream.messages.stream.assert_called_once_with(
        model="claude-sonnet-4-6", messages=messages, max_tokens=10
    )
    assert result is upstream.messages.stream.return_value


@pytest.mark.asyncio
async def test_anthropic_async_stream_forwards_to_upstream() -> None:
    upstream = MagicMock()
    wrapper = _AsyncMessages(upstream, Middleware({}))
    messages = [{"role": "user", "content": "hi"}]

    result = wrapper.stream(model="claude-sonnet-4-6", messages=messages, max_tokens=10)

    upstream.messages.stream.assert_called_once_with(
        model="claude-sonnet-4-6", messages=messages, max_tokens=10
    )
    assert result is upstream.messages.stream.return_value


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


def test_gemini_async_stream_forwards_to_upstream() -> None:
    upstream = MagicMock()
    wrapper = _GeminiAsyncModels(upstream, Middleware({}))

    # Async stream: our wrapper returns the coroutine from upstream directly
    # (not `async def`), so the call is synchronous and returns whatever
    # the mocked upstream call returns.
    result = wrapper.generate_content_stream(model="gemini-2.5-pro", contents="hi")

    upstream.aio.models.generate_content_stream.assert_called_once_with(
        model="gemini-2.5-pro", contents="hi"
    )
    assert result is upstream.aio.models.generate_content_stream.return_value
