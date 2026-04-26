"""AC-2: every wrapper request path emits exactly one root leanctx.compress span.

These tests use mock upstream SDKs so we don't need real API access. The
focus is on span emission shape, attribute population, and provider
context propagation (AC-4) per the 12-path enumeration in plan-otel.md.
"""

from __future__ import annotations

import asyncio
from typing import Any
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
from leanctx.observability.config import ObservabilityConfig
from tests.observability.conftest import otel_required


def _obs() -> ObservabilityConfig:
    return ObservabilityConfig(otel=True)


def _make_response_with_usage(usage_field: str = "usage") -> Any:
    """Create a mock response with a settable usage attribute."""

    class _Usage:
        input_tokens = 10
        output_tokens = 5

    class _Response:
        pass

    resp = _Response()
    setattr(resp, usage_field, _Usage())
    return resp


@otel_required
def test_anthropic_messages_create_emits_one_root_with_provider(spans: Any) -> None:
    """Path 1: leanctx.Anthropic.messages.create() sync."""
    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_make_response_with_usage())
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _Messages(upstream, mw, _obs())

    wrapper.create(messages=[{"role": "user", "content": "hi"}])

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "anthropic"


@otel_required
def test_anthropic_async_messages_create_emits_one_root(spans: Any) -> None:
    """Path 3: leanctx.AsyncAnthropic.messages.create() async."""
    upstream = MagicMock()
    upstream.messages.create = AsyncMock(return_value=_make_response_with_usage())
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _AsyncMessages(upstream, mw, _obs())

    asyncio.run(wrapper.create(messages=[{"role": "user", "content": "hi"}]))

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "anthropic"


@otel_required
def test_openai_completions_create_emits_one_root(spans: Any) -> None:
    """Path 5: leanctx.OpenAI.chat.completions.create() sync stream=False."""
    upstream = MagicMock()
    upstream.chat.completions.create = MagicMock(return_value=_make_response_with_usage())
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _Completions(upstream, mw, _obs())

    wrapper.create(messages=[{"role": "user", "content": "hi"}])

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "openai"


@otel_required
def test_openai_completions_create_stream_emits_one_root(spans: Any) -> None:
    """Path 6: OpenAI sync stream=True returns iterator wrapped with span ownership."""
    upstream = MagicMock()
    fake_stream = iter([1, 2, 3])
    upstream.chat.completions.create = MagicMock(return_value=fake_stream)
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _Completions(upstream, mw, _obs())

    result = wrapper.create(messages=[{"role": "user", "content": "hi"}], stream=True)
    chunks = list(result)
    assert chunks == [1, 2, 3]

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "openai"


@otel_required
def test_openai_async_completions_create_emits_one_root(spans: Any) -> None:
    """Path 7: leanctx.AsyncOpenAI.chat.completions.create() async stream=False."""
    upstream = MagicMock()
    upstream.chat.completions.create = AsyncMock(return_value=_make_response_with_usage())
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _AsyncCompletions(upstream, mw, _obs())

    asyncio.run(wrapper.create(messages=[{"role": "user", "content": "hi"}]))

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "openai"


@otel_required
def test_gemini_generate_content_emits_one_root(spans: Any) -> None:
    """Path 9: leanctx.Gemini.models.generate_content() sync."""
    upstream = MagicMock()
    upstream.models.generate_content = MagicMock(
        return_value=_make_response_with_usage("usage_metadata")
    )
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _GeminiModels(upstream, mw, _obs())

    wrapper.generate_content(model="gemini-test", contents="hello")

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "gemini"


@otel_required
def test_gemini_async_generate_content_emits_one_root(spans: Any) -> None:
    """Path 10: leanctx.Gemini.aio.models.generate_content() async."""
    upstream = MagicMock()
    upstream.aio.models.generate_content = AsyncMock(
        return_value=_make_response_with_usage("usage_metadata")
    )
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _GeminiAsyncModels(upstream, mw, _obs())

    asyncio.run(wrapper.generate_content(model="gemini-test", contents="hello"))

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.provider"] == "gemini"


@otel_required
def test_gemini_generate_content_stream_emits_one_root(spans: Any) -> None:
    """Path 11: Gemini sync generate_content_stream returns wrapped iterator."""
    upstream = MagicMock()
    upstream.models.generate_content_stream = MagicMock(return_value=iter([1, 2]))
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _GeminiModels(upstream, mw, _obs())

    result = wrapper.generate_content_stream(model="gemini-test", contents="hello")
    list(result)

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1


@otel_required
def test_gemini_opaque_bailout_method_recorded(spans: Any) -> None:
    """AC-3: Gemini contents with non-text parts (function_call) emit
    leanctx.method='opaque-bailout' instead of 'passthrough'."""
    upstream = MagicMock()
    upstream.models.generate_content = MagicMock(
        return_value=_make_response_with_usage("usage_metadata")
    )
    mw = Middleware({"mode": "on"}, observability=_obs())
    wrapper = _GeminiModels(upstream, mw, _obs())

    # Opaque shape: a Content-like object with a non-text part.
    opaque_contents = [{"role": "user", "parts": [{"function_call": {"name": "x"}}]}]

    wrapper.generate_content(model="gemini-test", contents=opaque_contents)

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 1
    assert roots[0].attributes["leanctx.method"] == "opaque-bailout"


@otel_required
def test_observability_disabled_means_zero_spans(spans: Any) -> None:
    """AC-2 negative: observability.otel=False produces zero spans."""
    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_make_response_with_usage())
    obs_off = ObservabilityConfig(otel=False)
    mw = Middleware({"mode": "off"}, observability=obs_off)
    wrapper = _Messages(upstream, mw, obs_off)

    wrapper.create(messages=[{"role": "user", "content": "hi"}])

    finished = spans.get_finished_spans()
    assert len(finished) == 0


@otel_required
def test_wrapper_routed_call_does_not_double_emit_compress_span(spans: Any) -> None:
    """AC-2 negative test + AC-6 row 1: wrapper-routed calls emit exactly
    ONE leanctx.compress span. Middleware's nested compression_span must
    be suppressed by the depth counter."""
    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_make_response_with_usage())
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _Messages(upstream, mw, _obs())

    wrapper.create(messages=[{"role": "user", "content": "hi"}])

    finished = spans.get_finished_spans()
    compress_spans = [s for s in finished if s.name == "leanctx.compress"]
    assert len(compress_spans) == 1, (
        f"expected exactly 1 leanctx.compress span (wrapper root, "
        f"middleware suppressed), got {len(compress_spans)}: "
        f"{[(s.name, s.attributes.get('leanctx.provider')) for s in compress_spans]}"
    )


@otel_required
def test_concurrent_async_calls_produce_independent_root_spans(spans: Any) -> None:
    """AC-6: asyncio.gather of two wrapper calls produces two roots, no
    parent-child confusion."""
    upstream = MagicMock()
    upstream.messages.create = AsyncMock(return_value=_make_response_with_usage())
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _AsyncMessages(upstream, mw, _obs())

    async def run_two() -> None:
        await asyncio.gather(
            wrapper.create(messages=[{"role": "user", "content": "a"}]),
            wrapper.create(messages=[{"role": "user", "content": "b"}]),
        )

    asyncio.run(run_two())

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 2
    # Neither span has the other as a parent.
    for s in roots:
        assert s.parent is None or not any(
            o.context.span_id == s.parent.span_id for o in roots
        )


@otel_required
def test_exception_unwind_resets_depth_counter(spans: Any) -> None:
    """AC-6 exception-unwind: a wrapper call where upstream raises must
    not strand the depth counter — a subsequent call still produces a root."""
    upstream = MagicMock()
    upstream.messages.create = MagicMock(side_effect=RuntimeError("boom"))
    mw = Middleware({"mode": "off"}, observability=_obs())
    wrapper = _Messages(upstream, mw, _obs())

    with pytest.raises(RuntimeError):
        wrapper.create(messages=[{"role": "user", "content": "hi"}])

    upstream.messages.create = MagicMock(return_value=_make_response_with_usage())
    wrapper.create(messages=[{"role": "user", "content": "again"}])

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    assert len(roots) == 2
    # First span is the error span; second is success.
    assert any(
        s.attributes is not None and s.attributes.get("leanctx.error") is True
        for s in roots
    )
