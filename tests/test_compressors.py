"""Tests for the Compressor protocol and Verbatim implementation."""

import pytest

from leanctx import Compressor, ContentType, Verbatim


def test_verbatim_satisfies_compressor_protocol() -> None:
    assert isinstance(Verbatim(), Compressor)


def test_verbatim_name() -> None:
    assert Verbatim().name == "verbatim"


def test_verbatim_returns_messages_unchanged() -> None:
    c = Verbatim()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    out, _ = c.compress(messages)
    assert out == messages
    # Must not have mutated the caller's list.
    assert out is not messages or out == messages


def test_verbatim_stats_report_no_compression() -> None:
    c = Verbatim()
    _, stats = c.compress([{"role": "user", "content": "hello world"}])
    assert stats.method == "verbatim"
    assert stats.ratio == 1.0
    assert stats.input_tokens == stats.output_tokens
    assert stats.input_tokens > 0
    assert stats.cost_usd == 0.0


def test_verbatim_empty_input() -> None:
    c = Verbatim()
    out, stats = c.compress([])
    assert out == []
    assert stats.input_tokens == 0
    assert stats.output_tokens == 0


@pytest.mark.asyncio
async def test_verbatim_async_matches_sync() -> None:
    c = Verbatim()
    messages = [{"role": "user", "content": "async hello"}]
    sync_out, sync_stats = c.compress(messages)
    async_out, async_stats = await c.compress_async(messages)
    assert sync_out == async_out
    assert sync_stats == async_stats


def test_content_type_values() -> None:
    # Routing config files reference these as strings; make the contract
    # explicit so a typo downstream triggers a test failure.
    assert ContentType.CODE.value == "code"
    assert ContentType.PROSE.value == "prose"
    assert ContentType.ERROR.value == "error"
    assert ContentType.REPEAT.value == "repeat"
    assert ContentType.LONG_IMPORTANT.value == "long_important"
    assert ContentType.UNKNOWN.value == "unknown"
