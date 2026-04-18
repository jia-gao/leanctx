"""Passthrough tests for the v0.0.x middleware."""

import pytest

from leanctx.middleware import CompressionStats, Middleware


def test_compress_messages_is_passthrough() -> None:
    mw = Middleware({})
    messages = [{"role": "user", "content": "hello world"}]
    out, _ = mw.compress_messages(messages)
    assert out == messages


def test_stats_report_passthrough() -> None:
    mw = Middleware({})
    _, stats = mw.compress_messages([{"role": "user", "content": "hi"}])
    assert isinstance(stats, CompressionStats)
    assert stats.method == "passthrough"
    assert stats.ratio == 1.0
    assert stats.cost_usd == 0.0


@pytest.mark.asyncio
async def test_compress_messages_async_is_passthrough() -> None:
    mw = Middleware({})
    messages = [{"role": "user", "content": "async hello"}]
    out, stats = await mw.compress_messages_async(messages)
    assert out == messages
    assert stats.method == "passthrough"


def test_config_is_stored() -> None:
    cfg = {"mode": "hybrid"}
    mw = Middleware(cfg)
    assert mw.config is cfg
