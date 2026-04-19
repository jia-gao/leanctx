"""Middleware tests — mode gating, threshold trigger, pipeline wiring."""

import pytest

from leanctx import CompressionStats, Middleware

# --------------------------------------------------------------------------- #
# Default / off mode — backward-compatible passthrough
# --------------------------------------------------------------------------- #


def test_default_config_is_passthrough() -> None:
    mw = Middleware({})
    messages = [{"role": "user", "content": "hello world"}]
    out, stats = mw.compress_messages(messages)
    assert out == messages
    assert stats.method == "passthrough"
    assert stats.ratio == 1.0


def test_explicit_off_mode_is_passthrough() -> None:
    mw = Middleware({"mode": "off"})
    out, stats = mw.compress_messages([{"role": "user", "content": "hi"}])
    assert stats.method == "passthrough"


@pytest.mark.asyncio
async def test_default_async_is_passthrough() -> None:
    mw = Middleware({})
    messages = [{"role": "user", "content": "async hello"}]
    out, stats = await mw.compress_messages_async(messages)
    assert out == messages
    assert stats.method == "passthrough"


def test_config_is_stored() -> None:
    cfg = {"mode": "hybrid"}
    mw = Middleware(cfg)
    assert mw.config is cfg


# --------------------------------------------------------------------------- #
# Active mode — pipeline runs, but all-Verbatim still preserves messages
# --------------------------------------------------------------------------- #


def test_on_mode_below_threshold_skips_pipeline() -> None:
    mw = Middleware({"mode": "on", "trigger": {"threshold_tokens": 100_000}})
    messages = [{"role": "user", "content": "short message"}]
    out, stats = mw.compress_messages(messages)
    assert out == messages
    assert stats.method == "below-threshold"


def test_on_mode_above_threshold_runs_pipeline_verbatim() -> None:
    # Force threshold to 0 so the pipeline always runs.
    mw = Middleware({"mode": "on", "trigger": {"threshold_tokens": 0}})
    messages = [{"role": "user", "content": "some prose content"}]
    out, stats = mw.compress_messages(messages)
    # Only Verbatim available in v0.0.x, so messages come through unchanged.
    assert out == messages
    assert stats.method == "verbatim"
    assert stats.input_tokens == stats.output_tokens
    assert stats.input_tokens > 0


def test_on_mode_drops_duplicates() -> None:
    mw = Middleware({"mode": "on", "trigger": {"threshold_tokens": 0}})
    msg = {"role": "user", "content": "duplicate tool output content"}
    out, _ = mw.compress_messages([msg, msg, msg])
    assert len(out) == 1


def test_on_mode_empty_input_returns_empty() -> None:
    mw = Middleware({"mode": "on"})
    out, stats = mw.compress_messages([])
    assert out == []
    assert stats.input_tokens == 0


@pytest.mark.asyncio
async def test_on_mode_async_pipeline_runs() -> None:
    mw = Middleware({"mode": "on", "trigger": {"threshold_tokens": 0}})
    messages = [{"role": "user", "content": "async prose"}]
    out, stats = await mw.compress_messages_async(messages)
    assert out == messages
    assert stats.method == "verbatim"


# --------------------------------------------------------------------------- #
# Config parsing — forward compatibility for v0.1 compressor names
# --------------------------------------------------------------------------- #


def test_unknown_compressor_in_routing_is_skipped_gracefully() -> None:
    # A forward-compatible config may reference compressor names that
    # don't exist yet (e.g. a future 'semantic' compressor). The router
    # logs a warning and falls back to the default (Verbatim).
    mw = Middleware(
        {
            "mode": "on",
            "trigger": {"threshold_tokens": 0},
            "routing": {"prose": "semantic"},  # unknown -> default
        }
    )
    messages = [{"role": "user", "content": "some prose to route"}]
    out, _ = mw.compress_messages(messages)
    assert out == messages  # default Verbatim preserves


def test_unknown_content_type_in_routing_is_skipped() -> None:
    mw = Middleware(
        {
            "mode": "on",
            "routing": {"not-a-real-type": "verbatim"},
        }
    )
    # Just verify construction doesn't raise.
    assert mw is not None


def test_stats_uses_passthrough_for_off_mode() -> None:
    _, stats = Middleware({"mode": "off"}).compress_messages(
        [{"role": "user", "content": "x"}]
    )
    assert isinstance(stats, CompressionStats)
    assert stats.method == "passthrough"
