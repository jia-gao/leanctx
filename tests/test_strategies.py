"""Tests for DedupStrategy and PurgeErrorsStrategy."""

from __future__ import annotations

from leanctx import DedupStrategy, PurgeErrorsStrategy, Strategy

# --------------------------------------------------------------------------- #
# DedupStrategy
# --------------------------------------------------------------------------- #


def test_dedup_satisfies_strategy_protocol() -> None:
    assert isinstance(DedupStrategy(), Strategy)


def test_dedup_passes_through_unique_messages() -> None:
    s = DedupStrategy()
    messages = [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    assert s.apply(messages) == messages


def test_dedup_drops_duplicate_content() -> None:
    s = DedupStrategy()
    duplicate = {"role": "user", "content": "same text"}
    messages = [duplicate, duplicate, duplicate]
    out = s.apply(messages)
    assert len(out) == 1


def test_dedup_is_persistent_across_calls() -> None:
    s = DedupStrategy()
    msg = {"role": "user", "content": "persistent"}
    first_call = s.apply([msg])
    second_call = s.apply([msg])
    assert first_call == [msg]
    assert second_call == []


def test_dedup_reset_clears_state() -> None:
    s = DedupStrategy()
    msg = {"role": "user", "content": "reset-me"}
    s.apply([msg])
    s.reset()
    out = s.apply([msg])
    assert out == [msg]


# --------------------------------------------------------------------------- #
# PurgeErrorsStrategy
# --------------------------------------------------------------------------- #


def test_purge_errors_satisfies_strategy_protocol() -> None:
    assert isinstance(PurgeErrorsStrategy(), Strategy)


def test_purge_errors_short_history_is_untouched() -> None:
    # Fewer messages than after_turns means nothing should be purged.
    s = PurgeErrorsStrategy(after_turns=4)
    messages = [
        {"role": "user", "content": "Traceback (most recent call last):\n  huge error"},
        {"role": "assistant", "content": "ok"},
    ]
    assert s.apply(messages) == messages


def test_purge_errors_purges_old_errors() -> None:
    s = PurgeErrorsStrategy(after_turns=2)
    old_error = {
        "role": "user",
        "content": "Traceback (most recent call last):\n" + ("big payload\n" * 50),
    }
    messages = [
        old_error,
        {"role": "assistant", "content": "handled"},
        {"role": "user", "content": "next question"},
        {"role": "assistant", "content": "answer"},
    ]
    out = s.apply(messages)
    # Only the first (old error) should be purged.
    assert out[0]["content"] == "[errored output purged for context compaction]"
    assert out[1:] == messages[1:]


def test_purge_errors_preserves_recent_errors() -> None:
    # The error itself is within the 'after_turns' tail, so it stays.
    s = PurgeErrorsStrategy(after_turns=3)
    recent_error = {
        "role": "user",
        "content": "Traceback (most recent call last):\n  bad call",
    }
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "r1"},
        recent_error,
        {"role": "assistant", "content": "r2"},
    ]
    out = s.apply(messages)
    # Only messages[0] is before the cutoff (len=4, after=3 -> cutoff=1).
    # messages[0] is prose, so it's left alone.
    # messages[2] is an error but within the tail, left alone.
    assert out == messages


def test_purge_errors_leaves_non_errors_alone() -> None:
    s = PurgeErrorsStrategy(after_turns=1)
    messages = [
        {"role": "user", "content": "normal prose message"},
        {"role": "assistant", "content": "normal response"},
        {"role": "user", "content": "recent"},
    ]
    out = s.apply(messages)
    assert out == messages


def test_purge_errors_custom_placeholder() -> None:
    s = PurgeErrorsStrategy(after_turns=1, placeholder="[gone]")
    old_error = {"role": "user", "content": "Error: boom\n" + ("x" * 100)}
    messages = [old_error, {"role": "assistant", "content": "ok"}]
    out = s.apply(messages)
    assert out[0]["content"] == "[gone]"
