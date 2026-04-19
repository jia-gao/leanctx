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


def test_dedup_does_not_carry_state_across_calls() -> None:
    # Correctness contract: dedup only applies within a single request.
    # Identical content in a later request must survive — it's a fresh
    # user query, not a stale duplicate.
    s = DedupStrategy()
    msg = {"role": "user", "content": "repeated user query"}
    first_call = s.apply([msg])
    second_call = s.apply([msg])
    assert first_call == [msg]
    assert second_call == [msg], "dedup must not leak state across apply() calls"


def test_dedup_reset_is_a_noop() -> None:
    # State is already per-call; reset() is preserved for backward compat
    # but does nothing. Calling it must not affect behavior.
    s = DedupStrategy()
    msg = {"role": "user", "content": "x"}
    s.apply([msg])
    s.reset()
    assert s.apply([msg, msg]) == [msg]


def test_dedup_preserves_different_roles_with_same_content() -> None:
    # Regression for the role-aware hash: "ok" from the user and "ok"
    # from the assistant are distinct turns, not duplicates.
    s = DedupStrategy()
    messages = [
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "ok"},
    ]
    assert s.apply(messages) == messages


def test_dedup_still_drops_duplicates_within_same_role() -> None:
    s = DedupStrategy()
    messages = [
        {"role": "user", "content": "same question"},
        {"role": "user", "content": "same question"},
    ]
    out = s.apply(messages)
    assert len(out) == 1


def test_dedup_skips_tool_result_messages() -> None:
    # Dropping a tool_result would orphan the matching tool_use — unsafe.
    s = DedupStrategy()
    tool_result = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "1", "content": "done"},
        ],
    }
    # Two tool_results with identical content but different tool_use_ids
    # would technically have the same text hash, but we never hash tool-
    # linked messages. Both survive.
    tool_result_b = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "2", "content": "done"},
        ],
    }
    out = s.apply([tool_result, tool_result_b])
    assert len(out) == 2


def test_dedup_skips_tool_use_messages() -> None:
    s = DedupStrategy()
    tool_use = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "1", "name": "ls", "input": {"path": "/"}},
        ],
    }
    tool_use_b = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "2", "name": "ls", "input": {"path": "/"}},
        ],
    }
    out = s.apply([tool_use, tool_use_b])
    assert len(out) == 2


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
