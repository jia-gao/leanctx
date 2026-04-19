"""Tests for leanctx._content.get_text_content.

Particularly the structured-block paths (tool_use, tool_result, document)
since those are the main real-world compression targets in agent traffic.
"""

from __future__ import annotations

import json

from leanctx._content import get_text_content
from leanctx.tokens import count_message_tokens

# --------------------------------------------------------------------------- #
# String content (OpenAI-style) — regression
# --------------------------------------------------------------------------- #


def test_string_content_returned_verbatim() -> None:
    assert get_text_content({"role": "user", "content": "hello"}) == "hello"


def test_empty_string_content() -> None:
    assert get_text_content({"role": "user", "content": ""}) == ""


def test_missing_content_key_returns_empty() -> None:
    assert get_text_content({"role": "user"}) == ""


# --------------------------------------------------------------------------- #
# Text blocks — regression
# --------------------------------------------------------------------------- #


def test_single_text_block() -> None:
    msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    assert get_text_content(msg) == "hello"


def test_multiple_text_blocks_joined_by_newline() -> None:
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"},
        ],
    }
    assert get_text_content(msg) == "one\ntwo"


# --------------------------------------------------------------------------- #
# tool_use blocks
# --------------------------------------------------------------------------- #


def test_tool_use_dict_input_serialized_with_tool_name() -> None:
    msg = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read_file",
                "input": {"path": "/etc/hosts", "limit": 100},
            }
        ],
    }
    out = get_text_content(msg)
    assert "read_file" in out
    assert "/etc/hosts" in out
    # JSON structure preserved so compression sees both keys
    assert json.dumps({"path": "/etc/hosts", "limit": 100}) in out


def test_tool_use_string_input() -> None:
    msg = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "1", "name": "search", "input": "kubernetes"}
        ],
    }
    out = get_text_content(msg)
    assert "search" in out
    assert "kubernetes" in out


def test_tool_use_without_name() -> None:
    msg = {"role": "assistant", "content": [{"type": "tool_use", "input": {"x": 1}}]}
    out = get_text_content(msg)
    assert out == json.dumps({"x": 1})  # no name prefix


def test_tool_use_null_input() -> None:
    msg = {"role": "assistant", "content": [{"type": "tool_use", "name": "noop"}]}
    assert get_text_content(msg) == ""


# --------------------------------------------------------------------------- #
# tool_result blocks — the main compression target
# --------------------------------------------------------------------------- #


def test_tool_result_string_content() -> None:
    big_output = "line\n" * 200
    msg = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "1", "content": big_output}],
    }
    assert get_text_content(msg) == big_output


def test_tool_result_nested_text_blocks() -> None:
    msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "1",
                "content": [
                    {"type": "text", "text": "first part"},
                    {"type": "text", "text": "second part"},
                ],
            }
        ],
    }
    assert get_text_content(msg) == "first part\nsecond part"


def test_tool_result_contributes_to_token_count() -> None:
    # Regression for the main bug Codex flagged: large tool_result payloads
    # must be visible to the compression trigger.
    big = "grep match line\n" * 500  # ~8 KB of text
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": big}
            ],
        }
    ]
    tokens = count_message_tokens(messages)
    assert tokens > 100, f"expected tool_result payload to be counted, got {tokens}"


# --------------------------------------------------------------------------- #
# document blocks
# --------------------------------------------------------------------------- #


def test_document_with_direct_text() -> None:
    msg = {
        "role": "user",
        "content": [{"type": "document", "text": "contract page 1"}],
    }
    assert get_text_content(msg) == "contract page 1"


def test_document_with_source_data() -> None:
    msg = {
        "role": "user",
        "content": [
            {"type": "document", "source": {"type": "text", "data": "page body"}}
        ],
    }
    assert get_text_content(msg) == "page body"


# --------------------------------------------------------------------------- #
# Mixed content & edge cases
# --------------------------------------------------------------------------- #


def test_mixed_text_and_tool_result() -> None:
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here's what I found:"},
            {
                "type": "tool_result",
                "tool_use_id": "1",
                "content": "big payload",
            },
        ],
    }
    out = get_text_content(msg)
    assert "Here's what I found:" in out
    assert "big payload" in out


def test_image_block_contributes_nothing() -> None:
    msg = {
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "data": "xxx"}},
            {"type": "text", "text": "describe this"},
        ],
    }
    # Only the text survives; image base64 is (correctly) ignored.
    assert get_text_content(msg) == "describe this"


def test_thinking_block_contributes_nothing() -> None:
    msg = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "internal reasoning here"},
            {"type": "text", "text": "final answer"},
        ],
    }
    assert get_text_content(msg) == "final answer"


def test_unknown_block_type_contributes_nothing() -> None:
    msg = {"role": "user", "content": [{"type": "future_type_2030", "payload": "x"}]}
    assert get_text_content(msg) == ""


def test_malformed_block_does_not_raise() -> None:
    # Non-dict items in the content list must be ignored, not blow up.
    msg = {"role": "user", "content": ["bare string", None, 42, {"type": "text", "text": "ok"}]}
    assert get_text_content(msg) == "ok"


def test_dict_content_not_wrapped_in_list() -> None:
    # Some providers have wrapped a single block in an object.
    msg = {"role": "user", "content": {"type": "text", "text": "unwrapped"}}
    assert get_text_content(msg) == "unwrapped"
