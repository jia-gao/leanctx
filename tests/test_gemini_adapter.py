"""Tests for Gemini contents <-> leanctx messages adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from leanctx._gemini_adapter import (
    ContentsShape,
    contents_to_messages,
    messages_to_contents,
)

# --------------------------------------------------------------------------- #
# Fake Content / Part objects that quack like google.genai types
# --------------------------------------------------------------------------- #


@dataclass
class _FakePart:
    text: str | None = None
    # Non-text parts leave .text = None; real SDK has function_call,
    # function_response, inline_data, etc.
    function_call: Any = None


@dataclass
class _FakeContent:
    role: str = "user"
    parts: list[Any] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# contents_to_messages — string shape
# --------------------------------------------------------------------------- #


def test_string_contents_becomes_single_user_message() -> None:
    messages, shape = contents_to_messages("hello world")
    assert messages == [{"role": "user", "content": "hello world"}]
    assert shape.kind == "string"


# --------------------------------------------------------------------------- #
# contents_to_messages — list shape
# --------------------------------------------------------------------------- #


def test_list_of_strings_becomes_user_messages() -> None:
    messages, shape = contents_to_messages(["hi", "ho"])
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "ho"},
    ]
    assert shape.kind == "list"


def test_list_of_content_objects_with_text_parts() -> None:
    contents = [
        _FakeContent(role="user", parts=[_FakePart(text="question")]),
        _FakeContent(role="model", parts=[_FakePart(text="answer")]),
    ]
    messages, shape = contents_to_messages(contents)
    assert messages == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert shape.kind == "list"


def test_list_of_content_dicts_with_text_parts() -> None:
    contents = [
        {"role": "user", "parts": [{"text": "q1"}]},
        {"role": "model", "parts": [{"text": "a1"}, {"text": "a1b"}]},
    ]
    messages, shape = contents_to_messages(contents)
    assert messages == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1\na1b"},
    ]
    assert shape.kind == "list"


def test_multiple_text_parts_joined_with_newline() -> None:
    contents = [
        _FakeContent(
            role="user", parts=[_FakePart(text="part one"), _FakePart(text="part two")]
        )
    ]
    messages, _ = contents_to_messages(contents)
    assert messages == [{"role": "user", "content": "part one\npart two"}]


def test_mixed_strings_and_content_objects() -> None:
    contents = [
        "raw user string",
        _FakeContent(role="model", parts=[_FakePart(text="model reply")]),
    ]
    messages, shape = contents_to_messages(contents)
    assert messages == [
        {"role": "user", "content": "raw user string"},
        {"role": "assistant", "content": "model reply"},
    ]
    assert shape.kind == "list"


# --------------------------------------------------------------------------- #
# contents_to_messages — opaque bailout
# --------------------------------------------------------------------------- #


def test_function_call_part_triggers_opaque_bailout() -> None:
    function_call_part = _FakePart(
        text=None, function_call={"name": "f", "args": {"x": 1}}
    )
    contents = [_FakeContent(role="model", parts=[function_call_part])]
    messages, shape = contents_to_messages(contents)
    assert messages == []
    assert shape.kind == "opaque"


def test_unknown_top_level_shape_is_opaque() -> None:
    messages, shape = contents_to_messages({"not": "supported"})
    assert messages == []
    assert shape.kind == "opaque"


def test_empty_text_part_is_skipped_not_opaque() -> None:
    # A part with text="" is still a text part; shouldn't trigger bailout.
    contents = [
        _FakeContent(
            role="user", parts=[_FakePart(text=""), _FakePart(text="real text")]
        )
    ]
    messages, shape = contents_to_messages(contents)
    assert shape.kind == "list"
    assert messages == [{"role": "user", "content": "real text"}]


# --------------------------------------------------------------------------- #
# messages_to_contents — roundtrip
# --------------------------------------------------------------------------- #


def test_string_shape_with_single_message_roundtrips_to_string() -> None:
    result = messages_to_contents(
        [{"role": "user", "content": "compressed"}],
        ContentsShape(kind="string"),
    )
    assert result == "compressed"


def test_string_shape_with_multiple_messages_falls_back_to_list() -> None:
    # If compression somehow returned multiple messages from a single
    # string input, we must still produce a valid contents shape.
    result = messages_to_contents(
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        ContentsShape(kind="string"),
    )
    assert isinstance(result, list)


def test_list_shape_produces_role_parts_dicts() -> None:
    result = messages_to_contents(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
        ContentsShape(kind="list"),
    )
    assert result == [
        {"role": "user", "parts": [{"text": "q"}]},
        {"role": "model", "parts": [{"text": "a"}]},  # assistant -> model
    ]


def test_list_shape_preserves_text_blocks_from_structured_content() -> None:
    # Compression may produce list-of-blocks content; we flatten to
    # Gemini's parts shape.
    result = messages_to_contents(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            }
        ],
        ContentsShape(kind="list"),
    )
    assert result == [
        {"role": "user", "parts": [{"text": "first"}, {"text": "second"}]}
    ]


def test_system_role_maps_to_user_on_output() -> None:
    # Gemini has no native system role.
    result = messages_to_contents(
        [{"role": "system", "content": "instruction"}], ContentsShape(kind="list")
    )
    assert result[0]["role"] == "user"


def test_opaque_shape_raises_if_roundtripped() -> None:
    with pytest.raises(ValueError, match="opaque"):
        messages_to_contents([], ContentsShape(kind="opaque"))


# --------------------------------------------------------------------------- #
# Roundtrip integration
# --------------------------------------------------------------------------- #


def test_roundtrip_string_through_no_op() -> None:
    # Simulate the common case: string in, same string out.
    contents = "the quick brown fox"
    messages, shape = contents_to_messages(contents)
    result = messages_to_contents(messages, shape)
    assert result == contents


def test_roundtrip_list_of_content_objects_through_no_op() -> None:
    contents = [
        _FakeContent(role="user", parts=[_FakePart(text="u1")]),
        _FakeContent(role="model", parts=[_FakePart(text="m1")]),
    ]
    messages, shape = contents_to_messages(contents)
    result = messages_to_contents(messages, shape)
    assert result == [
        {"role": "user", "parts": [{"text": "u1"}]},
        {"role": "model", "parts": [{"text": "m1"}]},
    ]
