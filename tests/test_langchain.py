"""Tests for leanctx.integrations.langchain."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import pytest

from leanctx.integrations.langchain import from_dicts, to_dicts

LANGCHAIN_AVAILABLE = importlib.util.find_spec("langchain_core") is not None


@dataclass
class _FakeLCMessage:
    """Stand-in for a LangChain BaseMessage (duck-typed)."""

    type: str
    content: str


def test_to_dicts_maps_human_to_user() -> None:
    msgs = [_FakeLCMessage(type="human", content="hello")]
    out = to_dicts(msgs)
    assert out == [{"role": "user", "content": "hello"}]


def test_to_dicts_maps_ai_to_assistant() -> None:
    msgs = [_FakeLCMessage(type="ai", content="hi back")]
    out = to_dicts(msgs)
    assert out == [{"role": "assistant", "content": "hi back"}]


def test_to_dicts_passes_system_through() -> None:
    msgs = [_FakeLCMessage(type="system", content="you are helpful")]
    out = to_dicts(msgs)
    assert out == [{"role": "system", "content": "you are helpful"}]


def test_to_dicts_unknown_type_preserved() -> None:
    msgs = [_FakeLCMessage(type="custom_type", content="x")]
    out = to_dicts(msgs)
    assert out == [{"role": "custom_type", "content": "x"}]


def test_to_dicts_missing_type_defaults_to_user() -> None:
    class WithoutType:
        content = "no type"

    out = to_dicts([WithoutType()])
    assert out == [{"role": "user", "content": "no type"}]


def test_to_dicts_missing_content_defaults_to_empty() -> None:
    class WithoutContent:
        type = "human"

    out = to_dicts([WithoutContent()])
    assert out == [{"role": "user", "content": ""}]


def test_to_dicts_round_trip_with_compressible_content() -> None:
    msgs = [
        _FakeLCMessage(type="human", content="question"),
        _FakeLCMessage(type="ai", content="answer"),
    ]
    dicts = to_dicts(msgs)
    assert dicts == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_to_dicts_preserves_tool_call_id() -> None:
    @dataclass
    class _FakeToolMsg:
        type: str
        content: str
        tool_call_id: str

    msgs = [_FakeToolMsg(type="tool", content="grep output", tool_call_id="call_42")]
    dicts = to_dicts(msgs)
    assert dicts[0]["role"] == "tool"
    assert dicts[0]["tool_call_id"] == "call_42"


def test_to_dicts_preserves_tool_calls_on_ai_message() -> None:
    @dataclass
    class _FakeAIWithTools:
        type: str
        content: str
        tool_calls: list[dict[str, Any]]

    calls = [{"id": "call_1", "name": "read_file", "args": {"path": "x"}}]
    msgs = [_FakeAIWithTools(type="ai", content="calling read_file", tool_calls=calls)]
    dicts = to_dicts(msgs)
    assert dicts[0]["tool_calls"] == calls


def test_to_dicts_preserves_name_on_function_message() -> None:
    @dataclass
    class _FakeFuncMsg:
        type: str
        content: str
        name: str

    msgs = [_FakeFuncMsg(type="function", content="result", name="do_thing")]
    dicts = to_dicts(msgs)
    assert dicts[0]["name"] == "do_thing"


@pytest.mark.skipif(LANGCHAIN_AVAILABLE, reason="langchain_core is installed")
def test_from_dicts_raises_without_langchain() -> None:
    with pytest.raises(ImportError, match="langchain_core"):
        from_dicts([{"role": "user", "content": "hi"}])


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain_core required")
def test_from_dicts_produces_langchain_messages() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    lc_messages = from_dicts(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert isinstance(lc_messages[0], SystemMessage)
    assert isinstance(lc_messages[1], HumanMessage)
    assert isinstance(lc_messages[2], AIMessage)
    assert lc_messages[0].content == "s"
    assert lc_messages[2].content == "a"
