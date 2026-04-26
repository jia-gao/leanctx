"""Lingua compressor tests. Uses an injected fake PromptCompressor so
CI doesn't need torch, transformers, or the 1.2 GB model weights."""

import importlib.util
from typing import Any
from unittest.mock import MagicMock

import pytest

from leanctx import Compressor, Lingua

LLMLINGUA_AVAILABLE = importlib.util.find_spec("llmlingua") is not None


def _mock_result(
    *, compressed: str = "short", origin: int = 100, compressed_tokens: int = 10
) -> dict[str, Any]:
    return {
        "compressed_prompt": compressed,
        "origin_tokens": origin,
        "compressed_tokens": compressed_tokens,
    }


def _fake_prompt_compressor(result: dict[str, Any]) -> MagicMock:
    fake = MagicMock()
    fake.compress_prompt.return_value = result
    return fake


# --------------------------------------------------------------------------- #
# Protocol + construction
# --------------------------------------------------------------------------- #


def test_lingua_satisfies_compressor_protocol() -> None:
    assert isinstance(Lingua(), Compressor)


def test_lingua_name() -> None:
    assert Lingua().name == "lingua"


def test_lingua_default_model_and_ratio() -> None:
    lingua = Lingua()
    assert "llmlingua-2" in lingua.model
    assert lingua.ratio == 0.5


def test_lingua_device_defaults_to_auto_detect() -> None:
    # None signals auto-detect on load. Users who want deterministic
    # behavior (Docker images, CI) can pin via device="cpu" | "mps" | "cuda".
    assert Lingua().device is None
    assert Lingua(device="cpu").device == "cpu"


def test_auto_device_returns_valid_device_name() -> None:
    from leanctx.compressors.lingua import _auto_device

    assert _auto_device() in {"cuda", "mps", "cpu"}


# --------------------------------------------------------------------------- #
# Compression path — exercised through an injected mock
# --------------------------------------------------------------------------- #


def test_lingua_compresses_text_via_mock() -> None:
    lingua = Lingua()
    lingua._prompt_compressor = _fake_prompt_compressor(
        _mock_result(compressed="short output", origin=120, compressed_tokens=12)
    )

    messages = [{"role": "user", "content": "long prose " * 200}]
    out, stats = lingua.compress(messages)

    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert out[0]["content"] == "short output"
    assert stats.method == "lingua"
    assert stats.input_tokens == 120
    assert stats.output_tokens == 12
    assert stats.ratio == pytest.approx(0.1)


def test_lingua_preserves_role() -> None:
    lingua = Lingua()
    lingua._prompt_compressor = _fake_prompt_compressor(_mock_result())
    messages = [{"role": "assistant", "content": "big response " * 100}]
    out, _ = lingua.compress(messages)
    assert out[0]["role"] == "assistant"


def test_lingua_calls_compress_prompt_with_configured_ratio() -> None:
    lingua = Lingua(ratio=0.3)
    fake = _fake_prompt_compressor(_mock_result())
    lingua._prompt_compressor = fake

    lingua.compress([{"role": "user", "content": "content to compress"}])

    fake.compress_prompt.assert_called_once()
    kwargs = fake.compress_prompt.call_args.kwargs
    assert kwargs["rate"] == 0.3


def test_lingua_falls_back_to_tiktoken_when_llmlingua_omits_counts() -> None:
    lingua = Lingua()
    # Empty/missing token fields → our own tokenizer takes over.
    lingua._prompt_compressor = _fake_prompt_compressor(
        {"compressed_prompt": "short"}
    )

    messages = [{"role": "user", "content": "much longer input text for counting"}]
    _, stats = lingua.compress(messages)

    assert stats.input_tokens > 0
    assert stats.output_tokens > 0
    assert stats.method == "lingua"


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_lingua_empty_messages_return_empty_without_loading_model() -> None:
    lingua = Lingua()
    # Note: _prompt_compressor stays None — verifies we never touched it.
    out, stats = lingua.compress([])
    assert out == []
    assert stats.method == "lingua"
    assert lingua._prompt_compressor is None


def test_lingua_whitespace_only_input_returns_unchanged() -> None:
    lingua = Lingua()
    messages = [{"role": "user", "content": "   \n\t  "}]
    out, stats = lingua.compress(messages)
    assert out == messages
    assert stats.method == "lingua"
    # Didn't invoke the model for whitespace-only input.
    assert lingua._prompt_compressor is None


@pytest.mark.asyncio
async def test_lingua_async_matches_sync() -> None:
    result = _mock_result(compressed="async compressed", origin=80, compressed_tokens=8)
    lingua_sync = Lingua()
    lingua_sync._prompt_compressor = _fake_prompt_compressor(result)
    lingua_async = Lingua()
    lingua_async._prompt_compressor = _fake_prompt_compressor(result)

    messages = [{"role": "user", "content": "async payload " * 50}]
    sync_out, sync_stats = lingua_sync.compress(messages)
    async_out, async_stats = await lingua_async.compress_async(messages)

    assert sync_out == async_out
    assert sync_stats == async_stats


# --------------------------------------------------------------------------- #
# Missing-dep path
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(LLMLINGUA_AVAILABLE, reason="llmlingua is installed")
def test_lingua_raises_import_error_when_llmlingua_missing() -> None:
    lingua = Lingua()
    # Trigger _load via a non-empty message.
    with pytest.raises(ImportError, match="'llmlingua' package"):
        lingua.compress([{"role": "user", "content": "trigger model load"}])


# --------------------------------------------------------------------------- #
# Block-aware compression (v0.2)
# --------------------------------------------------------------------------- #


def test_lingua_compresses_text_block_in_list_content() -> None:
    lingua = Lingua()
    lingua._prompt_compressor = _fake_prompt_compressor(
        _mock_result(compressed="short", origin=50, compressed_tokens=5)
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "long prose content " * 30}],
        }
    ]
    out, stats = lingua.compress(messages)

    # Block structure preserved — still a list, still a text block.
    assert isinstance(out[0]["content"], list)
    assert out[0]["content"][0]["type"] == "text"
    assert out[0]["content"][0]["text"] == "short"
    assert stats.method == "lingua"
    assert stats.input_tokens == 50
    assert stats.output_tokens == 5


def test_lingua_preserves_tool_use_blocks_verbatim() -> None:
    lingua = Lingua()
    # Mock should NOT be called for tool_use content.
    fake = _fake_prompt_compressor(_mock_result())
    lingua._prompt_compressor = fake

    tool_use_block = {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "read_file",
        "input": {"path": "/etc/hosts"},
    }
    messages = [{"role": "assistant", "content": [tool_use_block]}]
    out, _ = lingua.compress(messages)

    # Block preserved exactly — tool invocation semantics must not change.
    assert out[0]["content"][0] == tool_use_block
    fake.compress_prompt.assert_not_called()


def test_lingua_compresses_tool_result_string_content() -> None:
    lingua = Lingua()
    lingua._prompt_compressor = _fake_prompt_compressor(
        _mock_result(compressed="summarized output", origin=200, compressed_tokens=10)
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "very verbose tool output " * 50,
                }
            ],
        }
    ]
    out, stats = lingua.compress(messages)

    result_block = out[0]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "toolu_1"  # linkage preserved
    assert result_block["content"] == "summarized output"
    assert stats.ratio == pytest.approx(10 / 200)


def test_lingua_preserves_tool_result_with_code_fence_verbatim() -> None:
    """Regression guard: tool_result.content carrying a fenced code block
    must NOT be passed through Lingua — structural integrity (the
    agent-structural bench scenario) requires code preserved verbatim.
    Fix landed 2026-04-26 after agent-structural caught the regression."""
    lingua = Lingua()
    fake = _fake_prompt_compressor(_mock_result())
    lingua._prompt_compressor = fake

    code_payload = (
        "```python\n"
        "DB_POOL_SIZE = 20\n"
        "DB_QUERY_BUDGET = 7\n"
        "```"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": code_payload,
                }
            ],
        }
    ]
    out, _ = lingua.compress(messages)
    block = out[0]["content"][0]
    assert block["content"] == code_payload  # exact preservation
    fake.compress_prompt.assert_not_called()


def test_lingua_preserves_tool_result_with_traceback_verbatim() -> None:
    """Same guard for Python tracebacks in tool_result content."""
    lingua = Lingua()
    fake = _fake_prompt_compressor(_mock_result())
    lingua._prompt_compressor = fake

    tb = (
        "Traceback (most recent call last):\n"
        "  File 'app.py', line 142, in process_payment\n"
        "    conn = pool.acquire(timeout=5)\n"
        "TimeoutError: pool exhausted after 5s"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_3", "content": tb}
            ],
        }
    ]
    out, _ = lingua.compress(messages)
    assert out[0]["content"][0]["content"] == tb
    fake.compress_prompt.assert_not_called()


def test_lingua_compresses_tool_result_nested_text_blocks() -> None:
    lingua = Lingua()
    # Same mock result returned for each call; we're checking the block
    # structure survives regardless of what LLMLingua spits out.
    lingua._prompt_compressor = _fake_prompt_compressor(
        _mock_result(compressed="compact", origin=30, compressed_tokens=3)
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "part one of the output " * 20},
                        {"type": "text", "text": "part two of the output " * 20},
                    ],
                }
            ],
        }
    ]
    out, _ = lingua.compress(messages)

    inner = out[0]["content"][0]["content"]
    assert isinstance(inner, list)
    assert len(inner) == 2
    for inner_block in inner:
        assert inner_block["type"] == "text"
        assert inner_block["text"] == "compact"


def test_lingua_mixed_content_compresses_text_preserves_tool_use() -> None:
    lingua = Lingua()
    lingua._prompt_compressor = _fake_prompt_compressor(
        _mock_result(compressed="compact prose", origin=40, compressed_tokens=4)
    )
    tool_use = {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "search",
        "input": {"q": "kubernetes"},
    }
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "long reasoning prose " * 20},
                tool_use,
            ],
        }
    ]
    out, _ = lingua.compress(messages)

    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "compact prose"}
    assert blocks[1] == tool_use  # unchanged


def test_lingua_image_block_passthrough() -> None:
    lingua = Lingua()
    fake = _fake_prompt_compressor(_mock_result())
    lingua._prompt_compressor = fake

    image_block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
    }
    messages = [{"role": "user", "content": [image_block]}]
    out, _ = lingua.compress(messages)

    # Image preserved verbatim; model not consulted.
    assert out[0]["content"][0] == image_block
    fake.compress_prompt.assert_not_called()


def test_lingua_all_empty_input_skips_model_load() -> None:
    lingua = Lingua()
    # _prompt_compressor stays None — we verify the model was never loaded.
    messages = [{"role": "user", "content": [{"type": "image", "source": {"data": "x"}}]}]
    out, stats = lingua.compress(messages)
    assert out == messages
    assert stats.method == "lingua"
    assert lingua._prompt_compressor is None
