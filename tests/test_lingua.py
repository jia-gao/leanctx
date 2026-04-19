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


def test_lingua_preserves_role_from_first_message() -> None:
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
