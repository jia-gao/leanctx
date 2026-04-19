"""SelfLLM compressor tests.

Uses MagicMock to fake the upstream Anthropic client so we don't hit
the real API or require credentials in CI.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from leanctx import Compressor, SelfLLM

ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None


def _fake_anthropic_response(
    *, text: str = "short summary", input_tokens: int = 100, output_tokens: int = 10
) -> SimpleNamespace:
    """Build an object that quacks like an anthropic Message response."""
    return SimpleNamespace(
        content=[SimpleNamespace(text=text, type="text")],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _fake_client(response: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = response
    return client


# --------------------------------------------------------------------------- #
# Protocol + construction
# --------------------------------------------------------------------------- #


def test_selfllm_satisfies_compressor_protocol() -> None:
    assert isinstance(SelfLLM(), Compressor)


def test_selfllm_name() -> None:
    assert SelfLLM().name == "selfllm"


def test_selfllm_defaults() -> None:
    llm = SelfLLM()
    assert llm.provider == "anthropic"
    assert llm.model == "claude-haiku-4-5"
    assert llm.ratio == 0.3
    assert llm.max_summary_tokens == 500


def test_selfllm_unsupported_provider_raises() -> None:
    llm = SelfLLM(provider="cohere")
    with pytest.raises(ValueError, match="not supported"):
        llm.compress([{"role": "user", "content": "trigger load"}])


# --------------------------------------------------------------------------- #
# Compression via injected mock
# --------------------------------------------------------------------------- #


def test_selfllm_compresses_via_mock() -> None:
    llm = SelfLLM()
    llm._client = _fake_client(
        _fake_anthropic_response(text="summary", input_tokens=200, output_tokens=20)
    )

    messages = [{"role": "user", "content": "long input to compress " * 50}]
    out, stats = llm.compress(messages)

    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert out[0]["content"] == "summary"
    assert stats.method == "selfllm"
    assert stats.input_tokens == 200
    assert stats.output_tokens == 20
    assert stats.ratio == pytest.approx(0.1)


def test_selfllm_preserves_role_from_first_message() -> None:
    llm = SelfLLM()
    llm._client = _fake_client(_fake_anthropic_response())

    out, _ = llm.compress([{"role": "assistant", "content": "previous response text"}])
    assert out[0]["role"] == "assistant"


def test_selfllm_passes_model_and_max_tokens_to_upstream() -> None:
    llm = SelfLLM(model="claude-sonnet-4-6", max_summary_tokens=250)
    fake = _fake_client(_fake_anthropic_response())
    llm._client = fake

    llm.compress([{"role": "user", "content": "content"}])

    call_kwargs = fake.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["max_tokens"] == 250
    assert "system" in call_kwargs
    assert "messages" in call_kwargs


def test_selfllm_embeds_ratio_hint_in_user_prompt() -> None:
    llm = SelfLLM(ratio=0.2)
    fake = _fake_client(_fake_anthropic_response())
    llm._client = fake

    llm.compress([{"role": "user", "content": "x"}])

    user_msg = fake.messages.create.call_args.kwargs["messages"][0]
    # 0.2 -> "20%" hint in the prompt
    assert "20%" in user_msg["content"]


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_selfllm_empty_input_does_not_call_upstream() -> None:
    llm = SelfLLM()
    fake = MagicMock()
    llm._client = fake
    out, stats = llm.compress([])
    assert out == []
    assert stats.method == "selfllm"
    fake.messages.create.assert_not_called()


def test_selfllm_whitespace_input_does_not_call_upstream() -> None:
    llm = SelfLLM()
    fake = MagicMock()
    llm._client = fake
    messages = [{"role": "user", "content": "   \n\t  "}]
    out, _ = llm.compress(messages)
    assert out == messages
    fake.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_selfllm_async_matches_sync() -> None:
    response = _fake_anthropic_response(text="s", input_tokens=50, output_tokens=5)
    sync_llm = SelfLLM()
    sync_llm._client = _fake_client(response)
    async_llm = SelfLLM()
    async_llm._client = _fake_client(response)

    messages = [{"role": "user", "content": "async payload"}]
    sync_out, sync_stats = sync_llm.compress(messages)
    async_out, async_stats = await async_llm.compress_async(messages)

    assert sync_out == async_out
    assert sync_stats == async_stats


# --------------------------------------------------------------------------- #
# Missing-dep path
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(ANTHROPIC_AVAILABLE, reason="anthropic is installed")
def test_selfllm_raises_import_error_when_anthropic_missing() -> None:
    llm = SelfLLM()
    with pytest.raises(ImportError, match="'anthropic' package"):
        llm.compress([{"role": "user", "content": "trigger load"}])
