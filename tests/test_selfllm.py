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


def test_selfllm_unsupported_provider_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="not supported"):
        SelfLLM(provider="cohere")


def test_selfllm_default_model_resolves_per_provider() -> None:
    assert SelfLLM(provider="anthropic").model.startswith("claude-haiku")
    # OpenAI default must be a non-reasoning model so the completion
    # token budget actually produces visible output. gpt-4o-mini works;
    # gpt-5-nano (reasoning) burns the whole budget on hidden tokens.
    assert SelfLLM(provider="openai").model == "gpt-4o-mini"
    assert SelfLLM(provider="gemini").model.startswith("gemini-")


def test_selfllm_explicit_model_wins_over_default() -> None:
    assert SelfLLM(provider="openai", model="gpt-5.3-codex").model == "gpt-5.3-codex"


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


# --------------------------------------------------------------------------- #
# OpenAI provider
# --------------------------------------------------------------------------- #


def _fake_openai_response(
    *, text: str = "openai summary", prompt_tokens: int = 150, completion_tokens: int = 15
) -> SimpleNamespace:
    """Quacks like openai.types.chat.ChatCompletion."""
    message = SimpleNamespace(content=text, role="assistant")
    choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model="gpt-5-nano")


def test_selfllm_openai_compresses_via_mock() -> None:
    llm = SelfLLM(provider="openai")
    fake = MagicMock()
    fake.chat.completions.create.return_value = _fake_openai_response(
        text="compact", prompt_tokens=200, completion_tokens=20
    )
    llm._client = fake

    messages = [{"role": "user", "content": "long input " * 40}]
    out, stats = llm.compress(messages)

    assert out[0]["content"] == "compact"
    assert stats.method == "selfllm"
    assert stats.input_tokens == 200
    assert stats.output_tokens == 20
    assert stats.ratio == pytest.approx(0.1)


def test_selfllm_openai_passes_system_and_user_messages() -> None:
    llm = SelfLLM(provider="openai", max_summary_tokens=300)
    fake = MagicMock()
    fake.chat.completions.create.return_value = _fake_openai_response()
    llm._client = fake

    llm.compress([{"role": "user", "content": "payload"}])

    kwargs = fake.chat.completions.create.call_args.kwargs
    # System prompt goes in the messages list for OpenAI, not as a param.
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["max_completion_tokens"] == 300
    # Default model (gpt-4o-mini) is non-reasoning, so we should NOT
    # pass reasoning_effort.
    assert "reasoning_effort" not in kwargs


def test_selfllm_openai_sets_minimal_reasoning_effort_for_gpt5() -> None:
    # gpt-5 family is reasoning; without effort="minimal" the model
    # silently burns the completion-token budget on hidden reasoning.
    llm = SelfLLM(provider="openai", model="gpt-5-nano")
    fake = MagicMock()
    fake.chat.completions.create.return_value = _fake_openai_response()
    llm._client = fake

    llm.compress([{"role": "user", "content": "x"}])

    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == "minimal"


def test_selfllm_openai_sets_minimal_reasoning_effort_for_o_series() -> None:
    for model in ("o1-mini", "o3-mini", "o4-mini"):
        llm = SelfLLM(provider="openai", model=model)
        fake = MagicMock()
        fake.chat.completions.create.return_value = _fake_openai_response()
        llm._client = fake

        llm.compress([{"role": "user", "content": "x"}])

        kwargs = fake.chat.completions.create.call_args.kwargs
        assert kwargs.get("reasoning_effort") == "minimal", (
            f"expected minimal reasoning_effort for {model}"
        )


def test_selfllm_openai_no_reasoning_effort_for_non_reasoning_models() -> None:
    for model in ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4-turbo"):
        llm = SelfLLM(provider="openai", model=model)
        fake = MagicMock()
        fake.chat.completions.create.return_value = _fake_openai_response()
        llm._client = fake

        llm.compress([{"role": "user", "content": "x"}])

        kwargs = fake.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in kwargs, (
            f"unexpected reasoning_effort for non-reasoning model {model}"
        )


def test_selfllm_openai_empty_choices_handled() -> None:
    llm = SelfLLM(provider="openai")
    fake = MagicMock()
    # Rare but real: provider returns no choices (content filter, etc.)
    fake.chat.completions.create.return_value = SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=5, completion_tokens=0)
    )
    llm._client = fake

    out, stats = llm.compress([{"role": "user", "content": "x"}])
    assert out[0]["content"] == ""
    assert stats.output_tokens == 0


# --------------------------------------------------------------------------- #
# Gemini provider
# --------------------------------------------------------------------------- #


def _fake_gemini_response(
    *,
    text: str = "gemini summary",
    prompt_token_count: int = 180,
    candidates_token_count: int = 18,
) -> SimpleNamespace:
    """Quacks like google.genai types.GenerateContentResponse."""
    usage = SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
        total_token_count=prompt_token_count + candidates_token_count,
    )
    return SimpleNamespace(text=text, usage_metadata=usage)


def test_selfllm_gemini_compresses_via_mock() -> None:
    llm = SelfLLM(provider="gemini")
    fake = MagicMock()
    fake.models.generate_content.return_value = _fake_gemini_response(
        text="compact", prompt_token_count=200, candidates_token_count=20
    )
    llm._client = fake

    messages = [{"role": "user", "content": "long input " * 40}]
    out, stats = llm.compress(messages)

    assert out[0]["content"] == "compact"
    assert stats.method == "selfllm"
    assert stats.input_tokens == 200
    assert stats.output_tokens == 20


def test_selfllm_gemini_uses_system_instruction_in_config() -> None:
    llm = SelfLLM(provider="gemini", max_summary_tokens=250)
    fake = MagicMock()
    fake.models.generate_content.return_value = _fake_gemini_response()
    llm._client = fake

    llm.compress([{"role": "user", "content": "payload"}])

    kwargs = fake.models.generate_content.call_args.kwargs
    assert kwargs["model"].startswith("gemini-")
    assert "system_instruction" in kwargs["config"]
    assert kwargs["config"]["max_output_tokens"] == 250


def test_selfllm_gemini_disables_thinking_for_2_5_models() -> None:
    # Default model is gemini-2.5-flash, a thinking model. Without
    # thinking_budget=0 it silently burns the output-token budget on
    # hidden thinking tokens.
    llm = SelfLLM(provider="gemini")
    fake = MagicMock()
    fake.models.generate_content.return_value = _fake_gemini_response()
    llm._client = fake

    llm.compress([{"role": "user", "content": "x"}])

    kwargs = fake.models.generate_content.call_args.kwargs
    assert kwargs["config"]["thinking_config"] == {"thinking_budget": 0}


def test_selfllm_gemini_no_thinking_config_for_legacy_models() -> None:
    # Gemini 1.5 doesn't support thinking_config; passing it would error.
    llm = SelfLLM(provider="gemini", model="gemini-1.5-flash")
    fake = MagicMock()
    fake.models.generate_content.return_value = _fake_gemini_response()
    llm._client = fake

    llm.compress([{"role": "user", "content": "x"}])

    kwargs = fake.models.generate_content.call_args.kwargs
    assert "thinking_config" not in kwargs["config"]


def test_selfllm_gemini_missing_usage_metadata_handled() -> None:
    llm = SelfLLM(provider="gemini")
    fake = MagicMock()
    # Gemini streaming aggregates or errored responses may lack usage.
    fake.models.generate_content.return_value = SimpleNamespace(
        text="ok", usage_metadata=None
    )
    llm._client = fake

    out, stats = llm.compress([{"role": "user", "content": "x"}])
    assert out[0]["content"] == "ok"
    assert stats.input_tokens == 0
    assert stats.output_tokens == 0
