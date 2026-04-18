"""Token counting tests. Both tiktoken-available and fallback paths."""

import importlib.util

import pytest

from leanctx.tokens import count_message_tokens, count_tokens

TIKTOKEN_AVAILABLE = importlib.util.find_spec("tiktoken") is not None


def test_count_tokens_empty_string() -> None:
    assert count_tokens("") == 0


def test_count_tokens_short_string_nonzero() -> None:
    # Every non-empty string should report at least 1 token, either via
    # tiktoken or the max(1, ...) guard in the fallback path.
    assert count_tokens("hello") >= 1


def test_count_tokens_scales_with_length() -> None:
    short = count_tokens("hi")
    long = count_tokens("hi " * 1000)
    assert long > short * 100  # sanity: 1000x input → much bigger count


@pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
def test_count_tokens_tiktoken_gpt5_known_string() -> None:
    # "hello world" tokenizes to exactly 2 tokens in cl100k_base / o200k_base.
    assert count_tokens("hello world", model="gpt-5") == 2


@pytest.mark.skipif(TIKTOKEN_AVAILABLE, reason="test covers fallback only")
def test_count_tokens_fallback_approximates_char_over_four() -> None:
    # 40 chars / 4 = 10 tokens under the fallback estimator.
    assert count_tokens("a" * 40) == 10


def test_count_message_tokens_handles_string_content() -> None:
    messages = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hi there"},
    ]
    total = count_message_tokens(messages)
    assert total > 0


def test_count_message_tokens_handles_anthropic_blocks() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    total = count_message_tokens(messages)
    # Both text blocks should contribute.
    assert total >= 2


def test_count_message_tokens_ignores_non_text_blocks() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"data": "..."}},
                {"type": "text", "text": "hi"},
            ],
        }
    ]
    total = count_message_tokens(messages)
    # Image block contributes nothing; only "hi" is counted.
    assert total == count_tokens("hi")
