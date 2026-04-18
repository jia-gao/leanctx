"""Surface-level tests for the SDK wrappers. No network calls."""

import importlib.util

import pytest

ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None
OPENAI_AVAILABLE = importlib.util.find_spec("openai") is not None


def test_wrapper_classes_importable_from_top_level() -> None:
    from leanctx import Anthropic, AsyncAnthropic, AsyncOpenAI, OpenAI

    assert Anthropic is not None
    assert AsyncAnthropic is not None
    assert OpenAI is not None
    assert AsyncOpenAI is not None


@pytest.mark.skipif(ANTHROPIC_AVAILABLE, reason="anthropic SDK is installed")
def test_anthropic_raises_without_sdk() -> None:
    from leanctx import Anthropic

    with pytest.raises(ImportError, match="'anthropic' package"):
        Anthropic(api_key="test")


@pytest.mark.skipif(OPENAI_AVAILABLE, reason="openai SDK is installed")
def test_openai_raises_without_sdk() -> None:
    from leanctx import OpenAI

    with pytest.raises(ImportError, match="'openai' package"):
        OpenAI(api_key="test")


@pytest.mark.skipif(OPENAI_AVAILABLE, reason="openai SDK is installed")
def test_async_openai_raises_without_sdk() -> None:
    from leanctx import AsyncOpenAI

    with pytest.raises(ImportError, match="'openai' package"):
        AsyncOpenAI(api_key="test")
