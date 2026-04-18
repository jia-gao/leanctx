"""Surface-level tests for the SDK wrappers. No network calls."""

import importlib.util

import pytest

ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None
OPENAI_AVAILABLE = importlib.util.find_spec("openai") is not None

# find_spec("google.genai") raises ModuleNotFoundError when the `google`
# namespace package isn't present at all, rather than returning None like it
# does for single-segment modules. Fall back to a try-import.
try:
    from google import genai as _genai  # noqa: F401

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def test_wrapper_classes_importable_from_top_level() -> None:
    from leanctx import Anthropic, AsyncAnthropic, AsyncOpenAI, Gemini, OpenAI

    assert Anthropic is not None
    assert AsyncAnthropic is not None
    assert OpenAI is not None
    assert AsyncOpenAI is not None
    assert Gemini is not None


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


@pytest.mark.skipif(GEMINI_AVAILABLE, reason="google-genai SDK is installed")
def test_gemini_raises_without_sdk() -> None:
    from leanctx import Gemini

    with pytest.raises(ImportError, match="'google-genai' package"):
        Gemini(api_key="test")
