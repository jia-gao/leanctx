"""Surface-level tests for the SDK wrapper. No network calls."""

import importlib.util

import pytest

ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None


def test_wrapper_classes_importable_from_top_level() -> None:
    from leanctx import Anthropic, AsyncAnthropic

    assert Anthropic is not None
    assert AsyncAnthropic is not None


@pytest.mark.skipif(ANTHROPIC_AVAILABLE, reason="anthropic SDK is installed")
def test_anthropic_raises_without_sdk() -> None:
    from leanctx import Anthropic

    with pytest.raises(ImportError, match="'anthropic' package"):
        Anthropic(api_key="test")
