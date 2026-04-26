"""Smoke tests for the leanctx package."""

import leanctx


def test_version() -> None:
    assert leanctx.__version__ == "0.3.0"


def test_importable() -> None:
    assert hasattr(leanctx, "__version__")
