"""Smoke tests for the v0.0.0 reservation package."""

import leanctx


def test_version() -> None:
    assert leanctx.__version__ == "0.0.0"


def test_importable() -> None:
    assert hasattr(leanctx, "__version__")
