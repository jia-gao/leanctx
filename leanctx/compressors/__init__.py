"""Compressor strategies.

A :class:`Compressor` takes a span of chat messages and returns a possibly
smaller span with matching :class:`CompressionStats`. Concrete strategies:

* :class:`Verbatim` — no-op, used when content must not be altered (code,
  errors, tool schemas). The safe default.
* :class:`Lingua` (coming in v0.1) — wraps Microsoft's LLMLingua-2.
* :class:`SelfLLM` (coming in v0.1) — sends the span to the user's own LLM
  with a summarization prompt.

The :class:`Router` (also v0.1) selects a Compressor per content type.
"""

from leanctx.compressors.base import Compressor, ContentType
from leanctx.compressors.lingua import Lingua
from leanctx.compressors.verbatim import Verbatim

__all__ = ["Compressor", "ContentType", "Lingua", "Verbatim"]
