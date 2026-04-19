"""Deterministic preprocessing strategies.

Strategies run BEFORE compression in the Middleware pipeline. They're
pure Python, no LLM, no heavy models — cheap filters that shrink the
span before the compressor sees it. Running dedup first, for example,
means the LLMLingua-2 model doesn't waste cycles compressing a
duplicate tool output.

Available in v0.1:

* :class:`DedupStrategy` — drop duplicate messages (hash-based)
* :class:`PurgeErrorsStrategy` — strip verbose content from errored
  messages older than N turns, keep the error signal
"""

from leanctx.strategies.base import Strategy
from leanctx.strategies.dedup import DedupStrategy
from leanctx.strategies.purge_errors import PurgeErrorsStrategy

__all__ = ["DedupStrategy", "PurgeErrorsStrategy", "Strategy"]
