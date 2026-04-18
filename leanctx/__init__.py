"""leanctx — drop-in prompt compression for production LLM applications.

v0.0.x is a name-reservation release with the public SDK surface scaffolded
as a passthrough. The working release (v0.1) swaps the passthrough for the
real three-mode compression pipeline (local LLMLingua-2 / self_llm / hybrid).

See https://github.com/jia-gao/leanctx for progress.
"""

from leanctx.client import Anthropic, AsyncAnthropic
from leanctx.middleware import CompressionStats, Middleware

__version__ = "0.0.0"

__all__ = [
    "Anthropic",
    "AsyncAnthropic",
    "CompressionStats",
    "Middleware",
    "__version__",
]
