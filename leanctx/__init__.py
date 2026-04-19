"""leanctx — drop-in prompt compression for production LLM applications.

v0.0.x is a name-reservation release with the public SDK surface scaffolded
as a passthrough. The working release (v0.1) swaps the passthrough for the
real three-mode compression pipeline (local LLMLingua-2 / self_llm / hybrid).

See https://github.com/jia-gao/leanctx for progress.
"""

from leanctx.classifier import RepeatTracker, classify
from leanctx.client import Anthropic, AsyncAnthropic, AsyncOpenAI, Gemini, OpenAI
from leanctx.compressors import Compressor, ContentType, Lingua, Verbatim
from leanctx.middleware import Middleware
from leanctx.router import Router
from leanctx.stats import CompressionStats

__version__ = "0.0.0"

__all__ = [
    "Anthropic",
    "AsyncAnthropic",
    "AsyncOpenAI",
    "CompressionStats",
    "Compressor",
    "ContentType",
    "Gemini",
    "Lingua",
    "Middleware",
    "OpenAI",
    "RepeatTracker",
    "Router",
    "Verbatim",
    "__version__",
    "classify",
]
