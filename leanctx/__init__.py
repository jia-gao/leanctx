"""leanctx — drop-in prompt compression for production LLM applications.

v0.0.x is a name-reservation release with the public SDK surface scaffolded
as a passthrough. The working release (v0.1) swaps the passthrough for the
real three-mode compression pipeline (local LLMLingua-2 / self_llm / hybrid).

See https://github.com/jia-gao/leanctx for progress.
"""

from leanctx.classifier import RepeatTracker, classify
from leanctx.client import Anthropic, AsyncAnthropic, AsyncOpenAI, Gemini, OpenAI
from leanctx.compressors import Compressor, ContentType, Lingua, SelfLLM, Verbatim
from leanctx.middleware import Middleware
from leanctx.observability import MethodStatus, ObservabilityConfig
from leanctx.router import Router
from leanctx.stats import CompressionStats
from leanctx.strategies import DedupStrategy, PurgeErrorsStrategy, Strategy

__version__ = "0.3.0"

__all__ = [
    "Anthropic",
    "AsyncAnthropic",
    "AsyncOpenAI",
    "CompressionStats",
    "Compressor",
    "ContentType",
    "DedupStrategy",
    "Gemini",
    "Lingua",
    "MethodStatus",
    "Middleware",
    "ObservabilityConfig",
    "OpenAI",
    "PurgeErrorsStrategy",
    "RepeatTracker",
    "Router",
    "SelfLLM",
    "Strategy",
    "Verbatim",
    "__version__",
    "classify",
]
