"""Lingua — extractive compression via Microsoft LLMLingua-2.

Wraps the ``llmlingua`` package (MIT licensed, HF model cached locally).
The underlying model is a BERT-style token classifier trained to predict
which tokens in a prompt are keep-worthy; output is a subset of the
original tokens with surviving wording preserved verbatim.

The ``llmlingua`` package plus its ~1.2 GB model weights are an optional
dependency, loaded lazily on first use. Users install them via::

    pip install 'leanctx[lingua]'

Tests inject a fake PromptCompressor through ``_prompt_compressor`` so
CI can exercise the code path without torch / transformers / the model.
"""

from __future__ import annotations

import asyncio
from typing import Any

from leanctx._content import get_text_content
from leanctx.stats import CompressionStats
from leanctx.tokens import count_tokens

_LINGUA_INSTALL_HINT = (
    "The 'llmlingua' package is required for leanctx.Lingua. "
    "Install with: pip install 'leanctx[lingua]'"
)

_DEFAULT_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


class Lingua:
    """Extractive compressor backed by LLMLingua-2.

    Single-message compression: the Middleware calls ``compress`` with a
    one-message list, so we join + compress + repack into one message
    with the original role. Multi-message aggregation is the Middleware's
    job (via its message-by-message loop).
    """

    name = "lingua"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        ratio: float = 0.5,
    ) -> None:
        self.model = model
        # Same meaning as LLMLingua's ``rate``: fraction of tokens to KEEP.
        # 0.5 = keep half. Lower = more aggressive compression.
        self.ratio = ratio
        # Lazy-loaded on first compress() call. Tests assign a mock here
        # directly to skip the real model load.
        self._prompt_compressor: Any = None

    # ----------------------------------------------------------------- #
    # Compressor protocol
    # ----------------------------------------------------------------- #

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        if not messages:
            return messages, CompressionStats(method="lingua")

        text = "\n\n".join(get_text_content(m) for m in messages)
        if not text.strip():
            # Nothing to compress — avoid a model load for empty input.
            return messages, CompressionStats(method="lingua")

        compressor = self._load()
        result = compressor.compress_prompt(text, rate=self.ratio)

        compressed_text = str(result["compressed_prompt"])
        role = messages[0].get("role", "user")
        out: list[dict[str, Any]] = [{"role": role, "content": compressed_text}]

        # LLMLingua reports its own token counts; fall back to our tiktoken-
        # based counter when fields are missing.
        input_tokens = int(result.get("origin_tokens") or count_tokens(text))
        output_tokens = int(
            result.get("compressed_tokens") or count_tokens(compressed_text)
        )
        return out, CompressionStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ratio=(output_tokens / input_tokens) if input_tokens else 1.0,
            method="lingua",
        )

    async def compress_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        # LLMLingua is synchronous under the hood. Offload so we don't
        # block the event loop.
        return await asyncio.to_thread(self.compress, messages)

    # ----------------------------------------------------------------- #
    # Lazy loader
    # ----------------------------------------------------------------- #

    def _load(self) -> Any:
        if self._prompt_compressor is not None:
            return self._prompt_compressor
        try:
            import llmlingua
        except ImportError as e:
            raise ImportError(_LINGUA_INSTALL_HINT) from e
        self._prompt_compressor = llmlingua.PromptCompressor(
            model_name=self.model,
            use_llmlingua2=True,
        )
        return self._prompt_compressor
