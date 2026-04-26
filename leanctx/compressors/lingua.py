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

**Block-aware compression (v0.2):** Earlier versions flattened every
message into a single compressed text blob, which destroyed structured
blocks like ``tool_use`` and ``tool_result`` — breaking tool-call
linkage and dropping anything the classifier hadn't already filtered.

v0.2 preserves block structure:
* ``text`` blocks: compressed in place
* ``tool_result`` blocks: inner content recursively compressed
  (string → compressed string; list of blocks → each handled)
* ``tool_use`` blocks: preserved verbatim (the tool's input must not
  be rewritten — that would change the actual tool invocation)
* ``image``, ``thinking``, ``document``, unknown: preserved verbatim

Token counting in :class:`CompressionStats` sums over all blocks so
the reported ratio reflects the whole message, not just the portions
that were compressed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from leanctx._content import get_text_content
from leanctx.observability.compressor_hooks import compressor_span
from leanctx.observability.config import ObservabilityConfig
from leanctx.stats import CompressionStats
from leanctx.tokens import count_tokens

_LINGUA_INSTALL_HINT = (
    "The 'llmlingua' package is required for leanctx.Lingua. "
    "Install with: pip install 'leanctx[lingua]'"
)

_DEFAULT_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


def _auto_device() -> str:
    """Pick the best available device. CUDA > MPS (Apple Silicon) > CPU.

    LLMLingua's default is ``device_map="cuda"``, which blows up on machines
    without a CUDA GPU. We detect and fall back before handing off.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Lingua:
    """Block-aware extractive compressor backed by LLMLingua-2."""

    name = "lingua"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        ratio: float = 0.5,
        device: str | None = None,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self.model = model
        # Same meaning as LLMLingua's ``rate``: fraction of tokens to KEEP.
        # 0.5 = keep half. Lower = more aggressive compression.
        self.ratio = ratio
        # None = auto-detect on first load. Explicit values ("cuda", "mps",
        # "cpu") bypass detection — useful for Docker images where we want
        # deterministic behavior.
        self.device = device
        # Lazy-loaded on first compress() call. Tests assign a mock here
        # directly to skip the real model load.
        self._prompt_compressor: Any = None
        self.observability = observability or ObservabilityConfig()

    # ----------------------------------------------------------------- #
    # Compressor protocol
    # ----------------------------------------------------------------- #

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        with compressor_span(self.observability, name=self.name) as span:
            stats = self._compress_inner(messages)
            span.set_stats(stats[1])
            return stats

    def _compress_inner(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        if not messages:
            return messages, CompressionStats(method="lingua")

        # Short-circuit: skip the model load when there's nothing to
        # compress anywhere in the span.
        if not any(get_text_content(m).strip() for m in messages):
            return messages, CompressionStats(method="lingua")

        new_messages: list[dict[str, Any]] = []
        total_in = 0
        total_out = 0
        for msg in messages:
            new_msg, in_tok, out_tok = self._compress_message(msg)
            new_messages.append(new_msg)
            total_in += in_tok
            total_out += out_tok

        return new_messages, CompressionStats(
            input_tokens=total_in,
            output_tokens=total_out,
            ratio=(total_out / total_in) if total_in else 1.0,
            method="lingua",
        )

    async def compress_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        # LLMLingua is synchronous under the hood. Offload the inner
        # work, but keep the span entered in the calling thread/context
        # so the depth counter behaves correctly.
        with compressor_span(self.observability, name=self.name) as span:
            result = await asyncio.to_thread(self._compress_inner, messages)
            span.set_stats(result[1])
            return result

    # ----------------------------------------------------------------- #
    # Per-message / per-block compression
    # ----------------------------------------------------------------- #

    def _compress_message(
        self, msg: dict[str, Any]
    ) -> tuple[dict[str, Any], int, int]:
        """Compress the compressible text in a single message while
        preserving its structure. Returns ``(new_msg, in_tok, out_tok)``.
        """
        content = msg.get("content")

        # OpenAI-style plain string content — compress the whole thing.
        if isinstance(content, str):
            if not content.strip():
                return msg, 0, 0
            compressed, in_tok, out_tok = self._compress_text(content)
            return {**msg, "content": compressed}, in_tok, out_tok

        # Anthropic-style list of blocks — walk blocks, compress text
        # portions in place, preserve structured blocks verbatim.
        if isinstance(content, list):
            new_blocks: list[Any] = []
            total_in = 0
            total_out = 0
            for block in content:
                new_block, in_tok, out_tok = self._compress_block(block)
                new_blocks.append(new_block)
                total_in += in_tok
                total_out += out_tok
            return {**msg, "content": new_blocks}, total_in, total_out

        # Unknown shape (None, dict, etc.) — pass through.
        return msg, 0, 0

    def _compress_block(self, block: Any) -> tuple[Any, int, int]:
        """Compress text inside a structured block while preserving its
        type and any non-text fields. Returns ``(new_block, in_tok, out_tok)``.

        Passthrough block types (``tool_use``, ``image``, ``thinking``,
        ``document``, and unknown types) still contribute their token
        count to both input and output so the message-level ratio
        reflects the whole message, not just the compressed portion.
        """
        if not isinstance(block, dict):
            # Non-dict garbage — pass through, nothing to count.
            return block, 0, 0

        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "")
            if isinstance(text, str) and text.strip():
                compressed, in_tok, out_tok = self._compress_text(text)
                return {**block, "text": compressed}, in_tok, out_tok
            # Empty or non-string text field — pass through.
            return block, 0, 0

        if btype == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                if inner.strip():
                    compressed, in_tok, out_tok = self._compress_text(inner)
                    return {**block, "content": compressed}, in_tok, out_tok
                return block, 0, 0
            if isinstance(inner, list):
                new_inner: list[Any] = []
                total_in = 0
                total_out = 0
                for inner_block in inner:
                    nb, in_tok, out_tok = self._compress_block(inner_block)
                    new_inner.append(nb)
                    total_in += in_tok
                    total_out += out_tok
                return {**block, "content": new_inner}, total_in, total_out
            return block, 0, 0

        # Pass-through block types. Count tokens so the reported ratio
        # stays honest — a message that's mostly pass-through gets a
        # ratio close to 1.0 rather than looking wildly compressed.
        text = get_text_content({"content": [block]})
        tokens = count_tokens(text) if text else 0
        return block, tokens, tokens

    def _compress_text(self, text: str) -> tuple[str, int, int]:
        """Run LLMLingua on a raw text string."""
        compressor = self._load()
        result = compressor.compress_prompt(text, rate=self.ratio)
        compressed = str(result["compressed_prompt"])
        in_tok = int(result.get("origin_tokens") or count_tokens(text))
        out_tok = int(result.get("compressed_tokens") or count_tokens(compressed))
        return compressed, in_tok, out_tok

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
        device = self.device or _auto_device()
        self._prompt_compressor = llmlingua.PromptCompressor(
            model_name=self.model,
            use_llmlingua2=True,
            device_map=device,
        )
        return self._prompt_compressor
