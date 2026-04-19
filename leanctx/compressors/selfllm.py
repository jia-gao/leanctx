"""SelfLLM — compression by delegation to the user's own configured LLM.

Rather than running a local model (Lingua/LLMLingua-2), SelfLLM calls out
to an LLM — typically a smaller/cheaper sibling of the model the main
request targets, e.g. Haiku compressing for a Sonnet request — and asks
it to produce a compact summary.

Architecturally inspired by OpenCode DCP's pattern of asking the LLM to
produce compression summaries, but written from scratch: different
prompts, different dispatch (direct completion call, not tool-use), and
provider-agnostic shape. No code or prompts copied from DCP (which is
AGPL-3.0).

v0.1 supports provider="anthropic". OpenAI and Gemini land in a follow-up
when the prompt needs provider-specific tuning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from leanctx._content import get_text_content
from leanctx.stats import CompressionStats

_ANTHROPIC_INSTALL_HINT = (
    "The 'anthropic' package is required for SelfLLM(provider='anthropic'). "
    "Install with: pip install 'leanctx[anthropic]'"
)

_SYSTEM_PROMPT = """You are a context compression assistant.
Produce a compact, faithful summary of the provided content that
preserves everything a downstream model would need to continue the
conversation coherently.

Keep:
- specific facts, numbers, entity names, identifiers
- decisions that have already been made
- code snippets, file paths, and error messages verbatim
- the user's stated goals and constraints

Omit:
- repetitive phrasing and polite filler
- intermediate reasoning that reached the same conclusion
- commentary about what was compressed

Return only the summary. No preamble, no explanation.""".strip()


@dataclass(frozen=True)
class _Completion:
    """Provider-neutral completion response shape."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float = 0.0


class SelfLLM:
    """Compressor that delegates to an LLM for summarization.

    The LLM target is configured separately from the main request client
    so users can e.g. compress with Haiku while the main request goes to
    Sonnet. ``_client`` is lazy-loaded; tests inject a fake directly.
    """

    name = "selfllm"

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
        ratio: float = 0.3,
        max_summary_tokens: int = 500,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        # Ratio is a suggestion to the LLM, not a hard constraint — unlike
        # Lingua's ratio which is enforced by the classifier. Lower = more
        # aggressive summarization.
        self.ratio = ratio
        self.max_summary_tokens = max_summary_tokens
        self._client: Any = None

    # ----------------------------------------------------------------- #
    # Compressor protocol
    # ----------------------------------------------------------------- #

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        if not messages:
            return messages, CompressionStats(method="selfllm")

        text = "\n\n".join(get_text_content(m) for m in messages)
        if not text.strip():
            return messages, CompressionStats(method="selfllm")

        client = self._load()
        completion = self._call(client, text)

        role = messages[0].get("role", "user")
        out: list[dict[str, Any]] = [{"role": role, "content": completion.text}]

        return out, CompressionStats(
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            ratio=(completion.output_tokens / completion.input_tokens)
            if completion.input_tokens
            else 1.0,
            method="selfllm",
            cost_usd=completion.cost_usd,
        )

    async def compress_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        # v0.1 uses the sync SDK under the hood. Offload so callers in an
        # event loop don't block. v0.2 will wire AsyncAnthropic/AsyncOpenAI
        # for true non-blocking compression calls.
        return await asyncio.to_thread(self.compress, messages)

    # ----------------------------------------------------------------- #
    # Provider dispatch
    # ----------------------------------------------------------------- #

    def _load(self) -> Any:
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(_ANTHROPIC_INSTALL_HINT) from e
            self._client = anthropic.Anthropic(api_key=self.api_key)
            return self._client
        raise ValueError(
            f"SelfLLM provider {self.provider!r} not supported in v0.1 — "
            f"use 'anthropic' for now"
        )

    def _call(self, client: Any, text: str) -> _Completion:
        if self.provider == "anthropic":
            return self._call_anthropic(client, text)
        raise ValueError(f"provider {self.provider!r} not supported")

    def _call_anthropic(self, client: Any, text: str) -> _Completion:
        suggested_ratio = int(self.ratio * 100)
        user_prompt = (
            f"Compress the content below to roughly {suggested_ratio}% of its "
            f"original length while preserving all information a downstream "
            f"model would need to continue.\n\n<content>\n{text}\n</content>"
        )

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_summary_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Anthropic response: content is list[TextBlock], usage has
        # input_tokens / output_tokens.
        text_out = response.content[0].text if response.content else ""
        return _Completion(
            text=text_out,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=0.0,  # v0.2 adds a pricing table keyed by model
        )
