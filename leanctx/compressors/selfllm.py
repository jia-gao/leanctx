"""SelfLLM — compression by delegation to the user's own configured LLM.

Rather than running a local model (Lingua/LLMLingua-2), SelfLLM calls out
to an LLM — typically a smaller/cheaper sibling of the model the main
request targets, e.g. Haiku compressing for a Sonnet request, or
gpt-5-nano compressing for a gpt-5 request — and asks it to produce a
compact summary.

Architecturally inspired by OpenCode DCP's pattern of asking the LLM to
produce compression summaries, but written from scratch: different
prompts, different dispatch (direct completion call, not tool-use), and
provider-agnostic shape.

v0.2 supports ``anthropic`` / ``openai`` / ``gemini``. Provider-specific
prompt tuning (e.g. different phrasing for reasoning models) lands as
needed.
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
_OPENAI_INSTALL_HINT = (
    "The 'openai' package is required for SelfLLM(provider='openai'). "
    "Install with: pip install 'leanctx[openai]'"
)
_GEMINI_INSTALL_HINT = (
    "The 'google-genai' package is required for SelfLLM(provider='gemini'). "
    "Install with: pip install 'leanctx[gemini]'"
)

_SUPPORTED_PROVIDERS = frozenset({"anthropic", "openai", "gemini"})

# Sensible cheap default per provider. Users override via the ``model``
# kwarg when they want frontier quality on compression too.
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-nano",
    "gemini": "gemini-2.5-flash",
}

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

    Pick ``provider`` + ``model`` independently of the main request's
    target so you can e.g. compress with Haiku while the real request
    goes to Sonnet, or compress with gpt-5-nano while your app uses
    gpt-5. ``_client`` is lazy-loaded; tests inject a fake directly.
    """

    name = "selfllm"

    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
        ratio: float = 0.3,
        max_summary_tokens: int = 500,
    ) -> None:
        if provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"SelfLLM provider {provider!r} not supported — "
                f"use one of {sorted(_SUPPORTED_PROVIDERS)}"
            )
        self.provider = provider
        self.model = model or _DEFAULT_MODELS[provider]
        self.api_key = api_key
        # Ratio is a suggestion to the LLM, not a hard constraint — unlike
        # Lingua's ratio which is enforced by the model. Lower = more
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
        # v0.2 still uses the sync SDK under the hood. Offload so callers
        # in an event loop don't block. v0.3 will wire async provider
        # clients for true non-blocking compression calls.
        return await asyncio.to_thread(self.compress, messages)

    # ----------------------------------------------------------------- #
    # Provider dispatch
    # ----------------------------------------------------------------- #

    def _load(self) -> Any:
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            self._client = _load_anthropic_client(self.api_key)
        elif self.provider == "openai":
            self._client = _load_openai_client(self.api_key)
        elif self.provider == "gemini":
            self._client = _load_gemini_client(self.api_key)
        else:  # pragma: no cover — guarded in __init__
            raise ValueError(f"provider {self.provider!r}")
        return self._client

    def _call(self, client: Any, text: str) -> _Completion:
        user_prompt = self._user_prompt(text)
        if self.provider == "anthropic":
            return self._call_anthropic(client, user_prompt)
        if self.provider == "openai":
            return self._call_openai(client, user_prompt)
        if self.provider == "gemini":
            return self._call_gemini(client, user_prompt)
        raise ValueError(f"provider {self.provider!r}")  # pragma: no cover

    def _user_prompt(self, text: str) -> str:
        suggested_ratio = int(self.ratio * 100)
        return (
            f"Compress the content below to roughly {suggested_ratio}% of its "
            f"original length while preserving all information a downstream "
            f"model would need to continue.\n\n<content>\n{text}\n</content>"
        )

    def _call_anthropic(self, client: Any, user_prompt: str) -> _Completion:
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_summary_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_out = response.content[0].text if response.content else ""
        return _Completion(
            text=text_out,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _call_openai(self, client: Any, user_prompt: str) -> _Completion:
        response = client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_summary_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text_out = ""
        if response.choices:
            msg = response.choices[0].message
            text_out = (msg.content if msg and msg.content else "") or ""
        usage = response.usage
        return _Completion(
            text=text_out,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )

    def _call_gemini(self, client: Any, user_prompt: str) -> _Completion:
        # Gemini puts the system instruction in `config`, not in messages.
        response = client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config={
                "system_instruction": _SYSTEM_PROMPT,
                "max_output_tokens": self.max_summary_tokens,
            },
        )
        text_out = getattr(response, "text", "") or ""
        usage = getattr(response, "usage_metadata", None)
        return _Completion(
            text=text_out,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
        )


# --------------------------------------------------------------------------- #
# Per-provider client loaders
# --------------------------------------------------------------------------- #


def _load_anthropic_client(api_key: str | None) -> Any:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(_ANTHROPIC_INSTALL_HINT) from e
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _load_openai_client(api_key: str | None) -> Any:
    try:
        import openai
    except ImportError as e:
        raise ImportError(_OPENAI_INSTALL_HINT) from e
    return openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()


def _load_gemini_client(api_key: str | None) -> Any:
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(_GEMINI_INSTALL_HINT) from e
    return genai.Client(api_key=api_key) if api_key else genai.Client()
