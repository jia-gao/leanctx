"""Token counting across providers.

Middleware trigger logic (compress when a span exceeds N tokens) and
CompressionStats reporting both need a token count. Providers tokenize
differently — leanctx prefers tiktoken when available (accurate for OpenAI,
good approximation for Claude's similar BPE) and falls back to a cheap
character-based estimate when tiktoken isn't installed.

v0.1 will add:
  * Anthropic's hosted count_tokens endpoint for exact Claude counts
  * Gemini's client.models.count_tokens for Gemini
  * per-model encoding selection
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Rough conversion used when tiktoken isn't installed. The 4:1 char:token
# ratio is the conventional rule for English prose; code and non-English
# text can drift 2x in either direction.
_CHARS_PER_TOKEN_FALLBACK = 4


@lru_cache(maxsize=16)
def _load_tiktoken() -> Any | None:
    try:
        import tiktoken
    except ImportError:
        return None
    return tiktoken


@lru_cache(maxsize=64)
def _encoding_for(model: str | None) -> Any | None:
    tiktoken = _load_tiktoken()
    if tiktoken is None:
        return None
    if model is None:
        return tiktoken.get_encoding("cl100k_base")
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Unknown model — fall back to the default OpenAI encoding.
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str | None = None) -> int:
    """Count tokens in ``text`` for a given model.

    Uses tiktoken when installed (accurate for OpenAI, good approximation
    for Claude). Falls back to a character-based estimate when tiktoken
    isn't available.
    """
    if not text:
        return 0
    encoding = _encoding_for(model)
    if encoding is not None:
        return len(encoding.encode(text))
    return max(1, len(text) // _CHARS_PER_TOKEN_FALLBACK)


def count_message_tokens(
    messages: list[dict[str, Any]], model: str | None = None
) -> int:
    """Sum token counts across a list of chat messages.

    Includes only the ``content`` field of each message; role tokens and
    per-message framing overhead are v0.1 work.
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            # Anthropic-style content blocks: [{"type": "text", "text": "..."}]
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += count_tokens(block.get("text", ""), model)
    return total
