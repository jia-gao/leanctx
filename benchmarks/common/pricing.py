"""OpenRouter input pricing for the InsForge benchmark cost model.

The headline savings is reported in InsForge's own ``usage.prompt_tokens``; this
module turns those token deltas into dollars. Prices are **pinned** (USD per
*input* token) so a run is reproducible offline; ``refresh_pricing()`` (CLI:
``--refresh-pricing``) pulls current numbers from OpenRouter's public
``/api/v1/models`` endpoint.

Source: https://openrouter.ai/api/v1/models (pulled 2026-06-24).
"""
from __future__ import annotations

PINNED_DATE = "2026-06-24"

# USD per input (prompt) token, keyed by OpenRouter model slug.
_PINNED_INPUT_USD_PER_TOKEN: dict[str, float] = {
    "anthropic/claude-haiku-4.5": 1.0e-6,
    "anthropic/claude-sonnet-4.5": 3.0e-6,
    "anthropic/claude-sonnet-4": 3.0e-6,
    "openai/gpt-4o-mini": 0.15e-6,
}

# Provider-native model IDs → OpenRouter slug, so a direct-Anthropic run and the
# OpenRouter headline price the *same* model identically.
_ALIASES: dict[str, str] = {
    "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4.5",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4.5",
    "claude-sonnet-4-5-20250929": "anthropic/claude-sonnet-4.5",
}


def _canonical(model: str) -> str:
    """Map a provider-native id or bare slug to the canonical pricing key."""
    return _ALIASES.get(model, model)


def input_price_per_token(
    model: str, table: dict[str, float] | None = None
) -> float | None:
    """USD per input token for ``model``; ``None`` if the model is not priced."""
    return (table or _PINNED_INPUT_USD_PER_TOKEN).get(_canonical(model))


def cost_usd(
    prompt_tokens: int, model: str, table: dict[str, float] | None = None
) -> float | None:
    """Dollar cost of ``prompt_tokens`` input tokens for ``model``."""
    price = input_price_per_token(model, table)
    return None if price is None else prompt_tokens * price


def dollars_saved(
    tokens_off: int,
    tokens_on: int,
    model: str,
    table: dict[str, float] | None = None,
) -> float | None:
    """Dollars saved by compression: (off − on) input tokens × input price."""
    price = input_price_per_token(model, table)
    return None if price is None else (tokens_off - tokens_on) * price


def refresh_pricing(timeout: float = 10.0) -> dict[str, float]:
    """Pull current per-input-token prices from OpenRouter's public models API.

    Returns ``{model_slug: usd_per_input_token}``. No API key required (the
    models listing is public). Use to refresh the pinned table.
    """
    import json
    import urllib.request

    url = "https://openrouter.ai/api/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read())
    out: dict[str, float] = {}
    for m in data.get("data", []):
        slug = m.get("id")
        prompt = (m.get("pricing") or {}).get("prompt")
        if slug and prompt is not None:
            try:
                out[slug] = float(prompt)
            except (TypeError, ValueError):
                continue
    return out
