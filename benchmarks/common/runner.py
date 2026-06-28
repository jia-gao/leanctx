"""Leg runner + eval-LLM call for the benchmark harness (extracted from
bench_phase1, generalized for InsForge).

``run_leg`` is provider-agnostic: the compression step is an injected ``compress``
callable (a CR shim for ClawRouter, the leanctx sidecar for InsForge direct mode,
or identity when InsForge's gateway compresses server-side), and the chat call is
``eval_fn`` (defaults to ``call_eval_llm``, which speaks anthropic / openai /
openrouter / insforge). For savings-only workloads (no LongBench gold) a
``usage_probe`` captures the provider's ``usage.prompt_tokens`` with a 1-token
completion.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from benchmarks.common.longbench import (
    _build_lb_prompt,
    _extract_lb_answer,
    _lb_head_tail,
)
from benchmarks.common.tokens import _sum_tokens


def _build_eval_context(
    compressed_messages: list[dict],
    *,
    closed_book: bool = False,
) -> str:
    """Assemble the ``$DOC$`` context string the eval prompt is built from.

    Factored out of ``call_eval_llm`` so ``run_leg`` can compute the *same*
    string as a cache key for the shared-eval-draw optimisation: two legs whose
    compressed messages yield an identical context are guaranteed to send an
    identical prompt to the judge, so the second leg can reuse the first leg's
    answer instead of drawing a fresh (and, at temperature > 0, noisy) sample.
    """
    if closed_book:
        return "[no document provided]"
    return _lb_head_tail(
        " ".join(
            m["content"]
            for m in compressed_messages
            if isinstance(m.get("content"), str)
            and m.get("role") in ("user", "system", "assistant")
        ).strip()
    )


# Eval decoding temperature. LongBench scoring is a single-letter multiple-choice
# task; we want the judge as close to deterministic as practical so that the
# per-item accuracy signal reflects the *context* (and thus compression), not
# decoder sampling. 0.1 (rather than 0.0) keeps a hair of slack for providers
# that treat exactly-0 specially, while collapsing the ~20% identical-input flip
# rate observed at the API default of 1.0.
_EVAL_TEMPERATURE = 0.1

# OpenAI-compatible providers (one client, differing base_url / key). OpenRouter
# speaks this; InsForge's gateway does NOT (see _insforge_chat_call).
_OPENAI_COMPATIBLE = frozenset({"openai", "openrouter"})


def _insforge_prompt_tokens(data: Any) -> int:
    """Extract InsForge's ``usage.promptTokens`` — the *only* savings instrument
    in gateway mode — raising loudly if it is absent or non-positive.

    A real prompt is never 0 tokens, so a missing/0 count means the response
    shape changed (envelope drift, ``response.usage`` omitted on a cache hit,
    etc.), not "free compression". Coercing it to 0 silently inflated the
    OFF/ON savings headline; we fail the run instead so the number is never
    quietly wrong.
    """
    payload = data.get("data", data) if isinstance(data, dict) else {}
    usage = (payload.get("metadata") or {}).get("usage") or {}
    raw = usage.get("promptTokens")
    if raw is None:
        keys = sorted(data) if isinstance(data, dict) else type(data).__name__
        raise RuntimeError(
            "InsForge response missing metadata.usage.promptTokens — cannot "
            f"measure savings (top-level keys: {keys}). Response shape may have "
            "drifted from the pinned ref; re-check the gateway schema."
        )
    tok = int(raw)
    if tok <= 0:
        raise RuntimeError(
            f"InsForge reported promptTokens={tok}; refusing to record a "
            "0-token prompt as free compression (would inflate the savings)."
        )
    return tok


def _insforge_chat_call(
    prompt: str,
    model: str,
    max_tokens: int,
    *,
    base_url: str,
    api_key: str,
) -> tuple[str, int]:
    """One call to InsForge's Model Gateway (NOT OpenAI-compatible; see issue #11).

    POST ``{base_url}/api/ai/chat/completion`` with InsForge's flat schema
    (``model``/``messages``/``maxTokens``/``temperature``) + a user JWT, and read
    the prompt-token count from ``metadata.usage.promptTokens`` (camelCase). This
    is InsForge's *own* usage instrument — the whole point of the gateway run.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/api/ai/chat/completion"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "maxTokens": max_tokens,
        "temperature": _EVAL_TEMPERATURE,
    }
    resp = httpx.post(
        url,
        json=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    # successResponse() sends the result directly; tolerate a {data:...} wrapper too.
    payload = data.get("data", data) if isinstance(data, dict) else {}
    text = payload.get("text", "") or ""
    in_tok = _insforge_prompt_tokens(data)
    return text, in_tok


def _openai_compatible_call(
    prompt: str,
    model: str,
    max_tokens: int,
    *,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, int]:
    """One OpenAI-compatible chat completion; returns (text, prompt_tokens)."""
    import openai

    client = (
        openai.OpenAI(base_url=base_url, api_key=api_key)
        if (base_url or api_key)
        else openai.OpenAI()
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=_EVAL_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    in_tok = int(resp.usage.prompt_tokens) if resp.usage else 0
    return text, in_tok


def _resolve_openai_compatible(eval_cfg: dict, provider: str) -> tuple[str | None, str | None]:
    """(base_url, api_key) for an OpenAI-compatible provider, env-defaulted."""
    base_url = eval_cfg.get("base_url")
    api_key = eval_cfg.get("api_key")
    if provider == "openrouter":
        base_url = base_url or "https://openrouter.ai/api/v1"
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    return base_url, api_key


def call_eval_llm(
    compressed_messages: list[dict],
    item: dict,
    eval_cfg: dict,
    *,
    closed_book: bool = False,
) -> tuple[str | None, int]:
    """Build an LB prompt from compressed messages + item; call the eval LLM.

    When ``closed_book`` is True the document context is dropped entirely and
    the model must answer from priors alone — the control leg that rules out
    the null hypothesis "the context was irrelevant" (see reviewer feedback).

    ``provider`` may be ``anthropic`` (usage.input_tokens) or any OpenAI-compatible
    backend — ``openai`` / ``openrouter`` / ``insforge`` (usage.prompt_tokens);
    the latter two read ``base_url``/``api_key`` from ``eval_cfg`` (openrouter
    also env-defaults them). Returns (predicted_letter | None, input_tokens).
    """
    context = _build_eval_context(compressed_messages, closed_book=closed_book)
    prompt = _build_lb_prompt(item, context)

    provider = eval_cfg.get("provider", "anthropic")
    model = eval_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = int(eval_cfg.get("max_tokens", 64))

    text = ""
    in_tok = 0
    max_retries = 8
    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=_EVAL_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text if resp.content else ""
                in_tok = int(resp.usage.input_tokens)
            elif provider == "insforge":
                text, in_tok = _insforge_chat_call(
                    prompt, model, max_tokens,
                    base_url=eval_cfg["base_url"], api_key=eval_cfg["api_key"],
                )
            elif provider in _OPENAI_COMPATIBLE:
                base_url, api_key = _resolve_openai_compatible(eval_cfg, provider)
                text, in_tok = _openai_compatible_call(
                    prompt, model, max_tokens, base_url=base_url, api_key=api_key
                )
            else:
                raise ValueError(f"Unsupported eval provider: {provider!r}")
            break  # success
        except Exception as exc:
            is_rate_limit = "rate_limit" in type(exc).__name__.lower() or "429" in str(exc)
            if is_rate_limit and attempt < max_retries - 1:
                wait = min(2 ** attempt * 15, 120)  # 15s, 30s, 60s, 120s…
                print(f"  [rate-limit] waiting {wait}s before retry {attempt + 2}/{max_retries}")
                time.sleep(wait)
            else:
                raise

    return _extract_lb_answer(text), in_tok


def usage_probe_llm(messages: list[dict], probe_cfg: dict) -> int:
    """Send ``messages`` to the provider with ``max_tokens=1`` and return the
    provider's ``usage.prompt_tokens`` — the savings instrument for transcript
    (no-gold) workloads, where there is nothing to score but the input-token
    count is exactly what we want to measure off vs on.
    """
    provider = probe_cfg.get("provider", "openrouter")
    model = probe_cfg.get("model", "anthropic/claude-haiku-4.5")
    chat_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if isinstance(m.get("content"), str)
    ] or [{"role": "user", "content": ""}]

    max_retries = 8
    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model=model, max_tokens=1, messages=chat_messages
                )
                return int(resp.usage.input_tokens)
            if provider == "insforge":
                import httpx

                url = f"{probe_cfg['base_url'].rstrip('/')}/api/ai/chat/completion"
                r = httpx.post(
                    url,
                    json={"model": model, "messages": chat_messages, "maxTokens": 1},
                    headers={"Authorization": f"Bearer {probe_cfg['api_key']}"},
                    timeout=180,
                )
                r.raise_for_status()
                return _insforge_prompt_tokens(r.json())
            if provider in _OPENAI_COMPATIBLE:
                import openai

                base_url, api_key = _resolve_openai_compatible(probe_cfg, provider)
                client = (
                    openai.OpenAI(base_url=base_url, api_key=api_key)
                    if (base_url or api_key)
                    else openai.OpenAI()
                )
                resp = client.chat.completions.create(
                    model=model, max_tokens=1, messages=chat_messages
                )
                return int(resp.usage.prompt_tokens) if resp.usage else 0
            raise ValueError(f"Unsupported usage-probe provider: {provider!r}")
        except Exception as exc:
            is_rate_limit = "rate_limit" in type(exc).__name__.lower() or "429" in str(exc)
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(min(2 ** attempt * 15, 120))
            else:
                raise
    return 0


def run_leg(
    leg: str,
    items: list[dict[str, Any]],
    *,
    compress: Callable[[list[dict]], dict[str, Any]],
    lb_cfg: dict | None = None,
    eval_fn: Callable[..., tuple[str | None, int]] = call_eval_llm,
    closed_book: bool = False,
    msg_sink: dict[str, list[dict[str, Any]]] | None = None,
    reuse_eval: dict[str, dict[str, Any]] | None = None,
    eval_sink: dict[str, dict[str, Any]] | None = None,
    usage_probe: Callable[[list[dict]], int] | None = None,
) -> list[dict[str, Any]]:
    """Run one benchmark leg.

    ``compress(messages) -> result`` returns a dict with at least ``messages``
    (the compressed list) and optionally ``compressionRatio`` / ``stats``. For a
    closed-book leg, compression is skipped entirely.

    Shared eval draw (de-biasing): when ``reuse_eval`` is supplied (the prior
    leg's ``eval_sink``) and this item's eval context is byte-identical to the
    context the prior leg scored, the prior leg's answer is reused verbatim
    instead of issuing a fresh judge call. This makes verbatim-routed items —
    where leanctx changed nothing — contribute Δacc = 0 by construction, instead
    of injecting decoder noise. ``eval_sink`` records this leg's (context,
    answer, tokens) per item so a later leg can reuse them.

    ``usage_probe(messages) -> prompt_tokens`` is used for savings-only
    (no-``lb_cfg``) items to record the provider's ``usage_prompt_tokens``.
    """
    records = []
    for item in items:
        t0 = time.perf_counter()
        messages = item["messages"]
        tokens_raw = _sum_tokens(messages)
        if closed_book:
            # The control leg drops the document entirely, so there is nothing
            # to compress and no sidecar is needed — skip it to avoid depending
            # on a live service and to keep the leg cheap.
            compressed = messages
            compress_ms = 0
            comp_result: dict[str, Any] = {}
        else:
            # Time the compression call in isolation. An end-to-end duration_ms
            # would wrap the eval LLM (and its retries), so it cannot stand in
            # for compression latency — compress_ms is the real sidecar cost,
            # eval_ms is reported separately and never conflated.
            t_compress0 = time.perf_counter()
            comp_result = compress(messages)
            compress_ms = int((time.perf_counter() - t_compress0) * 1000)
            compressed = comp_result["messages"]
        tokens_compressed = _sum_tokens(compressed)
        # Retain the compressed messages for the post-run verbatim split.
        # Keyed by item_id; agent items without an id are skipped.
        item_id = item.get("item_id")
        if msg_sink is not None and item_id is not None:
            msg_sink[item_id] = compressed
        rec: dict[str, Any] = {
            "leg": leg,
            "workload": item.get("workload", "unknown"),
            "item_id": item.get("item_id"),
            "tokens_raw": tokens_raw,
            "tokens_compressed": tokens_compressed,
            "cr_compression_ratio": comp_result.get("compressionRatio"),
            "cr_stats": comp_result.get("stats"),
            "compress_ms": compress_ms,
            "closed_book": closed_book,
            "accuracy": None,
        }
        if lb_cfg and item.get("question"):
            context_key = _build_eval_context(compressed, closed_book=closed_book)
            prev = (
                reuse_eval.get(item_id)
                if reuse_eval is not None and item_id is not None
                else None
            )
            if prev is not None and prev.get("context") == context_key:
                # Identical prompt to the prior leg → reuse its draw (no judge
                # call, no decoder noise). eval_ms is 0 because nothing ran.
                answer, in_tok = prev["answer"], prev["in_tok"]
                rec["eval_ms"] = 0
                rec["eval_reused"] = True
            else:
                t_eval0 = time.perf_counter()
                answer, in_tok = eval_fn(
                    compressed, item, lb_cfg, closed_book=closed_book
                )
                rec["eval_ms"] = int((time.perf_counter() - t_eval0) * 1000)
                rec["eval_reused"] = False
            rec["accuracy"] = answer == item.get("gold")
            rec["lb_gold"] = item.get("gold")
            rec["lb_pred"] = answer
            rec["eval_input_tokens"] = in_tok
            # The provider's own usage.prompt_tokens — the InsForge headline
            # instrument. Identical to eval_input_tokens for LB items.
            rec["usage_prompt_tokens"] = in_tok
            rec["lb_domain"] = item.get("lb_domain", "")
            rec["lb_difficulty"] = item.get("lb_difficulty", "")
            rec["lb_length"] = item.get("lb_length", "")
            if eval_sink is not None and item_id is not None:
                eval_sink[item_id] = {
                    "context": context_key,
                    "answer": answer,
                    "in_tok": in_tok,
                }
        elif usage_probe is not None and not closed_book:
            # Savings-only (transcript) item: no gold to score, but the provider's
            # usage.prompt_tokens on the compressed messages is the headline.
            rec["usage_prompt_tokens"] = usage_probe(compressed)
        rec["duration_ms"] = int((time.perf_counter() - t0) * 1000)
        records.append(rec)
    return records
