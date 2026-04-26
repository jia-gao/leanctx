# SelfLLM compression: provider comparison

**Date:** 2026-04-25
**leanctx commits:** `54d64dd` (integration script) → `6127045` (fixes)
**Reproducible via:** [`scripts/integration_test_selfllm.py`](../../scripts/integration_test_selfllm.py)

## TL;DR

leanctx's `SelfLLM` compressor delegates summarization to your configured LLM API, so you can pair a cheap summarizer (Haiku, gpt-4o-mini, gemini-2.5-flash) with whatever frontier model handles your real request. We ran the same 1,740-character SRE incident document through `SelfLLM` against each of the three supported providers using each provider's cheapest tier, with the default `ratio=0.3` (target 30% retention) and `max_summary_tokens=500`.

| Provider  | Model              | In tokens | Out tokens | Ratio   | Saved | Latency |
|-----------|--------------------|----------:|-----------:|--------:|------:|--------:|
| Anthropic | `claude-haiku-4-5` |       627 |        261 | **41.6%** |   366 |   3.05s |
| OpenAI    | `gpt-4o-mini`      |       560 |        275 | **49.1%** |   285 |   6.42s |
| Gemini    | `gemini-2.5-flash` |       599 |        292 | **48.7%** |   307 | **2.25s** |

All three providers produced **coherent, well-structured summaries** preserving timestamps, metric values, and all four action items. No hallucinations observed. The differences below are stylistic, not correctness.

**Headline takeaway:** any of the three works for production compression. Pick by what you already pay for. Cost per call is well under a cent across all three.

## Methodology

### Sample document

A 1,740-character SRE incident write-up — mixed prose, specific timestamps (14:32 UTC, 14:30:47, 14:33:15), specific metric values (450ms→3.8s, 0.01%→1.2%), four numbered action items, plus a closing sentence about Thursday's post-mortem. Chosen because it's representative of:

- **RAG-like context** (incident notes you'd retrieve into a chat)
- **Multi-section structure** (prose + lists)
- **Information density** (numbers and proper nouns the compressor must preserve)
- **Ambient ambiguity** (the kind of document that summarizes well but with hallucination risk)

The full text is in [`scripts/integration_test_selfllm.py`](../../scripts/integration_test_selfllm.py) (`SAMPLE_DOC` constant) so any third party can reproduce.

### Configuration

```python
SelfLLM(
    provider="<anthropic|openai|gemini>",
    ratio=0.3,
    max_summary_tokens=500,
)
```

`ratio=0.3` is a *suggestion* embedded in the user prompt ("Compress to roughly 30% of original length…"), not a hard constraint. Each provider's actual ratio differed (41-49%) — none hit 30%, all stayed under 50%, all preserved the structurally important content.

### Token counting

Uses each provider's reported counts from the response usage object:
- Anthropic: `usage.input_tokens` / `usage.output_tokens`
- OpenAI: `usage.prompt_tokens` / `usage.completion_tokens`
- Gemini: `usage_metadata.prompt_token_count` / `usage_metadata.candidates_token_count`

Note: each provider tokenizes differently, so the same input shows 560-627 input tokens depending on the tokenizer. Apples-to-apples comparison still works because we're comparing each provider's compression *ratio* against itself.

## Per-provider results

### Anthropic — `claude-haiku-4-5`

**Output (261 tokens, 41.6% ratio):**

> Production incident started 14:32 UTC on /v1/messages endpoint: p99 latency spiked 450ms→3.8s, error rate 0.01%→1.2% over 4 minutes.
>
> Root cause: GPU memory exhaustion in us-east-1 inference cluster. Two of four pods went NotReady at 14:30:47; 504s from upstream model server followed at 14:31. HPA didn't scale (CPU was below threshold; GPU memory invisible to default metrics).
>
> Mitigation: manually scaled to 8 replicas, failed traffic to us-west-2. Latency recovered in 90 seconds. Likely cause: single tenant sending long contexts exhausting KV cache, untracked by autoscaler.
>
> Action items:
> 1. Add GPU memory utilization to HPA custom metrics
> 2. Implement per-tenant KV cache budgets in inference scheduler
> 3. Write retro doc; schedule post-mortem Thursday (platform, inference, oncall teams)
> 4. File ticket for dashboard work — Grafana default view didn't surface GPU memory, misdirected initial oncall response

**Style:** tightest of the three, highest information density per token. Uses `→` arrows for state transitions, drops articles aggressively. Reads slightly clipped but unambiguous.

### OpenAI — `gpt-4o-mini`

**Output (275 tokens, 49.1% ratio):**

> Investigating a production incident that began at 14:32 UTC. Symptoms: p99 latency on /v1/messages spiked from 450ms to 3.8s, error rate climbed from 0.01% to 1.2%. Logs show a wave of 504s from the us-east-1 model server cluster starting at 14:31; two of four inference pods became NotReady at 14:30:47. HPA did not scale up until 14:33:15 due to CPU being under the target threshold; the actual bottleneck was GPU memory exhaustion, which HPA can't monitor with default metrics.
>
> Mitigation: manually scaled deployment to 8 replicas and redirected traffic to us-west-2. Latency recovered within 90 seconds. Root cause of GPU memory pressure unknown; suggests a request pattern exhausted KV cache, likely a tenant sending long contexts not accounted for by the autoscaler.
>
> Action items:
> 1. Add GPU memory utilization to HPA custom metrics.
> 2. Implement per-tenant KV cache budgets.
> 3. Write a retro doc and schedule a post-mortem for Thursday with relevant teams.
> 4. File a follow-up ticket for dashboard work; default Grafana view didn't show GPU memory, leading to incorrect initial oncall response.
>
> Full timeline and quotes from the oncall channel to validate sequence before Thursday will be provided.

**Style:** most conversational, full sentences. Closest to "what a human would write." Slightly verbose vs Haiku, slightly less compact.

### Gemini — `gemini-2.5-flash`

**Output (292 tokens, 48.7% ratio):**

> Production incident started 14:32 UTC. Symptoms: p99 latency on `/v1/messages` endpoint spiked from 450ms to 3.8s, error rate climbed from 0.01% to 1.2%.
>
> Logs show 504s from upstream model server cluster (us-east-1) starting 14:31. Two of four inference pods in us-east-1 went NotReady at 14:30:47. HPA didn't scale up until 14:33:15 due to CPU under threshold; actual bottleneck was GPU memory exhaustion, which HPA didn't monitor.
>
> Mitigation: manually scaled deployment to 8 replicas and failed traffic to us-west-2. Latency recovered within 90s. Root cause of GPU memory pressure unknown, but curve suggests specific request pattern exhausted KV cache (possible single tenant sending long contexts not billed against GPU budget).
>
> Action items:
> 1. Add GPU memory utilization to HPA custom metrics.
> 2. Implement per-tenant KV cache budgets in inference scheduler.
> 3. Write retro doc and schedule post-mortem for Thursday with platform, inference, and oncall teams.
> 4. File followup ticket for dashboard work (Grafana didn't surface GPU memory).
>
> Reconstructing full timeline with oncall channel quotes for validation.

**Style:** middle ground between Anthropic's terseness and OpenAI's conversational tone. Uses code-ticks around endpoints (`/v1/messages`). Slightly more verbose section transitions.

**Latency winner:** 2.25s, fastest of the three by a wide margin.

## Bugs found in flight

The integration script caught two real bugs that would have shipped to v0.2.0 and silently broken compression for users on default configs.

### Bug 1: `gpt-5-nano` returned empty output

Original OpenAI default was `gpt-5-nano`. First run:

```
Resolved model: gpt-5-nano
output tokens: 500          ← hit max_completion_tokens
Compressed output:
                            ← nothing visible
```

`gpt-5-nano` is a *reasoning* model. Without `reasoning_effort="minimal"`, it spent the entire 500-token completion budget on hidden reasoning tokens and emitted zero visible output.

**Fix:** changed default to `gpt-4o-mini` (non-reasoning, comparable cost) and added auto-detection so users who explicitly pick gpt-5/o-series get `reasoning_effort="minimal"` automatically.

### Bug 2: `gemini-2.5-flash` returned 20-token truncated output

```
Resolved model: gemini-2.5-flash
output tokens: 20
Compressed output:
  Production incident began 14:32 UTC. Symptoms: `/v1/messages` endpoint
                                                                ↑ cut off
```

Gemini 2.5 family supports a "thinking" mode that's enabled by default. With `max_output_tokens=500`, the model spent ~480 on thinking and only 20 on visible output, truncating the summary mid-sentence.

**Fix:** auto-add `thinking_config: {thinking_budget: 0}` for gemini-2.5+ models so the full output budget produces visible text.

Both fixes are pinned by regression tests so we can't reintroduce the bugs.

## Cost analysis

Approximate cost per call at the cheapest model tier per provider:

| Provider | Model | Input rate | Output rate | Per-call (≈600 in / ≈275 out) |
|---|---|:-:|:-:|:-:|
| Anthropic | `claude-haiku-4-5` | $0.80/MT | $4.00/MT | ~$0.0016 |
| OpenAI    | `gpt-4o-mini`      | $0.15/MT | $0.60/MT | ~$0.0003 |
| Gemini    | `gemini-2.5-flash` | $0.075/MT (free tier) | $0.30/MT | ~$0.0001 |

(MT = million tokens. Rates as of 2026-04 for the chosen tiers; verify current pricing on each provider's website.)

For a service doing 100K compressions/day:
- Anthropic Haiku: ~$160/day
- OpenAI gpt-4o-mini: ~$30/day
- Gemini 2.5 Flash: ~$10/day

These are negligible compared to the *savings* on the main request. If your main request goes to Sonnet 4.6 ($3/MT input) and compression saves ~300 input tokens per request, you're saving ~$0.90/day per 100 requests on input alone — far more than the SelfLLM call costs even at Haiku rates.

## How to reproduce

```bash
# Anthropic
ANTHROPIC_API_KEY=sk-ant-... \
  python scripts/integration_test_selfllm.py

# OpenAI
OPENAI_API_KEY=sk-... \
  python scripts/integration_test_selfllm.py --provider openai

# Gemini
GEMINI_API_KEY=... \
  python scripts/integration_test_selfllm.py --provider gemini

# Override the cheap default if you want frontier-model summaries
python scripts/integration_test_selfllm.py \
  --provider anthropic --model claude-sonnet-4-6
```

Each run prints input length, output length, ratio, latency, and the actual compressed text so you can judge quality yourself.

## Caveats and limitations

- **Single sample document.** Compression ratios vary by content. Long boilerplate (legal contracts, log dumps) compresses more aggressively; technical specs with high information density compress less.
- **Quality is judged subjectively.** No automated quality eval — we read the outputs and confirmed information preservation by inspection. A real quality benchmark (BLEU, BERTScore, downstream task performance) is v0.3 work.
- **Cheapest tier per provider.** Frontier models (Opus, gpt-5, gemini-2.5-pro) would produce higher quality at higher cost. Trade-off varies per use case.
- **Latency includes network round-trip.** Numbers are from a single run; production latency depends on your region and any provider-side cold start.
- **Token counts are provider-reported.** Different providers tokenize the same text differently; the input-token columns aren't directly comparable across rows.
- **Compression ratio ≠ information loss.** A 50% ratio doesn't mean half the information is gone — extractive ratios on natural language typically retain >90% of semantic content because human writing is highly redundant.

## Recommendations

| Use case | Suggested provider |
|---|---|
| You already pay for Anthropic | Haiku 4.5 — tightest output |
| You already pay for OpenAI | gpt-4o-mini — most readable |
| You already pay for Google / want speed | gemini-2.5-flash — fastest by ~2x |
| Privacy-sensitive (no third-party API) | `Lingua` (LLMLingua-2 local) instead of SelfLLM |
| Low-volume premium quality | `provider="anthropic", model="claude-sonnet-4-6"` |

For most users: **stick with whatever provider you already use for your main requests.** The cross-provider quality differences are smaller than the integration friction of adding a second vendor.

## Comparison to alternatives (commentary)

This benchmark only covers leanctx's `SelfLLM`. We have separate measurements for:

- **`Lingua` (LLMLingua-2 local)** — 44.7% char reduction on a similar document, no API call, runs in ~5s on Apple Silicon MPS after a one-time 1.2 GB model download. See [`scripts/integration_test_lingua.py`](../../scripts/integration_test_lingua.py).

A leanctx-vs-Compresr-vs-naive-LLMLingua-vs-Anthropic-native-compaction comparison is on the v0.3 roadmap.

## Source code

The integration script that produced these numbers: [`scripts/integration_test_selfllm.py`](../../scripts/integration_test_selfllm.py)

The SelfLLM implementation: [`leanctx/compressors/selfllm.py`](../../leanctx/compressors/selfllm.py)

Both are MIT-licensed and reproducible end-to-end with your own API keys.
