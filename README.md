# leanctx

[![PyPI](https://img.shields.io/pypi/v/leanctx)](https://pypi.org/project/leanctx/)
[![Python](https://img.shields.io/pypi/pyversions/leanctx)](https://pypi.org/project/leanctx/)
[![License](https://img.shields.io/pypi/l/leanctx)](LICENSE)

**Drop-in prompt compression for production LLM applications.**
Cut your input-token bill by 10–40%，without sacrificing accuracy.

```python
# before
from openai import OpenAI

# after
from leanctx import OpenAI  # same interface, compressed requests
```

On the full **[LongBench v2](https://longbench2.github.io/)** set (N=503), layered on top of a compressor that is already running, leanctx removes **an extra 18.7 % of tokens** — rising to **36.7 % on prose-heavy traffic** — at a cost of 1.8 pp of accuracy. Every figure regenerates from [per-item records committed to this repo](#every-figure-here-regenerates-from-committed-data). Open-source models, runs locally, MIT-licensed. Your prompts and user data never leave your infrastructure by default.

[Quickstart](#quickstart-60-seconds) · [What makes it different](#what-makes-it-different) · [Benchmarks](#real-numbers) · [Integrations](#integrations) · [How it works](#how-it-works)

---

## What makes it different

**Loss-tolerance routing** — classify every segment of a prompt by how much
distortion it can survive, then compress each class differently.

```mermaid
flowchart TD
    A["Agent request"] --> B["Classify by loss tolerance"]
    B -->|zero tolerance| C["Verbatim<br/>byte for byte"]
    B -->|high tolerance| D["LLMLingua-2<br/>~50% removed"]
    B -->|opt-in| E["Self-LLM"]
    C --> F["Recompose + check invariants"]
    D --> F
    E --> F
    F -->|hold| G["Compressed → provider"]
    F -->|fail or timeout| H["Original → provider"]

    classDef zero fill:#e8f1fd,stroke:#2a78d6,color:#0b0b0b
    classDef high fill:#e9f4f0,stroke:#1baf7a,color:#0b0b0b
    classDef open fill:#f4f4f1,stroke:#898781,color:#0b0b0b
    class C zero
    class D high
    class H open
```

| Class | Content | Treatment | Effect |
|---|---|---|---|
| **Zero tolerance** | code, stack traces, `tool_use_id`, tool name and input, JSON | verbatim | 0 % altered |
| **High tolerance** | documentation, retrieved passages, logs, prior turns | LLMLingua-2, on-device | ~50 % removed |
| **Conditional** | low-confidence prose, oversized context | self-LLM, opt-in | 41–49 % removed |

Compression applied uniformly to a prompt will eventually damage the one part that
cannot survive being touched, and will do so unpredictably. The hard part is not
shortening text — it is deciding, inside a live request, what may be shortened at all.

---

## Quickstart (60 seconds)

```bash
pip install 'leanctx[openai,lingua]'    # or [anthropic], [gemini]
```

```python
from leanctx import OpenAI

client = OpenAI(
    leanctx_config={
        "mode": "on",
        "trigger": {"threshold_tokens": 2000},
        "routing": {"prose": "lingua"},  # route prose through LLMLingua-2
    },
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=512,
    messages=[{"role": "user", "content": LONG_DOCUMENT}],
)

print(response.usage.leanctx_tokens_saved)  # e.g. 1841
print(response.usage.leanctx_ratio)         # e.g. 0.49
```

> **First Lingua call loads ~1.2 GB of model weights** to `~/.cache/huggingface/`. Subsequent calls reuse the cache. Add `pip install 'leanctx[lingua]'` to opt in; without it, leanctx falls back to passthrough.

Verify the install with no API key needed:

```bash
leanctx bench list                                   # 7 registered scenarios
leanctx bench run agent-structural --workload agent  # 5 invariants enforced, exit 0 = pass
```

## Why this exists

You're building a production LLM app and your token bill is a line item:

- RAG apps with large retrieved documents
- Long-running conversational agents (LangChain / LangGraph / CrewAI)
- Document-processing pipelines
- Coding agents — Cursor-like / Claude-Code-like, with growing tool-call histories

Existing options have gaps:

- **Provider prompt caching** (Anthropic / OpenAI / Gemini) wins on stable prefixes — system prompts, tool definitions, retrieved-document pools. **It doesn't help with dynamic per-query content** (chat history, freshly retrieved docs, tool outputs). Compose with leanctx, don't choose between them.
- **Naive truncation** drops the middle of the document, exactly where many answers live. The LongBench v2 numbers above show this concretely.
- **Hosted compression APIs** (Compresr, Token Company) require sending your context to their servers. Closed-source models. leanctx is MIT-licensed, runs the model locally, and never makes outbound calls except to your existing provider.

## Real numbers

### Full LongBench v2 sweep — N=503, layered on a production compressor

The headline result. leanctx runs as a semantic pass **on top of** [ClawRouter](https://github.com/BlockRunAI/ClawRouter)'s seven structural compression layers, so the measured delta is what leanctx adds to a system that is *already* compressing. All 503 LongBench v2 questions (Tsinghua KEG, 8K–2M words), Claude Haiku 4.5 eval, temperature 0.1.

| | Avg tokens / request | vs raw | vs Leg A |
|---|---:|---:|---:|
| Raw (uncompressed) | 27,865 | — | — |
| Leg A — ClawRouter's 7 structural layers | 26,397 | −5.3 % | — |
| **Leg B — + leanctx Layer 8** | **21,470** | **−23.0 %** | **−18.7 %** |

| Accuracy | N | Leg A | Leg B | Δ |
|---|---:|---:|---:|---:|
| Overall | 503 | 45.3 % | 43.5 % | **−1.8 pp** |
| ↳ `verbatim`-routed (leanctx changed nothing) | 275 | 46.9 % | 46.9 % | 0.0 pp |
| ↳ `lingua`-routed (leanctx compressed) | 228 | 43.4 % | 39.5 % | −3.9 pp |

**Compression costs accuracy, it does not add it.** The claim is that the cost is small and bounded — 1.8 pp overall, against the −2 pp go/no-go gate set for the integration. Because 54.2 % of Layer-8 input tokens route to verbatim, the compression actually applied to eligible content is **40.8 %**; the verbatim half contributes exactly 0 to both the savings and the accuracy delta by construction.

Sub-buckets are not uniform, and the report gives all of them: `short`/`lingua` is −17.6 pp (N=68) and Single-Document QA −11.5 pp (N=78), while Long Structured Data Understanding is +9.4 pp (N=32). Sidecar latency 47 ms p50 on GPU. At Sonnet input pricing the savings are ~$78 per 1,000 requests.

Full report — per-bucket breakdowns by route × difficulty × length × domain, layer-by-layer contributions, cost model: [`benchmarks/clawrouter/full_long_bench_evaluation_result.md`](benchmarks/clawrouter/full_long_bench_evaluation_result.md).

#### Independent execution and audit

This result was not produced by the maintainer. The benchmark was executed and audited by outside contributors ([@YingjingLu](https://github.com/YingjingLu), [@QianXiaoMoRan9](https://github.com/QianXiaoMoRan9)). An audit of an earlier draft showed that a headline "+7.4 % on long context" was **eval noise rather than a compression effect** — 7 of the 8 net improved items came from the verbatim subset, where the input was byte-identical (McNemar p = 0.143). Two methodology fixes followed ([#7](https://github.com/jia-gao/leanctx/pull/7)): Leg B now reuses Leg A's answer whenever the compressed context is byte-identical, so verbatim items contribute Δ = 0 by construction instead of decoder noise, and eval temperature dropped to 0.1. A separate correction forced a letter choice in the closed-book control, which had been scoring 6 % — below the 25 % random floor for 4-way multiple choice — and inflating apparent context lift.

The numbers above are post-correction. Discussion: [issue #3](https://github.com/jia-gao/leanctx/issues/3).

#### Every figure here regenerates from committed data

The 503 per-item records are in the repo, so nothing above has to be taken on trust. The sweep was re-run from a clean checkout on separate hardware, at the documented defaults (`llmlingua-2-xlm-roberta-large-meetingbank`, ratio 0.5, threshold 1500), and the corpus-level accounting lands on the same values:

| Figure | Value | Independent re-run |
|---|---:|---:|
| Items | 503 | 503 |
| Routing mix | 228 lingua / 275 verbatim | 228 / 275 |
| Verbatim token share | 54.2 % | 54.2 % |
| Avg Layer-8 input | 26,397 | 26,396.75 |

Savings are hardware- and configuration-sensitive: the CPU re-run measures **18.7 %** blended / **40.8 %** on eligible content, against **24.1 % / 52.8 %** on the original GPU run. Two effects are visible in the records — 7 long items exceed the connector's 60 s timeout and fail open under CPU inference (a GPU run stays well inside it), and per-item compression declines as input grows, from 50.0 % on the smallest quartile to 37.7 % on the largest. The figures quoted throughout this README are the conservative ones.

Records, per-figure breakdown, and the script that produces them: [`results/full503_phase1_results.jsonl`](benchmarks/clawrouter/results/full503_phase1_results.jsonl) · [`short_route_counterfactual.md`](benchmarks/clawrouter/results/short_route_counterfactual.md) · `benchmarks/clawrouter/short_route_counterfactual.py`.

#### Savings as a function of traffic mix

The 503-item corpus is 45.8 % prose by token share, and blended savings scale linearly with that share — LLMLingua-2 is deterministic and extractive, so reweighting the per-item records to any target mix is exact arithmetic rather than resampling:

```
blended_savings(p) = p × r        p = prose token-share,  r = 40.8 % (95 % CI 37.8–43.7)
```

| Traffic mix | Prose share | Blended savings |
|---|---:|---:|
| Code-heavy agent traffic | 0.25 | 10.2 % |
| This corpus | 0.458 | 18.7 % |
| Balanced | 0.50 | 20.4 % |
| Prose-heavy (docs, RAG, transcripts) | 0.90 | 36.7 % |

So "what will leanctx save me?" has one honest answer: it depends on how much of your traffic is prose, and the curve above tells you. Full curve and CSV: [`prose_code_savings_curve.md`](benchmarks/clawrouter/results/prose_code_savings_curve.md).

### Ablation vs naive truncation — 15 items, directional

A separate, much smaller comparison against head+tail truncation at a 20K cap. Same model, same questions, same truncation across all three conditions.

| Method | Accuracy | Tokens kept | Reproduce |
|---|---:|---:|---|
| Baseline (head+tail truncation only) | 20.0 % (3/15) | 100 % of 20K cap | `leanctx bench run longbench-v2` |
| **leanctx Lingua** (ratio=0.5) | **40.0 % (6/15)** | **43 %** | `LEANCTX_LBV2_COMPRESSOR=lingua leanctx bench run longbench-v2` |
| leanctx SelfLLM (Haiku, ratio=0.3) | 26.7 % (4/15) | 1.4 % | `LEANCTX_LBV2_COMPRESSOR=selfllm leanctx bench run longbench-v2` |

The mechanism is real — truncation drops the middle of the document, while Lingua's extractive classifier keeps answer-bearing tokens distributed across it. **But n=15 is directional only: Fisher's exact two-sided p ≈ 0.18, so this table does not establish the effect.** Treat the N=503 sweep above as the load-bearing result. Per-question records: [`docs/blog/data/lbv2-2026-05-03/`](docs/blog/data/lbv2-2026-05-03/).

### Internal benchmark — coding-agent transcript

A realistic 9-message agent transcript — user question, file reads, grep, log dumps, failed edit, error trace — totaling ~2.1K tokens. Run through `leanctx.Anthropic` with content-aware routing (code → verbatim, errors → verbatim, prose → Lingua):

| Metric | Before | After | Reduction |
|---|:-:|:-:|:-:|
| Tokens | 2148 | 1384 | **35.6 %** |
| Tokens saved per request | | | **768** |

**What got preserved verbatim** (asserted programmatically by the `agent-structural` bench scenario):
- A 2 KB Python source file inside a `tool_result` block — byte-identical
- A Python traceback in an `is_error` tool result — byte-identical
- Every `tool_use_id` and the `name` / `input` of every `tool_use` block — tool linkage and tool-call payloads untouched
- `edit_file`'s `new_str` argument — the actual code edit isn't rewritten

**What actually compressed:**
- A 3.4 KB log dump shrank to 1.9 KB (45 % reduction) — the legitimate compression target
- Grep results and prose reasoning blocks shrank by 30–50 %

Reproduce: `leanctx bench run agent-structural --workload agent` — runs the real LLMLingua-2 model, ~30 s on Apple Silicon, no API key required. Status flips to `failure` with named invariants if any regress; CI-gateable.

### SelfLLM cross-provider comparison

Same 1.7 KB SRE-incident document through `SelfLLM` against each provider's cheapest tier:

| Provider  | Model              | Compression | Latency   | Cost per call |
|-----------|--------------------|:-----------:|:---------:|:-------------:|
| Anthropic | `claude-haiku-4-5` | **41.6 %**  | 3.05 s    | ~$0.0016      |
| OpenAI    | `gpt-4o-mini`      | **49.1 %**  | 6.42 s    | ~$0.0003      |
| Gemini    | `gemini-2.5-flash` | **48.7 %**  | **2.25 s** ⚡ | ~$0.0001      |

All three preserved every timestamp, metric value, and action item with no hallucination. Combined with `Lingua` (LLMLingua-2 local) hitting **44.7 %** char reduction on the same document at zero marginal cost, leanctx covers the full speed/cost/quality trade-off space.

Full methodology, per-provider output samples, cost analysis, bugs found in flight: [`docs/benchmarks/`](docs/benchmarks/).

## Integrations

Deployable integrations against third-party stacks, each with a working sidecar, a connector, and a measurement harness.

| Stack | What exists | Measured |
|---|---|---|
| **[ClawRouter](https://github.com/BlockRunAI/ClawRouter)** (BlockRunAI) | "Layer 8" sidecar + TypeScript connector — [`integrations/clawrouter/`](integrations/clawrouter/) | Full N=503 sweep, PASS on both gates ([report](benchmarks/clawrouter/full_long_bench_evaluation_result.md)) |

Integrations are opt-in and fail-open by construction: if the sidecar is unreachable, slow, or returns anything that fails the invariant check, the original uncompressed request goes upstream. A compression outage costs savings, never availability.

Running leanctx against your stack? Open an issue — measured results get listed here.

## How it works

### Invariants

Checked after recompose, before the request goes upstream:

- message count and ordering match the input
- every `tool_use_id`, tool name and tool input unchanged
- code and stack traces inside tool results byte-identical
- compression path responded within its timeout

**Any failure sends the original uncompressed request.** Fails open: a compression
outage costs savings, never availability.

### The pipeline

leanctx wraps your existing SDK call and applies a configurable compression pipeline
before the request hits the wire.

```
your code
   ↓
leanctx.Anthropic / OpenAI / Gemini    ← drop-in wrapper
   ↓
Middleware (mode=on/off, threshold)
   ↓
Per-message pipeline:
   classify (code | error | prose | …)
        ↓
   route to compressor:
        Verbatim  — never touch (code, errors, tool calls)
        Lingua    — LLMLingua-2 local, free marginal cost
        SelfLLM   — your configured LLM (Anthropic/OpenAI/Gemini), highest quality
   ↓
real Anthropic / OpenAI / Gemini SDK → API
```

Two layers of config:

- **`mode`** — `"on"` to compress, `"off"` to passthrough. Off is safe to leave deployed.
- **`routing`** — maps content types (code / error / prose / unknown / long_important) to compressors (verbatim / lingua / selfllm).

A fully-loaded production config:

```python
from leanctx import OpenAI

client = OpenAI(leanctx_config={
    "mode": "on",
    "trigger": {"threshold_tokens": 2000},  # don't bother below this
    "routing": {
        "code":           "verbatim",   # never touch code
        "error":          "verbatim",   # never touch stack traces
        "prose":          "lingua",     # local LLMLingua-2
        "long_important": "selfllm",    # cheap LLM summarization
    },
    "lingua":  {"ratio": 0.5, "device": "cpu"},
    "selfllm": {"model": "gpt-4o-mini", "api_key": "sk-...", "ratio": 0.3},
    "observability": {"otel": True},     # opt-in OpenTelemetry
})
```

## Compose with provider caching

leanctx is **complementary** to Anthropic / OpenAI / Gemini prompt caching, not competitive:

- **Provider caching wins** on stable prefixes: system prompts, tool definitions, retrieved-document pools that don't change across requests. Up to 90 % discount on cached reads.
- **leanctx wins** on dynamic per-query content: chat history, freshly retrieved docs, tool outputs, log dumps that vary every call.
- **They compose.** Mark your stable prefix with `cache_control` (provider-specific) and let leanctx compress the variable suffix. Both savings stack.

The OTel telemetry leanctx emits includes a `provider` label that you can correlate with provider-side cache-hit metrics in the same dashboard.

## Observability (v0.3)

leanctx emits OpenTelemetry spans + metrics for every compression call, opt-in via `leanctx_config["observability"]["otel"]`. The library is **API-only**: it never owns the OTel SDK or registers providers. The application configures OTel; leanctx emits.

```python
client = leanctx.Anthropic(
    leanctx_config={
        "mode": "on",
        "observability": {"otel": True},
    },
)
```

Each wrapper-routed call produces one root `leanctx.compress` span (provider, method, input_tokens, output_tokens, cost_usd, duration_ms) plus per-compressor child spans. Five metrics — 4 counters + 1 histogram — labeled by `provider`/`method`/`status`. Closed `leanctx.method` taxonomy: `passthrough` | `below-threshold` | `empty` | `opaque-bailout` | `verbatim` | `lingua` | `selfllm` | `hybrid`.

See [`docs/observability.md`](docs/observability.md) for the full attribute reference, stream-lifetime contract, app-side OTel SDK setup, and cardinality guidance.

## Reproducible benchmarks (v0.3)

The `leanctx bench` CLI ships seven named scenarios with versioned JSON output (`schema_version: "1"`):

```bash
leanctx bench list                                  # show registered scenarios
leanctx bench run lingua-local --workload rag       # offline, no API key
leanctx bench run agent-structural --workload agent # 5 invariants enforced
leanctx bench run anthropic-e2e --workload chat     # full stack, respx-mocked
leanctx bench run selfllm-anthropic --workload rag  # live API, set ANTHROPIC_API_KEY
leanctx bench run longbench-v2 --workload rag       # public LongBench v2 ablation
```

Versioned schema, multi-run isolation (`--runs N` constructs fresh client/middleware each run), clean diagnostics for missing extras / API keys (exit 3, no traceback). Built so downstream tooling can consume the JSON without breaking on schema changes.

## Install

```bash
pip install leanctx                              # core (passthrough only — useful for testing the wrapper)
pip install 'leanctx[anthropic,openai,gemini]'   # provider SDKs
pip install 'leanctx[lingua]'                    # + LLMLingua-2 local compression (~1.2 GB on first call)
pip install 'leanctx[otel]'                      # + OpenTelemetry API/SDK
pip install 'leanctx[bench]'                     # + respx for offline scenarios
pip install 'leanctx[longbench]'                 # + HuggingFace datasets for LongBench v2
pip install 'leanctx[server]'                    # + FastAPI/uvicorn HTTP compression sidecar
pip install 'leanctx[all]'                       # everything
```

### HTTP sidecar (for non-Python callers)

Run leanctx as a long-lived HTTP service so a TypeScript proxy, a LiteLLM callback, a Go gateway — any non-Python stack — can use leanctx compression over `POST /compress`:

```bash
pip install 'leanctx[server,lingua]'
leanctx-serve --port 8459
curl -s localhost:8459/compress -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"<long prose>"}]}'
```

It compresses `system`/`user`/`assistant` prose and forwards `tool` results + multimodal content verbatim, preserving message order/count. See [`docs/server.md`](docs/server.md).

Docker:

```bash
docker build -t leanctx:slim .                             # 341 MB, all provider SDKs
docker build -t leanctx:lingua --build-arg LINGUA=true .   # + LLMLingua-2, ~3 GB
```

## Supported providers

| Provider | Drop-in client | Streaming | Compression | SelfLLM target |
|---|:-:|:-:|:-:|:-:|
| Anthropic | `leanctx.Anthropic` / `AsyncAnthropic` | ✅ | ✅ | ✅ |
| OpenAI    | `leanctx.OpenAI` / `AsyncOpenAI` | ✅ | ✅ | ✅ |
| Gemini    | `leanctx.Gemini` (`.models` + `.aio.models`) | ✅ | ✅ \* | ✅ |

\* **Gemini text-only requests compress fully.** Requests that include `function_call`, `function_response`, or multimodal (`inline_data`) parts automatically bail out to passthrough — leanctx never rewrites tool-call payloads (would change tool semantics) and doesn't touch images. Multimodal + function-call compression is on the v0.3.x roadmap. Spans for these calls carry `leanctx.method = opaque-bailout` so you can monitor the share.

12 wrapper request paths instrumented (sync + async × stream + non-stream × 3 providers). Stream-path span lifetime closes at the first of: iterator exhaustion, explicit `.close()`, or `__del__` GC backstop — `duration_ms` covers the full stream lifetime.

## Status

[`v0.3.1`](https://github.com/jia-gao/leanctx/releases/tag/v0.3.1) is on PyPI. Built across a 5-round Codex-reviewed RLCR loop; 257 tests passing, ruff + mypy --strict clean across 40 source files.

## Roadmap

- [x] **v0.1** — Python SDK, drop-in wrappers, LLMLingua-2 + SelfLLM (Anthropic), classifier, router, dedup + purge-errors strategies, LangChain helpers, Docker
- [x] **v0.2** — SelfLLM on OpenAI + Gemini, block-aware compression (tool_use / tool_result preserved), Gemini contents normalization, LCEL `compress_runnable`
- [x] **v0.3** — OpenTelemetry observability across 12 wrapper paths, `leanctx bench` CLI (6 scenarios + versioned schema), `agent-structural` invariant enforcement, [public release `v0.3.1`](https://pypi.org/project/leanctx/) — 2026-04-26
- [x] **Full 503-item LongBench v2 sweep** — externally run and audited, methodology corrected, [published](benchmarks/clawrouter/full_long_bench_evaluation_result.md) — 2026-06-13; per-item records committed and independently re-run
- [ ] **v0.3.x** — ghcr.io Docker publish, OpenAI Responses-API intercept, multimodal + function-call compression for Gemini, LlamaIndex helpers, TypeScript SDK compression port
- [ ] **v0.4** — per-tenant attribution (with cardinality cap), Helm chart / K8s sidecar, stateful session dedup with explicit session IDs

## License

MIT. See [LICENSE](LICENSE).
