# leanctx

**Drop-in prompt compression for production LLM applications.**
Cut your LLM token bill by 40–60% without changing your code.

```python
# before
from openai import OpenAI

# after
from leanctx import OpenAI  # same interface, compressed requests
```

Open-source models. No API keys to anyone but your existing provider.
Your prompts and user data never leave your infrastructure by default.

---

> **Status:** v0.1 code is **feature-complete on `main`**. PyPI / npm still
> carry the `0.0.0` reservation while the `0.1.0` release is finalized.
> Track progress in the [roadmap](#roadmap) below.

## Who this is for

You're building a production LLM app and your token bill is a line item:

- RAG apps with large retrieved documents
- Long-running conversational agents
- LangChain / LangGraph / CrewAI workflows with growing tool chains
- Document-processing pipelines
- Anything where input tokens accumulate and you pay for every one

If your code calls a hosted LLM API in production and input tokens are a meaningful line item, this is for you.

## How it works

Three compression modes, one config switch:

- **`local`** — runs Microsoft's open-source LLMLingua-2 locally. Free marginal cost.
- **`self_llm`** — lets your own configured LLM do the compression. Highest quality.
- **`hybrid`** (default) — routes by content type: code stays verbatim, prose goes through LLMLingua-2, long important spans fall back to self_llm.

Content-aware routing means code blocks, diffs, stack traces, and tool schemas are preserved verbatim — no corrupted syntax.

```python
from leanctx import OpenAI

client = OpenAI(leanctx_config={
    "mode": "on",
    "trigger": {"threshold_tokens": 2000},
    "routing": {
        "code":           "verbatim",   # never touch code
        "error":          "verbatim",   # never touch stack traces
        "prose":          "lingua",     # local LLMLingua-2
        "long_important": "selfllm",    # cheap LLM summarization
    },
    "lingua":  {"ratio": 0.5, "device": "cpu"},
    "selfllm": {"model": "gpt-4o-mini", "api_key": "sk-...", "ratio": 0.3},
})

response = client.chat.completions.create(
    model="gpt-4o",
    max_tokens=1024,
    messages=[{"role": "user", "content": long_document}],
)

# Compression telemetry attached to the response
print(response.usage.leanctx_tokens_saved)
print(response.usage.leanctx_ratio)
```

## Real compression numbers

### Coding-agent workload (the main use case)

A realistic 9-message agent transcript — user question, file reads, grep, log dumps, failed edit, error trace — totaling ~2.1K tokens. Run through `leanctx.Anthropic` with `mode="on"` and content-aware routing (code → verbatim, errors → verbatim, prose → Lingua):

| Metric | Before | After | Reduction |
|---|:-:|:-:|:-:|
| Tokens | 2148 | 1384 | **35.6%** |
| Chars  | 7898 | 5701 | 27.8% |
| Tokens saved per request | | | **768** |

**What got preserved verbatim** (asserted programmatically):
- ✅ A 2 KB Python source file inside a `tool_result` block — byte-identical
- ✅ A Python traceback in an `is_error` tool result — byte-identical
- ✅ Every `tool_use_id` and the `name` / `input` of every `tool_use` block — so tool linkage and tool calls don't break
- ✅ `edit_file`'s `new_str` argument — so the actual code edit isn't rewritten

**What actually compressed:**
- A 3.4 KB log dump shrank to 1.9 KB (45% reduction) — the legitimate compression target
- A grep result and prose reasoning blocks shrank by 30-50%

Reproducible: `python scripts/integration_test_agent_workload.py` — runs the real LLMLingua-2 model, takes ~30s on Apple Silicon, no API key required.

### SelfLLM cross-provider comparison

Same 1.7 KB SRE-incident document through `SelfLLM` against each provider's cheapest tier:

| Provider  | Model              | Compression | Latency | Cost per call |
|-----------|--------------------|:-----------:|:-------:|:-------------:|
| Anthropic | `claude-haiku-4-5` | **41.6%**   | 3.05s   | ~$0.0016      |
| OpenAI    | `gpt-4o-mini`      | **49.1%**   | 6.42s   | ~$0.0003      |
| Gemini    | `gemini-2.5-flash` | **48.7%**   | **2.25s** ⚡ | ~$0.0001      |

All three preserved every timestamp, metric value, and action item with no hallucination. Combined with `Lingua` (LLMLingua-2 local) hitting **44.7% char reduction** on the same document at zero marginal cost, leanctx covers the full speed/cost/quality trade-off space.

Full methodology, per-provider output samples, cost analysis, and bugs we found in flight: [`docs/benchmarks/`](docs/benchmarks/).

## Observability (v0.3)

leanctx emits OpenTelemetry spans and metrics for every compression call, opt-in via `leanctx_config`. The library is **API-only**: it never owns the OTel SDK or registers providers. The application configures OTel; leanctx emits.

```python
client = leanctx.Anthropic(
    leanctx_config={
        "mode": "on",
        "observability": {"otel": True},
    }
)
```

Every wrapper-routed call produces one root `leanctx.compress` span with `provider`, `method`, `input_tokens`, `output_tokens`, `cost_usd`, and `duration_ms`, plus per-compressor child spans for granular tracing. Five metrics (4 counters + 1 histogram) are recorded with `provider`/`method`/`status` labels.

See [`docs/observability.md`](docs/observability.md) for the full attribute reference, span lifetime contract for streaming paths, sample app-side OTel SDK setup, and the closed `leanctx.method` taxonomy.

## Reproducible benchmarks (v0.3)

The `leanctx bench` CLI runs the offline integration scenarios with deterministic input and emits versioned JSON records:

```bash
leanctx bench list                                  # registered scenarios
leanctx bench run lingua-local --workload rag       # offline lingua compression
leanctx bench run agent-structural --workload agent # 5 structural-integrity invariants enforced
leanctx bench run anthropic-e2e --workload chat     # full stack, respx-mocked Anthropic
leanctx bench run selfllm-anthropic --workload rag  # live API (requires ANTHROPIC_API_KEY)
```

Output is one JSON record per run with `schema_version: "1"` and a documented field set so downstream tooling can consume it.

## Roadmap

- [x] v0.1 — Python SDK, drop-in Anthropic/OpenAI/Gemini wrappers, `local` (LLMLingua-2) + `self_llm` (Anthropic), content classifier, router, dedup + purge-errors strategies, LangChain format helpers, Docker image
- [x] v0.2 — `self_llm` on OpenAI + Gemini, block-aware compression (tool_use / tool_result preserved through Lingua), Gemini `contents` normalization (middleware actually runs), LangChain LCEL `compress_runnable`
- [ ] v0.2.0 release — bump version, publish to PyPI + npm (placeholders at 0.0.0 today)
- [ ] v0.3 — OTel observability, benchmark harness, ghcr.io Docker publish workflow, OpenAI responses-API intercept, multimodal + function-call compression for Gemini, LlamaIndex helpers, TypeScript SDK compression port
- [ ] v0.4 — Helm chart, Kubernetes sidecar proxy deployment, stateful session dedup with explicit session IDs

## Install

```bash
# Once v0.1.0 is published:
pip install leanctx
pip install 'leanctx[anthropic,openai,gemini]'  # pick your providers
pip install 'leanctx[lingua]'                   # + LLMLingua-2 local compression
pip install 'leanctx[all]'                      # everything

# Today (from source, main branch):
pip install git+https://github.com/jia-gao/leanctx.git
```

Docker images:

```bash
docker build -t leanctx:slim .                             # 341 MB, all provider SDKs
docker build -t leanctx:lingua --build-arg LINGUA=true .   # + LLMLingua-2, ~3 GB
```

## Supported providers

| Provider | Drop-in client | Streaming | Compression applied | SelfLLM target |
|---|:-:|:-:|:-:|:-:|
| Anthropic | ✅ `leanctx.Anthropic` / `AsyncAnthropic` | ✅ | ✅ | ✅ |
| OpenAI    | ✅ `leanctx.OpenAI` / `AsyncOpenAI` | ✅ | ✅ | ✅ |
| Gemini    | ✅ `leanctx.Gemini` (`.models` + `.aio.models`) | ✅ | ✅ * | ✅ |

**Gemini asterisk:** text-only requests compress fully. Requests that
include `function_call`, `function_response`, or multimodal
(`inline_data`) parts automatically bail out to passthrough — we
never rewrite tool-call payloads, as that would change tool
semantics. Multimodal + function-calling compression lands in v0.3.

## Architecture

```
your code
   ↓
leanctx.Anthropic / OpenAI / Gemini
   ↓
Middleware
   ├── Strategies (deterministic, no LLM):
   │     DedupStrategy, PurgeErrorsStrategy
   ↓
   ├── Per-message pipeline:
   │     classify → router → compressor
   ↓
Compressor:  Verbatim | Lingua (LLMLingua-2) | SelfLLM (your LLM)
   ↓
real Anthropic / OpenAI / Gemini SDK → API
```

## License

MIT. See [LICENSE](LICENSE).
