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

Measured end-to-end against the live APIs of all three providers using the same SRE-incident document. `SelfLLM` mode, default config, cheapest model per provider:

| Provider  | Model              | Compression | Latency | Cost per call |
|-----------|--------------------|:-----------:|:-------:|:-------------:|
| Anthropic | `claude-haiku-4-5` | **41.6%**   | 3.05s   | ~$0.0016      |
| OpenAI    | `gpt-4o-mini`      | **49.1%**   | 6.42s   | ~$0.0003      |
| Gemini    | `gemini-2.5-flash` | **48.7%**   | **2.25s** ⚡ | ~$0.0001      |

All three preserved every timestamp, metric value, and action item with no hallucination. Combined with `Lingua` (LLMLingua-2 local) compression hitting **44.7% char reduction** on the same document at zero marginal cost, leanctx covers the full speed/cost/quality trade-off space.

**Reproducible** via `scripts/integration_test_selfllm.py` and `scripts/integration_test_e2e.py` — bring your own API key (~$0.001 per run). Full methodology, per-provider output samples, cost analysis, and bugs we found in flight: [`docs/benchmarks/selfllm-providers.md`](docs/benchmarks/selfllm-providers.md).

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
