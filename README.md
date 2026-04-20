# leanctx

**Drop-in prompt compression for production LLM applications.**
Cut Anthropic / OpenAI / Gemini bills by 40–60% without changing your code.

```python
# before
from anthropic import Anthropic

# after
from leanctx import Anthropic  # same interface, compressed requests
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

If your code calls `anthropic.messages.create()` or `openai.chat.completions.create()` in production, this is for you.

## How it works

Three compression modes, one config switch:

- **`local`** — runs Microsoft's open-source LLMLingua-2 locally. Free marginal cost.
- **`self_llm`** — lets your own configured LLM do the compression. Highest quality.
- **`hybrid`** (default) — routes by content type: code stays verbatim, prose goes through LLMLingua-2, long important spans fall back to self_llm.

Content-aware routing means code blocks, diffs, stack traces, and tool schemas are preserved verbatim — no corrupted syntax.

```python
from leanctx import Anthropic

client = Anthropic(leanctx_config={
    "mode": "on",
    "trigger": {"threshold_tokens": 2000},
    "routing": {
        "code":           "verbatim",   # never touch code
        "error":          "verbatim",   # never touch stack traces
        "prose":          "lingua",     # local LLMLingua-2
        "long_important": "selfllm",    # cheap LLM summarization
    },
    "lingua":  {"ratio": 0.5, "device": "cpu"},
    "selfllm": {"model": "claude-haiku-4-5", "api_key": "sk-...", "ratio": 0.3},
})

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": long_document}],
)

# Compression telemetry attached to the response
print(response.usage.leanctx_tokens_saved)
print(response.usage.leanctx_ratio)
```

## Real compression numbers

Measured end-to-end with the real LLMLingua-2 model (`scripts/integration_test_e2e.py`):

- 4,450 chars in → 2,462 chars sent on the wire to `api.anthropic.com` (**44.7% reduction**)
- 395 tokens saved per request at `mode="on", ratio=0.5`
- `response.usage.leanctx_method == "lingua"` verifies the pipeline executed

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
