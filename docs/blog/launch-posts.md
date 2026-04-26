# v0.3 launch — platform-specific drafts

Source: [docs/blog/v0.3-launch.md](v0.3-launch.md). All numbers below come from a real `leanctx bench run` JSON capture on 2026-04-26 (Apple Silicon MPS).

## Hacker News — Show HN

**Title (≤ 80 chars):**

> Show HN: leanctx – drop-in prompt compression with OTel and a reproducible bench

(Alternative if you want to lead with numbers:)

> Show HN: leanctx 0.3 – 50% prompt token cuts on agent traffic, OTel spans included

**URL:** `https://github.com/jia-gao/leanctx`

**First comment (post immediately after submitting):**

> Author here. leanctx is a drop-in wrapper around the official Anthropic / OpenAI / google-genai SDKs that compresses messages before they hit the wire. v0.1 / v0.2 shipped the compressors (LLMLingua-2 local, plus a SelfLLM mode that delegates to Haiku / gpt-4o-mini / gemini-2.5-flash). v0.3 adds the two things people kept asking for:
>
> 1. **OpenTelemetry spans + metrics on every wrapper call.** API-only — leanctx never owns your TracerProvider; it emits via `opentelemetry.trace.get_tracer("leanctx")` and inherits whatever you've set up. 12 wrapper paths instrumented (Anthropic / OpenAI / Gemini × sync + async × stream + non-stream), 4 counters + 1 histogram, closed `leanctx.method` taxonomy. Stream-path span lifetime closes at first of {iterator exhaustion, `.close()`, `__del__` GC backstop}.
>
> 2. **A `leanctx bench` CLI** with six named scenarios and versioned JSON output. Reproducible: `pip install 'leanctx[lingua,bench]' && leanctx bench run lingua-local --workload rag` gives you the same number I'd give you. The `agent-structural` scenario verifies five invariants on a coding-agent transcript (tool_use_id linkage, code blocks preserved verbatim, tracebacks preserved verbatim, tool input dicts unchanged, log spans compressed) and exits non-zero on any regression — so it's CI-gateable.
>
> Real numbers from the bench run earlier today: 51% token reduction on a 1.7 KB SRE-incident RAG document, 53% on an 8-turn chat, 50% on a 9-message coding-agent transcript with all 5 structural invariants green.
>
> Honest framing in the comparison table in the writeup: this isn't a replacement for Anthropic prompt caching (those target different windows of the prompt and compose), it's not LiteLLM (that's routing, not compression), and it's not Compresr (closed-weights hosted; leanctx runs the model locally). The honest gap vs naive LLMLingua-2 is the wrapper layer + classifier + tool-aware block handling.
>
> Writeup with the full numbers + design rationale: https://github.com/jia-gao/leanctx/blob/main/docs/blog/v0.3-launch.md
>
> Happy to answer questions about the OTel design (asymmetric span emission via a depth counter), the bench CLI architecture, or why per-tenant attribution got deferred to v0.4 (cardinality cap design isn't free).

**Posting tips:**
- Submit Sunday evening PT (≈ 18:00–20:00 PT) — Show HN's window before the late-night engineers and the Monday-morning crowd.
- Don't ask for upvotes anywhere; HN dings that hard.
- Be in the thread for the first hour answering questions — that's when the post lives or dies.
- If someone asks "vs Compresr": stay generous; the answer is in the comparison table.

---

## r/MachineLearning

**Subreddit rules:** [P], [D], [R] tags expected. Use `[P]` for projects.

**Title (≤ 300 chars):**

> [P] leanctx 0.3 — drop-in prompt compression for production LLM apps with OpenTelemetry spans + a reproducible bench CLI. 50% token reduction on a coding-agent workload with all 5 structural invariants preserved.

**Body:**

> A v0.3 release of an OSS Python library I've been working on. Wraps the Anthropic / OpenAI / google-genai SDKs and compresses messages before they hit the wire. v0.1 / v0.2 shipped the actual compressors (LLMLingua-2 local + a SelfLLM mode that delegates summarization to Haiku / gpt-4o-mini / gemini-2.5-flash). v0.3 makes it production-evaluable.
>
> **What v0.3 adds:**
>
> 1. **OpenTelemetry spans + metrics**, opt-in via `[otel]` extra. API-only — the library never registers a TracerProvider; emits via `opentelemetry.trace.get_tracer("leanctx")`. 12 wrapper paths instrumented across Anthropic / OpenAI / Gemini, sync + async, stream + non-stream. Stream-path span lifetime closes at first of {iterator exhaustion, `.close()`, `__del__`}; duration_ms covers full stream lifetime, not setup. Five OTel metrics (4 counters + 1 histogram) labeled by provider / method / status.
>
> 2. **`leanctx bench` CLI** with six named scenarios: `lingua-local`, `anthropic-e2e`, `selfllm-{anthropic,openai,gemini}`, `agent-structural`. Versioned JSON output (`schema_version: "1"`). Multi-run isolation guarantees fresh client/middleware state per run.
>
> **Headline numbers** (real `leanctx bench run` output, captured 2026-04-26 on Apple Silicon MPS, ratio=0.5):
>
> | Workload | Input → Output | Reduction | Duration |
> |---|---|---|---|
> | RAG (1.7 KB SRE doc) | 621 → 302 tok | 51 % | 12.1 s |
> | 8-turn chat | 59 → 28 tok | 53 % | 8.1 s |
> | 9-msg coding agent | 846 → 423 tok | 50 % | 7.3 s |
>
> The agent-structural scenario additionally verifies that fenced code blocks, Python tracebacks, `tool_use_id` linkage, and tool-input dicts are preserved exactly. Status flips to failure with named invariants if any regress — works as a CI gate.
>
> **Comparisons (honest):**
>
> - vs Anthropic prompt caching: orthogonal. Caching targets stable prefixes; leanctx compresses variable content. They compose.
> - vs naive LLMLingua-2 directly: leanctx adds the SDK wrapper layer, content classifier (code → Verbatim, prose → Lingua), router, tool-aware block handling, OTel telemetry. None of that is in LLMLingua-2 the library.
> - vs LiteLLM / Portkey / Helicone: those are routing/caching/gateways; they don't do prompt compression.
> - vs Compresr (YC W26): closed weights, hosted API. leanctx runs the model locally.
>
> Writeup: https://github.com/jia-gao/leanctx/blob/main/docs/blog/v0.3-launch.md
> Repo: https://github.com/jia-gao/leanctx (MIT)
> Pip: `pip install leanctx==0.3.1` then `pip install 'leanctx[anthropic,lingua,bench]'`
>
> Happy to answer questions about the design — the asymmetric span-emission rule and the stream-owning-iterator pattern were the most interesting calls to make.

---

## r/LangChain

**Title:**

> leanctx 0.3 — drop-in prompt compression for LangChain users (LCEL `compress_runnable` + OTel observability + reproducible bench)

**Body:**

> Posting because the LangChain integration is the v0.2 piece that's most relevant here, and v0.3 just shipped the observability + bench layer on top.
>
> leanctx is a Python library that wraps the Anthropic / OpenAI / google-genai SDKs and compresses messages before they hit the wire. The LangChain integration is `compress_runnable(cfg)` — fits straight into an LCEL pipeline:
>
> ```python
> from leanctx.integrations.langchain import compress_runnable
> from langchain_anthropic import ChatAnthropic
>
> chain = compress_runnable({
>     "mode": "on",
>     "trigger": {"threshold_tokens": 2000},
>     "routing": {"prose": "lingua", "code": "verbatim", "error": "verbatim"},
>     "observability": {"otel": True},
> }) | ChatAnthropic(model="claude-sonnet-4-6")
> ```
>
> No wrapping at the SDK boundary; the runnable compresses the messages list and passes it through. Tool calls survive the round-trip (the v0.2 work that made `from_dicts` preserve `tool_call_id` and `tool_calls`).
>
> **What's new in v0.3 for LangChain users specifically:**
>
> - Spans + metrics surface every compression call to whatever OTel provider your LangSmith / Langfuse / Arize stack is already configured with. Library is API-only — no SDK or exporter ownership inside leanctx.
> - `leanctx bench` lets you run the same scenarios against your own LangChain workloads to verify before deploying. Six scenarios cover offline (deterministic) and live-provider (non-deterministic) modes.
>
> **Real numbers:** 50 % token reduction on a 9-message coding-agent transcript while preserving every code block, traceback, and tool_use_id linkage. The `agent-structural` scenario CI-gates that contract.
>
> Writeup: https://github.com/jia-gao/leanctx/blob/main/docs/blog/v0.3-launch.md
> Repo: https://github.com/jia-gao/leanctx (MIT, `pip install 'leanctx[anthropic,lingua,bench]'`)

---

## dev.to

dev.to mostly wants the blog content as-is. Two things to add when crossposting:

**Cover image suggestion:** screenshot of `leanctx bench list` output (the missing-extras-diagnostic decoration is visually distinctive and shows the CLI is real).

**Tags:**
- `python`
- `llm`
- `opentelemetry`
- `observability`
- `opensource`

**Canonical URL:** point at the repo blog file: `https://github.com/jia-gao/leanctx/blob/main/docs/blog/v0.3-launch.md`. Search-engine-wise this avoids the duplicate-content penalty.

**Body:** copy `docs/blog/v0.3-launch.md` verbatim. dev.to renders the table syntax fine.

---

## LinkedIn

LinkedIn rewards short + concrete. Three paragraphs max.

**Post:**

> Just shipped leanctx v0.3 — adds OpenTelemetry observability and a reproducible `leanctx bench` CLI to the prompt-compression library I've been working on. The headline number: 50% token reduction on a coding-agent transcript while preserving every code block, traceback, and tool-call linkage exactly. CI-gateable via the agent-structural bench scenario.
>
> Two design choices worth flagging: (1) leanctx is API-only with respect to OTel — it never instantiates a TracerProvider or registers an exporter; your application owns the SDK. (2) The bench CLI emits versioned JSON (`schema_version: "1"`) so downstream tooling can consume it without breaking when the schema changes. Six named scenarios across offline + live-provider modes.
>
> Built across a 5-round Claude Code + Codex review loop — every round Codex reviewed the diff and Claude addressed the findings until the contract held. The full plan, round summaries, and the resulting commit chain are public in the repo. ~5500 lines added, 244 tests, all 11 acceptance criteria met.
>
> Writeup → https://github.com/jia-gao/leanctx/blob/main/docs/blog/v0.3-launch.md
> `pip install leanctx==0.3.1`

**Posting tips:**
- Sunday evening (most LinkedIn engagement on Sunday + Monday for B2B).
- No emojis (LinkedIn algorithm penalizes them at >3 per post for technical posts).
- Tag people who'd find it relevant — but only if you actually know them and the post is relevant to their work. Spray-tagging burns reputation.
- If you have a connection in AI infra you want to surface, reply to the thread early with a question to that connection (gets them notified without tagging in the OP).

---

## Posting order suggestion

1. **dev.to first** (~18:00 PT Sunday) — establishes the canonical writeup URL with a real comments thread.
2. **HN second** (~18:30 PT) — links to the GitHub repo, not the blog. The blog is for users who click through.
3. **r/MachineLearning + r/LangChain** (~19:00 PT) — once HN is moving.
4. **LinkedIn** (~20:00 PT) — links to the writeup, not HN. LinkedIn audience is less HN-overlapped than you'd expect; keep them separate.

Don't post to all four within the same 5 minutes — looks like a campaign rather than a launch. Spacing them an hour apart preserves credibility.
