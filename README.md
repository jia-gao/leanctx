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

> **Status:** v0.0.0 — name reservation. v0.1 (working release) coming in ~4 weeks.
> [Watch the repo](https://github.com/jia-gao/leanctx/subscription) to be notified.

## Who this is for

You're building a production LLM app and your token bill is a line item:

- RAG apps with large retrieved documents
- Long-running conversational agents
- LangChain / LangGraph / CrewAI workflows with growing tool chains
- Document-processing pipelines
- Anything where input tokens accumulate and you pay for every one

If your code calls `anthropic.messages.create()` or `openai.chat.completions.create()` in production, this is for you.

## How it works (coming in v0.1)

Three compression modes, one config switch:

- **`local`** — runs Microsoft's open-source LLMLingua-2 locally. Free marginal cost.
- **`self_llm`** — lets your own configured LLM do the compression. Highest quality.
- **`hybrid`** (default) — routes by content type: code stays verbatim, prose goes through LLMLingua-2, long important spans fall back to self_llm.

Content-aware routing means code blocks, diffs, stack traces, and tool schemas are preserved verbatim — no corrupted syntax.

## Roadmap

- [ ] v0.1 — Python SDK, `local` mode (LLMLingua-2), Anthropic + OpenAI drop-in clients
- [ ] v0.2 — `self_llm` mode, Gemini client, LangChain / LlamaIndex integrations
- [ ] v0.3 — TypeScript SDK, Docker image, OTel observability
- [ ] v0.4 — Helm chart, Kubernetes sidecar deployment

## Install (placeholder)

```bash
pip install leanctx  # not yet functional — reservation only
```

## Credits

Architecturally inspired by [OpenCode DCP](https://github.com/Opencode-DCP/opencode-dynamic-context-pruning) (AGPL-3.0, not copied). leanctx is a clean-room implementation under MIT.

Ships Microsoft's [LLMLingua-2](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank) (MIT).

## License

MIT. See [LICENSE](LICENSE).
