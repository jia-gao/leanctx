# leanctx (TypeScript)

Drop-in prompt compression for production LLM applications. TypeScript SDK.

> **Status:** v0.0.0 — skeleton. The Python SDK is feature-complete for v0.1
> (LLMLingua-2 + SelfLLM + strategies); the TS SDK currently mirrors the
> public surface with a passthrough implementation. Real compression lands
> when v0.1 releases.

## Install

```bash
npm install leanctx
# and your provider SDK:
npm install @anthropic-ai/sdk        # for Anthropic
npm install openai                   # for OpenAI
```

## Usage (preview)

```ts
import { Anthropic } from "leanctx";

const client = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
    leanctxConfig: {
        mode: "on",
        trigger: { thresholdTokens: 4000 },
        routing: { prose: "lingua" },  // v0.1
    },
});

const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 1024,
    messages: [{ role: "user", content: longDocument }],
});

// v0.1 will attach compression telemetry:
// response.usage.leanctxTokensSaved
```

## Why prefer the Python SDK for now

The Python SDK already ships real compression (LLMLingua-2 local, SelfLLM
via Haiku-style delegation, dedup + purge-errors strategies) and full
end-to-end tests. TS is currently passthrough only.

For cross-language deployments, consider running leanctx as an HTTP proxy
(v0.3+) and pointing your TS client at it.

## License

MIT. See the repo root `LICENSE`.
