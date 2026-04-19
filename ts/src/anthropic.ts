/**
 * Anthropic wrapper — drop-in replacement for `@anthropic-ai/sdk`.
 *
 * Usage::
 *
 *     import { Anthropic } from "leanctx";
 *     const client = new Anthropic({ apiKey: "sk-ant-..." });
 *     const response = await client.messages.create({ ... });
 *     // response.usage.leanctxTokensSaved is attached
 *
 * v0.0.x is passthrough — the wrapper forwards to the real SDK without
 * compression. Compression lands in v0.1 via Middleware.
 */

import Anthropic_SDK from "@anthropic-ai/sdk";
import type { LeanctxConfig } from "./middleware.js";
import { Middleware } from "./middleware.js";

type AnthropicClientOptions = ConstructorParameters<typeof Anthropic_SDK>[0];

export interface LeanctxClientOptions {
    leanctxConfig?: LeanctxConfig;
}

export class Anthropic {
    private _upstream: Anthropic_SDK;
    private _middleware: Middleware;
    readonly messages: MessagesWrapper;

    constructor(options: AnthropicClientOptions & LeanctxClientOptions = {}) {
        const { leanctxConfig, ...upstreamOptions } = options;
        this._upstream = new Anthropic_SDK(upstreamOptions);
        this._middleware = new Middleware(leanctxConfig ?? {});
        this.messages = new MessagesWrapper(this._upstream, this._middleware);
    }

    // Forward any other property access to the upstream client. The cast
    // matches the Python __getattr__ fallback: non-intercepted attributes
    // (completions, batches, models, etc.) pass through untouched.
    get upstream(): Anthropic_SDK {
        return this._upstream;
    }
}

class MessagesWrapper {
    constructor(
        private readonly _upstream: Anthropic_SDK,
        private readonly _middleware: Middleware,
    ) {}

    async create(params: Parameters<Anthropic_SDK["messages"]["create"]>[0]): Promise<unknown> {
        // The Anthropic SDK's create signature is a discriminated union
        // (stream vs non-stream); we just forward both paths. v0.1 wires
        // the middleware to actually compress params.messages.
        if ("messages" in params && Array.isArray(params.messages)) {
            const [compressed] = this._middleware.compressMessages(
                params.messages as Parameters<typeof this._middleware.compressMessages>[0],
            );
            params = { ...params, messages: compressed } as typeof params;
        }
        return this._upstream.messages.create(params);
    }

    stream(params: Parameters<Anthropic_SDK["messages"]["stream"]>[0]): unknown {
        if ("messages" in params && Array.isArray(params.messages)) {
            const [compressed] = this._middleware.compressMessages(
                params.messages as Parameters<typeof this._middleware.compressMessages>[0],
            );
            params = { ...params, messages: compressed } as typeof params;
        }
        return this._upstream.messages.stream(params);
    }
}
