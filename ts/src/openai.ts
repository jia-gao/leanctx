/**
 * OpenAI wrapper — drop-in replacement for `openai`.
 *
 * Usage::
 *
 *     import { OpenAI } from "leanctx";
 *     const client = new OpenAI({ apiKey: "sk-..." });
 *     const response = await client.chat.completions.create({ ... });
 *
 * v0.0.x is passthrough.
 */

import OpenAI_SDK from "openai";
import type { LeanctxConfig } from "./middleware.js";
import { Middleware } from "./middleware.js";
import { attachTelemetry } from "./telemetry.js";

type OpenAIClientOptions = ConstructorParameters<typeof OpenAI_SDK>[0];

export interface OpenAILeanctxClientOptions {
    leanctxConfig?: LeanctxConfig;
}

export class OpenAI {
    private _upstream: OpenAI_SDK;
    private _middleware: Middleware;
    readonly chat: ChatWrapper;

    constructor(options: OpenAIClientOptions & OpenAILeanctxClientOptions = {}) {
        const { leanctxConfig, ...upstreamOptions } = options;
        this._upstream = new OpenAI_SDK(upstreamOptions);
        this._middleware = new Middleware(leanctxConfig ?? {});
        this.chat = new ChatWrapper(this._upstream, this._middleware);
    }

    get upstream(): OpenAI_SDK {
        return this._upstream;
    }
}

class ChatWrapper {
    readonly completions: CompletionsWrapper;

    constructor(upstream: OpenAI_SDK, middleware: Middleware) {
        this.completions = new CompletionsWrapper(upstream, middleware);
    }
}

class CompletionsWrapper {
    constructor(
        private readonly _upstream: OpenAI_SDK,
        private readonly _middleware: Middleware,
    ) {}

    async create(
        params: Parameters<OpenAI_SDK["chat"]["completions"]["create"]>[0],
    ): Promise<unknown> {
        let stats = undefined;
        if ("messages" in params && Array.isArray(params.messages)) {
            const [compressed, s] = this._middleware.compressMessages(
                params.messages as Parameters<typeof this._middleware.compressMessages>[0],
            );
            params = { ...params, messages: compressed } as typeof params;
            stats = s;
        }
        const response = await this._upstream.chat.completions.create(params);
        // Only attach when we actually ran the middleware (stream=false
        // returns a ChatCompletion; stream=true returns an iterator
        // whose chunks don't carry usage until the final chunk — v0.2).
        if (stats !== undefined && !(params as { stream?: boolean }).stream) {
            attachTelemetry(response, stats);
        }
        return response;
    }
}
