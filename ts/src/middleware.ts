/**
 * Middleware — provider-agnostic compression orchestrator (skeleton).
 *
 * v0.0.x is a passthrough: compressMessages returns input unchanged
 * with zero-valued stats. v0.1 will plug in the Compressor router
 * (local LLMLingua-2 / self_llm / hybrid), matching the Python SDK's
 * architecture.
 */

import type { CompressionStats } from "./stats.js";
import { passthroughStats } from "./stats.js";

export interface LeanctxConfig {
    mode?: "off" | "passthrough" | "on" | "hybrid";
    trigger?: {
        thresholdTokens?: number;
    };
    routing?: Record<string, string>;
    lingua?: Record<string, unknown>;
    selfllm?: Record<string, unknown>;
    strategies?: {
        dedup?: boolean;
        purgeErrors?: boolean | { afterTurns?: number };
    };
}

export interface ChatMessage {
    role: string;
    content: string | Array<{ type: string; text?: string }>;
    [key: string]: unknown;
}

export class Middleware {
    readonly config: LeanctxConfig;

    constructor(config: LeanctxConfig = {}) {
        this.config = config;
    }

    compressMessages(messages: ChatMessage[]): [ChatMessage[], CompressionStats] {
        // v0.0.x: passthrough.
        return [messages, passthroughStats()];
    }

    async compressMessagesAsync(
        messages: ChatMessage[],
    ): Promise<[ChatMessage[], CompressionStats]> {
        return [messages, passthroughStats()];
    }
}
