/**
 * leanctx — drop-in prompt compression for production LLM applications.
 *
 * v0.0.x is a passthrough skeleton matching the Python SDK's public
 * surface. The working release (v0.1) will swap passthrough for the
 * real three-mode compression pipeline.
 *
 * See https://github.com/jia-gao/leanctx for the Python SDK (which is
 * further along) and overall progress.
 */

export { Anthropic } from "./anthropic.js";
export type { LeanctxClientOptions } from "./anthropic.js";

export { OpenAI } from "./openai.js";
export type { OpenAILeanctxClientOptions } from "./openai.js";

export { Middleware } from "./middleware.js";
export type { ChatMessage, LeanctxConfig } from "./middleware.js";

export type { CompressionStats } from "./stats.js";
export { passthroughStats } from "./stats.js";

export { attachTelemetry } from "./telemetry.js";

export const VERSION = "0.0.0";
