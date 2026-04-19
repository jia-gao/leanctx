/**
 * Attach leanctx compression telemetry to a provider response.
 *
 * Mirrors the Python SDK's `_attach_telemetry` — uses three fields on
 * the response's usage object so downstream observability pipelines see
 * a uniform shape across both SDKs:
 *
 *   usage.leanctxTokensSaved
 *   usage.leanctxRatio
 *   usage.leanctxMethod
 *
 * Providers that don't expose a `usage` field (e.g. streaming responses
 * before the final chunk) are silently skipped.
 */

import type { CompressionStats } from "./stats.js";

export function attachTelemetry(
    response: unknown,
    stats: CompressionStats,
    field: string = "usage",
): void {
    if (response === null || typeof response !== "object") {
        return;
    }
    const usage = (response as Record<string, unknown>)[field];
    if (usage === null || typeof usage !== "object") {
        return;
    }
    const target = usage as Record<string, unknown>;
    target.leanctxTokensSaved = stats.inputTokens - stats.outputTokens;
    target.leanctxRatio = stats.ratio;
    target.leanctxMethod = stats.method;
}
