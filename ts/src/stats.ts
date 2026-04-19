/**
 * CompressionStats — telemetry shape shared by every compression call.
 *
 * Attached to each response as `usage.leanctxTokensSaved`,
 * `usage.leanctxRatio`, and `usage.leanctxMethod` — camelCase variants
 * of the Python SDK's underscore_case for JS idiom.
 */

export interface CompressionStats {
    inputTokens: number;
    outputTokens: number;
    ratio: number;
    method: string;
    costUsd: number;
}

export function passthroughStats(): CompressionStats {
    return {
        inputTokens: 0,
        outputTokens: 0,
        ratio: 1.0,
        method: "passthrough",
        costUsd: 0.0,
    };
}
