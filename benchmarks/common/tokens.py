"""Token counting for the benchmark harness (extracted from bench_phase1).

Wraps ``leanctx.tokens.count_tokens`` over a message list — only plain-string
content contributes, matching what the sidecar compresses.
"""
from __future__ import annotations


def _sum_tokens(messages: list[dict]) -> int:
    from leanctx.tokens import count_tokens

    combined = " ".join(
        m["content"] if isinstance(m.get("content"), str) else ""
        for m in messages
    )
    return count_tokens(combined)
