"""Bundled workload fixtures.

Three workloads ship with leanctx:

- ``rag``  — a single 1.7 KB SRE-incident document; matches the
  benchmark in docs/benchmarks/agent-workload.md's RAG harness
- ``chat`` — multi-turn conversation, ~3 KB across 8 turns
- ``agent`` — 9-message coding-agent transcript with tool_use /
  tool_result blocks (matches scripts/integration_test_agent_workload.py)

Workloads are returned as a list of message dicts in the leanctx
canonical shape (``role``, ``content``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

WorkloadName = str


def _rag() -> list[dict[str, Any]]:
    document = (
        "Service degradation timeline:\n"
        "14:02 — checkout-api p99 latency rises from 120ms to 480ms.\n"
        "14:04 — order-events Kafka topic partition 7 lag spikes to 200k.\n"
        "14:05 — pager triggers on the SRE rotation.\n"
        "14:08 — first responder confirms checkout-api pod CPU is at 95%.\n"
        "14:11 — restart of payment-gateway-* pods returns latency to baseline.\n"
        "14:12 — root cause is a connection-pool exhaustion in payment-gateway after a "
        "deploy that increased the per-request DB query count from 3 to 7 without "
        "raising the pool size.\n"
        "14:30 — fix lands: pool size raised from 20 to 50, plus an alert on pool "
        "saturation > 80%.\n"
    ) * 4
    return [
        {"role": "user", "content": document},
        {
            "role": "user",
            "content": "Summarize this incident in two sentences.",
        },
    ]


def _chat() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "What is the average annual rainfall in Seattle?"},
        {"role": "assistant", "content": "About 37 inches per year."},
        {"role": "user", "content": "How does that compare to Portland?"},
        {
            "role": "assistant",
            "content": "Portland averages around 43 inches — slightly wetter than Seattle.",
        },
        {"role": "user", "content": "What about San Francisco?"},
        {"role": "assistant", "content": "San Francisco gets about 23 inches annually."},
        {"role": "user", "content": "Now give me a summary of all three."},
    ]


def _agent() -> list[dict[str, Any]]:
    # Compact agent-workload fixture: tool_use + tool_result with a code
    # block, an error trace, and a long log span. Mirrors the structural
    # invariants exercised by scripts/integration_test_agent_workload.py.
    return [
        {
            "role": "user",
            "content": "Find the bug that's making checkouts return 502.",
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll grep the recent error logs."},
                {
                    "type": "tool_use",
                    "id": "tu_001",
                    "name": "grep_logs",
                    "input": {"pattern": "ERROR.*checkout", "since": "30m"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_001",
                    "content": (
                        "2026-04-25T14:02:13Z ERROR checkout-api: payment_gateway timeout\n"
                        "2026-04-25T14:02:14Z ERROR checkout-api: payment_gateway timeout\n"
                        "2026-04-25T14:02:14Z ERROR checkout-api: payment_gateway timeout\n"
                        + ("2026-04-25T14:02:15Z INFO routine heartbeat\n" * 30)
                        + "2026-04-25T14:03:01Z ERROR payment-gateway: connection pool exhausted"
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Pool exhaustion. Let me view the config."},
                {
                    "type": "tool_use",
                    "id": "tu_002",
                    "name": "read_file",
                    "input": {"path": "payment-gateway/config/db.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_002",
                    "content": (
                        "```python\n"
                        "DB_POOL_SIZE = 20  # raised from 5 in 2024\n"
                        "DB_POOL_TIMEOUT = 5\n"
                        "DB_QUERY_BUDGET = 7  # was 3 before deploy 2026-04-25 14:00\n"
                        "```"
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": (
                "DB_QUERY_BUDGET went from 3 to 7 in today's deploy without raising "
                "DB_POOL_SIZE — that's the regression. Recommend raising pool size to 50 "
                "and adding an alert on pool saturation > 80%."
            ),
        },
        {"role": "user", "content": "Confirm the deploy SHA and the relevant traceback."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Traceback (most recent call last):\n"
                        "  File 'app.py', line 142, in process_payment\n"
                        "    conn = pool.acquire(timeout=5)\n"
                        "  File 'pool.py', line 87, in acquire\n"
                        "    raise TimeoutError('pool exhausted after 5s')\n"
                        "TimeoutError: pool exhausted after 5s"
                    ),
                }
            ],
        },
        {"role": "user", "content": "Thanks. Summarize the fix in one paragraph."},
    ]


_WORKLOADS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "rag": _rag,
    "chat": _chat,
    "agent": _agent,
}


def list_workloads() -> list[str]:
    return sorted(_WORKLOADS)


def load_workload(name: str) -> list[dict[str, Any]]:
    if name not in _WORKLOADS:
        known = ", ".join(sorted(_WORKLOADS))
        raise KeyError(f"unknown workload {name!r}; known: {known}")
    return _WORKLOADS[name]()
