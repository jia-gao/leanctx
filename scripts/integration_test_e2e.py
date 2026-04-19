"""End-to-end integration test: leanctx.Anthropic wrapper with real
Lingua compression and a respx-mocked Anthropic API.

This is the full v0.1 pipeline exercised top to bottom:

    leanctx.Anthropic
        ↓
    Middleware (mode=on, threshold, routing)
        ↓
    classify / route / compress
        ↓
    Lingua (real llmlingua + LLMLingua-2 model)
        ↓
    anthropic.Anthropic (real SDK)
        ↓
    httpx / respx (mocked at the socket boundary)

Requires: ``pip install 'leanctx[dev,lingua]'``
"""

from __future__ import annotations

# Force HuggingFace offline BEFORE any imports that may transitively load
# transformers — otherwise loading LLMLingua-2 makes network calls to
# huggingface.co that respx would intercept and fail.
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json  # noqa: E402
import time  # noqa: E402

import respx  # noqa: E402
from httpx import Response  # noqa: E402

from leanctx import Anthropic  # noqa: E402

LONG_PROSE = (
    "Cloud-native architectures have become the dominant paradigm for deploying "
    "modern applications. The shift from monolithic architectures to microservices "
    "was driven by scalability, resilience, and independent deployment cycles. "
    "Kubernetes adopters report improved productivity, faster time-to-market, and "
    "better utilization. However, the transition demands investment in observability, "
    "service meshes, and developer tooling to manage complexity. "
) * 10


def _mock_response() -> dict:
    return {
        "id": "msg_01INTEGRATION",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ack"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 50,
            "output_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def main() -> None:
    print("=== leanctx.Anthropic + real Lingua + mocked Anthropic API ===\n")

    with respx.mock(base_url="https://api.anthropic.com") as mock:
        route = mock.post("/v1/messages").mock(
            return_value=Response(200, json=_mock_response())
        )

        client = Anthropic(
            api_key="sk-test",
            leanctx_config={
                "mode": "on",
                "trigger": {"threshold_tokens": 100},
                "routing": {"prose": "lingua"},
            },
        )

        print(f"Sending request with {len(LONG_PROSE)} chars of prose...")
        t0 = time.monotonic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": LONG_PROSE}],
        )
        elapsed = time.monotonic() - t0
        print(f"  full request cycle: {elapsed:.2f}s\n")

    # What actually reached the Anthropic endpoint?
    sent_body = json.loads(route.calls[0].request.content)
    sent_content = sent_body["messages"][0]["content"]

    print("=== What we sent to api.anthropic.com ===")
    print(f"  messages sent:    {len(sent_body['messages'])}")
    print(f"  first msg chars:  {len(sent_content)}")
    print(f"  original chars:   {len(LONG_PROSE)}")
    print(f"  char reduction:   {1 - len(sent_content) / len(LONG_PROSE):.1%}\n")

    print("=== Preview of compressed bytes that hit the wire ===")
    print(sent_content[:500] + ("..." if len(sent_content) > 500 else ""))
    print()

    print("=== leanctx telemetry on response ===")
    print(f"  leanctx_method:       {response.usage.leanctx_method}")
    print(f"  leanctx_ratio:        {response.usage.leanctx_ratio:.1%}")
    print(f"  leanctx_tokens_saved: {response.usage.leanctx_tokens_saved}")
    print()

    assert response.usage.leanctx_method == "lingua", (
        f"expected method=lingua, got {response.usage.leanctx_method}"
    )
    assert len(sent_content) < len(LONG_PROSE), (
        f"expected compression; sent {len(sent_content)} vs original {len(LONG_PROSE)}"
    )
    assert response.usage.leanctx_tokens_saved > 0, "expected positive token savings"

    print("OK — v0.1 pipeline end-to-end works.")


if __name__ == "__main__":
    main()
