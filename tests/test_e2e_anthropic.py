"""End-to-end integration tests against the real anthropic SDK.

Mocks at the transport layer, so we exercise every layer — the real
``anthropic.Anthropic`` client, its HTTP stack, our intercept wrappers —
without a real API key or network access. If this passes, the drop-in claim
(``from leanctx import Anthropic``) actually holds.

These tests deliberately do **not** use respx. anthropic 1.0 moved its
transport from ``httpx`` to ``httpx2``; respx patches ``httpx``, so under 1.0
it silently stops intercepting and the "mocked" calls become real network
calls that fail on auth. Binding the mock to whichever HTTP library the
installed SDK actually imported keeps these tests honest across both
generations — and makes a future transport swap fail loudly rather than
quietly escaping to the network.
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import pytest

ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None

pytestmark = pytest.mark.skipif(
    not ANTHROPIC_AVAILABLE,
    reason="anthropic is required for e2e tests",
)


def _http_mod() -> Any:
    """The HTTP library the installed anthropic SDK is built on.

    anthropic >= 1.0 uses httpx2, earlier releases use httpx. Read it off the
    SDK rather than guessing, so the mock transport is the exact type its
    client will accept even when both libraries are installed.
    """
    import anthropic._base_client as base

    mod = getattr(base, "httpx2", None) or getattr(base, "httpx", None)
    if mod is None:  # pragma: no cover - only if the SDK restructures again
        pytest.skip("cannot determine the anthropic SDK's HTTP library")
    return mod


def _response_body(text: str = "hi back", model: str = "claude-sonnet-4-6") -> dict[str, Any]:
    return {
        "id": "msg_01TEST",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


class _Recorder:
    """Captures the requests the SDK actually put on the wire."""

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.requests: list[Any] = []

    @property
    def called(self) -> bool:
        return bool(self.requests)

    def sent_json(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    def client(self) -> Any:
        http = _http_mod()

        def handler(request: Any) -> Any:
            self.requests.append(request)
            return http.Response(200, json=self._body)

        return http.Client(transport=http.MockTransport(handler))


def test_anthropic_wrapper_returns_real_response_shape() -> None:
    from leanctx import Anthropic

    rec = _Recorder(_response_body())
    client = Anthropic(api_key="sk-test", http_client=rec.client())
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )

    # Real anthropic response shape flowed through untouched.
    assert response.model == "claude-sonnet-4-6"
    assert response.content[0].text == "hi back"
    assert response.usage.input_tokens == 10


def test_leanctx_telemetry_attached_to_usage() -> None:
    from leanctx import Anthropic

    rec = _Recorder(_response_body())
    client = Anthropic(api_key="sk-test", http_client=rec.client())
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )

    # Passthrough mode still attaches the three leanctx_* fields so
    # downstream observability pipelines see a uniform shape.
    assert hasattr(response.usage, "leanctx_tokens_saved")
    assert hasattr(response.usage, "leanctx_ratio")
    assert hasattr(response.usage, "leanctx_method")
    assert response.usage.leanctx_method == "passthrough"


def test_pipeline_runs_when_mode_is_on() -> None:
    """With mode=on + threshold=0, every request hits the pipeline.

    All routing paths fall back to Verbatim in v0.0.x, so output messages
    match input. The test verifies the pipeline actually executed — the
    leanctx_method on the response is "verbatim", not "passthrough".
    """
    from leanctx import Anthropic

    rec = _Recorder(_response_body(text="ok"))
    client = Anthropic(
        api_key="sk-test",
        http_client=rec.client(),
        leanctx_config={"mode": "on", "trigger": {"threshold_tokens": 0}},
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "prose message through pipeline"}],
    )

    assert rec.called
    # The HTTP body our wrapper sent upstream still has the original
    # message — Verbatim didn't alter content — but it passed through
    # our pipeline, which is what we set out to prove.
    sent = rec.sent_json()
    assert len(sent["messages"]) == 1
    assert sent["messages"][0]["role"] == "user"

    assert response.usage.leanctx_method == "verbatim"


def test_custom_base_url_honored() -> None:
    """Users on Bedrock, Vertex, or a proxy pass base_url through."""
    from leanctx import Anthropic

    rec = _Recorder(_response_body())
    client = Anthropic(
        api_key="sk-test",
        base_url="https://proxy.example.com",
        http_client=rec.client(),
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.content[0].text == "hi back"
    # The override actually reached the wire rather than defaulting to
    # api.anthropic.com — that is the whole point of the parameter.
    assert rec.called
    assert str(rec.requests[0].url).startswith("https://proxy.example.com")
