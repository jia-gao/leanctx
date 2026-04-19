"""End-to-end integration tests against the real anthropic SDK.

Uses respx to mock the HTTP call, so we exercise every layer — the real
anthropic.Anthropic client, httpx, our intercept wrappers — without a
real API key or network access. If this passes, the drop-in claim
(``from leanctx import Anthropic``) actually holds.
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import pytest

ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None
RESPX_AVAILABLE = importlib.util.find_spec("respx") is not None

pytestmark = pytest.mark.skipif(
    not (ANTHROPIC_AVAILABLE and RESPX_AVAILABLE),
    reason="anthropic and respx are required for e2e tests",
)


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


def test_anthropic_wrapper_returns_real_response_shape() -> None:
    import respx
    from httpx import Response

    from leanctx import Anthropic

    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=_response_body()))

        client = Anthropic(api_key="sk-test")
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
    import respx
    from httpx import Response

    from leanctx import Anthropic

    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=_response_body()))

        client = Anthropic(api_key="sk-test")
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
    import respx
    from httpx import Response

    from leanctx import Anthropic

    with respx.mock(base_url="https://api.anthropic.com") as mock:
        route = mock.post("/v1/messages").mock(
            return_value=Response(200, json=_response_body(text="ok"))
        )

        client = Anthropic(
            api_key="sk-test",
            leanctx_config={"mode": "on", "trigger": {"threshold_tokens": 0}},
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "prose message through pipeline"}],
        )

    assert route.called
    # The HTTP body our wrapper sent upstream still has the original
    # message — Verbatim didn't alter content — but it passed through
    # our pipeline, which is what we set out to prove.
    sent = json.loads(route.calls[0].request.content)
    assert len(sent["messages"]) == 1
    assert sent["messages"][0]["role"] == "user"

    assert response.usage.leanctx_method == "verbatim"


def test_custom_base_url_honored() -> None:
    """Users on Bedrock, Vertex, or a proxy pass base_url through."""
    import respx
    from httpx import Response

    from leanctx import Anthropic

    with respx.mock(base_url="https://proxy.example.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=_response_body()))

        client = Anthropic(api_key="sk-test", base_url="https://proxy.example.com")
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert response.content[0].text == "hi back"
