"""Runner: anthropic-e2e — full leanctx.Anthropic + middleware path against a
transport-mocked Anthropic API.

Exercises the wrapper → middleware → compressor → upstream stack
without hitting the real Anthropic API. Requires the ``[anthropic]`` extra.
The mock is injected into the client as an HTTP transport, so it holds for
both anthropic < 1.0 (httpx) and >= 1.0 (httpx2).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from leanctx.bench.scenarios import register
from leanctx.bench.schema import BenchRecord
from leanctx.bench.workloads import load_workload


def _mock_anthropic_response() -> dict[str, Any]:
    return {
        "id": "msg_BENCH",
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


def _sdk_http_module() -> Any:
    """The HTTP library the installed anthropic SDK is built on.

    anthropic >= 1.0 moved its transport to ``httpx2``; earlier releases use
    ``httpx``. Read it off the SDK instead of importing a fixed name, so the
    mock transport is the type its client accepts under either generation.

    This scenario used to mock with respx, which patches ``httpx`` globally.
    Under anthropic 1.0 that patch no longer applies, and the "mocked"
    benchmark would have issued a **real** request to api.anthropic.com with a
    placeholder key. Injecting the transport into the client cannot silently
    escape to the network that way.
    """
    import anthropic._base_client as base  # noqa: PLC0415

    mod = getattr(base, "httpx2", None) or getattr(base, "httpx", None)
    if mod is None:  # pragma: no cover - only if the SDK restructures again
        raise RuntimeError(
            "cannot determine the anthropic SDK's HTTP library; "
            "the anthropic-e2e scenario cannot mock it safely"
        )
    return mod


@register(
    "anthropic-e2e",
    description="Full leanctx.Anthropic stack against a mocked Anthropic API.",
    required_extras=("anthropic",),
)
def run(*, workload: str, **opts: object) -> BenchRecord:
    try:
        from leanctx import Anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "the [anthropic] extra is required for the anthropic-e2e scenario. "
            "Install with: pip install 'leanctx[anthropic]'"
        ) from exc

    http = _sdk_http_module()
    body = _mock_anthropic_response()

    def _handler(request: Any) -> Any:
        return http.Response(200, json=body)

    messages = load_workload(workload)
    t0 = time.perf_counter()
    client = Anthropic(
        api_key="sk-test",
        http_client=http.Client(transport=http.MockTransport(_handler)),
        leanctx_config={
            "mode": "on",
            "trigger": {"threshold_tokens": 100},
        },
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=10, messages=messages
    )
    duration_ms = int((time.perf_counter() - t0) * 1000)

    usage = resp.usage
    in_tok = int(getattr(usage, "input_tokens", 0))
    out_tok = int(getattr(usage, "output_tokens", 0))
    saved = int(getattr(usage, "leanctx_tokens_saved", 0))
    method = str(getattr(usage, "leanctx_method", "passthrough"))
    cost_usd = float(getattr(usage, "leanctx_cost_usd", 0.0))
    ratio = float(getattr(usage, "leanctx_ratio", 1.0))

    return BenchRecord(
        leanctx_version=_lc_version(),
        scenario="anthropic-e2e",
        workload=workload,
        status="success",
        request_provider="anthropic",
        request_model="claude-sonnet-4-6",
        compressor=method,
        input_tokens=in_tok,
        output_tokens=out_tok,
        tokens_saved=saved,
        ratio=ratio,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        warmup=False,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _lc_version() -> str:
    from leanctx import __version__  # noqa: PLC0415

    return __version__
