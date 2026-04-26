"""Runner: anthropic-e2e — full leanctx.Anthropic + middleware path,
respx-mocked Anthropic API.

Exercises the wrapper → middleware → compressor → upstream stack
without hitting the real Anthropic API. Requires the ``[bench]`` extra
(for respx) and the ``[anthropic]`` extra.
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


@register(
    "anthropic-e2e",
    description="Full leanctx.Anthropic stack with respx-mocked Anthropic API.",
    required_extras=("anthropic", "bench"),
)
def run(*, workload: str, **opts: object) -> BenchRecord:
    try:
        import respx  # noqa: PLC0415
        from httpx import Response  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "the [bench] extra is required for the anthropic-e2e scenario. "
            "Install with: pip install 'leanctx[bench]'"
        ) from exc

    try:
        from leanctx import Anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "the [anthropic] extra is required for the anthropic-e2e scenario. "
            "Install with: pip install 'leanctx[anthropic]'"
        ) from exc

    messages = load_workload(workload)
    t0 = time.perf_counter()
    with respx.mock(
        base_url="https://api.anthropic.com", assert_all_called=False
    ) as mock:
        mock.post("/v1/messages").mock(
            return_value=Response(200, json=_mock_anthropic_response())
        )
        client = Anthropic(
            api_key="sk-test",
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
