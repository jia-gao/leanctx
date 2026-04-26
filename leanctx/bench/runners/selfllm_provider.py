"""Runners: selfllm-{anthropic,openai,gemini} — live-provider compression.

Each runner constructs a SelfLLM compressor targeting the named provider
and runs it against the workload. They report per-call distributions
rather than asserting exact values (live API output is non-deterministic).

Each requires the corresponding provider extra (anthropic / openai /
gemini) AND its API-key env var:

    selfllm-anthropic → ANTHROPIC_API_KEY
    selfllm-openai    → OPENAI_API_KEY
    selfllm-gemini    → GEMINI_API_KEY (or GOOGLE_API_KEY)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from leanctx.bench.scenarios import register
from leanctx.bench.schema import BenchRecord
from leanctx.bench.workloads import load_workload

_PROVIDER_DEFAULTS = {
    "anthropic": ("claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    "openai": ("gpt-4o-mini", "OPENAI_API_KEY"),
    "gemini": ("gemini-2.5-flash", "GEMINI_API_KEY"),
}


def _run_selfllm(provider: str, workload: str) -> BenchRecord:
    try:
        from leanctx.compressors import SelfLLM  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            f"the [{provider}] extra is required for selfllm-{provider}. "
            f"Install with: pip install 'leanctx[{provider}]'"
        ) from exc

    model, env_var = _PROVIDER_DEFAULTS[provider]
    api_key = os.environ.get(env_var) or (
        os.environ.get("GOOGLE_API_KEY") if provider == "gemini" else None
    )
    if not api_key:
        raise RuntimeError(
            f"selfllm-{provider} requires environment variable {env_var}."
        )

    messages = load_workload(workload)
    compressor = SelfLLM(provider=provider, model=model, api_key=api_key)
    t0 = time.perf_counter()
    _, stats = compressor.compress(messages)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    return BenchRecord(
        leanctx_version=_lc_version(),
        scenario=f"selfllm-{provider}",
        workload=workload,
        status="success",
        compression_provider=provider,
        compression_model=model,
        compressor="selfllm",
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
        tokens_saved=stats.input_tokens - stats.output_tokens,
        ratio=stats.ratio,
        cost_usd=stats.cost_usd,
        duration_ms=duration_ms,
        warmup=False,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


@register(
    "selfllm-anthropic",
    description="SelfLLM compression via Anthropic (Haiku) live API.",
    required_extras=("anthropic",),
    required_env=("ANTHROPIC_API_KEY",),
)
def run_anthropic(*, workload: str, **opts: object) -> BenchRecord:
    return _run_selfllm("anthropic", workload)


@register(
    "selfllm-openai",
    description="SelfLLM compression via OpenAI (gpt-4o-mini) live API.",
    required_extras=("openai",),
    required_env=("OPENAI_API_KEY",),
)
def run_openai(*, workload: str, **opts: object) -> BenchRecord:
    return _run_selfllm("openai", workload)


@register(
    "selfllm-gemini",
    description="SelfLLM compression via Gemini (2.5-flash) live API.",
    required_extras=("gemini",),
    required_env=("GEMINI_API_KEY",),
)
def run_gemini(*, workload: str, **opts: object) -> BenchRecord:
    return _run_selfllm("gemini", workload)


def _lc_version() -> str:
    from leanctx import __version__  # noqa: PLC0415

    return __version__
