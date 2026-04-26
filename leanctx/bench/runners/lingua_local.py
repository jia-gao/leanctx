"""Runner: lingua-local — offline LLMLingua-2 compression on a workload.

Uses no provider HTTP — purely local model. Requires the ``[lingua]``
extra; the runner reports a clean diagnostic if it's missing.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from leanctx.bench.scenarios import register
from leanctx.bench.schema import BenchRecord
from leanctx.bench.workloads import load_workload


@register(
    "lingua-local",
    description="Offline LLMLingua-2 compression on a single workload.",
    required_extras=("lingua",),
)
def run(*, workload: str, **opts: object) -> BenchRecord:
    try:
        from leanctx.compressors import Lingua  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "the [lingua] extra is required for the lingua-local scenario. "
            "Install with: pip install 'leanctx[lingua]'"
        ) from exc

    messages = load_workload(workload)
    raw_ratio = opts.get("ratio", 0.5)
    ratio = float(raw_ratio) if isinstance(raw_ratio, (int, float)) else 0.5
    lingua = Lingua(ratio=ratio)
    t0 = time.perf_counter()
    _, stats = lingua.compress(messages)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    revision: str | None = None
    pc = lingua._prompt_compressor
    if pc is not None:
        revision = getattr(pc, "model_name", None) or lingua.model

    return BenchRecord(
        leanctx_version=_lc_version(),
        scenario="lingua-local",
        workload=workload,
        status="success",
        compressor="lingua",
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
        tokens_saved=stats.input_tokens - stats.output_tokens,
        ratio=stats.ratio,
        cost_usd=stats.cost_usd,
        duration_ms=duration_ms,
        warmup=False,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        lingua_model_revision=revision,
    )


def _lc_version() -> str:
    from leanctx import __version__  # noqa: PLC0415

    return __version__
