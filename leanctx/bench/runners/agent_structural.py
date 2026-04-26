"""Runner: agent-structural — verifies the 5 structural-integrity invariants
on the agent workload (preserves the contract from
scripts/integration_test_agent_workload.py).

Invariants checked (each binary; all must hold for status=success):
1. tool_use_id linkage preserved between tool_use and tool_result blocks
2. code blocks preserved verbatim (Lingua must not rewrite them)
3. error/traceback content preserved verbatim
4. tool_use input dict preserved (compression must not mutate tool args)
5. log span (verbose tool_result content) demonstrably compressed
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from leanctx.bench.scenarios import register
from leanctx.bench.schema import BenchRecord
from leanctx.bench.workloads import load_workload


@register(
    "agent-structural",
    description="Structural-integrity invariants on the agent workload.",
    required_extras=("lingua",),
)
def run(*, workload: str, **opts: object) -> BenchRecord:
    if workload != "agent":
        raise ValueError("agent-structural requires --workload agent")

    try:
        from leanctx.compressors import Lingua  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "the [lingua] extra is required for agent-structural. "
            "Install with: pip install 'leanctx[lingua]'"
        ) from exc

    original = load_workload(workload)
    lingua = Lingua(ratio=0.5)
    t0 = time.perf_counter()
    compressed, stats = lingua.compress(original)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    invariants = {
        "tool_linkage": _check_tool_linkage(compressed, original),
        "code_verbatim": _check_code_verbatim(compressed, original),
        "error_verbatim": _check_error_verbatim(compressed, original),
        "tool_input_preserved": _check_tool_input(compressed, original),
        "log_compressed": _check_log_compressed(compressed, original),
    }

    status = "success" if all(invariants.values()) else "failure"

    return BenchRecord(
        leanctx_version=_lc_version(),
        scenario="agent-structural",
        workload=workload,
        status=status,
        compressor="lingua",
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
        tokens_saved=stats.input_tokens - stats.output_tokens,
        ratio=stats.ratio,
        cost_usd=stats.cost_usd,
        duration_ms=duration_ms,
        warmup=False,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        invariants=invariants,
        error=None if status == "success" else _failed_invariants(invariants),
    )


def _failed_invariants(inv: dict[str, bool]) -> str:
    failed = [k for k, ok in inv.items() if not ok]
    return f"failed invariants: {', '.join(failed)}"


def _iter_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten message contents into a list of block dicts."""
    out: list[dict[str, Any]] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    out.append(block)
    return out


def _check_tool_linkage(
    compressed: list[dict[str, Any]], original: list[dict[str, Any]]
) -> bool:
    orig_ids = {b["id"] for b in _iter_blocks(original) if b.get("type") == "tool_use"}
    comp_ids = {b["id"] for b in _iter_blocks(compressed) if b.get("type") == "tool_use"}
    if orig_ids != comp_ids:
        return False
    orig_refs = {
        b["tool_use_id"]
        for b in _iter_blocks(original)
        if b.get("type") == "tool_result"
    }
    comp_refs = {
        b["tool_use_id"]
        for b in _iter_blocks(compressed)
        if b.get("type") == "tool_result"
    }
    return orig_refs == comp_refs


def _check_code_verbatim(
    compressed: list[dict[str, Any]], original: list[dict[str, Any]]
) -> bool:
    """Code-fenced blocks (```...```) must appear verbatim in compressed text."""
    import re  # noqa: PLC0415

    pat = re.compile(r"```[\s\S]*?```")
    orig_text = "\n".join(_text_blobs(original))
    comp_text = "\n".join(_text_blobs(compressed))
    return all(code in comp_text for code in pat.findall(orig_text))


def _check_error_verbatim(
    compressed: list[dict[str, Any]], original: list[dict[str, Any]]
) -> bool:
    orig_text = "\n".join(_text_blobs(original))
    comp_text = "\n".join(_text_blobs(compressed))
    if "Traceback" in orig_text and "Traceback" not in comp_text:
        return False
    return "TimeoutError" not in orig_text or "TimeoutError" in comp_text


def _check_tool_input(
    compressed: list[dict[str, Any]], original: list[dict[str, Any]]
) -> bool:
    orig_inputs = {
        b["id"]: b["input"]
        for b in _iter_blocks(original)
        if b.get("type") == "tool_use"
    }
    comp_inputs = {
        b["id"]: b.get("input")
        for b in _iter_blocks(compressed)
        if b.get("type") == "tool_use"
    }
    return orig_inputs == comp_inputs


def _check_log_compressed(
    compressed: list[dict[str, Any]], original: list[dict[str, Any]]
) -> bool:
    """The verbose log block (tool_result with ~30 INFO heartbeats) must shrink."""
    orig_logs = [
        b["content"]
        for b in _iter_blocks(original)
        if b.get("type") == "tool_result" and "heartbeat" in str(b.get("content", ""))
    ]
    comp_logs = [
        b["content"]
        for b in _iter_blocks(compressed)
        if b.get("type") == "tool_result" and "heartbeat" in str(b.get("content", ""))
    ]
    if not orig_logs:
        return True
    if not comp_logs:
        return True  # log was elided entirely; that's also "compressed"
    return len(comp_logs[0]) < len(orig_logs[0])


def _text_blobs(messages: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        out.append(str(block.get("text", "")))
                    elif block.get("type") == "tool_result":
                        rc = block.get("content")
                        if isinstance(rc, str):
                            out.append(rc)
    return out


def _lc_version() -> str:
    from leanctx import __version__  # noqa: PLC0415

    return __version__
