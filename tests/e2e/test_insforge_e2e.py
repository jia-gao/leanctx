"""End-to-end InsForge harness test (direct mode).

Runs a tiny LongBench subset off vs on through a real provider + the live GPU
sidecar, asserting the savings/accuracy/report pipeline works on real
``usage.prompt_tokens``. Uses ``--upstream anthropic`` (the key present in CI/dev
.env); the OpenRouter / live-gateway paths share the same run_leg/scoring code.

Marked e2e. Skips without ANTHROPIC_API_KEY. Cost: a few cents.
"""
from __future__ import annotations

import os

import pytest

from benchmarks.insforge import bench_insforge as bi


@pytest.mark.e2e
def test_insforge_direct_anthropic_small_subset(tmp_path):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    out = tmp_path / "if.jsonl"
    report = tmp_path / "if.md"
    rc = bi.main([
        "--upstream", "anthropic",
        "--model", "claude-haiku-4-5-20251001",
        "--lb-n", "3",
        "--no-closed-book",
        "--out", str(out), "--report", str(report),
    ])
    assert rc in (0, 1)  # PASS or NO-GO, but it ran end to end
    assert out.exists() and report.exists()

    import json

    records = [json.loads(line) for line in out.read_text().splitlines()]
    off = [r for r in records if r["leg"] == "OFF" and r.get("usage_prompt_tokens")]
    on = [r for r in records if r["leg"] == "ON" and r.get("usage_prompt_tokens")]
    assert off and on, "no usage tokens recorded — provider call failed"
    # provider returned real prompt-token counts
    assert all(r["usage_prompt_tokens"] > 0 for r in off + on)
    text = report.read_text()
    assert "usage.prompt_tokens" in text
    assert "LongBench v2 accuracy" in text
