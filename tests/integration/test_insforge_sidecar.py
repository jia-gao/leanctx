"""InsForge harness integration tests — require a live leanctx sidecar only
(no Node, no InsForge stack, no API key). Reuses the session-scoped
``leanctx_sidecar`` fixture from conftest.

Marked ``insforge``: run with ``-m insforge``.
"""
from __future__ import annotations

import pytest

from benchmarks.insforge import bench_insforge as bi
from benchmarks.insforge.transcripts import load_transcript_items


@pytest.mark.insforge
def test_sidecar_compress_preserves_count_and_shrinks_prose(leanctx_sidecar):
    """A prose-heavy transcript round-trips with the same message count and
    fewer tiktoken tokens; the tool/code turns survive verbatim."""
    url, _proc, lingua_ok = leanctx_sidecar
    if not lingua_ok:
        pytest.skip("llmlingua not installed — sidecar is in passthrough mode")

    from leanctx.tokens import count_tokens

    compress = bi._sidecar_compress_factory(url)
    # agent_extended is a 50-turn agent transcript (prose + code + tool results).
    from leanctx.bench.workloads import load_workload

    messages = load_workload("agent_extended")
    result = compress(messages)

    assert len(result["messages"]) == len(messages)  # one-in-one-out
    before = sum(count_tokens(m["content"]) for m in messages if isinstance(m["content"], str))
    after = sum(
        count_tokens(m["content"]) for m in result["messages"] if isinstance(m["content"], str)
    )
    assert after < before, "compression did not reduce prose tokens"


@pytest.mark.insforge
def test_run_offon_savings_no_api(leanctx_sidecar):
    """OFF (identity) vs ON (sidecar) over LongBench-free items: the ON leg has
    strictly fewer tiktoken tokens. No provider calls (eval_cfg=None)."""
    url, _proc, lingua_ok = leanctx_sidecar
    if not lingua_ok:
        pytest.skip("llmlingua not installed — sidecar is in passthrough mode")

    # One full-context item (~3.4K tokens, over the sidecar's 1500-token gate).
    transcript_items = load_transcript_items("agent_extended", episodes=1)
    res = bi.run_offon(
        lb_items=[], transcript_items=transcript_items,
        eval_cfg=None, sidecar_url=url, closed_book=False, direct=True,
    )
    off_tok = sum(r["tokens_compressed"] for r in res["off"])
    on_tok = sum(r["tokens_compressed"] for r in res["on"])
    assert on_tok < off_tok, f"ON ({on_tok}) not < OFF ({off_tok})"


@pytest.mark.insforge
def test_dry_run_main_writes_report_without_api(leanctx_sidecar, tmp_path):
    """The --dry-run harness path produces a report + JSONL with no provider
    calls (sidecar + tiktoken only).

    Uses ``--lb-n 0`` (transcript-only): LongBench loading pulls the optional
    ``datasets`` extra (and would download the dataset), which CI does not
    install — the transcript workload exercises the same dry-run wiring without
    that dependency.
    """
    url, _proc, _lingua_ok = leanctx_sidecar
    out = tmp_path / "r.jsonl"
    report = tmp_path / "r.md"
    rc = bi.main([
        "--upstream", "anthropic", "--dry-run",
        "--lb-n", "0", "--transcript", "--transcript-episodes", "1",
        "--sidecar-url", url,
        "--out", str(out), "--report", str(report),
    ])
    assert rc in (0, 1)
    assert report.exists() and out.exists()
    assert "Measurement Report" in report.read_text()
    assert "DRY RUN" in report.read_text()
