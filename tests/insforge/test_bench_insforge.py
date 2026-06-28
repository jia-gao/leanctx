"""Unit tests for the InsForge measurement harness (no I/O, no API, no Node).

Covers: pricing + cost math, the transcript loader / SWE-bench adapter, the
chat-completion.service.ts patch, compute_metrics on the provider usage field,
the usage-probe run_leg path, and the report renderer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.common import pricing
from benchmarks.common.runner import run_leg
from benchmarks.common.scoring import GateResult, compute_metrics
from benchmarks.insforge import bench_insforge as bi
from benchmarks.insforge.transcripts import (
    load_transcript_items,
    swebench_traj_to_messages,
)

_FIXTURES = Path(__file__).resolve().parents[1].parent / "benchmarks" / "insforge" / "fixtures"


# ── pricing ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_pricing_haiku_alias_resolves():
    # provider-native id and OpenRouter slug price identically
    assert pricing.input_price_per_token("anthropic/claude-haiku-4.5") == pytest.approx(1e-6)
    assert pricing.input_price_per_token("claude-haiku-4-5-20251001") == pytest.approx(1e-6)


@pytest.mark.unit
def test_pricing_unknown_model_is_none():
    assert pricing.input_price_per_token("made/up-model") is None
    assert pricing.cost_usd(1000, "made/up-model") is None
    assert pricing.dollars_saved(1000, 500, "made/up-model") is None


@pytest.mark.unit
def test_pricing_dollars_saved():
    # 400 input tokens saved at $1/Mtok = $0.0004
    assert pricing.dollars_saved(1000, 600, "anthropic/claude-haiku-4.5") == pytest.approx(4e-4)
    assert pricing.cost_usd(1_000_000, "anthropic/claude-haiku-4.5") == pytest.approx(1.0)


@pytest.mark.unit
def test_pricing_custom_table_overrides_pinned():
    table = {"x/y": 2e-6}
    assert pricing.input_price_per_token("x/y", table) == pytest.approx(2e-6)
    assert pricing.input_price_per_token("anthropic/claude-haiku-4.5", table) is None


# ── transcript loader / SWE-bench adapter ───────────────────────────────────


@pytest.mark.unit
def test_agent_transcript_episodes_subset():
    items = load_transcript_items("agent_extended", episodes=2)
    assert len(items) == 2
    assert all(it["workload"] == "transcript" for it in items)
    assert all(it["item_id"].startswith("transcript::agent_extended::ep") for it in items)
    # each episode is a non-empty message list
    assert all(len(it["messages"]) > 0 for it in items)


@pytest.mark.unit
def test_agent_transcript_turns_cap():
    items = load_transcript_items("agent_extended", turns=5)
    total = sum(len(it["messages"]) for it in items)
    assert total == 5


@pytest.mark.unit
def test_swebench_adapter_history_shape():
    items = load_transcript_items(str(_FIXTURES / "swebench_sample.traj.json"))
    assert len(items) >= 1
    msgs = items[0]["messages"]
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert "user" in roles and "assistant" in roles
    assert all(isinstance(m["content"], str) for m in msgs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "traj",
    [
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        {"history": [{"role": "user", "content": "a"}]},
        {"messages": [{"role": "assistant", "content": "b"}]},
        {"trajectory": [{"messages": [{"role": "user", "content": "c"}]}]},
        {"trajectory": [{"response": "model out", "observation": "tool out"}]},
    ],
)
def test_swebench_adapter_tolerates_shapes(traj):
    msgs = swebench_traj_to_messages(traj)
    assert msgs and all("role" in m and isinstance(m["content"], str) for m in msgs)


@pytest.mark.unit
def test_swebench_adapter_flattens_block_content():
    traj = [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"text": "world"}]}]
    msgs = swebench_traj_to_messages(traj)
    assert msgs[0]["content"] == "hello\nworld"


# ── chat-completion.service.ts patch ────────────────────────────────────────


@pytest.mark.unit
def test_patch_wraps_both_call_sites_on_real_v2_2_2():
    src = (_FIXTURES / "chat-completion.service.v2.2.2.ts").read_text()
    patched = bi.patch_chat_completion_ts(src)
    # connector is INLINED (no npm dep) after the import block
    assert "async function compressMessages" in patched
    assert patched.index("import") < patched.index("async function compressMessages")
    assert patched.count("await compressMessages(this.formatMessages(messages))") == 2
    # the un-wrapped anchor is gone
    assert bi._INSFORGE_ANCHOR not in patched.replace(bi._INSFORGE_REPLACEMENT, "")


@pytest.mark.unit
def test_patch_is_idempotent():
    src = (_FIXTURES / "chat-completion.service.v2.2.2.ts").read_text()
    once = bi.patch_chat_completion_ts(src)
    assert bi.patch_chat_completion_ts(once) == once


@pytest.mark.unit
def test_patch_raises_when_anchor_missing():
    with pytest.raises(RuntimeError, match="occurrences"):
        bi.patch_chat_completion_ts("const x = 1;\n")


# ── compute_metrics on the provider usage field ─────────────────────────────


@pytest.mark.unit
def test_compute_metrics_usage_field_drives_savings():
    off = [
        {"tokens_raw": 1000, "tokens_compressed": 1000,
         "usage_prompt_tokens": 1000, "accuracy": True},
        {"tokens_raw": 1000, "tokens_compressed": 1000,
         "usage_prompt_tokens": 1000, "accuracy": True},
    ]
    on = [
        {"tokens_raw": 1000, "tokens_compressed": 600,
         "usage_prompt_tokens": 600, "accuracy": True},
        {"tokens_raw": 1000, "tokens_compressed": 700,
         "usage_prompt_tokens": 700, "accuracy": False},
    ]
    m = compute_metrics(off, on, token_field="usage_prompt_tokens",
                        input_price_per_token=1e-6)
    # (2000 - 1300) / 2000 = 0.35
    assert m["delta_tokens"] == pytest.approx(0.35)
    # accuracy 1.0 -> 0.5
    assert m["delta_accuracy"] == pytest.approx(-0.5)


@pytest.mark.unit
def test_compute_metrics_default_field_unchanged_for_cr():
    off = [{"tokens_raw": 100, "tokens_compressed": 80, "accuracy": True}]
    on = [{"tokens_raw": 100, "tokens_compressed": 40, "accuracy": True}]
    m = compute_metrics(off, on)  # default token_field=tokens_compressed
    assert m["delta_tokens"] == pytest.approx(0.5)


@pytest.mark.unit
def test_compute_metrics_empty_pool_is_zeroed_not_crash():
    """An empty aligned pool (e.g. a transcript-only run, transcript excluded)
    yields zeroed metrics instead of a ZeroDivisionError that discards the run."""
    m = compute_metrics([], [], token_field="usage_prompt_tokens")
    assert m["delta_tokens"] == 0.0
    assert m["tokens_a"] == 0 and m["tokens_b"] == 0
    assert m["acc_closed_book"] is None


# ── run_leg usage-probe path (no network) ───────────────────────────────────


@pytest.mark.unit
def test_run_leg_usage_probe_records_prompt_tokens():
    item = {"messages": [{"role": "user", "content": "hi"}],
            "workload": "transcript", "item_id": "t0"}
    recs = run_leg(
        "ON", [item],
        compress=lambda m: {"messages": m},
        usage_probe=lambda m: 123,
    )
    assert recs[0]["usage_prompt_tokens"] == 123
    assert recs[0]["accuracy"] is None  # savings-only, no gold


@pytest.mark.unit
def test_run_leg_injected_compress_and_eval():
    item = {"messages": [{"role": "user", "content": "doc " * 50}],
            "workload": "lb", "item_id": "i1",
            "question": "q", "choice_A": "a", "choice_B": "b",
            "choice_C": "c", "choice_D": "d", "gold": "B"}
    recs = run_leg(
        "ON", [item],
        compress=lambda m: {"messages": [{"role": "user", "content": "shrunk"}]},
        lb_cfg={"provider": "x"},
        eval_fn=lambda compressed, it, cfg, *, closed_book=False: ("B", 42),
    )
    assert recs[0]["accuracy"] is True
    assert recs[0]["usage_prompt_tokens"] == 42
    assert recs[0]["eval_input_tokens"] == 42


# ── report renderer ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_format_insforge_report_headline_and_dollars():
    off = [{"item_id": "i1", "workload": "lb_if", "tokens_raw": 1000,
            "usage_prompt_tokens": 1000, "tokens_compressed": 1000, "accuracy": True}]
    on = [{"item_id": "i1", "workload": "lb_if", "tokens_raw": 1000,
           "usage_prompt_tokens": 600, "tokens_compressed": 600, "accuracy": True}]
    mu = compute_metrics(off, on, token_field="usage_prompt_tokens", input_price_per_token=1e-6)
    mt = compute_metrics(off, on, token_field="tokens_compressed")
    text = bi.format_insforge_report(
        off, on, [], model="anthropic/claude-haiku-4.5",
        gate=GateResult(passed=True), metrics_usage=mu, metrics_tiktoken=mt,
        verbatim_metrics=None, direct=True, dry_run=False,
        savings_threshold=0.20, accuracy_drop=0.02,
    )
    assert "usage.prompt_tokens" in text
    assert "Dollars saved" in text
    assert "$0.0004" in text          # 400 tokens * $1/Mtok
    assert "40.0%" in text            # savings


@pytest.mark.unit
def test_report_headline_excludes_transcript():
    """Transcript still appears (informational) but the headline + gate savings
    are LongBench-only — never the blended figure (#4/#8). A huge, highly
    compressible transcript must not drag the headline toward its own ratio."""
    off = [
        {"item_id": "lb1", "workload": "lb_if", "tokens_raw": 1000,
         "usage_prompt_tokens": 1000, "tokens_compressed": 1000, "accuracy": True},
        {"item_id": "t0", "workload": "transcript", "tokens_raw": 50_000,
         "usage_prompt_tokens": 50_000, "tokens_compressed": 50_000},
    ]
    on = [
        {"item_id": "lb1", "workload": "lb_if", "tokens_raw": 1000,
         "usage_prompt_tokens": 600, "tokens_compressed": 600, "accuracy": True},
        {"item_id": "t0", "workload": "transcript", "tokens_raw": 50_000,
         "usage_prompt_tokens": 10_000, "tokens_compressed": 10_000},
    ]
    # Gate pool = LongBench only (what main()'s _pool now produces).
    lb_off = [r for r in off if r["workload"] != "transcript"]
    lb_on = [r for r in on if r["workload"] != "transcript"]
    mu = compute_metrics(lb_off, lb_on, token_field="usage_prompt_tokens",
                         input_price_per_token=1e-6)
    text = bi.format_insforge_report(
        off, on, [], model="anthropic/claude-haiku-4.5",
        gate=GateResult(passed=True), metrics_usage=mu, metrics_tiktoken=mu,
        verbatim_metrics=None, direct=True, dry_run=False,
        savings_threshold=0.20, accuracy_drop=0.02,
    )
    assert "LongBench (gated)" in text
    assert "excluded from gate" in text
    assert "40.0%" in text   # LongBench-only headline (1000 -> 600)
    assert "80.0%" in text   # transcript's own ratio, informational row only
    # the blended figure (≈79.2%) must never be the headline
    assert "79.2%" not in text


# ── InsForge gateway client + bootstrap (mocked HTTP) ───────────────────────


class _FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.mark.unit
def test_insforge_chat_call_reads_metadata_usage(monkeypatch):
    import httpx

    from benchmarks.common.runner import _insforge_chat_call

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["auth"] = headers.get("Authorization")
        return _FakeResp(200, {
            "text": "The correct answer is (B)",
            "metadata": {"model": "x", "usage": {"promptTokens": 1234}},
        })

    monkeypatch.setattr(httpx, "post", fake_post)
    text, in_tok = _insforge_chat_call(
        "prompt", "anthropic/claude-haiku-4.5", 64,
        base_url="http://localhost:7130", api_key="JWT123",
    )
    assert text == "The correct answer is (B)"
    assert in_tok == 1234
    assert captured["url"].endswith("/api/ai/chat/completion")
    assert captured["body"]["maxTokens"] == 64  # camelCase, not max_tokens
    assert captured["auth"] == "Bearer JWT123"


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        {"text": "ok", "metadata": {}},                          # usage absent
        {"text": "ok", "metadata": {"usage": {}}},               # promptTokens absent
        {"text": "ok", "metadata": {"usage": {"promptTokens": 0}}},  # 0 (impossible)
    ],
)
def test_insforge_chat_call_raises_on_unusable_usage(monkeypatch, payload):
    """A missing/0 prompt-token count must fail loudly, not be coerced to 0 —
    coercion silently inflated the OFF/ON savings headline (#3)."""
    import httpx

    from benchmarks.common.runner import _insforge_chat_call

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(200, payload))
    with pytest.raises(RuntimeError, match="promptTokens"):
        _insforge_chat_call("p", "m", 64, base_url="http://x", api_key="k")


@pytest.mark.unit
def test_bootstrap_register_returns_jwt(monkeypatch):
    import benchmarks.insforge.bench_insforge as b

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/api/auth/users"):
            return _FakeResp(201, {"accessToken": "tok-register"})
        raise AssertionError("should not reach login when register succeeds")

    monkeypatch.setattr(b.httpx, "post", fake_post)
    tok = b.bootstrap_gateway("http://localhost:7130",
                              {"USER_EMAIL": "a@b.c", "USER_PASSWORD": "pw"})
    assert tok == "tok-register"


@pytest.mark.unit
def test_bootstrap_falls_back_to_login(monkeypatch):
    import benchmarks.insforge.bench_insforge as b

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/api/auth/users"):
            return _FakeResp(409, {}, text="user exists")  # register fails
        if url.endswith("/api/auth/sessions"):
            return _FakeResp(200, {"accessToken": "tok-login"})
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(b.httpx, "post", fake_post)
    tok = b.bootstrap_gateway("http://localhost:7130",
                              {"USER_EMAIL": "a@b.c", "USER_PASSWORD": "pw"})
    assert tok == "tok-login"


@pytest.mark.unit
def test_format_insforge_report_marks_dry_run():
    off = [{"item_id": "i1", "workload": "t", "tokens_raw": 100, "tokens_compressed": 100}]
    on = [{"item_id": "i1", "workload": "t", "tokens_raw": 100, "tokens_compressed": 60}]
    mt = compute_metrics(off, on, token_field="tokens_compressed")
    text = bi.format_insforge_report(
        off, on, [], model="anthropic/claude-haiku-4.5",
        gate=GateResult(passed=False, fail_reason="x"), metrics_usage=mt,
        metrics_tiktoken=mt, verbatim_metrics=None, direct=True, dry_run=True,
        savings_threshold=0.20, accuracy_drop=0.02,
    )
    assert "DRY RUN" in text


# ── significance gate + offline rescore ──────────────────────────────────────


def _paired_lb_records(pairs: list[tuple[bool, bool]]) -> tuple[list, list]:
    """Build aligned OFF/ON LongBench record lists from (off, on) correctness."""
    off, on = [], []
    for i, (a, b) in enumerate(pairs):
        iid = f"i{i}"
        off.append({"item_id": iid, "workload": "lb_if", "leg": "OFF",
                    "tokens_raw": 1000, "usage_prompt_tokens": 1000,
                    "tokens_compressed": 1000, "accuracy": a})
        on.append({"item_id": iid, "workload": "lb_if", "leg": "ON",
                   "tokens_raw": 1000, "usage_prompt_tokens": 700,
                   "tokens_compressed": 700, "accuracy": b,
                   "lx_route": "lingua", "lx_verbatim_tokens": 0,
                   "lx_compressed_in_tokens": 1000, "lx_compressed_out_tokens": 700})
    return off, on


@pytest.mark.unit
def test_report_shows_mcnemar_ci_and_within_noise():
    # Small within-noise drop: 3 regressions vs 2 improvements ⇒ not significant.
    pairs = [(True, True)] * 20 + [(False, False)] * 20 \
        + [(True, False)] * 3 + [(False, True)] * 2
    off, on = _paired_lb_records(pairs)
    mu = compute_metrics(off, on, token_field="usage_prompt_tokens",
                         input_price_per_token=1e-6)
    text = bi.format_insforge_report(
        off, on, [], model="anthropic/claude-haiku-4.5",
        gate=GateResult(passed=True), metrics_usage=mu, metrics_tiktoken=mu,
        verbatim_metrics=None, direct=True, dry_run=False,
        savings_threshold=0.20, accuracy_drop=0.02,
    )
    assert "McNemar" in text
    assert "95% CI" in text
    assert "within run-to-run noise" in text


@pytest.mark.unit
def test_rescore_from_jsonl_round_trips_and_gates_go(tmp_path):
    # A within-noise run written to JSONL re-renders to GO with no provider call.
    pairs = [(True, True)] * 30 + [(False, False)] * 30 \
        + [(True, False)] * 4 + [(False, True)] * 3
    off, on = _paired_lb_records(pairs)
    jsonl = tmp_path / "results.jsonl"
    with jsonl.open("w") as f:
        for r in off + on:
            f.write(json.dumps(r) + "\n")

    report, gate = bi.rescore_from_jsonl(
        jsonl, model="anthropic/claude-haiku-4.5",
        savings_threshold=0.20, accuracy_drop=0.02,
    )
    assert gate.passed is True               # CI upper bound above −2%
    assert "McNemar" in report
    assert "30.0%" in report                 # savings rendered (1000 → 700)
    # §2b reconstructed from the saved lx_* fields (no raw messages needed).
    assert "Excluding Verbatim" in report


@pytest.mark.unit
def test_report_reframes_savings_shortfall_as_target_not_nogo():
    # Accuracy fine, savings under the 20% target: the top line must read as the
    # honest result, not a flat NO-GO, and savings is labeled an internal target.
    pairs = [(True, True)] * 30 + [(False, False)] * 30 \
        + [(True, False)] * 3 + [(False, True)] * 2
    off, on = _paired_lb_records(pairs)
    # Shave savings to ~12% (under the 20% target) via the usage tokens.
    for r in on:
        r["usage_prompt_tokens"] = 880
        r["tokens_compressed"] = 880
    mu = compute_metrics(off, on, token_field="usage_prompt_tokens",
                         input_price_per_token=1e-6)
    gate = bi.apply_gate(mu, savings_threshold=0.20, accuracy_drop=0.02,
                         savings_is_target=True)
    text = bi.format_insforge_report(
        off, on, [], model="anthropic/claude-haiku-4.5",
        gate=gate, metrics_usage=mu, metrics_tiktoken=mu,
        verbatim_metrics=None, direct=True, dry_run=False,
        savings_threshold=0.20, accuracy_drop=0.02,
    )
    assert gate.passed is True and gate.savings_met is False
    assert "NO-GO" not in text
    assert "fewer prompt tokens" in text
    assert "internal target" in text
    assert "under target" in text
