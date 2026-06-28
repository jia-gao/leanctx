"""Behavioral unit tests for the Phase 1 harness internals the reviewer
flagged as untested: answer extraction, token summing, the gold-matching /
scoring pipeline (via run_leg with the network mocked out), latency
percentiles, the closed-book control, and the LongBench sampler.

All marked unit — no real network, no real model, no real dataset.
"""
from __future__ import annotations

import sys
import types

import pytest

import benchmarks.clawrouter.bench_phase1 as harness
from benchmarks.clawrouter.bench_phase1 import (
    GateResult,
    _build_eval_context,
    _build_lb_prompt,
    _extract_lb_answer,
    _load_lb_items,
    _percentile,
    _sum_tokens,
    _verbatim_split_item,
    aggregate_verbatim_metrics,
    call_eval_llm,
    compute_metrics,
    format_report,
    run_leg,
    verbatim_split_by_item,
    write_by_item_dumps,
)

# ── _extract_lb_answer ──────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected",
    [
        ("The correct answer is (B).", "B"),
        ("blah blah The correct answer is (D) because…", "D"),
        ("The correct answer is C", "C"),  # bare, no parens
        ("The correct answer is **(A)**", "A"),  # markdown bold stripped
        ("I think it is probably A.", None),  # no template phrase
        ("The correct answer is (E)", None),  # out of A–D range
        ("", None),
    ],
)
def test_extract_lb_answer(text, expected):
    assert _extract_lb_answer(text) == expected


@pytest.mark.unit
def test_closed_book_prompt_forces_a_parseable_letter():
    """The forced-choice instruction lives in the shared template, and a
    response in the exact format it dictates parses to a single letter — even
    in the closed-book leg where there is no document. Locks the
    template↔parser contract that keeps the control off the 0% abstain floor.

    (The e2e counterpart, ``test_e2e_closed_book_responses_parse_to_letter``,
    asserts a *real* closed-book judge call lands a letter.)
    """
    item = {
        "question": "Q?", "choice_A": "a", "choice_B": "b",
        "choice_C": "c", "choice_D": "d",
    }
    prompt = _build_lb_prompt(item, "[no document provided]")
    # The forced-choice instruction must be present in the prompt both legs see.
    assert "commit to exactly one option" in prompt
    assert "never reply that you cannot answer" in prompt
    # A response in the dictated format parses to exactly one A–D letter.
    assert _extract_lb_answer("The correct answer is (C).") == "C"


# ── _sum_tokens ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_sum_tokens_counts_string_content():
    msgs = [{"role": "user", "content": "hello world"}]
    assert _sum_tokens(msgs) > 0


@pytest.mark.unit
def test_sum_tokens_ignores_non_string_content():
    # structured blocks (list content) contribute nothing — only plain strings
    only_blocks = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    assert _sum_tokens(only_blocks) == 0


@pytest.mark.unit
def test_sum_tokens_scales_with_length():
    short = _sum_tokens([{"role": "user", "content": "word"}])
    long = _sum_tokens([{"role": "user", "content": "word " * 200}])
    assert long > short


# ── run_leg: gold matching & scoring pipeline ───────────────────────────────


@pytest.fixture
def lb_item():
    return {
        "messages": [{"role": "user", "content": "a long document " * 100}],
        "workload": "lb_s1",
        "item_id": "lb_0001",
        "question": "What is the answer?",
        "choice_A": "alpha",
        "choice_B": "beta",
        "choice_C": "gamma",
        "choice_D": "delta",
        "gold": "B",
        "lb_domain": "single",
        "lb_difficulty": "hard",
        "lb_length": "long",
    }


@pytest.mark.unit
def test_run_leg_scores_correct_answer(monkeypatch, lb_item):
    monkeypatch.setattr(
        harness, "compress_via_shim",
        lambda messages, url: {"messages": messages, "stats": {}},
    )
    # model returns the gold letter → accuracy True
    monkeypatch.setattr(
        harness, "call_eval_llm",
        lambda compressed, item, cfg, *, closed_book=False: ("B", 1234),
    )
    [rec] = run_leg("B", [lb_item], "http://x", lb_cfg={"provider": "anthropic"})
    assert rec["accuracy"] is True
    assert rec["lb_gold"] == "B"
    assert rec["lb_pred"] == "B"
    assert rec["eval_input_tokens"] == 1234
    assert rec["compress_ms"] >= 0
    assert "eval_ms" in rec


@pytest.mark.unit
def test_run_leg_scores_wrong_answer(monkeypatch, lb_item):
    monkeypatch.setattr(
        harness, "compress_via_shim",
        lambda messages, url: {"messages": messages, "stats": {}},
    )
    monkeypatch.setattr(
        harness, "call_eval_llm",
        lambda compressed, item, cfg, *, closed_book=False: ("C", 10),
    )
    [rec] = run_leg("B", [lb_item], "http://x", lb_cfg={"provider": "anthropic"})
    assert rec["accuracy"] is False
    assert rec["lb_pred"] == "C"


@pytest.mark.unit
def test_run_leg_unparseable_answer_is_incorrect(monkeypatch, lb_item):
    monkeypatch.setattr(
        harness, "compress_via_shim",
        lambda messages, url: {"messages": messages, "stats": {}},
    )
    monkeypatch.setattr(
        harness, "call_eval_llm",
        lambda compressed, item, cfg, *, closed_book=False: (None, 10),
    )
    [rec] = run_leg("B", [lb_item], "http://x", lb_cfg={"provider": "anthropic"})
    assert rec["accuracy"] is False  # None != gold
    assert rec["lb_pred"] is None


@pytest.mark.unit
def test_run_leg_closed_book_skips_shim(monkeypatch, lb_item):
    """Closed-book leg must not touch the shim (it runs after shims die)."""
    def boom(*a, **k):  # pragma: no cover - asserts it's never called
        raise AssertionError("compress_via_shim must not be called closed-book")

    monkeypatch.setattr(harness, "compress_via_shim", boom)

    seen = {}

    def fake_eval(compressed, item, cfg, *, closed_book=False):
        seen["closed_book"] = closed_book
        return ("B", 5)

    monkeypatch.setattr(harness, "call_eval_llm", fake_eval)
    [rec] = run_leg(
        "C", [lb_item], shim_url="", lb_cfg={"provider": "anthropic"},
        closed_book=True,
    )
    assert seen["closed_book"] is True
    assert rec["closed_book"] is True
    assert rec["compress_ms"] == 0


# ── _build_eval_context ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_build_eval_context_joins_text_roles():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
        {"role": "tool", "content": "ignored-role"},
        {"role": "user", "content": [{"type": "text", "text": "blocks-ignored"}]},
    ]
    ctx = _build_eval_context(msgs)
    assert "hello" in ctx and "world" in ctx
    assert "ignored-role" not in ctx  # role not in {user,system,assistant}
    assert "blocks-ignored" not in ctx  # non-string content skipped


@pytest.mark.unit
def test_build_eval_context_closed_book_is_placeholder():
    msgs = [{"role": "user", "content": "should not appear"}]
    assert _build_eval_context(msgs, closed_book=True) == "[no document provided]"
    assert "should not appear" not in _build_eval_context(msgs, closed_book=True)


# ── _build_lb_prompt: closed-book forced choice ─────────────────────────────


_LB_ITEM = {
    "question": "Q?",
    "choice_A": "a",
    "choice_B": "b",
    "choice_C": "c",
    "choice_D": "d",
}


@pytest.mark.unit
def test_build_lb_prompt_forces_a_letter_in_both_legs():
    # The forced-choice instruction lives in the shared template, so a hedge is
    # never scored below the 25% random floor (which would inflate context lift)
    # regardless of leg — and the prompt is byte-identical apart from $DOC$.
    open_book = _build_lb_prompt(_LB_ITEM, "the document")
    closed_book = _build_lb_prompt(_LB_ITEM, "[no document provided]")
    for prompt in (open_book, closed_book):
        assert "must commit to exactly one option" in prompt
        assert "never reply that you cannot answer" in prompt
    # Identical scaffolding: the only difference is the $DOC$ slot.
    assert open_book.replace("the document", "[no document provided]") == closed_book


# ── call_eval_llm: low eval temperature ─────────────────────────────────────


def _fake_anthropic(captured: dict, *, text: str = "The correct answer is (B)"):
    """A stand-in ``anthropic`` module whose client records create() kwargs."""
    class _Resp:
        content = [types.SimpleNamespace(text=text)]
        usage = types.SimpleNamespace(input_tokens=42)

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Client:
        def __init__(self):
            self.messages = _Messages()

    mod = types.ModuleType("anthropic")
    mod.Anthropic = _Client
    return mod


def _fake_openai(captured: dict, *, text: str = "The correct answer is (B)"):
    class _Resp:
        choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
        usage = types.SimpleNamespace(prompt_tokens=42)

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    mod = types.ModuleType("openai")
    mod.OpenAI = _Client
    return mod


@pytest.fixture
def eval_call_item():
    return {
        "question": "q?", "choice_A": "a", "choice_B": "b",
        "choice_C": "c", "choice_D": "d", "gold": "B",
    }


@pytest.mark.unit
def test_call_eval_llm_anthropic_uses_low_temperature(monkeypatch, eval_call_item):
    captured: dict = {}
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(captured))
    answer, in_tok = call_eval_llm(
        [{"role": "user", "content": "doc"}],
        eval_call_item,
        {"provider": "anthropic", "model": "m", "max_tokens": 16},
    )
    assert captured["temperature"] == 0.1
    assert answer == "B"
    assert in_tok == 42


@pytest.mark.unit
def test_call_eval_llm_openai_uses_low_temperature(monkeypatch, eval_call_item):
    captured: dict = {}
    monkeypatch.setitem(sys.modules, "openai", _fake_openai(captured))
    answer, in_tok = call_eval_llm(
        [{"role": "user", "content": "doc"}],
        eval_call_item,
        {"provider": "openai", "model": "m", "max_tokens": 16},
    )
    assert captured["temperature"] == 0.1
    assert answer == "B"
    assert in_tok == 42


# ── run_leg: shared eval draw (de-biasing) ──────────────────────────────────


@pytest.mark.unit
def test_run_leg_reuses_eval_for_identical_context(monkeypatch, lb_item):
    """Leg B reuses Leg A's answer when the eval context is byte-identical —
    no second judge call, and the reused flag is recorded."""
    monkeypatch.setattr(
        harness, "compress_via_shim",
        lambda messages, url: {"messages": messages, "stats": {}},
    )
    calls = {"n": 0}

    def counting_eval(compressed, item, cfg, *, closed_book=False):
        calls["n"] += 1
        return ("B", 7)

    monkeypatch.setattr(harness, "call_eval_llm", counting_eval)

    eval_a: dict = {}
    [rec_a] = run_leg(
        "A", [lb_item], "http://x", lb_cfg={"provider": "anthropic"}, eval_sink=eval_a
    )
    assert calls["n"] == 1
    assert rec_a["eval_reused"] is False
    assert eval_a[lb_item["item_id"]]["answer"] == "B"

    # Leg B sees the same compressed messages → identical context → reuse.
    [rec_b] = run_leg(
        "B", [lb_item], "http://x", lb_cfg={"provider": "anthropic"}, reuse_eval=eval_a
    )
    assert calls["n"] == 1, "judge must not be called again for identical input"
    assert rec_b["eval_reused"] is True
    assert rec_b["accuracy"] is True
    assert rec_b["lb_pred"] == "B"
    assert rec_b["eval_input_tokens"] == 7
    assert rec_b["eval_ms"] == 0


@pytest.mark.unit
def test_run_leg_does_not_reuse_when_context_differs(monkeypatch, lb_item):
    """Genuinely-compressed items (different context) get a fresh draw."""
    monkeypatch.setattr(
        harness, "compress_via_shim",
        lambda messages, url: {"messages": messages, "stats": {}},
    )
    calls = {"n": 0}

    def counting_eval(compressed, item, cfg, *, closed_book=False):
        calls["n"] += 1
        return ("B", 7)

    monkeypatch.setattr(harness, "call_eval_llm", counting_eval)

    eval_a: dict = {}
    run_leg("A", [lb_item], "http://x", lb_cfg={"provider": "anthropic"}, eval_sink=eval_a)
    assert calls["n"] == 1

    # Same item_id but a different (compressed) document → context differs.
    item_b = {**lb_item, "messages": [{"role": "user", "content": "a much shorter doc"}]}
    [rec_b] = run_leg(
        "B", [item_b], "http://x", lb_cfg={"provider": "anthropic"}, reuse_eval=eval_a
    )
    assert calls["n"] == 2, "differing context must trigger a fresh judge call"
    assert rec_b["eval_reused"] is False


# ── _percentile ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_percentile_empty_is_none():
    assert _percentile([], 0.5) is None


@pytest.mark.unit
def test_percentile_single_value():
    assert _percentile([42.0], 0.95) == 42.0


@pytest.mark.unit
def test_percentile_median_and_p95():
    data = [float(i) for i in range(1, 101)]  # 1..100
    assert _percentile(data, 0.50) == pytest.approx(50.0, abs=1.0)
    assert _percentile(data, 0.95) == pytest.approx(95.0, abs=1.0)


# ── compute_metrics: latency isolation + closed-book ────────────────────────


@pytest.mark.unit
def test_compute_metrics_sidecar_latency_from_compress_ms():
    """Sidecar latency comes from Leg B compress_ms (NOT end-to-end duration_ms);
    eval-LLM latency comes from Leg A (Leg B reuses identical-input draws at
    eval_ms=0 under the shared draw, which would read as a ~0ms 'free' judge)."""
    records_a = [
        {"tokens_raw": 100, "tokens_compressed": 80, "accuracy": True, "eval_ms": 9000},
        {"tokens_raw": 100, "tokens_compressed": 80, "accuracy": True, "eval_ms": 12000},
    ]
    records_b = [
        {"tokens_raw": 100, "tokens_compressed": 50, "accuracy": True,
         "compress_ms": 300, "eval_ms": 0, "duration_ms": 9300},     # reused → 0ms
        {"tokens_raw": 100, "tokens_compressed": 50, "accuracy": True,
         "compress_ms": 400, "eval_ms": 12000, "duration_ms": 12400},
    ]
    metrics = compute_metrics(records_a, records_b)
    # p50 is in the 300–400 ms range, nowhere near the 9–12 s end-to-end times
    assert metrics["sidecar_p50_ms"] in (300, 400)
    assert metrics["sidecar_p50_ms"] < 1000
    # eval p50 reflects the real Leg A judge cost (9–12 s), not Leg B's 0ms reuse
    assert metrics["eval_p50_ms"] >= 9000


@pytest.mark.unit
def test_compute_metrics_closed_book_accuracy_and_none():
    records_a = [{"tokens_raw": 100, "tokens_compressed": 80, "accuracy": True}]
    records_b = [{"tokens_raw": 100, "tokens_compressed": 50, "accuracy": True}]
    # no closed-book records → None
    assert compute_metrics(records_a, records_b).get("acc_closed_book") is None
    # with closed-book records → averaged
    records_cb = [
        {"accuracy": True}, {"accuracy": False}, {"accuracy": False}, {"accuracy": False},
    ]
    m = compute_metrics(records_a, records_b, records_cb)
    assert m["acc_closed_book"] == pytest.approx(0.25)


# ── _load_lb_items: random, reproducible, oversampled ───────────────────────


def _fake_dataset(monkeypatch):
    """Patch datasets.load_dataset to return a synthetic LongBench-ish pool.

    The real ``datasets`` package is never exercised — it is fully mocked — and
    it is not a ``[dev]`` dependency, so CI installs without it. When it is
    absent we inject a lightweight stub module so ``_load_lb_items``'s
    ``from datasets import load_dataset`` still resolves.
    """
    import sys
    import types

    items = []
    for length in ("short", "medium", "long"):
        for diff in ("easy", "hard"):
            for i in range(20):
                items.append({
                    "_id": f"{length}_{diff}_{i}",
                    "context": f"ctx {length} {diff} {i} " * 5,
                    "question": "q?",
                    "choice_A": "a", "choice_B": "b",
                    "choice_C": "c", "choice_D": "d",
                    "answer": "A",
                    "domain": "d", "difficulty": diff, "length": length,
                })

    try:
        import datasets
    except ModuleNotFoundError:
        datasets = types.ModuleType("datasets")
        monkeypatch.setitem(sys.modules, "datasets", datasets)

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: items, raising=False)


@pytest.mark.unit
def test_load_lb_items_is_reproducible(monkeypatch):
    _fake_dataset(monkeypatch)
    a = _load_lb_items(limit=12, seed=7)
    _fake_dataset(monkeypatch)
    b = _load_lb_items(limit=12, seed=7)
    assert [x["item_id"] for x in a] == [x["item_id"] for x in b]
    assert len(a) == 12


@pytest.mark.unit
def test_load_lb_items_different_seed_differs(monkeypatch):
    _fake_dataset(monkeypatch)
    a = _load_lb_items(limit=12, seed=1)
    _fake_dataset(monkeypatch)
    b = _load_lb_items(limit=12, seed=2)
    assert [x["item_id"] for x in a] != [x["item_id"] for x in b]


@pytest.mark.unit
def test_load_lb_items_is_not_first_n(monkeypatch):
    """Regression: sampling must not just take the first-N items per cell."""
    _fake_dataset(monkeypatch)
    sample = _load_lb_items(limit=18, seed=99)
    # the deterministic first-N bug would pick only "_0", "_1", … suffixes
    suffixes = {x["item_id"].rsplit("_", 1)[1] for x in sample}
    assert suffixes != {str(i) for i in range(len(sample))}
    assert any(int(s) >= 5 for s in suffixes)  # reaches deeper into the pool


@pytest.mark.unit
def test_load_lb_items_oversamples_long(monkeypatch):
    _fake_dataset(monkeypatch)
    sample = _load_lb_items(limit=30, seed=3, oversample_long=True)
    lengths = [x["lb_length"] for x in sample]
    n_long = lengths.count("long")
    n_short = lengths.count("short")
    # long is double-weighted, so it should out-represent short
    assert n_long > n_short


# ── Verbatim-excluded compression metrics (Phase B) ─────────────────────────

_PROSE = "the quick brown fox jumps over the lazy dog " * 40   # plain prose
_CODE = "```python\nimport os\nimport sys\ndef f():\n    return os\n```"


def _msg(content: str, role: str = "user") -> dict:
    return {"role": role, "content": content}


@pytest.mark.unit
def test_verbatim_split_item_prose_is_compressed():
    a = [_msg(_PROSE)]
    b = [_msg("fox dog")]  # pretend-compressed (fewer tokens)
    v, c_in, c_out, route = _verbatim_split_item(a, b, threshold=10)
    assert v == 0
    assert c_in == _sum_tokens(a)
    assert c_out == _sum_tokens(b)
    assert c_out < c_in
    assert route == "lingua"


@pytest.mark.unit
def test_verbatim_split_item_code_is_verbatim():
    a = [_msg(_CODE)]
    b = [_msg(_CODE)]  # verbatim → unchanged
    v, c_in, c_out, route = _verbatim_split_item(a, b, threshold=10)
    assert c_in == 0 and c_out == 0
    assert v == _sum_tokens(a)
    assert route == "verbatim"


# A JSON blob long/dense enough to classify as ContentType.STRUCTURED.
_STRUCTURED = '{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}' * 20


@pytest.mark.unit
def test_verbatim_split_item_structured_is_compressed():
    """Regression guard: structured data is routed to the (gentler) lingua pass
    by the middleware, so the split must book it as *compressed* — not verbatim.
    If a new compressed ContentType is omitted here, the route label drifts from
    the sidecar and the route/reuse invariant trips at run time (exit 3)."""
    from leanctx.classifier import classify
    from leanctx.compressors import ContentType

    assert classify(_msg(_STRUCTURED)) is ContentType.STRUCTURED  # fixture sanity
    a = [_msg(_STRUCTURED)]
    b = [_msg('{"a": 1, "b": 2}')]  # pretend-gently-compressed (fewer tokens)
    v, c_in, c_out, route = _verbatim_split_item(a, b, threshold=10)
    assert v == 0
    assert c_in == _sum_tokens(a) and c_out == _sum_tokens(b)
    assert route == "lingua"


@pytest.mark.unit
def test_verbatim_split_item_below_threshold_all_verbatim():
    """A request under the token gate is passed through → everything verbatim."""
    a = [_msg(_PROSE)]
    b = [_msg("fox dog")]
    v, c_in, c_out, route = _verbatim_split_item(a, b, threshold=10_000_000)
    assert c_in == 0 and c_out == 0
    assert v == _sum_tokens(a)
    assert route == "verbatim"


@pytest.mark.unit
def test_verbatim_split_item_ineligible_role_is_verbatim():
    """tool-role / non-string content is never eligible → verbatim."""
    a = [_msg(_PROSE, role="tool")]
    b = [_msg(_PROSE, role="tool")]
    v, c_in, c_out, route = _verbatim_split_item(a, b, threshold=10)
    assert c_in == 0 and v == _sum_tokens(a)
    assert route == "verbatim"


@pytest.mark.unit
def test_verbatim_split_item_hybrid():
    a = [_msg(_PROSE), _msg(_CODE)]
    b = [_msg("fox dog"), _msg(_CODE)]
    v, c_in, c_out, route = _verbatim_split_item(a, b, threshold=10)
    assert c_in == _sum_tokens([a[0]])
    assert v == _sum_tokens([a[1]])
    assert route == "hybrid"


@pytest.mark.unit
def test_verbatim_split_item_misaligned_returns_none():
    """Different message counts (CR non-determinism) → caller-skippable None."""
    assert _verbatim_split_item([_msg("a")], [_msg("a"), _msg("b")], threshold=10) is None


@pytest.mark.unit
def test_verbatim_split_by_item_pairs_only_shared_ids():
    a = {"x": [_msg(_PROSE)], "only_a": [_msg(_PROSE)]}
    b = {"x": [_msg("fox dog")], "only_b": [_msg("y")]}
    out = verbatim_split_by_item(a, b, threshold=10)
    assert set(out) == {"x"}
    assert out["x"]["lx_route"] == "lingua"
    assert out["x"]["lx_compressed_in_tokens"] > 0


@pytest.mark.unit
def test_aggregate_verbatim_metrics_decomposition_identity():
    """overall_savings == (1 - verbatim_share) * nonverbatim_savings (exact)."""
    per_item = {
        "a": {
            "lx_verbatim_tokens": 100,
            "lx_compressed_in_tokens": 100,
            "lx_compressed_out_tokens": 40,
            "lx_route": "hybrid",
        }
    }
    m = aggregate_verbatim_metrics(per_item)
    assert m is not None
    assert m["verbatim_share"] == pytest.approx(0.5)
    assert m["nonverbatim_ratio"] == pytest.approx(0.4)
    assert m["nonverbatim_savings"] == pytest.approx(0.6)
    assert m["overall_savings"] == pytest.approx(0.3)
    assert m["token_delta"] == 60
    # the headline identity that motivates the whole section
    assert m["overall_savings"] == pytest.approx(
        (1 - m["verbatim_share"]) * m["nonverbatim_savings"]
    )


@pytest.mark.unit
def test_aggregate_verbatim_metrics_all_verbatim_no_zero_division():
    per_item = {
        "a": {
            "lx_verbatim_tokens": 100,
            "lx_compressed_in_tokens": 0,
            "lx_compressed_out_tokens": 0,
            "lx_route": "verbatim",
        }
    }
    m = aggregate_verbatim_metrics(per_item)
    assert m is not None
    assert m["nonverbatim_ratio"] is None
    assert m["nonverbatim_savings"] is None
    assert m["verbatim_share"] == pytest.approx(1.0)
    assert m["overall_savings"] == pytest.approx(0.0)
    assert m["token_delta"] == 0


@pytest.mark.unit
def test_aggregate_verbatim_metrics_empty_is_none():
    assert aggregate_verbatim_metrics({}) is None


# ── format_report: verbatim section rendering ────────────────────────────────


def _minimal_records():
    records_a = [{"tokens_raw": 100, "tokens_compressed": 90, "accuracy": True}]
    records_b = [{"tokens_raw": 100, "tokens_compressed": 50, "accuracy": True}]
    return records_a, records_b


@pytest.mark.unit
def test_format_report_includes_verbatim_section():
    records_a, records_b = _minimal_records()
    metrics = compute_metrics(records_a, records_b)
    vm = aggregate_verbatim_metrics({
        "a": {
            "lx_verbatim_tokens": 100,
            "lx_compressed_in_tokens": 100,
            "lx_compressed_out_tokens": 40,
            "lx_route": "hybrid",
        }
    })
    text = format_report(
        metrics, GateResult(passed=True),
        records_a=records_a, records_b=records_b,
        verbatim_metrics=vm,
    )
    assert "2b. Compression Excluding Verbatim Content" in text
    assert "Verbatim share" in text
    assert "Non-verbatim only" in text
    assert "Decomposition" in text


@pytest.mark.unit
def test_format_report_omits_verbatim_section_when_absent():
    records_a, records_b = _minimal_records()
    metrics = compute_metrics(records_a, records_b)
    text = format_report(
        metrics, GateResult(passed=True),
        records_a=records_a, records_b=records_b,
        verbatim_metrics=None,
    )
    assert "Compression Excluding Verbatim Content" not in text


# ── write_by_item_dumps ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_write_by_item_dumps_includes_all_fields_and_messages(tmp_path):
    import json

    records_a = [
        {"leg": "A", "item_id": "abc", "workload": "lb_s1",
         "tokens_compressed": 100, "accuracy": False},
        # agent warmup item has no id — must be skipped
        {"leg": "A", "item_id": None, "workload": "agent_s1",
         "tokens_compressed": 9},
    ]
    records_b = [
        {"leg": "B", "item_id": "abc", "workload": "lb_s1",
         "tokens_compressed": 60, "accuracy": True, "lx_route": "lingua"},
    ]
    records_cb = [{"leg": "C", "item_id": "abc", "accuracy": True}]
    msgs_a = {"abc": [{"role": "user", "content": "before layer 8"}]}
    msgs_b = {"abc": [{"role": "user", "content": "after"}]}

    out_dir = tmp_path / "by_item"
    written = write_by_item_dumps(
        out_dir, records_a, records_b, records_cb, msgs_a, msgs_b
    )

    # Only the item with an id is dumped.
    assert [p.name for p in written] == ["abc.json"]
    dump = json.loads(written[0].read_text())

    # All JSONL fields are preserved per leg.
    assert dump["leg_a"]["tokens_compressed"] == 100
    assert dump["leg_b"]["lx_route"] == "lingua"
    assert dump["leg_closed_book"]["accuracy"] is True
    # Full messages on either side of Layer 8.
    assert dump["input_before_layer8"][0]["content"] == "before layer 8"
    assert dump["output_after_layer8"][0]["content"] == "after"


@pytest.mark.unit
def test_write_by_item_dumps_sanitizes_filename(tmp_path):
    written = write_by_item_dumps(
        tmp_path, [], [], [],
        {"a/b:c": [{"role": "user", "content": "x"}]},
        {"a/b:c": [{"role": "user", "content": "y"}]},
    )
    assert written[0].name == "a_b_c.json"


@pytest.mark.unit
def test_write_by_item_dumps_handles_missing_leg(tmp_path):
    import json

    # Item present only in the message sinks (no flat records) still dumps.
    written = write_by_item_dumps(
        tmp_path, [], [], [],
        {"only": [{"role": "user", "content": "in"}]},
        {},
    )
    dump = json.loads(written[0].read_text())
    assert dump["leg_a"] is None
    assert dump["leg_b"] is None
    assert dump["output_after_layer8"] is None
    assert dump["input_before_layer8"][0]["content"] == "in"
