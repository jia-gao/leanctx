"""Markdown report rendering for the benchmark harness (extracted from
bench_phase1). Provider-agnostic; pricing/labels are passed in by the caller.
"""
from __future__ import annotations

from typing import Any

from benchmarks.common.scoring import GateResult
from benchmarks.common.verbatim import acc_by_route


def format_report(
    metrics: dict[str, Any],
    gate: GateResult,
    records_a: list[dict[str, Any]] | None = None,
    records_b: list[dict[str, Any]] | None = None,
    *,
    run_date: str = "",
    eval_model: str = "",
    cr_commit: str = "",
    savings_threshold: float = 0.20,
    accuracy_drop: float = 0.02,
    verbatim_metrics: dict[str, Any] | None = None,
) -> str:
    """Generate a rich Markdown Phase 1 report.

    Falls back to a compact 7-line summary when records are not supplied
    (keeps Sprint 4 unit tests green).
    """
    verdict = "PASS" if gate.passed else "NO-GO"

    # ── compact fallback (used by unit tests) ────────────────────────────
    if records_a is None or records_b is None:
        lines = [
            f"## Phase 1 Report — {verdict}",
            "",
            f"delta_tokens      : {metrics.get('delta_tokens', 'N/A'):.4f}"
            if isinstance(metrics.get("delta_tokens"), float)
            else f"delta_tokens      : {metrics.get('delta_tokens', 'N/A')}",
            f"delta_accuracy    : {metrics.get('delta_accuracy', 'N/A'):.4f}"
            if isinstance(metrics.get("delta_accuracy"), float)
            else f"delta_accuracy    : {metrics.get('delta_accuracy', 'N/A')}",
            f"e2e_ratio         : {metrics.get('e2e_ratio', 'N/A'):.4f}"
            if isinstance(metrics.get("e2e_ratio"), float)
            else f"e2e_ratio         : {metrics.get('e2e_ratio', 'N/A')}",
            f"cr_savings        : {metrics.get('cr_savings', 'N/A'):.4f}"
            if isinstance(metrics.get("cr_savings"), float)
            else f"cr_savings        : {metrics.get('cr_savings', 'N/A')}",
            f"cost_saved_per_1k : {metrics.get('cost_saved_per_1k', 'N/A'):.4f}"
            if isinstance(metrics.get("cost_saved_per_1k"), float)
            else f"cost_saved_per_1k : {metrics.get('cost_saved_per_1k', 'N/A')}",
            f"sidecar_p50_ms    : {metrics.get('sidecar_p50_ms', 'N/A')}",
        ]
        if not gate.passed:
            lines.append(f"\nFail reason: {gate.fail_reason}")
        return "\n".join(lines)

    # ── rich report ──────────────────────────────────────────────────────
    import datetime

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    date_str = run_date or now_utc.strftime("%Y-%m-%d %H:%M UTC")
    verdict_icon = "✅" if gate.passed else "❌"

    def pct(v: float | None) -> str:
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    def fmt_f(v: float | None, decimals: int = 4) -> str:
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    lb_a = [r for r in records_a if r.get("accuracy") is not None]
    lb_b = [r for r in records_b if r.get("accuracy") is not None]

    n_lb = len(lb_a)
    acc_a = metrics.get("acc_a", 0.0)
    acc_b = metrics.get("acc_b", 0.0)
    delta_acc = metrics.get("delta_accuracy", 0.0)

    tokens_raw_avg = sum(r["tokens_raw"] for r in records_a) / max(len(records_a), 1)
    tokens_a_avg = metrics.get("tokens_a", 0) / max(len(records_a), 1)
    tokens_b_avg = metrics.get("tokens_b", 0) / max(len(records_b), 1)

    # go/no-go row helpers
    def gate_row(label: str, threshold: str, actual: str, passed_cond: bool) -> str:
        return f"| {label} | {threshold} | {actual} | {'✅ PASS' if passed_cond else '❌ FAIL'} |"
    savings_ok = metrics.get("delta_tokens", 0.0) >= savings_threshold
    accuracy_ok = delta_acc >= -accuracy_drop

    # CR layer stats (average across all records_a)
    def _avg_stat(key: str) -> float:
        vals = [r.get("cr_stats", {}).get(key, 0) for r in records_a if r.get("cr_stats")]
        return sum(vals) / len(vals) if vals else 0.0

    cr_layers = [
        ("L1 Dedup",      "duplicatesRemoved",       _avg_stat("duplicatesRemoved")),
        ("L2 Whitespace", "whitespaceSavedChars",     _avg_stat("whitespaceSavedChars")),
        ("L3 Dictionary", "dictionarySubstitutions",  _avg_stat("dictionarySubstitutions")),
        ("L4 Paths",      "pathsShortened",           _avg_stat("pathsShortened")),
        ("L5 JSON",       "jsonCompactedChars",       _avg_stat("jsonCompactedChars")),
        ("L6 Observation","observationsCompressed",   _avg_stat("observationsCompressed")),
        ("L7 Codebook",   "dynamicSubstitutions",     _avg_stat("dynamicSubstitutions")),
    ]

    # accuracy breakdown helper — each metadata bucket (e.g. easy/hard) is
    # further split by leanctx route, so the verbatim row (Δ pinned to 0 by the
    # shared eval draw) is separated from the lingua row (the genuine
    # compression signal), with a per-bucket overall row. Route is carried only
    # by Leg B; both legs are compared on the same item partition by item_id.
    def _acc_table_by_route(key: str) -> list[str]:
        route_by_id = {
            r.get("item_id"): r.get("lx_route")
            for r in lb_b
            if r.get("item_id") is not None and r.get("lx_route")
        }
        a_by_id = {r.get("item_id"): r for r in lb_a if r.get("item_id") is not None}
        b_by_id = {r.get("item_id"): r for r in lb_b if r.get("item_id") is not None}

        buckets: dict[str, dict[str, list[tuple[bool, bool]]]] = {}
        for item_id, route in route_by_id.items():
            ra, rb = a_by_id.get(item_id), b_by_id.get(item_id)
            if ra is None or rb is None:
                continue
            if ra.get("accuracy") is None or rb.get("accuracy") is None:
                continue
            bucket = ra.get(key, "") or "unknown"
            pair = (bool(ra["accuracy"]), bool(rb["accuracy"]))
            buckets.setdefault(bucket, {}).setdefault(route, []).append(pair)

        def _cells(pairs: list[tuple[bool, bool]]) -> str:
            n = len(pairs)
            aa = sum(a for a, _ in pairs) / n
            bb = sum(b for _, b in pairs) / n
            return f"{n} | {pct(aa)} | {pct(bb)} | {pct(bb - aa)}"

        rows = ["| | route | N | Leg A | Leg B | Δ |", "|---|---|---|---|---|---|"]
        for bucket in sorted(buckets):
            routes = buckets[bucket]
            for route in ("verbatim", "lingua", "hybrid"):
                if route in routes:
                    rows.append(f"| {bucket} | {route} | {_cells(routes[route])} |")
            overall = [p for ps in routes.values() for p in ps]
            rows.append(f"| {bucket} | **overall** | {_cells(overall)} |")
        return rows

    def _ms(v: float | None) -> str:
        return f"{v:,.0f} ms" if v is not None else "N/A"

    p50_str = _ms(metrics.get("sidecar_p50_ms"))
    p95_str = _ms(metrics.get("sidecar_p95_ms"))
    eval_p50_str = _ms(metrics.get("eval_p50_ms"))
    eval_p95_str = _ms(metrics.get("eval_p95_ms"))

    # ── build report ─────────────────────────────────────────────────────
    out: list[str] = []
    A = out.append

    A("# ClawRouter × leanctx — Phase 1 Benchmark Report")
    A("")
    A(f"**Verdict: {verdict_icon} {verdict}**  ·  "
      f"Date: {date_str}  ·  "
      f"LB N={n_lb}  ·  "
      + (f"Eval: {eval_model}  ·  " if eval_model else "")
      + (f"CR commit: `{cr_commit[:8]}`" if cr_commit else ""))
    A("")
    A("---")
    A("")

    # 1. Go/No-Go gate
    A("## 1. Go / No-Go Gate")
    A("")
    A("| Condition | Threshold | Actual | Result |")
    A("|-----------|-----------|--------|--------|")
    A(gate_row(
        "Δ tokens — extra savings from Layer 8",
        f"≥ {savings_threshold * 100:.0f}%",
        pct(metrics.get("delta_tokens")),
        savings_ok,
    ))
    A(gate_row(
        "Δ accuracy — LongBench quality",
        f"≥ −{accuracy_drop * 100:.0f}%",
        pct(delta_acc),
        accuracy_ok,
    ))
    A("")
    if not gate.passed:
        A(f"> ⚠️  **Fail reason:** {gate.fail_reason}")
        A("")

    # 2. Token compression
    A("## 2. Token Compression")
    A("")
    A("| Stage | Avg tokens / request | vs raw | vs Leg A |")
    A("|-------|---------------------|--------|----------|")
    A(f"| Raw (uncompressed) | {tokens_raw_avg:,.0f} | — | — |")
    A(f"| **Leg A** — CR 7 layers | {tokens_a_avg:,.0f} | "
      f"−{pct(metrics.get('cr_savings'))} | — |")
    A(f"| **Leg B** — CR + leanctx Layer 8 | {tokens_b_avg:,.0f} | "
      f"−{pct(1 - metrics.get('e2e_ratio', 1.0))} | "
      f"−{pct(metrics.get('delta_tokens'))} |")
    A("")

    # 2b. Compression excluding verbatim content
    if verbatim_metrics is not None:
        vm = verbatim_metrics
        rc = vm["route_counts"]
        n = vm["n_items"]
        overall_ratio = (
            vm["layer8_out_tokens"] / vm["layer8_in_tokens"]
            if vm["layer8_in_tokens"]
            else None
        )
        A("## 2b. Compression Excluding Verbatim Content")
        A("")
        A("_Layer 8 routes code, errors, and structured content to **Verbatim "
          "(0%)**. This view reports compression on only the content actually "
          "sent to the compressor — a figure invariant to how much of the sample "
          "is verbatim-routed (see `oversampling_analysis.html`)._")
        A("")
        A(f"**Verbatim share:** {pct(vm['verbatim_share'])} of Layer-8 input "
          f"tokens were preserved verbatim. Routing mix over {n} items: "
          f"{rc.get('lingua', 0)} compressed · {rc.get('verbatim', 0)} verbatim · "
          f"{rc.get('hybrid', 0)} mixed.")
        A("")
        A("| View | Avg input / req | Avg output / req | Ratio | Savings | Δ tokens / req |")
        A("|------|-----------------|------------------|-------|---------|----------------|")
        A(f"| Overall (incl. verbatim) | {vm['avg_layer8_in']:,.0f} | "
          f"{vm['avg_layer8_out']:,.0f} | {fmt_f(overall_ratio, 3)} | "
          f"{pct(vm['overall_savings'])} | {vm['token_delta'] / n:,.0f} |")
        A(f"| **Non-verbatim only** | {vm['avg_nv_in']:,.0f} | "
          f"{vm['avg_nv_out']:,.0f} | {fmt_f(vm['nonverbatim_ratio'], 3)} | "
          f"{pct(vm['nonverbatim_savings'])} | {vm['token_delta'] / n:,.0f} |")
        A("")
        A(f"_Decomposition: overall {pct(vm['overall_savings'])} ≈ "
          f"(1 − {pct(vm['verbatim_share'])}) × non-verbatim "
          f"{pct(vm['nonverbatim_savings'])}. The absolute token delta is "
          f"identical in both views — verbatim content contributes 0._")
        A("")

    # 3. LongBench accuracy
    A("## 3. LongBench v2 Accuracy")
    A("")
    if n_lb > 0:
        A("| | N | Leg A (CR only) | Leg B (CR + leanctx) | Δ |")
        A("|---|---|---|---|---|")
        A(f"| **Overall** | {n_lb} | {pct(acc_a)} | {pct(acc_b)} | {pct(delta_acc)} |")
        A("")
        acc_cb = metrics.get("acc_closed_book")
        if acc_cb is not None:
            lift_a = acc_a - acc_cb
            lift_b = acc_b - acc_cb
            A("**Closed-book control** (no document — answers from priors only): "
              f"{pct(acc_cb)}. Context lift: Leg A {pct(lift_a)}, Leg B {pct(lift_b)}.")
            if lift_b <= 0:
                A("")
                A("> ⚠️  Leg B accuracy does not beat the closed-book baseline — "
                  "the compressed context may not be carrying the answer. Treat the "
                  "accuracy result with caution.")
            A("")
        # The verbatim-row Δ=0 mechanism (assert_route_reuse_invariant) and the
        # headline/by-route 'overall' parity (assert_overall_parity) are checked
        # by validate_run_invariants() in main() *after* all artifacts are
        # written — so a methodology violation surfaces loudly without discarding
        # a completed, fully-paid-for run. The renderer itself stays pure.
        route_rows = acc_by_route(lb_a, lb_b)
        if route_rows:
            A("### By leanctx route")
            A("")
            A("_`verbatim` = leanctx passed the context through unchanged; "
              "`lingua` = leanctx actually compressed it. With the shared eval "
              "draw, identical-input (verbatim) items score identically on both "
              "legs, so any non-zero Δ here lives entirely in `lingua`._")
            A("")
            A("| route | N | Leg A | Leg B | Δ |")
            A("|---|---|---|---|---|")
            for label, n, ra, rb in route_rows:
                disp = f"**{label}**" if label == "overall" else label
                A(f"| {disp} | {n} | {pct(ra)} | {pct(rb)} | {pct(rb - ra)} |")
            A("")
        for _hdr, _key in (
            ("By difficulty", "lb_difficulty"),
            ("By length", "lb_length"),
            ("By domain", "lb_domain"),
        ):
            tbl = _acc_table_by_route(_key)
            if len(tbl) > 2:  # header + separator only ⇒ no data to show
                A(f"### {_hdr}")
                A("")
                out.extend(tbl)
                A("")
    else:
        A("_No LongBench items in this run._")
        A("")

    # 4. CR layer contribution (Leg A)
    A("## 4. ClawRouter Layer Contributions (Leg A avg)")
    A("")
    A("| Layer | Statistic | Avg value |")
    A("|-------|-----------|-----------|")
    for name, stat, val in cr_layers:
        A(f"| {name} | {stat} | {val:,.1f} |")
    layer8_delta = (
        (metrics.get("tokens_a", 0) - metrics.get("tokens_b", 0)) / max(len(records_a), 1)
    )
    A(f"| **L8 leanctx** | Δ tokens / item | **{layer8_delta:,.0f}** |")
    A("")

    # 5. Latency
    A("## 5. Latency")
    A("")
    A("_Sidecar = the Layer 8 compression call in isolation (Leg B). Eval LLM "
      "time is the LongBench judge call, sourced from **Leg A** (every item gets "
      "a real call there; Leg B reuses identical-input draws at 0ms under the "
      "shared draw, which would otherwise read as a free judge). It is shown "
      "separately and is **not** part of the sidecar figure._")
    A("")
    A("| Metric | P50 | P95 |")
    A("|--------|-----|-----|")
    A(f"| Sidecar (Layer 8 compression) | {p50_str} | {p95_str} |")
    A(f"| Eval LLM (LongBench answer) | {eval_p50_str} | {eval_p95_str} |")
    A("")

    # 6. Cost analysis
    A("## 6. Cost Analysis")
    A("")
    cost = metrics.get("cost_saved_per_1k", 0.0)
    delta_t = metrics.get("delta_tokens", 0.0)
    A("Assuming Claude Sonnet input pricing ($15 / 1M tokens):")
    A("")
    A("```")
    A(f"Δ tokens      = {pct(delta_t)} of avg {tokens_raw_avg:,.0f} raw tokens")
    A(f"Savings / req = {delta_t * tokens_raw_avg:,.0f} tokens")
    A(f"Cost saved    = {delta_t * tokens_raw_avg * 15 / 1e6:.4f} USD / request")
    A(f"              = ${cost:,.2f} per 1 000 requests")
    A("```")
    A("")
    A("---")
    A("")
    short_commit = cr_commit[:8] if cr_commit else "unknown"
    A(f"_Generated by `bench_phase1.py` · CR commit `{short_commit}`_")

    return "\n".join(out)
