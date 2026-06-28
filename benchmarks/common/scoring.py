"""Scoring + go/no-go gate for the benchmark harness (extracted from bench_phase1).

Provider-agnostic: ``compute_metrics`` aggregates per-item records (token counts
+ accuracy) and ``apply_gate`` applies the savings/accuracy thresholds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any


@dataclass
class GateResult:
    passed: bool
    fail_reason: str = ""
    # Whether the savings figure met its bar. When savings is treated as a soft
    # target (``apply_gate(savings_is_target=True)``) a shortfall sets this False
    # *without* failing ``passed`` — the verdict then reflects correctness alone,
    # and the report frames savings as a labeled target rather than a NO-GO.
    savings_met: bool = True


def mcnemar_paired(
    pairs: list[tuple[bool, bool]], *, confidence: float = 0.95
) -> dict[str, Any]:
    """Paired McNemar test + accuracy-diff CI from per-item OFF/ON correctness.

    ``pairs`` are ``(off_correct, on_correct)`` booleans for the **same** items
    scored on both legs — exactly the aligned per-item records the harness
    already saves, so this never needs a fresh LLM call. The test conditions on
    the *discordant* pairs only (one leg right, the other wrong); concordant
    pairs carry no signal about a difference and drop out.

    Returns:
      ``delta``        accuracy difference ON − OFF (= (b01 − b10) / n)
      ``ci_lo/ci_hi``  Wald CI on that paired difference at ``confidence``
      ``p_value``      *exact* two-sided binomial McNemar p over discordant pairs
      ``z``            signed normal approx (negative ⇒ ON worse); diagnostic only
      ``b10``          regressions: OFF correct → ON wrong
      ``b01``          improvements: OFF wrong → ON correct
      ``n_discordant`` b01 + b10
      ``n``            total paired items

    Conservative by construction: an empty or fully-concordant sample yields
    ``p_value=1.0`` and a degenerate (zero-width) CI, so a significance-based
    gate can never *fail* on no evidence.
    """
    n = len(pairs)
    b10 = sum(1 for off, on in pairs if off and not on)  # regression
    b01 = sum(1 for off, on in pairs if on and not off)  # improvement
    n_disc = b01 + b10
    delta = (b01 - b10) / n if n else 0.0

    # Exact two-sided McNemar: discordant outcomes are Binomial(n_disc, 0.5)
    # under H0. Two-sided p = 2·P(X ≤ min(b01, b10)), capped at 1.
    if n_disc == 0:
        p_value = 1.0
    else:
        k = min(b01, b10)
        tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc)
        p_value = min(2.0 * tail, 1.0)

    z = (b01 - b10) / math.sqrt(n_disc) if n_disc else 0.0

    # Wald CI on the paired difference of proportions. Var of (b01 − b10)/n is
    # (b01 + b10 − (b01 − b10)²/n) / n²; degenerates to 0 when n or n_disc is 0.
    if n and n_disc:
        var = (n_disc - (b01 - b10) ** 2 / n) / n ** 2
        se = math.sqrt(max(var, 0.0))
    else:
        se = 0.0
    z_crit = NormalDist().inv_cdf((1 + confidence) / 2)
    return {
        "delta": delta,
        "ci_lo": delta - z_crit * se,
        "ci_hi": delta + z_crit * se,
        "p_value": p_value,
        "z": z,
        "b10": b10,
        "b01": b01,
        "n_discordant": n_disc,
        "n": n,
        "confidence": confidence,
    }


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in [0, 1]); None for an empty sample."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    idx = min(int(round(q * (len(s) - 1))), len(s) - 1)
    return float(s[idx])


def compute_metrics(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    records_cb: list[dict[str, Any]] | None = None,
    *,
    token_field: str = "tokens_compressed",
    input_price_per_token: float = 15 / 1e6,
) -> dict[str, Any]:
    """Aggregate per-item leg records into the headline metrics.

    ``token_field`` selects which per-record count drives the savings figure —
    ``"tokens_compressed"`` (tiktoken, the ClawRouter default) or, for InsForge,
    ``"usage_prompt_tokens"`` (the provider's own ``usage.prompt_tokens``).
    ``input_price_per_token`` sets the cost model (default $15/1M = Sonnet; pass
    the model's OpenRouter price for InsForge).

    An empty aligned pool (e.g. a transcript-only run, since the transcript is
    excluded from the metric pool, or every item dropped) yields a zeroed metric
    dict rather than a ``ZeroDivisionError`` that would discard a finished run.
    """
    if not records_a or not records_b:
        return {
            "delta_tokens": 0.0,
            "delta_accuracy": 0.0,
            "e2e_ratio": 0.0,
            "cr_savings": 0.0,
            "cost_saved_per_1k": 0.0,
            "sidecar_p50_ms": None,
            "sidecar_p95_ms": None,
            "eval_p50_ms": None,
            "eval_p95_ms": None,
            "tokens_a": 0,
            "tokens_b": 0,
            "acc_a": 0.0,
            "acc_b": 0.0,
            "acc_closed_book": None,
            "accuracy_sig": mcnemar_paired([]),
        }

    tokens_a = sum(r[token_field] for r in records_a)
    tokens_b = sum(r[token_field] for r in records_b)

    avg_raw = sum(r["tokens_raw"] for r in records_a) / len(records_a)
    avg_a = tokens_a / len(records_a)
    avg_b = tokens_b / len(records_b)

    delta_tokens = (tokens_a - tokens_b) / tokens_a if tokens_a > 0 else 0.0
    cr_savings = (avg_raw - avg_a) / avg_raw if avg_raw > 0 else 0.0
    e2e_ratio = avg_b / avg_raw if avg_raw > 0 else 0.0

    acc_a_vals = [r["accuracy"] for r in records_a if r.get("accuracy") is not None]
    acc_b_vals = [r["accuracy"] for r in records_b if r.get("accuracy") is not None]
    acc_a = sum(acc_a_vals) / len(acc_a_vals) if acc_a_vals else 0.0
    acc_b = sum(acc_b_vals) / len(acc_b_vals) if acc_b_vals else 0.0
    delta_accuracy = acc_b - acc_a

    # Paired significance: same items run OFF and ON, so the accuracy delta is a
    # *paired* comparison, not two independent samples. Align by item_id and run
    # McNemar over the per-item (off, on) correctness — this is what lets the
    # gate tell a real regression from run-to-run noise. Falls back gracefully
    # (empty → p=1.0) when records carry no item_id (e.g. unit fixtures).
    a_by_id = {
        r["item_id"]: r
        for r in records_a
        if r.get("item_id") and r.get("accuracy") is not None
    }
    b_by_id = {
        r["item_id"]: r
        for r in records_b
        if r.get("item_id") and r.get("accuracy") is not None
    }
    pairs = [
        (bool(a_by_id[i]["accuracy"]), bool(b_by_id[i]["accuracy"]))
        for i in a_by_id.keys() & b_by_id.keys()
    ]
    accuracy_sig = mcnemar_paired(pairs)

    # Closed-book control: accuracy when the model answers from priors alone.
    # If Leg A/B accuracy is at or below this, the context did no work and the
    # "compression preserved information" claim is unsupported.
    cb_vals = [
        r["accuracy"]
        for r in (records_cb or [])
        if r.get("accuracy") is not None
    ]
    acc_closed_book: float | None = (
        sum(cb_vals) / len(cb_vals) if cb_vals else None
    )

    # Cost saved over 1K conversations at the configured input price.
    cost_saved_per_1k = delta_tokens * avg_raw * input_price_per_token * 1000

    # Sidecar (Layer 8) latency is the compression call in isolation, taken
    # from Leg B records (the only leg with the sidecar in the path). The eval
    # LLM time is reported separately and never folded into this figure.
    compress_ms_b = [
        r["compress_ms"] for r in records_b if r.get("compress_ms") is not None
    ]
    # Eval-LLM latency is sourced from Leg A, not Leg B: under the shared eval
    # draw, verbatim Leg-B items reuse Leg A's answer and record eval_ms=0, so a
    # Leg-B P50 would read ~0ms and misrepresent the judge as "free". Leg A
    # always issues a real judge call (no prior to reuse), so its distribution is
    # the honest per-call cost.
    eval_ms_a = [r["eval_ms"] for r in records_a if r.get("eval_ms") is not None]

    return {
        "delta_tokens": delta_tokens,
        "delta_accuracy": delta_accuracy,
        "e2e_ratio": e2e_ratio,
        "cr_savings": cr_savings,
        "cost_saved_per_1k": cost_saved_per_1k,
        "sidecar_p50_ms": _percentile(compress_ms_b, 0.50),
        "sidecar_p95_ms": _percentile(compress_ms_b, 0.95),
        "eval_p50_ms": _percentile(eval_ms_a, 0.50),
        "eval_p95_ms": _percentile(eval_ms_a, 0.95),
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "acc_closed_book": acc_closed_book,
        "accuracy_sig": accuracy_sig,
    }


def apply_gate(
    metrics: dict[str, Any],
    *,
    savings_threshold: float = 0.20,
    accuracy_drop: float = 0.02,
    savings_is_target: bool = False,
) -> GateResult:
    """Apply the savings + accuracy gates.

    ``savings_is_target`` decides the *severity* of a savings shortfall:
    ``False`` (default) treats the savings threshold as a hard gate — under it
    is a NO-GO. ``True`` treats it as a soft internal target — a shortfall is
    recorded (``savings_met=False``) but does **not** fail the run, so the
    verdict reflects correctness (accuracy) alone. Accuracy is always a hard
    correctness gate either way: compression that *significantly* degrades
    answers is a real regression regardless of how much it saves.
    """
    savings_met = metrics["delta_tokens"] >= savings_threshold
    if not savings_met and not savings_is_target:
        return GateResult(
            passed=False,
            savings_met=False,
            fail_reason=(
                f"savings delta {metrics['delta_tokens']:.4f} below threshold {savings_threshold}"
            ),
        )

    # Accuracy gate — significance-based when per-item pairs are available.
    # A ±tolerance threshold on the point estimate alone flips GO/NO-GO on
    # run-to-run noise (the eval's own SE over N=503 is ~1 tolerance). Instead,
    # fail only when the regression is significant *beyond* tolerance — i.e.
    # even the optimistic edge of the 95% CI sits below −tolerance. A point
    # estimate that dips under −tolerance but whose CI still reaches above it is
    # within noise and passes. Falls back to the point-estimate rule when no
    # paired stats are present (e.g. hand-built metrics dicts in unit tests).
    sig = metrics.get("accuracy_sig")
    if sig is not None:
        if sig["ci_hi"] < -accuracy_drop:
            return GateResult(
                passed=False,
                savings_met=savings_met,
                fail_reason=(
                    f"accuracy regression significant beyond tolerance: Δ "
                    f"{sig['delta']:+.4f} (95% CI [{sig['ci_lo']:+.4f}, "
                    f"{sig['ci_hi']:+.4f}], McNemar p={sig['p_value']:.4f}); "
                    f"CI upper bound below allowed −{accuracy_drop}"
                ),
            )
    elif metrics["delta_accuracy"] < -accuracy_drop:
        return GateResult(
            passed=False,
            savings_met=savings_met,
            fail_reason=(
                f"accuracy dropped {-metrics['delta_accuracy']:.4f} exceeds allowed {accuracy_drop}"
            ),
        )
    return GateResult(passed=True, savings_met=savings_met)
