"""Significance-based accuracy gate — paired McNemar test + CI.

Pure-math unit tests (no I/O, no LLM). The headline fixture mirrors the real
InsForge full-503 run the reviewer flagged: Δ −2.2% that is *within noise*, so
the gate must read it as GO, not NO-GO.
"""
from __future__ import annotations

import pytest

from benchmarks.common.scoring import apply_gate, compute_metrics, mcnemar_paired

# The full-503 contingency table, reconstructed from the saved by_item records:
#   201 both-correct, 247 both-wrong, 33 regressions (OFF✓ON✗), 22 improvements.
FULL503_PAIRS = (
    [(True, True)] * 201
    + [(False, False)] * 247
    + [(True, False)] * 33
    + [(False, True)] * 22
)


# ── mcnemar_paired math ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_mcnemar_counts_and_delta_match_real_run():
    sig = mcnemar_paired(FULL503_PAIRS)
    assert sig["n"] == 503
    assert sig["b10"] == 33  # regressions
    assert sig["b01"] == 22  # improvements
    assert sig["n_discordant"] == 55
    assert sig["delta"] == pytest.approx(-11 / 503, abs=1e-6)


@pytest.mark.unit
def test_mcnemar_full503_is_not_significant():
    # Exact two-sided binomial p over 55 discordant pairs ≈ 0.18, CI crosses 0.
    sig = mcnemar_paired(FULL503_PAIRS)
    assert sig["p_value"] == pytest.approx(0.177, abs=0.01)
    assert sig["ci_lo"] == pytest.approx(-0.0507, abs=1e-3)
    assert sig["ci_hi"] == pytest.approx(0.0070, abs=1e-3)
    assert sig["ci_hi"] > 0  # upper bound above zero ⇒ within noise


@pytest.mark.unit
def test_mcnemar_empty_is_degenerate_not_crash():
    sig = mcnemar_paired([])
    assert sig["n"] == 0
    assert sig["n_discordant"] == 0
    assert sig["p_value"] == 1.0
    assert sig["ci_lo"] == sig["ci_hi"] == 0.0


@pytest.mark.unit
def test_mcnemar_all_concordant_is_not_significant():
    sig = mcnemar_paired([(True, True)] * 50 + [(False, False)] * 50)
    assert sig["n_discordant"] == 0
    assert sig["p_value"] == 1.0
    assert sig["delta"] == 0.0


@pytest.mark.unit
def test_mcnemar_large_one_sided_regression_is_significant():
    # 40 regressions vs 5 improvements over many items ⇒ p < 0.05, CI below 0.
    pairs = [(True, True)] * 400 + [(True, False)] * 40 + [(False, True)] * 5
    sig = mcnemar_paired(pairs)
    assert sig["p_value"] < 0.05
    assert sig["ci_hi"] < 0  # whole CI below zero ⇒ real regression
    assert sig["z"] < 0  # signed: ON worse


# ── significance-based gate ──────────────────────────────────────────────────


@pytest.mark.unit
def test_gate_passes_on_within_noise_drop():
    # The exact case that previously read NO-GO on the point estimate alone.
    metrics = {"delta_tokens": 0.232, "accuracy_sig": mcnemar_paired(FULL503_PAIRS)}
    result = apply_gate(metrics, savings_threshold=0.20, accuracy_drop=0.02)
    assert result.passed is True


@pytest.mark.unit
def test_gate_fails_on_significant_regression():
    pairs = [(True, True)] * 400 + [(True, False)] * 40 + [(False, True)] * 5
    metrics = {"delta_tokens": 0.232, "accuracy_sig": mcnemar_paired(pairs)}
    result = apply_gate(metrics, savings_threshold=0.20, accuracy_drop=0.02)
    assert result.passed is False
    assert "accuracy" in result.fail_reason.lower()


@pytest.mark.unit
def test_gate_savings_still_enforced_under_significance():
    metrics = {"delta_tokens": 0.10, "accuracy_sig": mcnemar_paired(FULL503_PAIRS)}
    result = apply_gate(metrics, savings_threshold=0.20, accuracy_drop=0.02)
    assert result.passed is False
    assert "savings" in result.fail_reason.lower()


@pytest.mark.unit
def test_gate_savings_as_target_does_not_fail_run():
    # Soft-target mode (InsForge): a savings shortfall is recorded but the run
    # passes when correctness holds — the verdict reflects accuracy alone.
    metrics = {"delta_tokens": 0.176, "accuracy_sig": mcnemar_paired(FULL503_PAIRS)}
    result = apply_gate(metrics, savings_threshold=0.20, accuracy_drop=0.02,
                        savings_is_target=True)
    assert result.passed is True
    assert result.savings_met is False  # under target, but not a NO-GO


@pytest.mark.unit
def test_gate_savings_as_target_still_fails_on_real_regression():
    # Even as a soft target, accuracy stays a hard correctness gate.
    pairs = [(True, True)] * 400 + [(True, False)] * 40 + [(False, True)] * 5
    metrics = {"delta_tokens": 0.10, "accuracy_sig": mcnemar_paired(pairs)}
    result = apply_gate(metrics, savings_threshold=0.20, accuracy_drop=0.02,
                        savings_is_target=True)
    assert result.passed is False
    assert "accuracy" in result.fail_reason.lower()
    assert result.savings_met is False


@pytest.mark.unit
def test_gate_falls_back_to_point_estimate_without_pairs():
    # No accuracy_sig (hand-built metrics) ⇒ legacy point-estimate rule applies.
    metrics = {"delta_tokens": 0.25, "delta_accuracy": -0.05}
    result = apply_gate(metrics, savings_threshold=0.20, accuracy_drop=0.02)
    assert result.passed is False
    assert "accuracy" in result.fail_reason.lower()


# ── compute_metrics wires the paired stat in ─────────────────────────────────


@pytest.mark.unit
def test_compute_metrics_populates_accuracy_sig_paired_by_item():
    off = [
        {"item_id": "a", "workload": "lb", "tokens_raw": 100,
         "tokens_compressed": 100, "usage_prompt_tokens": 100, "accuracy": True},
        {"item_id": "b", "workload": "lb", "tokens_raw": 100,
         "tokens_compressed": 100, "usage_prompt_tokens": 100, "accuracy": True},
    ]
    on = [
        {"item_id": "a", "workload": "lb", "tokens_raw": 100,
         "tokens_compressed": 80, "usage_prompt_tokens": 80, "accuracy": True},
        {"item_id": "b", "workload": "lb", "tokens_raw": 100,
         "tokens_compressed": 80, "usage_prompt_tokens": 80, "accuracy": False},
    ]
    m = compute_metrics(off, on)
    sig = m["accuracy_sig"]
    assert sig["n"] == 2
    assert sig["b10"] == 1  # item b regressed
    assert sig["b01"] == 0
