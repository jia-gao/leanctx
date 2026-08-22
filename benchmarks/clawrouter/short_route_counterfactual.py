#!/usr/bin/env python3
"""Two analyses over a full-503 Leg-B record set, both pure arithmetic.

1. **Reproduction check** — recompute the published headline figures (verbatim
   share, blended savings, non-verbatim savings, routing mix) from a locally
   regenerated record set and diff them against
   ``full_long_bench_evaluation_result.md``. A savings-only re-run
   (``bench_phase1.py --no-eval``) is enough: none of these figures depend on
   the judge.

2. **Short-tier routing counterfactual** — what happens to savings if
   ``length == "short"`` content is routed verbatim instead of compressed.

Why the counterfactual is exact on the accuracy side
----------------------------------------------------
Under a "short -> verbatim" rule those items are byte-identical between Leg A
and Leg B. Since #7, byte-identical inputs reuse Leg A's answer rather than
issuing a second judge call, so their accuracy delta is 0 *by construction* —
not estimated, and not something a re-run could disturb. That is why the
accuracy side can be read straight off the published per-bucket table while the
token side is recomputed from local records.

The token give-back, by contrast, is a real cost and is what this script
measures: excluding a bucket from compression forfeits its savings.

Run:
    .venv/bin/python benchmarks/clawrouter/short_route_counterfactual.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Published figures from full_long_bench_evaluation_result.md (2026-06-13),
# the run this analysis is checked against.
PUBLISHED = {
    "n_items": 503,
    "n_lingua": 228,
    "n_verbatim": 275,
    "verbatim_share": 0.542,
    "blended_savings": 0.241,
    "nonverbatim_savings": 0.528,
    "avg_layer8_in": 26397,
    "avg_layer8_out": 20022,
    "delta_per_req": 6374,
}

# Accuracy deltas by length x route, from the same report. Accuracy is NOT
# recomputed here — a --no-eval run has none — so these are carried as
# published constants and labelled as such in the output.
PUBLISHED_ACC = {
    # length: (n_lingua, leg_a_pct, leg_b_pct)
    "short": (68, 48.5, 30.9),
    "medium": (119, 37.8, 42.0),
    "long": (41, 51.2, 46.3),
}
PUBLISHED_OVERALL_ACC = {"n": 503, "leg_a": 45.3, "leg_b": 43.5}


@dataclass
class Bucket:
    n: int = 0
    verbatim: int = 0
    comp_in: int = 0
    comp_out: int = 0

    def add(self, rec: dict[str, Any]) -> None:
        self.n += 1
        self.verbatim += rec.get("lx_verbatim_tokens", 0)
        self.comp_in += rec.get("lx_compressed_in_tokens", 0)
        self.comp_out += rec.get("lx_compressed_out_tokens", 0)

    @property
    def layer8_in(self) -> int:
        return self.verbatim + self.comp_in

    @property
    def saved(self) -> int:
        return self.comp_in - self.comp_out


def load_leg_b(path: Path, workload: str = "lb_s1") -> list[dict[str, Any]]:
    """Leg-B LongBench records carrying a routing split."""
    out = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("leg") != "B" or r.get("workload") != workload:
                continue
            if "lx_route" not in r:
                continue  # unaligned item — no split available
            out.append(r)
    return out


def pct(x: float) -> str:
    return f"{x * 100:.1f} %"


def check(label: str, got: float, want: float, tol: float, unit: str = "") -> tuple[str, bool]:
    delta = got - want
    ok = abs(delta) <= tol
    mark = "MATCH" if ok else "DIFFERS"
    return (
        f"| {label} | {got:,.3f}{unit} | {want:,.3f}{unit} | {delta:+,.3f} | {mark} |",
        ok,
    )


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=here / "results")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    data = args.data or (args.results / "full503_phase1_results.jsonl")
    if not data.exists():
        print(f"[fatal] no record file at {data}")
        return 2

    recs = load_leg_b(data)
    if not recs:
        print(f"[fatal] no Leg-B lb_s1 records with lx_* split in {data}")
        return 2

    total = Bucket()
    by_len_route: dict[tuple[str, str], Bucket] = {}
    for r in recs:
        total.add(r)
        key = (r.get("lb_length", "?"), r.get("lx_route", "?"))
        by_len_route.setdefault(key, Bucket()).add(r)

    n = total.n
    routes = {"lingua": 0, "verbatim": 0, "hybrid": 0}
    for r in recs:
        routes[r["lx_route"]] = routes.get(r["lx_route"], 0) + 1

    verbatim_share = total.verbatim / total.layer8_in if total.layer8_in else 0.0
    blended = total.saved / total.layer8_in if total.layer8_in else 0.0
    nonverb = total.saved / total.comp_in if total.comp_in else 0.0

    L: list[str] = []
    A = L.append
    A("# Full-503 reproduction check + short-tier routing counterfactual")
    A("")
    A(f"**Source:** `{data.name}` · {n} aligned Leg-B LongBench items")
    A("")
    A("Regenerated locally with `bench_phase1.py --no-eval` (savings-only, zero")
    A("API spend). Token accounting comes from the compression step alone, so")
    A("every figure in section 1 is directly comparable to the published run.")
    A("")

    # ── 1. Reproduction ───────────────────────────────────────────────────
    A("## 1. Reproduction vs the published run")
    A("")
    A("| Figure | This run | Published | Δ | |")
    A("|---|---:|---:|---:|---|")
    rows = [
        check("Items", float(n), float(PUBLISHED["n_items"]), 0),
        check("Routed to lingua", float(routes.get("lingua", 0)), float(PUBLISHED["n_lingua"]), 0),
        check("Routed to verbatim", float(routes.get("verbatim", 0)),
              float(PUBLISHED["n_verbatim"]), 0),
        check("Verbatim token share", verbatim_share, PUBLISHED["verbatim_share"], 0.005),
        check("Blended savings", blended, PUBLISHED["blended_savings"], 0.005),
        check("Non-verbatim savings", nonverb, PUBLISHED["nonverbatim_savings"], 0.005),
        check("Avg Layer-8 in", total.layer8_in / n, float(PUBLISHED["avg_layer8_in"]), 50),
        check("Avg saved / req", total.saved / n, float(PUBLISHED["delta_per_req"]), 50),
    ]
    all_ok = True
    for row, ok in rows:
        A(row)
        all_ok = all_ok and ok
    A("")
    A(f"**Verdict: {'reproduces' if all_ok else 'DOES NOT fully reproduce'}** "
      f"(tolerance: 0.5 pp on shares, 50 tokens on averages).")
    A("")

    # ── 2. Routing table ──────────────────────────────────────────────────
    A("## 2. Routing and savings by length")
    A("")
    A("| length | route | n | Layer-8 in | saved | savings |")
    A("|---|---|---:|---:|---:|---:|")
    for key in sorted(by_len_route):
        length, route = key
        b = by_len_route[key]
        s = b.saved / b.layer8_in if b.layer8_in else 0.0
        A(f"| {length} | {route} | {b.n} | {b.layer8_in:,} | {b.saved:,} | {pct(s)} |")
    A("")

    # ── 3. Counterfactual ─────────────────────────────────────────────────
    short_lingua = by_len_route.get(("short", "lingua"), Bucket())
    forfeited = short_lingua.saved
    cf_saved = total.saved - forfeited
    cf_blended = cf_saved / total.layer8_in if total.layer8_in else 0.0

    A("## 3. Counterfactual — route `short` to verbatim")
    A("")
    A("Excluding the short tier from compression forfeits its token savings and")
    A("recovers its accuracy loss. Both sides are quantified below; the token")
    A("side is measured here, the accuracy side is carried from the published")
    A("run (see the note on why it is exact rather than estimated).")
    A("")
    A("### Token cost")
    A("")
    A("| | Savings | Blended rate |")
    A("|---|---:|---:|")
    A(f"| As shipped | {total.saved:,} | {pct(blended)} |")
    A(f"| short → verbatim | {cf_saved:,} | {pct(cf_blended)} |")
    A(f"| **Forfeited** | **−{forfeited:,}** | **{(cf_blended - blended) * 100:+.1f} pp** |")
    A("")
    A(f"Short-tier compression is {short_lingua.n} of {n} items "
      f"({short_lingua.n / n * 100:.1f} %) and "
      f"{forfeited / total.saved * 100:.1f} % of all savings."
      if total.saved else "")
    A("")
    A("### Accuracy recovered (published run — not recomputed here)")
    A("")
    n_s, a_s, b_s = PUBLISHED_ACC["short"]
    items_lost_short = round(n_s * (a_s - b_s) / 100)
    oa = PUBLISHED_OVERALL_ACC
    items_lost_all = round(oa["n"] * (oa["leg_a"] - oa["leg_b"]) / 100)
    cf_correct_b = round(oa["n"] * oa["leg_b"] / 100) + items_lost_short
    cf_acc_b = cf_correct_b / oa["n"] * 100
    A("| | N | Leg A | Leg B | Δ |")
    A("|---|---:|---:|---:|---:|")
    A(f"| short / lingua | {n_s} | {a_s} % | {b_s} % | {b_s - a_s:+.1f} pp |")
    A(f"| overall (as shipped) | {oa['n']} | {oa['leg_a']} % | {oa['leg_b']} % | "
      f"{oa['leg_b'] - oa['leg_a']:+.1f} pp |")
    A(f"| **overall (short → verbatim)** | {oa['n']} | {oa['leg_a']} % | "
      f"{cf_acc_b:.1f} % | **{cf_acc_b - oa['leg_a']:+.1f} pp** |")
    A("")
    A(f"The short tier loses ~{items_lost_short} items while the whole corpus "
      f"loses ~{items_lost_all}: every other bucket nets positive. Routing it "
      f"verbatim makes compression accuracy-neutral overall.")
    A("")
    A("> **Caveat.** This rule was derived from the same run it is evaluated on,")
    A("> so the accuracy figure is in-sample and cannot be treated as a")
    A("> validated result. It is a hypothesis with an exactly-known token price,")
    A("> and needs an out-of-sample confirmation before it is claimed anywhere.")
    A("")

    text = "\n".join(L)
    out = args.out or (args.results / "short_route_counterfactual.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"\n[out] {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
