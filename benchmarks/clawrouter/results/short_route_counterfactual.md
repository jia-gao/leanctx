# Full-503 reproduction check + short-tier routing counterfactual

**Source:** `full503_phase1_results.jsonl` · 503 aligned Leg-B LongBench items

Regenerated locally with `bench_phase1.py --no-eval` (savings-only, zero
API spend). Token accounting comes from the compression step alone, so
every figure in section 1 is directly comparable to the published run.

## 1. Reproduction vs the published run

| Figure | This run | Published | Δ | |
|---|---:|---:|---:|---|
| Items | 503.000 | 503.000 | +0.000 | MATCH |
| Routed to lingua | 228.000 | 228.000 | +0.000 | MATCH |
| Routed to verbatim | 275.000 | 275.000 | +0.000 | MATCH |
| Verbatim token share | 0.542 | 0.542 | +0.000 | MATCH |
| Blended savings | 0.187 | 0.241 | -0.054 | DIFFERS |
| Non-verbatim savings | 0.408 | 0.528 | -0.120 | DIFFERS |
| Avg Layer-8 in | 26,396.753 | 26,397.000 | -0.247 | MATCH |
| Avg saved / req | 4,926.642 | 6,374.000 | -1,447.358 | DIFFERS |

**Verdict: DOES NOT fully reproduce** (tolerance: 0.5 pp on shares, 50 tokens on averages).

## 2. Routing and savings by length

| length | route | n | Layer-8 in | saved | savings |
|---|---|---:|---:|---:|---:|
| long | lingua | 41 | 1,403,476 | 448,530 | 32.0 % |
| long | verbatim | 67 | 1,909,258 | 0 | 0.0 % |
| medium | lingua | 119 | 3,436,503 | 1,462,024 | 42.5 % |
| medium | verbatim | 96 | 2,629,671 | 0 | 0.0 % |
| short | lingua | 68 | 1,235,924 | 567,547 | 45.9 % |
| short | verbatim | 112 | 2,662,735 | 0 | 0.0 % |

## 3. Counterfactual — route `short` to verbatim

Excluding the short tier from compression forfeits its token savings and
recovers its accuracy loss. Both sides are quantified below; the token
side is measured here, the accuracy side is carried from the published
run (see the note on why it is exact rather than estimated).

### Token cost

| | Savings | Blended rate |
|---|---:|---:|
| As shipped | 2,478,101 | 18.7 % |
| short → verbatim | 1,910,554 | 14.4 % |
| **Forfeited** | **−567,547** | **-4.3 pp** |

Short-tier compression is 68 of 503 items (13.5 %) and 22.9 % of all savings.

### Accuracy recovered (published run — not recomputed here)

| | N | Leg A | Leg B | Δ |
|---|---:|---:|---:|---:|
| short / lingua | 68 | 48.5 % | 30.9 % | -17.6 pp |
| overall (as shipped) | 503 | 45.3 % | 43.5 % | -1.8 pp |
| **overall (short → verbatim)** | 503 | 45.3 % | 45.9 % | **+0.6 pp** |

The short tier loses ~12 items while the whole corpus loses ~9: every other bucket nets positive. Routing it verbatim makes compression accuracy-neutral overall.

> **Caveat.** This rule was derived from the same run it is evaluated on,
> so the accuracy figure is in-sample and cannot be treated as a
> validated result. It is a hypothesis with an exactly-known token price,
> and needs an out-of-sample confirmation before it is claimed anywhere.
