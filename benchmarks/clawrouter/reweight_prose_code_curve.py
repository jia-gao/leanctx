#!/usr/bin/env python3
"""Method A — prose-vs-code savings curve by reweighting the full-503 run.

Reweights the existing LongBench-v2 full-503 Leg-B records (no new benchmark
run, no API spend) into a savings-as-a-function-of-mix curve along a single
axis: prose share vs code share.

Why this is exact, not estimated
--------------------------------
leanctx routes PROSE -> Lingua and everything else -> Verbatim (0%).
LLMLingua-2 is deterministic, so per-item token savings is a fixed function of
content. On a *token-share* axis the blended savings is therefore exactly:

    savings(p) = p * r          p = prose token-share, r = per-prose-token rate

a straight line through the origin. Reweighting to any target mix is arithmetic.

Inputs
------
full503_phase1_results.jsonl — the 503 LongBench Leg-B records carrying the
per-item route split (lx_verbatim_tokens / lx_compressed_in_tokens /
lx_compressed_out_tokens), produced by bench_phase1.py.

Outputs (written next to this script's --results dir)
-----------------------------------------------------
prose_code_savings_curve.png   — the plot (token-share + request-share panels)
prose_code_savings_curve.csv   — curve points for downstream use
prose_code_savings_curve.md    — the human-readable report

Run:
    .venv/bin/python benchmarks/clawrouter/reweight_prose_code_curve.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Savings targets that name the three traffic regimes (from issue #2 discussion).
REGIMES = [
    ("code-heavy", 0.10),
    ("balanced", 0.25),
    ("prose-heavy", 0.45),
]


@dataclass
class Corpus:
    n: int
    prose_tokens: int  # compressible (Lingua) input tokens
    code_tokens: int  # verbatim input tokens
    prose_out_tokens: int  # compressible output tokens
    n_prose_req: int
    n_code_req: int
    mean_prose_input: float  # I_p
    mean_code_input: float  # I_c
    prose_items: list[tuple[int, int]]  # (in, out) per prose request, for bootstrap
    by_domain: dict[str, tuple[int, int]]  # domain -> (in, out) for prose items

    @property
    def input_tokens(self) -> int:
        return self.prose_tokens + self.code_tokens

    @property
    def saved_tokens(self) -> int:
        return self.prose_tokens - self.prose_out_tokens

    @property
    def r(self) -> float:
        """Per-prose-token savings rate (the slope)."""
        return self.saved_tokens / self.prose_tokens

    @property
    def p0(self) -> float:
        """Natural prose token-share of this corpus."""
        return self.prose_tokens / self.input_tokens

    @property
    def overall_savings(self) -> float:
        return self.saved_tokens / self.input_tokens


def load_corpus(path: Path, *, leg: str = "B", workload: str = "lb_s1") -> Corpus:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    items = [
        r
        for r in recs
        if r.get("leg") == leg
        and "lx_route" in r
        and r.get("workload") == workload
    ]
    prose_tokens = code_tokens = prose_out = 0
    prose_items: list[tuple[int, int]] = []
    n_prose = n_code = 0
    prose_inputs: list[int] = []
    code_inputs: list[int] = []
    by_domain: dict[str, list[int]] = {}
    for r in items:
        vin = r["lx_verbatim_tokens"]
        cin = r["lx_compressed_in_tokens"]
        cout = r["lx_compressed_out_tokens"]
        prose_tokens += cin
        code_tokens += vin
        prose_out += cout
        if r["lx_route"] == "lingua":
            n_prose += 1
            prose_items.append((cin, cout))
            prose_inputs.append(cin)
            d = by_domain.setdefault(r.get("lb_domain", "?"), [0, 0])
            d[0] += cin
            d[1] += cout
        else:
            n_code += 1
            code_inputs.append(vin)
    return Corpus(
        n=len(items),
        prose_tokens=prose_tokens,
        code_tokens=code_tokens,
        prose_out_tokens=prose_out,
        n_prose_req=n_prose,
        n_code_req=n_code,
        mean_prose_input=sum(prose_inputs) / len(prose_inputs),
        mean_code_input=sum(code_inputs) / len(code_inputs),
        prose_items=prose_items,
        by_domain={k: (v[0], v[1]) for k, v in by_domain.items()},
    )


def bootstrap_r(corpus: Corpus, n_boot: int = 2000, seed: int = 1234) -> tuple[float, float]:
    """95% CI on the slope r via item-level resampling of prose requests."""
    rng = random.Random(seed)
    items = corpus.prose_items
    n = len(items)
    rs: list[float] = []
    for _ in range(n_boot):
        sin = sout = 0
        for _ in range(n):
            cin, cout = items[rng.randrange(n)]
            sin += cin
            sout += cout
        rs.append((sin - sout) / sin)
    rs.sort()
    lo = rs[int(0.025 * n_boot)]
    hi = rs[int(0.975 * n_boot)]
    return lo, hi


def request_share_savings(q: float, ip: float, ic: float, r: float) -> float:
    """Blended savings at prose *request*-share q.

    A prose request saves r * (its input); a code request saves 0. With mean
    prose input I_p and mean code input I_c, the token-weighted savings is:

        savings(q) = q*I_p*r / (q*I_p + (1-q)*I_c)

    which equals p(q)*r where p(q) is the induced prose token-share. The curve
    bends away from the token-share line whenever I_c != I_p.
    """
    num = q * ip * r
    den = q * ip + (1.0 - q) * ic
    return num / den if den else 0.0


def make_plot(corpus: Corpus, r_lo: float, r_hi: float, out: Path) -> None:
    r = corpus.r
    ps = [i / 100 for i in range(101)]
    fig, (axT, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Panel A: token-share axis (exactly linear) ---------------------
    axT.fill_between(ps, [p * r_lo for p in ps], [p * r_hi for p in ps],
                     color="#4c72b0", alpha=0.18, label=f"95% CI on slope ({r_lo:.3f}–{r_hi:.3f})")
    axT.plot(ps, [p * r for p in ps], color="#4c72b0", lw=2.2,
             label=f"savings = p · r   (r = {r:.3f})")
    # corpus point
    axT.scatter([corpus.p0], [corpus.overall_savings], color="#c44e52", s=120,
                zorder=5, marker="*",
                label=f"full-503 corpus (p={corpus.p0:.3f}, {corpus.overall_savings:.1%})")
    # regime markers
    for name, sav in REGIMES:
        p = sav / r
        axT.scatter([p], [sav], color="#55a868", s=45, zorder=5)
        axT.annotate(f"{name}\n(p={p:.2f} → {sav:.0%})", (p, sav),
                     textcoords="offset points", xytext=(8, -4), fontsize=8)
    axT.set_xlim(0, 1)
    axT.set_ylim(0, 0.6)
    axT.set_xlabel("Prose token-share  p")
    axT.set_ylabel("Blended token savings")
    axT.set_title("A · Token-share axis — exactly linear")
    axT.grid(alpha=0.25)
    axT.legend(loc="upper left", fontsize=8)

    # ---- Panel B: request-share axis (bend) -----------------------------
    qs = ps
    axR.plot(qs, [q * r for q in qs], color="#999999", lw=1.4, ls="--",
             label="token-share line (reference)")
    emp = [request_share_savings(q, corpus.mean_prose_input, corpus.mean_code_input, r) for q in qs]
    axR.plot(qs, emp, color="#4c72b0", lw=2.2,
             label=f"LongBench request-share (I_c/I_p={corpus.mean_code_input/corpus.mean_prose_input:.2f})")
    # illustrative: code requests 3x larger (agent-like dumps)
    illo = [request_share_savings(q, 1.0, 3.0, r) for q in qs]
    axR.plot(qs, illo, color="#dd8452", lw=2.0, ls=":",
             label="illustrative: code reqs 3× larger (agent-like)")
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 0.6)
    axR.set_xlabel("Prose request-share  q")
    axR.set_ylabel("Blended token savings")
    axR.set_title("B · Request-share axis — bends when I_c ≠ I_p")
    axR.grid(alpha=0.25)
    axR.legend(loc="upper left", fontsize=8)

    fig.suptitle("leanctx Layer-8 — prose-vs-code savings curve (Method A, full-503 reweight)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_csv(corpus: Corpus, out: Path) -> None:
    r = corpus.r
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prose_token_share", "blended_savings_token_axis",
                    "blended_savings_request_axis_longbench"])
        for i in range(101):
            p = i / 100
            req = request_share_savings(p, corpus.mean_prose_input, corpus.mean_code_input, r)
            w.writerow([f"{p:.2f}", f"{p * r:.4f}", f"{req:.4f}"])


def write_report(corpus: Corpus, r_lo: float, r_hi: float, png: Path, csvf: Path, out: Path) -> None:
    r = corpus.r
    ic_ip = corpus.mean_code_input / corpus.mean_prose_input
    # regime rows
    regime_rows = "\n".join(
        f"| {name} | {sav / r:.3f} | {sav:.1%} |"
        for name, sav in REGIMES
    )
    # token-share table
    pts = [0.0, 0.10, 0.25, corpus.p0, 0.50, 0.75, 0.90, 1.0]
    share_rows = "\n".join(
        f"| {p:.3f}{' ← corpus' if abs(p - corpus.p0) < 1e-9 else ''} | "
        f"{p * r:.1%} | {p * r_lo:.1%} – {p * r_hi:.1%} |"
        for p in pts
    )
    # by-domain band
    dom_rows = "\n".join(
        f"| {d} | {ci:,} | {(ci - co) / ci:.3f} |"
        for d, (ci, co) in sorted(corpus.by_domain.items(), key=lambda x: -x[1][0])
    )
    overall_lo = corpus.p0 * r_lo
    overall_hi = corpus.p0 * r_hi
    out.write_text(f"""# Prose-vs-code savings curve — Method A (full-503 reweight)

**Date:** 2026-06-23 · **Source:** `full503_phase1_results.jsonl` (503 LongBench-v2 Leg-B items)
**Method:** reweight existing per-item route splits — **no new benchmark run, no API spend**.
**Scope:** single axis, prose-share vs code-share. Accuracy is out of scope (LongBench v2 covers it).

> Generated by `benchmarks/clawrouter/reweight_prose_code_curve.py`. Re-run to regenerate.

---

## TL;DR

On a **token-share** axis the curve is an exact straight line through the origin:

```
blended_savings(p) = p · r        p = prose token-share,  r = {r:.4f}
```

- **Slope r = {r:.1%}** (per-prose-token savings rate; 95% CI {r_lo:.1%}–{r_hi:.1%}).
- The full-503 corpus sits at **p₀ = {corpus.p0:.3f}** → **{corpus.overall_savings:.1%}** blended savings
  (95% CI {overall_lo:.1%}–{overall_hi:.1%}), reproducing the published 24.1% gate number.
- Because LLMLingua-2 is deterministic, every point on the line is **exact arithmetic**, not an estimate.

![prose-vs-code savings curve]({png.name})

Curve data: [`{csvf.name}`]({csvf.name}).

---

## 1. The line (token-share axis)

| Prose token-share p | Blended savings (p·r) | 95% CI |
|---|---|---|
{share_rows}

The three traffic regimes named in issue #2 map to these prose-shares:

| Regime | Prose token-share needed | Blended savings |
|---|---|---|
{regime_rows}

So leanctx clears the 20% gate once traffic is **≳ {0.20 / r:.0%} prose by tokens**; below that it
under-delivers, above ~85% prose it approaches the ~{r:.0%} compressor ceiling.

---

## 2. Token-share vs request-share (the bend)

Real users think in *requests* ("what % of my requests are coding?"), not tokens. The request-share
curve is:

```
savings(q) = q·I_p·r / (q·I_p + (1−q)·I_c)
```

where I_p, I_c are the mean input tokens of a prose vs a code request. It bends **below** the token-share
line whenever code requests are larger (I_c > I_p), because code then eats a disproportionate token share.

**On LongBench this bend is negligible:** I_p = {corpus.mean_prose_input:,.0f} tok,
I_c = {corpus.mean_code_input:,.0f} tok → **I_c/I_p = {ic_ip:.2f}**. LongBench items are all ~26K-token
documents regardless of type, so request-share ≈ token-share and both axes are linear here.

⚠️ **This is a LongBench artifact, not a general result.** In real agent traffic, code/tool/log dumps are
much larger than chat turns (I_c/I_p ≈ 3–10×), which bends the request-share curve well below linear — a
50%-prose-by-requests stream can be ~25%-prose-by-tokens. Panel B shows an illustrative I_c/I_p = 3 curve.
Quantifying the real bend is a **Method B** question (clean prose + code corpora at realistic sizes).

---

## 3. The slope band — how r varies by prose sub-type

The single slope `r` hides variation across prose content types. Per-domain r (prose/Lingua items only):

| Prose content domain | Compressible input tokens | r |
|---|---|---|
{dom_rows}

Spread is ~{(max((ci - co) / ci for ci, co in corpus.by_domain.values()) - min((ci - co) / ci for ci, co in corpus.by_domain.values())) * 100:.0f} points
({min((ci - co) / ci for ci, co in corpus.by_domain.values()):.1%}–{max((ci - co) / ci for ci, co in corpus.by_domain.values()):.1%}).
Structured-data prose compresses least; dialogue/single-doc prose most. Real traffic's r depends on which
prose sub-types dominate — another reason to validate the slope on representative prose (Method B / MeetingBank).

---

## 4. Caveats (read before quoting a number)

1. **"Code" here = all LongBench verbatim-routed content** (code repos, structured data, errors) — not
   literally source code. The axis is really *compressible-prose vs everything-else*.
2. **r = {r:.1%} is LongBench-flavored.** LongBench prose is academic QA context; chat / RAG / transcripts
   compress differently. MeetingBank (Lingua-2's own eval) is the honest best-case anchor.
3. **The request-share bend is corpus-specific** (driven by I_c/I_p, which is ~1 on LongBench but ≫1 in
   agent traffic). Its *existence* is general; its *magnitude* needs Method B.
4. **Production point is one point on this line.** The Anthropic Economic Index puts ~35% of Claude traffic
   in coding tasks — but that is a task-category share, not a code-*token* share (a coding task is mostly
   prose: instructions, explanations). Treat it as loose orientation only.

---

## 5. Reproduce

```bash
.venv/bin/python benchmarks/clawrouter/reweight_prose_code_curve.py
```

Reads `full503_phase1_results.jsonl`; writes `{png.name}`, `{csvf.name}`, and this report.
""")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=here / "results",
                    help="dir holding full503_phase1_results.jsonl and where outputs land")
    ap.add_argument("--data", type=Path, default=None,
                    help="override path to the full-503 JSONL")
    ap.add_argument("--leg", default="B",
                    help="leg label carrying the route split (ClawRouter: B; InsForge: ON)")
    ap.add_argument("--workload", default="lb_s1",
                    help="workload tag to filter (ClawRouter: lb_s1; InsForge: lb_if)")
    args = ap.parse_args()

    data = args.data or (args.results / "full503_phase1_results.jsonl")
    corpus = load_corpus(data, leg=args.leg, workload=args.workload)
    r_lo, r_hi = bootstrap_r(corpus)

    png = args.results / "prose_code_savings_curve.png"
    csvf = args.results / "prose_code_savings_curve.csv"
    md = args.results / "prose_code_savings_curve.md"

    make_plot(corpus, r_lo, r_hi, png)
    write_csv(corpus, csvf)
    write_report(corpus, r_lo, r_hi, png, csvf, md)

    print(f"corpus: N={corpus.n}  prose_req={corpus.n_prose_req}  code_req={corpus.n_code_req}")
    print(f"r = {corpus.r:.4f}  (95% CI {r_lo:.4f}-{r_hi:.4f})")
    print(f"p0 = {corpus.p0:.4f}  overall savings = {corpus.overall_savings:.4f}")
    print(f"I_c/I_p = {corpus.mean_code_input / corpus.mean_prose_input:.3f}")
    print(f"wrote: {png}\n       {csvf}\n       {md}")


if __name__ == "__main__":
    main()
