"""Verbatim-vs-compressed token split, by-route accuracy, run invariants, and
per-item dumps (extracted from bench_phase1). Provider-agnostic.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from benchmarks.common.tokens import _sum_tokens

_SIDECAR_THRESHOLD_DEFAULT = 1500


def _sidecar_threshold() -> int:
    return int(os.environ.get("LEANCTX_SERVER_THRESHOLD", str(_SIDECAR_THRESHOLD_DEFAULT)))


def _verbatim_split_item(
    msgs_a: list[dict[str, Any]],
    msgs_b: list[dict[str, Any]],
    *,
    threshold: int,
) -> tuple[int, int, int, str] | None:
    """Bucket one item's Layer-8 input into verbatim vs compressed tokens.

    ``msgs_a`` is the CR-only output (== what leanctx saw inside Leg B), aligned
    message-for-message with ``msgs_b`` (the CR + leanctx output). Reuses
    leanctx's *real* eligibility roles and ``classify()`` so there is no copied
    routing logic to drift.

    Returns ``(verbatim_tokens, compressed_in, compressed_out, route)`` where
    ``route`` is ``"verbatim"`` | ``"lingua"`` | ``"hybrid"``, or ``None`` when
    the two legs cannot be aligned (different message counts → caller skips).
    """
    from leanctx.classifier import classify
    from leanctx.compressors import ContentType
    from leanctx.server import _COMPRESSIBLE_ROLES

    if len(msgs_a) != len(msgs_b):
        return None

    eligible = {
        i
        for i, m in enumerate(msgs_a)
        if m.get("role") in _COMPRESSIBLE_ROLES and isinstance(m.get("content"), str)
    }
    # Request-level gate: if the eligible subset is under the threshold the
    # whole request is passed through, so nothing is compressed.
    eligible_tokens = sum(_sum_tokens([msgs_a[i]]) for i in eligible)
    request_compressible = eligible_tokens >= threshold

    # Content types the default router sends to the lingua pass: prose at the
    # standard ratio, structured data (JSON/dialogue logs) at the gentler
    # ``structured_ratio`` (middleware auto-route). Both are *compressed* — they
    # must mirror the sidecar here or the route label drifts from the real
    # routing and the route/reuse invariant trips (which is exactly what a new
    # ContentType silently omitted from this set would do).
    _COMPRESSED_TYPES = (ContentType.PROSE, ContentType.STRUCTURED)

    verbatim = comp_in = comp_out = 0
    n_compressed = 0
    for i, (ma, mb) in enumerate(zip(msgs_a, msgs_b, strict=True)):
        ta = _sum_tokens([ma])
        is_compressed = (
            request_compressible
            and i in eligible
            and classify(ma) in _COMPRESSED_TYPES
        )
        if is_compressed:
            comp_in += ta
            comp_out += _sum_tokens([mb])
            n_compressed += 1
        else:
            verbatim += ta

    if n_compressed == 0:
        route = "verbatim"
    elif n_compressed == len(msgs_a):
        route = "lingua"
    else:
        route = "hybrid"
    return verbatim, comp_in, comp_out, route


def verbatim_split_by_item(
    msgs_a_by_id: dict[str, list[dict[str, Any]]],
    msgs_b_by_id: dict[str, list[dict[str, Any]]],
    *,
    threshold: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-item verbatim/non-verbatim token split for every item present in
    both leg maps. Returns ``{item_id: {lx_* fields}}`` (items that cannot be
    aligned are omitted)."""
    if threshold is None:
        threshold = _sidecar_threshold()
    out: dict[str, dict[str, Any]] = {}
    for item_id in msgs_a_by_id:
        if item_id not in msgs_b_by_id:
            continue
        split = _verbatim_split_item(
            msgs_a_by_id[item_id], msgs_b_by_id[item_id], threshold=threshold
        )
        if split is None:
            continue
        v, c_in, c_out, route = split
        out[item_id] = {
            "lx_verbatim_tokens": v,
            "lx_compressed_in_tokens": c_in,
            "lx_compressed_out_tokens": c_out,
            "lx_route": route,
        }
    return out


def aggregate_verbatim_metrics(
    per_item: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate per-item splits into the figures the report needs. Returns
    ``None`` when there is nothing to analyse."""
    if not per_item:
        return None

    tot_v = sum(d["lx_verbatim_tokens"] for d in per_item.values())
    tot_in = sum(d["lx_compressed_in_tokens"] for d in per_item.values())
    tot_out = sum(d["lx_compressed_out_tokens"] for d in per_item.values())
    n = len(per_item)

    route_counts = {"verbatim": 0, "lingua": 0, "hybrid": 0}
    for d in per_item.values():
        route_counts[d["lx_route"]] = route_counts.get(d["lx_route"], 0) + 1

    layer8_in = tot_v + tot_in
    layer8_out = tot_v + tot_out
    verbatim_share = tot_v / layer8_in if layer8_in else 0.0
    nv_ratio = (tot_out / tot_in) if tot_in else None
    nv_savings = (1.0 - nv_ratio) if nv_ratio is not None else None
    overall_savings = (layer8_in - layer8_out) / layer8_in if layer8_in else 0.0

    return {
        "n_items": n,
        "verbatim_tokens": tot_v,
        "compressed_in_tokens": tot_in,
        "compressed_out_tokens": tot_out,
        "layer8_in_tokens": layer8_in,
        "layer8_out_tokens": layer8_out,
        "verbatim_share": verbatim_share,
        "nonverbatim_ratio": nv_ratio,
        "nonverbatim_savings": nv_savings,
        "overall_savings": overall_savings,
        "token_delta": tot_in - tot_out,
        "route_counts": route_counts,
        "avg_layer8_in": layer8_in / n,
        "avg_layer8_out": layer8_out / n,
        "avg_nv_in": tot_in / n,
        "avg_nv_out": tot_out / n,
    }


def write_by_item_dumps(
    out_dir: Path,
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    records_cb: list[dict[str, Any]],
    msgs_a_by_id: dict[str, list[dict[str, Any]]],
    msgs_b_by_id: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    """Write one deep-dive JSON file per benchmark item under ``out_dir``.

    Each file carries every field the flat JSONL holds for that item — the
    Leg A record (CR only), the Leg B record (CR + leanctx Layer 8, already
    enriched with the ``lx_*`` verbatim split), and the closed-book control
    when present — plus the **full message payloads** on either side of
    Layer 8 that the JSONL deliberately omits:

      * ``input_before_layer8``  — the CR-only output, i.e. exactly what
        leanctx saw as input inside Leg B (CR's 7 layers are deterministic).
      * ``output_after_layer8``  — the CR + leanctx Layer 8 output.

    Only items carrying an ``item_id`` are dumped (the agent warmup item has
    none and is not tracked in the message sinks). Returns the files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {r["item_id"]: r for r in records if r.get("item_id") is not None}

    a_by_id = _index(records_a)
    b_by_id = _index(records_b)
    cb_by_id = _index(records_cb)

    item_ids = (
        set(a_by_id) | set(b_by_id) | set(cb_by_id)
        | set(msgs_a_by_id) | set(msgs_b_by_id)
    )

    written: list[Path] = []
    for item_id in sorted(item_ids):
        rec_a = a_by_id.get(item_id)
        rec_b = b_by_id.get(item_id)
        dump = {
            "item_id": item_id,
            "workload": (rec_b or rec_a or {}).get("workload"),
            "leg_a": rec_a,
            "leg_b": rec_b,
            "leg_closed_book": cb_by_id.get(item_id),
            "input_before_layer8": msgs_a_by_id.get(item_id),
            "output_after_layer8": msgs_b_by_id.get(item_id),
        }
        # item_id is a dataset hash, but sanitize defensively for filesystem use.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(item_id))
        path = out_dir / f"{safe}.json"
        path.write_text(json.dumps(dump, indent=2, ensure_ascii=False))
        written.append(path)
    return written


def acc_by_route(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
) -> list[tuple[str, int, float, float]]:
    """Accuracy split by leanctx route — ``verbatim`` / ``lingua`` / ``hybrid``
    plus an ``overall`` row — as ``(label, N, acc_a, acc_b)``.

    Only Leg B carries ``lx_route``; each item's route is joined back onto the
    Leg A record by ``item_id`` so both legs are compared on the *same*
    partition. This is the lens that separates real compression effects (the
    ``lingua`` row, where leanctx actually rewrote the context) from eval noise
    on untouched items (the ``verbatim`` row, which — with the shared eval draw —
    is pinned to Δ = 0 by construction). Items lacking a route or an accuracy on
    either leg are skipped.
    """
    route_by_id = {
        r.get("item_id"): r.get("lx_route")
        for r in records_b
        if r.get("item_id") is not None and r.get("lx_route")
    }
    a_by_id = {r.get("item_id"): r for r in records_a if r.get("item_id") is not None}
    b_by_id = {r.get("item_id"): r for r in records_b if r.get("item_id") is not None}

    buckets: dict[str, list[tuple[bool, bool]]] = {}
    overall: list[tuple[bool, bool]] = []
    for item_id, route in route_by_id.items():
        ra, rb = a_by_id.get(item_id), b_by_id.get(item_id)
        if ra is None or rb is None:
            continue
        if ra.get("accuracy") is None or rb.get("accuracy") is None:
            continue
        pair = (bool(ra["accuracy"]), bool(rb["accuracy"]))
        buckets.setdefault(route, []).append(pair)
        overall.append(pair)

    def _row(label: str, pairs: list[tuple[bool, bool]]) -> tuple[str, int, float, float]:
        n = len(pairs)
        acc_a = sum(a for a, _ in pairs) / n if n else 0.0
        acc_b = sum(b for _, b in pairs) / n if n else 0.0
        return (label, n, acc_a, acc_b)

    rows = [
        _row(route, buckets[route])
        for route in ("verbatim", "lingua", "hybrid")
        if route in buckets
    ]
    if overall:
        rows.append(_row("overall", overall))
    return rows


def assert_route_reuse_invariant(records_b: list[dict[str, Any]]) -> None:
    """Make the ``verbatim`` row's *Δ = 0* mechanical, not coincidental.

    ``acc_by_route`` pins the verbatim row to Δ = 0 *only if* every
    verbatim-labelled item actually reused Leg A's eval draw — byte-identical
    prompt ⇒ same answer object ⇒ same accuracy. But the route label and the
    reuse decision are produced by two **independent** paths: the label comes
    from leanctx's ``classify()`` + threshold replay (``_verbatim_split_item``),
    while reuse fires on a byte-identical ``_build_eval_context`` (``:1284``).
    Nothing else cross-checks them, so the headline guarantee is otherwise just
    prose, contingent on the classifier replay always matching the sidecar.

    This consumes the ``eval_reused`` flag and asserts the two paths agree:
    every ``verbatim`` item was reused (Δ pinned to 0) and no ``lingua`` /
    ``hybrid`` item was (a compressed item must be scored on its own draw, not
    silently inherit Leg A's). It also catches the truncation false-hit the
    review flagged — a genuinely-compressed item whose change lands in the
    dropped middle would reuse on a ``lingua`` label and trip the second clause.

    Only Leg B records carry a route; items without a route or without an eval
    (``eval_reused`` unset) are outside ``acc_by_route``'s set and are skipped.
    Raises ``AssertionError`` on any divergence.
    """
    violations: list[str] = []
    for r in records_b:
        route = r.get("lx_route")
        if not route or r.get("eval_reused") is None:
            continue
        reused = bool(r["eval_reused"])
        item_id = r.get("item_id")
        if route == "verbatim" and not reused:
            violations.append(
                f"  {item_id}: route=verbatim but eval_reused=False "
                "→ Δ not pinned to 0 (verbatim row would carry decoder noise)"
            )
        elif route in ("lingua", "hybrid") and reused:
            violations.append(
                f"  {item_id}: route={route} but eval_reused=True "
                "→ compressed item silently scored as a no-op (false cache hit)"
            )
    if violations:
        raise AssertionError(
            "route/reuse invariant violated — 'verbatim Δ=0' is not mechanical:\n"
            + "\n".join(violations)
        )


def assert_overall_parity(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
) -> None:
    """Keep the two ``overall`` accuracy figures from silently diverging.

    Section 3's headline pools *every* scored item (accuracy not ``None``);
    ``acc_by_route``'s ``overall`` row pools only items that *also* carry a
    route, joined Leg A↔B by ``item_id``. On the current single-message LB
    workload these are the same set, so the two figures agree — but nothing
    structurally guarantees it. This asserts the sets coincide: Leg A and Leg B
    score the same item_ids, and every scored Leg B item carries a route. If
    they ever drift (e.g. a scored item without a route label), the headline and
    the by-route ``overall`` would quietly disagree; this fails loudly instead.

    Skipped entirely when no item carries a route (no by-route table to desync).
    Raises ``AssertionError`` on divergence.
    """
    a_ids = {r.get("item_id") for r in records_a if r.get("accuracy") is not None}
    b_ids = {r.get("item_id") for r in records_b if r.get("accuracy") is not None}
    routed = {
        r.get("item_id")
        for r in records_b
        if r.get("accuracy") is not None and r.get("lx_route")
    }
    if not routed:
        return  # no by-route 'overall' row exists, nothing to reconcile

    problems: list[str] = []
    if a_ids != b_ids:
        problems.append(
            f"  Leg A/B scored sets differ: A-only={sorted(a_ids - b_ids)}, "
            f"B-only={sorted(b_ids - a_ids)}"
        )
    if b_ids != routed:
        problems.append(
            f"  scored Leg B items missing a route: {sorted(b_ids - routed)}"
        )
    if problems:
        raise AssertionError(
            "headline vs by-route 'overall' set parity violated — the two "
            "'overall' accuracies are computed from different sets:\n"
            + "\n".join(problems)
        )


def validate_run_invariants(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
) -> None:
    """Methodology self-checks for a finished run. Raises ``AssertionError`` on
    violation.

    These used to live inside ``format_report``, but ``main`` renders the report
    *before* it persists the JSONL/report/dumps — so a raise there discarded a
    completed, fully-paid-for run with nothing written. ``main`` now calls this
    only *after* every artifact is on disk, so a violation is surfaced loudly
    (non-zero exit) without destroying the run's output. ``format_report`` is
    kept a pure formatter.

    Operates on the same scored (accuracy-bearing) Leg-A/Leg-B subsets the report
    pools, so it validates exactly what the report shows.
    """
    lb_a = [r for r in records_a if r.get("accuracy") is not None]
    lb_b = [r for r in records_b if r.get("accuracy") is not None]
    assert_route_reuse_invariant(lb_b)
    assert_overall_parity(lb_a, lb_b)


def align_metric_pools(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    *,
    exclude_workloads: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(all_a, all_b)`` filtered to the *shared* metric pool.

    An item is kept only if **both** legs produced it with ``tokens_compressed >
    0`` and its workload is not excluded. Filtering each leg independently could
    drop an item from one leg only — e.g. a leg compresses it to 0 tokens, or the
    sidecar fail-opens — which would (a) make ``compute_metrics``' ``delta_tokens``
    compare mismatched populations (skewing the savings gate) and (b) trip the
    headline-vs-by-route parity check. After ``exclude_workloads`` the survivors
    are LongBench items, which always carry an ``item_id``.
    """
    def _in_pool(r: dict[str, Any]) -> bool:
        return r["tokens_compressed"] > 0 and r.get("workload") not in exclude_workloads

    a_pool_ids = {r.get("item_id") for r in records_a if _in_pool(r)}
    b_pool_ids = {r.get("item_id") for r in records_b if _in_pool(r)}
    shared_ids = a_pool_ids & b_pool_ids
    all_a = [r for r in records_a if _in_pool(r) and r.get("item_id") in shared_ids]
    all_b = [r for r in records_b if _in_pool(r) and r.get("item_id") in shared_ids]
    return all_a, all_b
