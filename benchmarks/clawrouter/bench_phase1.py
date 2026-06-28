"""Phase 1 go/no-go benchmark harness for ClawRouter × leanctx integration.

Architecture:
  Phase A: clone + patch + build ClawRouter at pinned commit
  Phase B: spawn leanctx sidecar + CR shim
  Phase C: two-leg run (Leg A = CR only, Leg B = CR + leanctx Layer 8)
  Phase D: score, apply gate, write report
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# ── Re-exported provider-agnostic harness core (benchmarks.common) ─────
# Extracted so the InsForge harness can reuse the scorer / LongBench /
# verbatim / report logic; imported here to preserve the historical
# ``benchmarks.clawrouter.bench_phase1`` import surface. See __all__.
from benchmarks.common.longbench import (
    _LB_ANS_RE_BARE,
    _LB_ANS_RE_PAREN,
    _LB_PROMPT_TEMPLATE,
    _build_lb_prompt,
    _extract_lb_answer,
    _lb_head_tail,
    _load_lb_items,
)
from benchmarks.common.report import format_report
from benchmarks.common.runner import (
    _build_eval_context,
    call_eval_llm,
)
from benchmarks.common.runner import run_leg as _runner_run_leg
from benchmarks.common.scoring import (
    GateResult,
    _percentile,
    apply_gate,
    compute_metrics,
    mcnemar_paired,
)
from benchmarks.common.tokens import _sum_tokens
from benchmarks.common.verbatim import (
    _SIDECAR_THRESHOLD_DEFAULT,
    _sidecar_threshold,
    _verbatim_split_item,
    acc_by_route,
    aggregate_verbatim_metrics,
    align_metric_pools,
    assert_overall_parity,
    assert_route_reuse_invariant,
    validate_run_invariants,
    verbatim_split_by_item,
    write_by_item_dumps,
)

# Names re-exported above for the historical bench_phase1 import surface.
__all__ = [
    "GateResult",
    "_LB_ANS_RE_BARE",
    "_LB_ANS_RE_PAREN",
    "_LB_PROMPT_TEMPLATE",
    "_SIDECAR_THRESHOLD_DEFAULT",
    "_build_lb_prompt",
    "_build_eval_context",
    "_extract_lb_answer",
    "_lb_head_tail",
    "_load_lb_items",
    "_percentile",
    "_sidecar_threshold",
    "_sum_tokens",
    "_verbatim_split_item",
    "acc_by_route",
    "aggregate_verbatim_metrics",
    "align_metric_pools",
    "apply_gate",
    "apply_patch",
    "assert_overall_parity",
    "assert_route_reuse_invariant",
    "call_eval_llm",
    "compress_via_shim",
    "compute_metrics",
    "format_report",
    "health_check",
    "mcnemar_paired",
    "run_leg",
    "setup_clawrouter",
    "spawn_cr_shim",
    "spawn_leanctx_sidecar",
    "validate_run_invariants",
    "verbatim_split_by_item",
    "wait_for_health",
    "write_by_item_dumps",
    "write_shim_file",
    "main",
]

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

# ── Repository + defaults ──────────────────────────────────────────────────

CR_REPO_URL = "https://github.com/BlockRunAI/ClawRouter.git"
CR_DEFAULT_COMMIT = "89269507b2173200222d03c1d4a0f80665b525d1"

# Productized Layer-8 connector — integrations/clawrouter/connector, published
# as `@leanctx/clawrouter-connector`. The benchmark builds + installs this into
# the cloned CR and imports it, instead of carrying its own hand-maintained copy
# of the Layer-8 hook (the former LAYER8_TS_BLOCK). Single source of truth.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONNECTOR_DIR = _REPO_ROOT / "integrations" / "clawrouter" / "connector"
_CONNECTOR_PKG = "@leanctx/clawrouter-connector"

# ── Layer 8 injection ──────────────────────────────────────────────────────
# The connector owns all the behaviour (env gate, eligibility, fetch, timeout,
# fail-open, one-in-one-out guard). The patch just imports it and calls it
# immediately before the anchor line
#   const compressedChars = calculateTotalChars(result);
# When LEANCTX_SIDECAR_URL is unset the connector is a no-op, so CR behaviour is
# unchanged — exactly as a real ClawRouter user would wire it (see the
# connector README).

_LAYER8_IMPORT = f'import {{ leanctxLayer8 }} from "{_CONNECTOR_PKG}";\n'

LAYER8_CALL_BLOCK = """\
  // ── Layer 8: leanctx LLMLingua-2 prose pass (opt-in, fail-open) ──────────
  // Productized connector (integrations/clawrouter/connector). No-op unless
  // LEANCTX_SIDECAR_URL is set; any error/timeout returns `result` unchanged.
  result = await leanctxLayer8(result);
  // ── end Layer 8 ──────────────────────────────────────────────────────────"""

# ── CR shim — Node.js ESM source written into workdir during setup ─────────

CR_SHIM_MJS = """\
import { createServer } from "http";
import { compressContext } from "./dist/compression/index.js";

const PORT = parseInt(process.env.CR_SHIM_PORT ?? "8461");

// CompressionConfig uses a nested `layers` object — flat keys are ignored.
const FULL_CONFIG = {
  layers: {
    deduplication: true,
    whitespace: true,
    dictionary: true,
    paths: true,
    jsonCompact: true,
    observation: true,
    dynamicCodebook: true,
  },
};

createServer(async (req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    res.writeHead(200).end("ok");
    return;
  }
  if (req.method !== "POST") { res.writeHead(405).end(); return; }
  let body = "";
  for await (const chunk of req) body += chunk;
  try {
    const { messages, config } = JSON.parse(body);
    const callerCfg = config ?? {};
    const mergedCfg = {
      ...FULL_CONFIG,
      ...callerCfg,
      layers: { ...FULL_CONFIG.layers, ...(callerCfg.layers ?? {}) },
    };
    const result = await compressContext(messages, mergedCfg);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(result));
  } catch (e) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: String(e) }));
  }
}).listen(PORT, () => console.log(`CR shim listening on :${PORT}`));
"""

# ═══════════════════════════════════════════════════════════════════════════
# Sprint 2 — Patch machinery
# ═══════════════════════════════════════════════════════════════════════════

_TS_ANCHOR = "  const compressedChars = calculateTotalChars(result);"
_TSUP_OLD = 'entry: ["src/index.ts", "src/cli.ts"],'
_TSUP_NEW = 'entry: ["src/index.ts", "src/cli.ts", "src/compression/index.ts"],'


def apply_patch(cr_dir: Path) -> None:
    _patch_compression_ts(cr_dir)
    _patch_tsup_config(cr_dir)


def _patch_compression_ts(cr_dir: Path) -> None:
    target = cr_dir / "src" / "compression" / "index.ts"
    src = target.read_text()
    if _TS_ANCHOR not in src:
        raise RuntimeError(
            f"Patch A anchor not found in {target}. "
            f"Expected: {_TS_ANCHOR!r}. Re-check --cr-commit."
        )
    if "leanctxLayer8" in src:
        return  # idempotent
    src = _LAYER8_IMPORT + src
    src = src.replace(_TS_ANCHOR, LAYER8_CALL_BLOCK + "\n" + _TS_ANCHOR)
    target.write_text(src)
    print(f"[patch A] Layer 8 connector wired into {target.relative_to(cr_dir)}")


def _patch_tsup_config(cr_dir: Path) -> None:
    target = cr_dir / "tsup.config.ts"
    src = target.read_text()
    if _TSUP_OLD not in src:
        if "src/compression/index.ts" in src:
            return  # idempotent
        raise RuntimeError(
            f"Patch B anchor not found in {target}. Re-check --cr-commit."
        )
    target.write_text(src.replace(_TSUP_OLD, _TSUP_NEW))
    print(f"[patch B] Compression entry added to {target.relative_to(cr_dir)}")


# ═══════════════════════════════════════════════════════════════════════════
# Sprint 3 — Shim file writer
# ═══════════════════════════════════════════════════════════════════════════


def write_shim_file(workdir: Path) -> Path:
    path = workdir / "cr_shim.mjs"
    path.write_text(CR_SHIM_MJS)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Sprint 5 — Service lifecycle
# ═══════════════════════════════════════════════════════════════════════════


def health_check(url: str) -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_health(
    proc: subprocess.Popen | None,
    url: str,
    timeout: float = 30,
) -> None:
    deadline = time.time() + timeout
    while not health_check(url):
        if time.time() > deadline:
            raise TimeoutError(f"{url} did not become healthy within {timeout}s")
        time.sleep(1)


def spawn_cr_shim(
    workdir: Path,
    port: int,
    extra_env: dict[str, str],
) -> subprocess.Popen:
    env = {**os.environ, "CR_SHIM_PORT": str(port), **extra_env}
    proc = subprocess.Popen(["node", str(workdir / "cr_shim.mjs")], env=env)
    wait_for_health(proc, f"http://127.0.0.1:{port}")
    return proc


def spawn_leanctx_sidecar(
    url: str,
    lingua_ratio: float,
    device: str = "auto",
) -> subprocess.Popen | None:
    if health_check(url):
        return None  # already running
    import sys as _sys

    # Resolve leanctx-serve from the same venv as the running interpreter.
    venv_bin = str(Path(_sys.executable).parent)
    # Pass the device through explicitly. An empty value lets the server
    # auto-detect; anything else (cuda / cpu / mps) is honored verbatim. The
    # server logs the *resolved* device at model load so the choice is
    # provable from logs, not assumed.
    device_env = "" if device == "auto" else device
    env = {
        **os.environ,
        "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
        "LEANCTX_SERVER_LINGUA_RATIO": str(lingua_ratio),
        "LEANCTX_SERVER_LINGUA_DEVICE": device_env,
    }
    print(f"[sidecar] LEANCTX_SERVER_LINGUA_DEVICE={device_env or '(auto)'}")
    proc = subprocess.Popen(["leanctx-serve"], env=env)
    # 120 s: includes one-time LLMLingua-2 weight download + warmup pass.
    wait_for_health(proc, url, timeout=120)
    return proc


# ═══════════════════════════════════════════════════════════════════════════
# Sprint 6 — CR shim integration helpers
# ═══════════════════════════════════════════════════════════════════════════


def install_connector(cr_dir: Path) -> None:
    """Build the productized Layer-8 connector and install it into the CR clone.

    The Layer-8 hook now lives in ``integrations/clawrouter/connector`` as
    ``@leanctx/clawrouter-connector``; the benchmark no longer carries its own
    copy. We build it (``tsc`` → ``dist``) and ``npm install`` it into the cloned
    CR so tsup can resolve the ``leanctxLayer8`` import when bundling Layer 8.
    """
    if not _CONNECTOR_DIR.exists():
        raise RuntimeError(
            f"Layer-8 connector not found at {_CONNECTOR_DIR}. "
            "Expected integrations/clawrouter/connector in the repo."
        )
    subprocess.run(["npm", "ci"], cwd=_CONNECTOR_DIR, check=True)
    subprocess.run(["npm", "run", "build"], cwd=_CONNECTOR_DIR, check=True)
    subprocess.run(["npm", "install", str(_CONNECTOR_DIR)], cwd=cr_dir, check=True)
    print(f"[connector] built + installed {_CONNECTOR_PKG} from {_CONNECTOR_DIR}")


def setup_clawrouter(workdir: Path, commit: str = CR_DEFAULT_COMMIT) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    if not (workdir / ".git").exists():
        subprocess.run(["git", "clone", CR_REPO_URL, str(workdir)], check=True)
    subprocess.run(["git", "checkout", commit], cwd=workdir, check=True)
    subprocess.run(["npm", "ci"], cwd=workdir, check=True)
    install_connector(workdir)
    apply_patch(workdir)
    subprocess.run(["npm", "run", "build"], cwd=workdir, check=True)


def compress_via_shim(messages: list[dict], shim_url: str) -> dict:
    # 120 s: allows for Layer 8 Lingua inference on CPU (~38 s for agent_extended).
    r = httpx.post(shim_url, json={"messages": messages}, timeout=120)
    r.raise_for_status()
    return r.json()


def run_leg(
    leg: str,
    items: list[dict[str, Any]],
    shim_url: str,
    lb_cfg: dict | None,
    *,
    closed_book: bool = False,
    msg_sink: dict[str, list[dict[str, Any]]] | None = None,
    reuse_eval: dict[str, dict[str, Any]] | None = None,
    eval_sink: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """ClawRouter leg runner — back-compat wrapper over the generic
    ``benchmarks.common.runner.run_leg``.

    Binds the CR-shim compressor (``compress_via_shim`` at ``shim_url``) and the
    module-level ``call_eval_llm`` (so the methodology tests can monkeypatch
    ``harness.compress_via_shim`` / ``harness.call_eval_llm``). The generic
    runner owns the scoring + shared-eval-draw logic; this keeps the historical
    ``run_leg(leg, items, shim_url, lb_cfg, …)`` signature.
    """
    def _compress(messages: list[dict]) -> dict[str, Any]:
        return compress_via_shim(messages, shim_url)

    return _runner_run_leg(
        leg,
        items,
        compress=_compress,
        lb_cfg=lb_cfg,
        eval_fn=call_eval_llm,
        closed_book=closed_book,
        msg_sink=msg_sink,
        reuse_eval=reuse_eval,
        eval_sink=eval_sink,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point (Sprint 8)
# ═══════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 1 harness. Returns 0 (PASS) or 1 (NO-GO)."""
    import argparse

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="ClawRouter × leanctx Phase 1 harness")
    parser.add_argument("--cr-commit", default=CR_DEFAULT_COMMIT)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/clawrouter_bench"))
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--dry-run-patch", action="store_true")
    parser.add_argument("--agent-stages", type=int, choices=[1, 2], default=1)
    parser.add_argument("--lb-stages", type=int, choices=[1, 2], default=1)
    parser.add_argument("--lb-n", type=int, default=5,
                        help="LongBench questions per stage (default: 5 for stage 1, 60 for full)")
    parser.add_argument("--fail-fast", action="store_true", default=True)
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8459")
    parser.add_argument("--shim-port", type=int, default=8461)
    parser.add_argument("--lingua-ratio", type=float, default=0.5)
    parser.add_argument("--lingua-device",
                        default=os.environ.get("LEANCTX_SERVER_LINGUA_DEVICE", "auto"),
                        help="Device for Layer 8 Lingua: auto|cuda|cpu|mps. "
                             "Passed to the sidecar; resolved device is logged.")
    parser.add_argument("--eval-provider",
                        default=os.environ.get("BENCHMARK_EVAL_PROVIDER", "anthropic"))
    parser.add_argument("--eval-model",
                        default=os.environ.get("BENCHMARK_EVAL_MODEL", "claude-sonnet-4-6"))
    parser.add_argument("--savings-threshold", type=float, default=0.20)
    parser.add_argument("--accuracy-drop", type=float, default=0.02)
    parser.add_argument("--sample-seed", type=int, default=1234,
                        help="Seed for random LongBench sampling (reproducibility).")
    parser.add_argument("--no-oversample-long", dest="oversample_long",
                        action="store_false", default=True,
                        help="Disable double-weighting of long-context LB items.")
    parser.add_argument("--closed-book", dest="closed_book", action="store_true",
                        default=True,
                        help="Run the no-context control leg (default: on).")
    parser.add_argument("--no-closed-book", dest="closed_book", action="store_false")
    parser.add_argument("--out", type=Path, default=Path("./phase1_results.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("./phase1_report.md"))
    parser.add_argument(
        "--by-item-dir", type=Path, default=None,
        help="Directory for per-item deep-dive dumps (full input before / output "
             "after Layer 8 + all JSONL fields). Default: <out dir>/by_item.",
    )
    args = parser.parse_args(argv)

    if args.dry_run_patch:
        print(_LAYER8_IMPORT + LAYER8_CALL_BLOCK)
        return 0

    # Token-count validation. Savings figures derive from leanctx.tokens, which
    # silently falls back to a len(text)//4 char estimate when tiktoken is
    # absent — which would make the savings a character ratio, not a token
    # ratio. Hard-fail here so the reported number is always a real token count.
    from leanctx.tokens import _load_tiktoken

    if _load_tiktoken() is None:
        print(
            "[fatal] tiktoken is not installed — token counts would be a "
            "char/4 estimate, not real tokens. Install with: "
            "pip install 'leanctx[tokens]'",
            flush=True,
        )
        return 2
    print("[tokens] tiktoken present — token counts are exact (cl100k_base).")

    workdir = args.workdir
    sidecar_url = args.sidecar_url
    eval_cfg: dict[str, Any] = {
        "provider": args.eval_provider,
        "model": args.eval_model,
        "max_tokens": 1024,
    }

    # ── Phase A: Setup ────────────────────────────────────────────────────
    already_built = (workdir / "dist" / "compression" / "index.js").exists()
    if not args.skip_setup and not already_built:
        print("[setup] Cloning + building ClawRouter…")
        setup_clawrouter(workdir, commit=args.cr_commit)
    write_shim_file(workdir)

    # ── Phase B: Spawn sidecar ────────────────────────────────────────────
    print(f"[sidecar] Checking {sidecar_url} …")
    sidecar_proc = spawn_leanctx_sidecar(
        sidecar_url, lingua_ratio=args.lingua_ratio, device=args.lingua_device
    )

    # ── Phase C: Leg A (CR only) ──────────────────────────────────────────
    print("[leg A] Starting CR shim without sidecar …")
    shim_a = spawn_cr_shim(workdir, port=args.shim_port, extra_env={})
    shim_a_url = f"http://127.0.0.1:{args.shim_port}"

    # Compressed-message sinks for the verbatim split (Phase B).
    msgs_a_by_id: dict[str, list[dict[str, Any]]] = {}
    msgs_b_by_id: dict[str, list[dict[str, Any]]] = {}
    # Leg A's per-item eval draws (context + answer + tokens). Leg B reuses an
    # entry whenever its eval context is identical, collapsing decoder noise on
    # verbatim-routed items to exactly zero.
    eval_a_by_id: dict[str, dict[str, Any]] = {}

    records_a: list[dict[str, Any]] = []
    try:
        from leanctx.bench.workloads import load_workload

        agent_msgs = load_workload("agent")
        agent_items_a = [{"messages": agent_msgs, "workload": "agent_s1"}]
        print("[leg A] Running agent_s1 …")
        records_a.extend(run_leg("A", agent_items_a, shim_a_url, lb_cfg=None))

        if args.lb_stages >= 1:
            n = args.lb_n
            print(f"[leg A] Loading {n} LongBench questions "
                  f"(seed={args.sample_seed}, oversample_long={args.oversample_long}) …")
            lb_items = _load_lb_items(
                limit=n,
                workload_tag="lb_s1",
                seed=args.sample_seed,
                oversample_long=args.oversample_long,
            )
            print(f"[leg A] Running lb_s1 ({n} items) …")
            records_a.extend(
                run_leg(
                    "A", lb_items, shim_a_url, lb_cfg=eval_cfg,
                    msg_sink=msgs_a_by_id, eval_sink=eval_a_by_id,
                )
            )
    finally:
        shim_a.terminate()
        shim_a.wait()

    # ── Phase C: Leg B (CR + leanctx) ────────────────────────────────────
    print("[leg B] Starting CR shim with sidecar …")
    shim_b = spawn_cr_shim(
        workdir,
        port=args.shim_port + 1,
        extra_env={"LEANCTX_SIDECAR_URL": sidecar_url},
    )
    shim_b_url = f"http://127.0.0.1:{args.shim_port + 1}"

    records_b: list[dict[str, Any]] = []
    try:
        agent_items_b = [{"messages": load_workload("agent"), "workload": "agent_s1"}]
        print("[leg B] Running agent_s1 …")
        records_b.extend(run_leg("B", agent_items_b, shim_b_url, lb_cfg=None))

        if args.lb_stages >= 1:
            # reuse same LB items from Leg A
            print(f"[leg B] Running lb_s1 ({len(lb_items)} items) …")
            records_b.extend(
                run_leg(
                    "B", lb_items, shim_b_url, lb_cfg=eval_cfg,
                    msg_sink=msgs_b_by_id, reuse_eval=eval_a_by_id,
                )
            )
    finally:
        shim_b.terminate()
        shim_b.wait()

    # ── Phase C': Leg C (closed-book control) ─────────────────────────────
    # No document at all — the model answers LongBench from priors. This is the
    # control that rules out "the context was irrelevant"; it needs no shim and
    # no sidecar, just the eval LLM. Runs over the same LB items as Leg A/B.
    records_cb: list[dict[str, Any]] = []
    if args.closed_book and args.lb_stages >= 1 and lb_items:
        print(f"[leg C] Closed-book control ({len(lb_items)} items, no context) …")
        records_cb.extend(
            run_leg("C", lb_items, shim_url="", lb_cfg=eval_cfg, closed_book=True)
        )

    # ── Phase D: Score + report ───────────────────────────────────────────
    if sidecar_proc is not None:
        sidecar_proc.terminate()
        sidecar_proc.wait()

    # agent_s1 is a synthetic 79-token stub that exercises the agent code path
    # only; keep it in the JSONL for debugging but exclude it from the metric
    # pools so it never lands in the savings-gate denominator or accuracy split.
    # (The per-workload table that used to surface it was already dropped.)
    _METRIC_EXCLUDE = {"agent_s1"}
    all_a, all_b = align_metric_pools(
        records_a, records_b, exclude_workloads=_METRIC_EXCLUDE
    )

    import datetime as _dt
    run_date = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metrics = compute_metrics(all_a, all_b, records_cb)
    gate = apply_gate(metrics, savings_threshold=args.savings_threshold,
                      accuracy_drop=args.accuracy_drop)

    # Verbatim-excluded compression view (Phase B): split each item's Layer-8
    # input into verbatim vs compressed tokens, then enrich the Leg B records
    # so the per-item split lands in the JSONL too.
    per_item_split = verbatim_split_by_item(msgs_a_by_id, msgs_b_by_id)
    verbatim_metrics = aggregate_verbatim_metrics(per_item_split)
    for rec in records_b:
        rec_id = rec.get("item_id")
        extra = per_item_split.get(rec_id) if rec_id is not None else None
        if extra:
            rec.update(extra)

    report_text = format_report(
        metrics, gate,
        records_a=all_a, records_b=all_b,
        run_date=run_date,
        eval_model=f"{args.eval_provider}/{args.eval_model}",
        cr_commit=args.cr_commit,
        savings_threshold=args.savings_threshold,
        accuracy_drop=args.accuracy_drop,
        verbatim_metrics=verbatim_metrics,
    )

    # Write JSONL
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rec in records_a + records_b + records_cb:
            f.write(json.dumps(rec) + "\n")

    # Write per-item deep-dive dumps (all JSONL fields + full input before /
    # output after Layer 8). records_b carries the lx_* split merged above.
    by_item_dir = args.by_item_dir or (args.out.parent / "by_item")
    item_files = write_by_item_dumps(
        by_item_dir, records_a, records_b, records_cb, msgs_a_by_id, msgs_b_by_id
    )

    # Write report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text)

    print(report_text)
    print(f"\n[out] JSONL → {args.out}")
    print(f"[out] report → {args.report}")
    print(f"[out] per-item dumps → {by_item_dir} ({len(item_files)} items)")

    # Methodology self-checks run LAST, after every artifact is on disk, so a
    # violation is surfaced loudly (exit 3) without discarding the run's output.
    try:
        validate_run_invariants(all_a, all_b)
    except AssertionError as exc:
        print(
            "\n[INVALID] methodology self-check failed — the run's metrics are "
            "untrustworthy, but all artifacts above were still written:\n"
            f"{exc}",
            flush=True,
        )
        return 3

    return 0 if gate.passed else 1


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
