"""leanctx bench CLI entry point.

Two subcommands:

    leanctx bench list                          # show registered scenarios
    leanctx bench run <scenario> --workload <w> # run one scenario × workload

``list`` is dependency-free and works without any optional extras
installed; missing extras / API keys are surfaced as informational
markers next to the scenario name.

``run`` exit codes:
    0 — success
    1 — scenario reported status=failure (e.g., agent-structural invariant failed)
    2 — invalid arguments / unknown scenario / unknown workload
    3 — missing extra or API key (clear diagnostic, no traceback)
    4 — schema validation of the produced record failed
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Iterable
from typing import Any

from leanctx.bench import scenarios
from leanctx.bench.schema import BenchRecord, validate_record
from leanctx.bench.workloads import list_workloads


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leanctx bench",
        description="Reproducible compression benchmarks.",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_list = sub.add_parser("list", help="List registered scenarios.")
    p_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    p_run = sub.add_parser("run", help="Run a single scenario.")
    p_run.add_argument("scenario", help="Scenario name (see `leanctx bench list`).")
    p_run.add_argument(
        "--workload",
        default="rag",
        help="Workload selector (default: rag). One of: " + ", ".join(list_workloads()),
    )
    p_run.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1).")
    p_run.add_argument(
        "--output",
        default="-",
        help="Where to write JSON records (- = stdout, default).",
    )
    p_run.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Compression ratio (offline scenarios only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "run":
        return _cmd_run(args)
    return 2  # pragma: no cover


def _cmd_list(args: argparse.Namespace) -> int:
    rows = []
    for info in scenarios.list_scenarios():
        rows.append(
            {
                "name": info.name,
                "description": info.description,
                "required_extras": list(info.required_extras),
                "missing_extras": _missing_extras(info.required_extras),
                "required_env": list(info.required_env),
                "missing_env": [v for v in info.required_env if not os.environ.get(v)],
            }
        )

    if args.json:
        print(json.dumps({"scenarios": rows}, separators=(",", ":")))
        return 0

    for r in rows:
        marks = []
        if r["missing_extras"]:
            marks.append(f"missing extras: {','.join(r['missing_extras'])}")
        if r["missing_env"]:
            marks.append(f"missing env: {','.join(r['missing_env'])}")
        suffix = f"  [{'; '.join(marks)}]" if marks else ""
        print(f"  {r['name']:24s}  {r['description']}{suffix}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        info, runner = scenarios.get(args.scenario)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.workload not in list_workloads():
        print(
            f"error: unknown workload {args.workload!r}; known: {', '.join(list_workloads())}",
            file=sys.stderr,
        )
        return 2

    missing_extras = _missing_extras(info.required_extras)
    if missing_extras:
        print(
            f"error: scenario {info.name!r} requires the {','.join(missing_extras)} extra. "
            f"Install with: pip install 'leanctx[{','.join(missing_extras)}]'",
            file=sys.stderr,
        )
        return 3

    missing_env = [v for v in info.required_env if not os.environ.get(v)]
    if missing_env:
        print(
            f"error: scenario {info.name!r} requires environment variable(s) "
            f"{','.join(missing_env)}",
            file=sys.stderr,
        )
        return 3

    out_stream = _open_output(args.output)
    overall_status = 0
    try:
        opts: dict[str, Any] = {}
        if args.ratio is not None:
            opts["ratio"] = args.ratio
        for _i in range(max(1, int(args.runs))):
            try:
                record = runner(workload=args.workload, **opts)
            except RuntimeError as exc:
                # Runtime error from the runner (e.g., missing extra
                # caught at runner level). Treat as scenario-level failure.
                print(f"error: {exc}", file=sys.stderr)
                return 3
            if not isinstance(record, BenchRecord):
                print("error: runner did not return a BenchRecord", file=sys.stderr)
                return 4
            errors = validate_record(record.to_dict())
            if errors:
                print(f"error: record schema validation failed: {errors}", file=sys.stderr)
                return 4
            out_stream.write(record.to_json() + "\n")
            out_stream.flush()
            if record.status != "success":
                overall_status = 1
    finally:
        if out_stream is not sys.stdout:
            out_stream.close()
    return overall_status


def _open_output(path: str) -> Any:
    if path == "-" or path == "":
        return sys.stdout
    return open(path, "w", encoding="utf-8")


def _missing_extras(required: Iterable[str]) -> list[str]:
    """Return the extras whose import names aren't satisfied."""
    missing: list[str] = []
    for name in required:
        if not _extra_installed(name):
            missing.append(name)
    return missing


# Map extra name → import name to probe.
_EXTRA_PROBES: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google.genai",
    "lingua": "llmlingua",
    "tokens": "tiktoken",
    "bench": "respx",
    "otel": "opentelemetry",
}


def _extra_installed(name: str) -> bool:
    probe = _EXTRA_PROBES.get(name)
    if probe is None:
        return True  # unknown extra; assume satisfied
    try:
        importlib.import_module(probe)
        return True
    except ImportError:
        return False


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
