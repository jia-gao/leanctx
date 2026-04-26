"""Coding-agent workload integration test (5 structural-integrity invariants).

v0.3 compatibility wrapper: delegates to `leanctx bench run agent-structural`
in-process. To run via the new CLI directly:

    leanctx bench run agent-structural --workload agent

The agent-structural scenario enforces all 5 structural invariants
(tool_use_id linkage, code verbatim, error verbatim, tool_use input
preserved, log compressed) and exits non-zero if any fails.

Requires: ``pip install 'leanctx[lingua]'``.
"""

from __future__ import annotations

import sys

from leanctx.bench import scenarios


def main() -> int:
    info, runner = scenarios.get("agent-structural")
    record = runner(workload="agent")

    print("=== leanctx bench run agent-structural --workload agent ===")
    print(f"  status:          {record.status}")
    print(f"  compressor:      {record.compressor}")
    print(f"  input  tokens:   {record.input_tokens}")
    print(f"  output tokens:   {record.output_tokens}")
    print(f"  tokens saved:    {record.tokens_saved}")
    print(f"  ratio:           {record.ratio:.1%}")
    print(f"  duration:        {record.duration_ms} ms")
    if record.invariants is not None:
        print("  invariants:")
        for k, v in record.invariants.items():
            mark = "✓" if v else "✗"
            print(f"    {mark} {k}")

    if record.status != "success":
        print(f"FAIL: {record.error}", file=sys.stderr)
        return 1
    print("\nOK — all 5 structural invariants preserved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
