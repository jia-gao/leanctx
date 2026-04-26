"""End-to-end integration test: leanctx.Anthropic wrapper with respx-mocked
Anthropic API.

v0.3 compatibility wrapper: delegates to `leanctx bench run anthropic-e2e`
in-process. To run via the new CLI directly:

    leanctx bench run anthropic-e2e --workload rag

Requires: ``pip install 'leanctx[anthropic,bench]'`` (and optionally
[lingua] when middleware mode=on routes prose to Lingua, which the
anthropic-e2e scenario does at threshold_tokens=100).
"""

from __future__ import annotations

import sys

from leanctx.bench import scenarios


def main() -> int:
    info, runner = scenarios.get("anthropic-e2e")
    record = runner(workload="rag")

    print("=== leanctx bench run anthropic-e2e --workload rag ===")
    print(f"  scenario:        {record.scenario}")
    print(f"  status:          {record.status}")
    print(f"  request:         {record.request_provider}/{record.request_model}")
    print(f"  compressor:      {record.compressor}")
    print(f"  input  tokens:   {record.input_tokens}")
    print(f"  output tokens:   {record.output_tokens}")
    print(f"  tokens saved:    {record.tokens_saved}")
    print(f"  ratio:           {record.ratio:.1%}")
    print(f"  cost_usd:        {record.cost_usd}")
    print(f"  duration:        {record.duration_ms} ms")

    if record.status != "success":
        print(f"FAIL: {record.error}", file=sys.stderr)
        return 1
    print("\nOK — v0.3 pipeline end-to-end works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
