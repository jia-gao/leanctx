"""Local integration test for the Lingua compressor.

v0.3 compatibility wrapper: delegates to `leanctx bench run lingua-local`
in-process so the legacy invocation continues to work but the actual
logic lives in the bench scenario. To run via the new CLI directly:

    leanctx bench run lingua-local --workload rag

Requires: ``pip install 'leanctx[lingua]'``.
"""

from __future__ import annotations

import sys

from leanctx.bench import scenarios


def main() -> int:
    info, runner = scenarios.get("lingua-local")
    record = runner(workload="rag")

    print("=== leanctx bench run lingua-local --workload rag ===")
    print(f"  scenario:      {record.scenario}")
    print(f"  workload:      {record.workload}")
    print(f"  status:        {record.status}")
    print(f"  method:        {record.compressor}")
    print(f"  input  tokens: {record.input_tokens}")
    print(f"  output tokens: {record.output_tokens}")
    print(f"  ratio:         {record.ratio:.1%}")
    print(f"  tokens saved:  {record.tokens_saved}")
    print(f"  duration:      {record.duration_ms} ms")
    if record.lingua_model_revision:
        print(f"  lingua model:  {record.lingua_model_revision}")

    if record.status != "success":
        print(f"FAIL: {record.error}", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
