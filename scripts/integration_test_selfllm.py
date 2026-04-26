"""SelfLLM cross-provider integration test.

v0.3 compatibility wrapper: delegates to the corresponding
`leanctx bench run selfllm-{anthropic,openai,gemini}` scenario in-process.

To run via the new CLI directly:

    leanctx bench run selfllm-anthropic --workload rag
    leanctx bench run selfllm-openai    --workload rag
    leanctx bench run selfllm-gemini    --workload rag

Required environment variables:

    ANTHROPIC_API_KEY                 (for --provider anthropic, the default)
    OPENAI_API_KEY                    (for --provider openai)
    GEMINI_API_KEY or GOOGLE_API_KEY  (for --provider gemini)

Legacy CLI shape preserved:

    python scripts/integration_test_selfllm.py
    python scripts/integration_test_selfllm.py --provider openai
    python scripts/integration_test_selfllm.py --provider gemini
"""

from __future__ import annotations

import argparse
import sys

from leanctx.bench import scenarios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider",
        choices=("anthropic", "openai", "gemini"),
        default="anthropic",
    )
    parser.add_argument(
        "--workload",
        default="rag",
        help="Workload selector (default: rag).",
    )
    args = parser.parse_args()

    scenario_name = f"selfllm-{args.provider}"
    info, runner = scenarios.get(scenario_name)
    try:
        record = runner(workload=args.workload)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 3

    print(f"=== leanctx bench run {scenario_name} --workload {args.workload} ===")
    print(f"  status:          {record.status}")
    print(f"  compression:     {record.compression_provider}/{record.compression_model}")
    print(f"  input  tokens:   {record.input_tokens}")
    print(f"  output tokens:   {record.output_tokens}")
    print(f"  tokens saved:    {record.tokens_saved}")
    print(f"  ratio:           {record.ratio:.1%}")
    print(f"  cost_usd:        {record.cost_usd}")
    print(f"  duration:        {record.duration_ms} ms")

    if record.status != "success":
        print(f"FAIL: {record.error}", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
