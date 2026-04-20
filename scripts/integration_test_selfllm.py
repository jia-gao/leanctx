"""Integration test for SelfLLM — real API call to your configured provider.

Unlike Lingua (local model, free marginal cost), SelfLLM delegates
compression to an LLM API. This script exercises the real network
path with a realistic sample document.

Requires an API key in the environment:
    ANTHROPIC_API_KEY   (for --provider anthropic, the default)
    OPENAI_API_KEY      (for --provider openai)
    GEMINI_API_KEY      (for --provider gemini)

Cost per run: typically a fraction of a cent — SelfLLM defaults to the
cheapest tier per provider (Haiku / gpt-5-nano / gemini-2.5-flash).

Run with:

    .venv/bin/python scripts/integration_test_selfllm.py
    .venv/bin/python scripts/integration_test_selfllm.py --provider openai
    .venv/bin/python scripts/integration_test_selfllm.py --provider gemini
    .venv/bin/python scripts/integration_test_selfllm.py --ratio 0.2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from leanctx import SelfLLM

_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

SAMPLE_DOC = """
We're investigating a production incident that started at 14:32 UTC.
The symptoms reported by oncall: p99 latency on the /v1/messages
endpoint spiked from 450ms to 3.8s over a 4-minute window, and error
rate on the same endpoint climbed from 0.01% to 1.2%.

Logs from the inference fleet show a wave of 504s from the upstream
model server cluster in us-east-1 starting around 14:31. Two of the
four inference pods in that region went NotReady at 14:30:47, and
HPA didn't scale up until 14:33:15 because CPU was already under the
target threshold — the real bottleneck was GPU memory exhaustion,
which HPA can't see with the default metrics.

Our mitigation: we manually scaled the deployment to 8 replicas and
failed traffic over to us-west-2. Latency recovered within 90 seconds
of the failover. We haven't identified the root cause of the GPU
memory pressure yet, but the shape of the curve suggests a specific
request pattern exhausted KV cache on those pods — possibly a single
tenant sending very long contexts that the autoscaler didn't know to
bill against GPU budget.

Action items:
1. Add GPU memory utilization to the HPA custom metrics so scale-out
   actually tracks the real bottleneck.
2. Implement per-tenant KV cache budgets in the inference scheduler
   so a single long-context request can't starve the pod.
3. Write a retro doc and schedule a post-mortem for Thursday with
   platform, inference, and oncall teams.
4. File a followup ticket for the dashboard work — the default
   Grafana view didn't surface GPU memory, which is why the first
   oncall response pointed at the wrong subsystem.

The full timeline is reconstructed below with quotes from the oncall
channel so we can validate the sequence before Thursday.
""".strip()


def _get_api_key(provider: str) -> str:
    env_var = _ENV_VAR[provider]
    key = os.environ.get(env_var)
    if not key:
        print(
            f"ERROR: {env_var} not set. SelfLLM(provider='{provider}') "
            f"needs an API key to make a real compression call.",
            file=sys.stderr,
        )
        print(
            f"\nSet it temporarily for this run:\n"
            f"    export {env_var}=... && "
            f"python scripts/integration_test_selfllm.py --provider {provider}",
            file=sys.stderr,
        )
        sys.exit(2)
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=list(_ENV_VAR),
        help="Which LLM provider to use for compression. Default: anthropic.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.3,
        help="Target compression ratio suggested to the LLM (0.3 = aim for 30%%). Default: 0.3.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default cheap model per provider.",
    )
    parser.add_argument(
        "--max-summary-tokens",
        type=int,
        default=500,
        dest="max_summary_tokens",
        help="Ceiling on summary length. Default: 500.",
    )
    args = parser.parse_args()

    api_key = _get_api_key(args.provider)

    print("=== SelfLLM integration test ===\n")
    print(f"Provider:  {args.provider}")
    print(f"Ratio:     {args.ratio}")
    print(f"Max tokens out: {args.max_summary_tokens}\n")

    llm = SelfLLM(
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        ratio=args.ratio,
        max_summary_tokens=args.max_summary_tokens,
    )
    print(f"Resolved model: {llm.model}\n")

    messages = [{"role": "user", "content": SAMPLE_DOC}]
    print(f"Compressing sample document ({len(SAMPLE_DOC)} chars)...")
    t0 = time.monotonic()
    out, stats = llm.compress(messages)
    elapsed = time.monotonic() - t0
    print(f"  completed in {elapsed:.2f}s\n")

    print("=== Results ===")
    print(f"  method:        {stats.method}")
    print(f"  input tokens:  {stats.input_tokens}")
    print(f"  output tokens: {stats.output_tokens}")
    print(f"  ratio:         {stats.ratio:.1%}")
    print(f"  saved:         {stats.input_tokens - stats.output_tokens} tokens")

    print("\n=== Input (first 500 chars) ===")
    print(SAMPLE_DOC[:500] + ("..." if len(SAMPLE_DOC) > 500 else ""))

    print("\n=== Compressed output ===")
    print(out[0]["content"])

    # Sanity check
    assert stats.input_tokens > stats.output_tokens, (
        f"expected compression; got input={stats.input_tokens} "
        f"output={stats.output_tokens}"
    )
    assert stats.method == "selfllm"
    print("\nOK")


if __name__ == "__main__":
    main()
