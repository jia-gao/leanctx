"""Local integration test for the Lingua compressor.

Requires: ``pip install 'leanctx[lingua]'`` — pulls torch, transformers,
llmlingua. First run downloads ~1.2 GB of model weights to
``~/.cache/huggingface/``.

Run with:

    .venv/bin/python scripts/integration_test_lingua.py
"""

from __future__ import annotations

import time

from leanctx import Lingua

SAMPLE_DOC = """
Cloud-native architectures have become the dominant paradigm for deploying
modern applications. The shift from monolithic architectures to microservices
was driven by the need for scalability, resilience, and independent deployment
cycles. Organizations adopting Kubernetes as their container orchestration
platform report increased developer productivity, faster time to market, and
improved infrastructure utilization. However, the transition is not without
challenges — teams must invest in observability, service mesh technologies,
and developer tooling to manage the growing complexity.

Distributed tracing provides visibility into request flows across services,
helping teams debug issues that span multiple components. Metrics collection
via Prometheus and visualization via Grafana have become standard practice
across the industry. Log aggregation through the ELK stack or Loki rounds
out the observability picture. For teams running LLM workloads on Kubernetes,
additional specialized tooling is required: GPU scheduling, model serving
frameworks like vLLM or SGLang, and inference gateways that handle caching,
routing, and cost attribution across multiple tenants.
""".strip() * 3


def main() -> None:
    print("Constructing Lingua (model load on first use)...")
    t0 = time.monotonic()
    lingua = Lingua(ratio=0.5)
    lingua._load()
    load_time = time.monotonic() - t0
    print(f"  model loaded in {load_time:.1f}s")

    print("\nCompressing sample document...")
    t0 = time.monotonic()
    messages = [{"role": "user", "content": SAMPLE_DOC}]
    compressed, stats = lingua.compress(messages)
    elapsed = time.monotonic() - t0
    print(f"  compressed in {elapsed:.2f}s")

    print("\n=== Results ===")
    print(f"  method:        {stats.method}")
    print(f"  input  tokens: {stats.input_tokens}")
    print(f"  output tokens: {stats.output_tokens}")
    print(f"  ratio:         {stats.ratio:.1%}")
    print(f"  saved:         {stats.input_tokens - stats.output_tokens} tokens")

    print("\n=== Input (first 400 chars) ===")
    print(messages[0]["content"][:400] + ("..." if len(messages[0]["content"]) > 400 else ""))
    print("\n=== Compressed output (first 400 chars) ===")
    print(compressed[0]["content"][:400] + ("..." if len(compressed[0]["content"]) > 400 else ""))

    assert stats.input_tokens > stats.output_tokens, (
        f"expected compression; got input={stats.input_tokens}, output={stats.output_tokens}"
    )
    assert stats.ratio < 1.0, f"expected ratio < 1; got {stats.ratio}"
    print("\nOK")


if __name__ == "__main__":
    main()
