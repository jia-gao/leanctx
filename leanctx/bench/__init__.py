"""leanctx bench — reproducible compression benchmarks.

A versioned-JSON CLI that wraps the existing integration_test_*.py
families as named scenarios. Users run::

    leanctx bench list                                  # registered scenarios
    leanctx bench run lingua-local --workload rag       # one offline run
    leanctx bench run agent-structural --workload agent # invariants enforcement
    leanctx bench run lingua-local --runs 5             # 5 runs, fresh state each

The package is imported lazily — no top-level imports that pull
optional extras (anthropic, openai, lingua, otel). Scenarios that need
extras / API keys are listed by ``leanctx bench list`` even when those
deps are missing; ``leanctx bench run`` reports a clear diagnostic
naming the missing extra or env var.
"""

from __future__ import annotations

from leanctx.bench.schema import BenchRecord, validate_record

__all__ = ["BenchRecord", "validate_record"]
