# Draft: OpenTelemetry observability + benchmark harness for leanctx

**Status:** draft for `humanize:gen-plan`. Will be replaced by a structured plan.
**Target version:** v0.3
**Estimated rounds:** 4-6 in RLCR

---

## Goal

Make leanctx **production-observable** without forcing a dep, and ship a **reproducible benchmark harness** that turns the existing one-off integration scripts into a single CLI users (and our launch blog) can run.

Two threads, related enough to plan together:

1. **OTel exports** — emit OpenTelemetry metrics + spans for every compression
   call so users running Datadog / Honeycomb / Grafana / Prometheus see what
   leanctx is doing in their existing stack.
2. **`leanctx bench` CLI** — collapse `scripts/integration_test_*.py` into a
   single `leanctx bench --workload {rag,chat,agent}` command that emits
   structured JSON, optionally with the OTel pipeline live so users can
   verify telemetry shape before going to prod.

Both are additive. No existing behavior should change.

---

## Motivation

We have measured numbers in `docs/benchmarks/` but no easy way for a user to
verify the same numbers on their own workload, and no machine-readable output
they can pipe into their own tooling. Codex and the README both note this gap.

Production users also need:
- **Per-tenant attribution** — "tenant X saved Y tokens this month" → an OTel
  metric with a tenant-id attribute is the natural way.
- **Per-route attribution** — "the /support endpoint compressed 60%, /chat
  compressed 12% — why?" → spans with route attributes.
- **Cost tracking** — `leanctx_cost_usd` already exists in `CompressionStats`;
  exposing it via OTel makes finance dashboards trivial.

Existing telemetry mechanism (the `usage.leanctx_*` fields attached to provider
responses) is great for one-off checks but doesn't aggregate. OTel is the
aggregator.

---

## High-level approach

### Optional dependency, zero forced install

leanctx must keep its current install footprint. OTel is opt-in via a new
extra:

```bash
pip install 'leanctx[otel]'   # adds opentelemetry-api + sdk
```

Users who don't install `[otel]` see no behavior change. The integration is
a soft import: if `opentelemetry` isn't importable, every emit is a no-op.

### One emission point: Middleware

`Middleware.compress_messages` is the single chokepoint every compression
call flows through. Wrap it with an OTel span and emit metrics there. No
need to touch each Compressor.

```
client.messages.create()
   ↓
_Messages.create() — wraps in span "leanctx.client.create" (provider attr)
   ↓
Middleware.compress_messages() — wraps in span "leanctx.compress" with
                                  full stats attached on success
   ↓
classifier / router / compressor — child spans optional (configurable)
```

### Four counters + one histogram

| Metric | Type | Unit | Labels |
|---|---|---|---|
| `leanctx.compress.calls` | counter | requests | provider, mode, method, status |
| `leanctx.compress.input_tokens` | counter | tokens | provider, mode, method |
| `leanctx.compress.output_tokens` | counter | tokens | provider, mode, method |
| `leanctx.compress.cost_usd` | counter | usd | provider, mode, method |
| `leanctx.compress.duration_ms` | histogram | ms | provider, mode, method |

Tokens-saved is `input - output`, computed downstream from the two counters
to avoid double-bookkeeping.

### Span attributes

Per `compress_messages` call:

```
leanctx.mode             = "on" | "off" | "passthrough"
leanctx.method           = "verbatim" | "lingua" | "selfllm" | "hybrid" | ...
leanctx.input_tokens     = int
leanctx.output_tokens    = int
leanctx.ratio            = float
leanctx.cost_usd         = float
leanctx.message_count    = int (number of messages in span)
```

Plus a `leanctx.error` boolean + `error.type` / `error.message` if an
exception escapes. Keep cardinality bounded — no message content, no IDs.

### Configuration

Two layers:

1. **Standard OTel env vars work out of the box.** If
   `OTEL_EXPORTER_OTLP_ENDPOINT` is set when leanctx is imported,
   `[otel]` extra is installed, and `leanctx_config` doesn't explicitly
   disable observability, leanctx auto-attaches.
2. **Per-client override** in `leanctx_config`:
   ```python
   "observability": {
       "otel": True,                    # default: auto from env
       "service_name": "my-app",        # default: "leanctx"
       "child_spans": False,            # default: False — only top-level
       "extra_attributes": {"region": "us-west-2"},  # passed to all spans
   }
   ```

### Benchmark harness

New CLI entry point: `leanctx bench` (via `pyproject.toml [project.scripts]`).

```bash
leanctx bench --workload rag                    # default settings
leanctx bench --workload agent --runs 5         # multiple runs, statistics
leanctx bench --workload chat --provider openai --output result.json
leanctx bench --workload all --json             # ndjson stream of all
```

Workloads are versioned fixtures shipped with the package:
- `rag` — the existing 1.7 KB SRE-incident document
- `chat` — multi-turn conversation, ~3 KB across 8 turns
- `agent` — the 9-message coding-agent transcript from
  `scripts/integration_test_agent_workload.py`

Output (one record per run):
```json
{
  "workload": "agent",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "compressor": "lingua",
  "input_tokens": 2148,
  "output_tokens": 1384,
  "ratio": 0.644,
  "tokens_saved": 768,
  "duration_ms": 26400,
  "cost_usd": 0.0,
  "leanctx_version": "0.3.0",
  "lingua_model": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
  "timestamp": "2026-04-25T18:45:12Z"
}
```

When OTel is configured, `bench` also emits the same spans/metrics so users
can dry-run their telemetry pipeline against known fixtures before deploying.

---

## Detailed feature breakdown

### 1. `leanctx/observability/` package

```
leanctx/observability/
    __init__.py        # public surface: configure(), instrument()
    config.py          # ObservabilityConfig dataclass + env loading
    otel.py            # the actual OTel integration; imports gated
    metrics.py         # named-counter holders, lazy-init
    span.py            # span context manager helpers
```

The `otel.py` module is the only one that imports `opentelemetry`. Everything
else uses internal types, so disabling OTel just makes `otel.py` a no-op.

### 2. Middleware integration

In `compress_messages` (and `compress_messages_async`), wrap the body in:

```python
with start_compression_span(self._observability) as span:
    ...
    span.set_stats(stats)
```

`start_compression_span` returns a no-op context manager when observability is
disabled.

### 3. Bench harness

Files:
```
leanctx/bench/
    __init__.py
    cli.py             # argparse, --workload, --provider, --output
    workloads.py       # fixture data, dispatch by name
    runner.py          # the Run* logic — produce a BenchResult
    report.py          # JSON / table output formatters
```

Hook into `pyproject.toml`:
```toml
[project.scripts]
leanctx = "leanctx.bench.cli:main"
```

So `pip install leanctx` puts a `leanctx` binary on `$PATH`.

### 4. Tests

```
tests/observability/
    test_config.py         # env var parsing, dataclass defaults
    test_metrics.py        # counter/histogram update logic with fake recorder
    test_span_context.py   # span attributes set correctly
    test_no_otel_installed.py   # behavior when opentelemetry import fails

tests/bench/
    test_cli.py            # arg parsing, --json output shape
    test_workloads.py      # fixture sanity, deterministic results
    test_runner.py         # mocked Compressor: stats end up in BenchResult
```

### 5. Documentation

- `docs/observability.md` — full setup guide: install, env vars, what gets
  emitted, sample Grafana dashboard
- `README.md` — short "Observability" section pointing at the docs
- `docs/benchmarks/agent-workload.md` — update reproduction step to use
  `leanctx bench --workload agent` instead of the standalone script
- `docs/benchmarks/selfllm-providers.md` — same

---

## Acceptance criteria

The RLCR loop is done when ALL of:

1. `pip install leanctx` (no extras) → existing 170 tests still pass; no
   `opentelemetry` import; importing `leanctx` adds <50 ms to startup.
2. `pip install 'leanctx[otel]'` + `OTEL_EXPORTER_OTLP_ENDPOINT=...` →
   one `client.messages.create()` call produces:
   - One span named `leanctx.compress` with the documented attributes
   - 4 counters and 1 histogram updated by the right amounts
   Verified by an in-test recorder, no real OTel collector required.
3. `leanctx_config={"observability": {"otel": False}}` overrides the env var.
4. `leanctx bench --workload agent --output -` produces valid JSON matching
   the schema documented above. Same workload run twice produces identical
   `input_tokens` / structural fields (only timestamp + duration vary).
5. `leanctx bench --workload all --provider anthropic` runs all three
   workloads against the same provider and emits ndjson.
6. Bench CLI exits non-zero on any compression error; integration with our
   structural-integrity asserts (the agent-workload script's 5 invariants)
   is preserved.
7. `docs/observability.md` exists, covers env-var setup, includes a sample
   Grafana panel JSON.
8. CI passes: ruff clean, mypy strict on all new files, all tests green.

---

## Out of scope (explicitly v0.4+)

- Multi-tenant scoping (passing tenant_id per-request) — needs a request-
  context mechanism we don't have yet.
- Custom samplers / exemplars — defer to user's OTel SDK config.
- Distributed tracing across services — leanctx is a library, not a service;
  this is the user's job.
- Live dashboard / standalone monitor — bench writes JSON, that's enough.
- Prometheus pull-mode endpoint — the OTel exporter handles that route.
- Logs (vs metrics + traces). One obvious follow-up but adds another OTel
  signal type to maintain.

---

## Open questions for `gen-plan` to resolve

1. **Span naming convention** — `leanctx.compress` vs
   `compress.middleware.compress_messages`? Standard OTel naming is
   `<library>.<operation>`. Probably `leanctx.compress`.

2. **Default `service_name`** when user doesn't set it — `"leanctx"` or
   inherit from `OTEL_SERVICE_NAME` env? OTel's own convention says env wins.

3. **Cost-counter precision** — `cost_usd` is a float. OTel counters expect
   integers or doubles. Use `Decimal` internally and convert at emit time, or
   accept float-precision drift?

4. **Bench `--workload all`** semantics — sequential or parallel? Sequential
   is simpler and avoids OTel context confusion. Parallel adds another reason
   to skew compression numbers via cold-cache effects.

5. **Bench output schema versioning** — embed `"schema_version": 1` in JSON
   so future schema changes don't break downstream consumers.

6. **Should bench be installable separately** (e.g. `leanctx-bench` package)
   to avoid bloating leanctx's install? Probably no — it's tiny and uses
   leanctx code paths. But worth a 30s discussion.

---

## Risks

- **OTel SDK churn** — OTel Python SDK has had breaking changes between
  minor versions. Pin to a known-stable range (`>=1.30,<2.0`?) and add
  matrix CI for the lower + upper bound.
- **Cost-counter accuracy** — providers price per-MT-tokens. We currently
  carry `cost_usd=0.0` for all but a few code paths. The OTel work shouldn't
  paper over the missing cost data; instead, document that `cost_usd` is best-
  effort and depends on `Compressor` reporting it.
- **Bench fixture rot** — the LLMLingua-2 model changes ratios over time.
  Embed a `lingua_model_revision` field in the bench output so historical
  comparisons remain meaningful.

---

## Files likely to change

- `leanctx/observability/` — new package, ~5 files
- `leanctx/bench/` — new package, ~4 files
- `leanctx/middleware.py` — add observability hook
- `leanctx/__init__.py` — re-export new public types
- `pyproject.toml` — `[project.scripts]` entry, new `[otel]` extra, mypy
  override for `opentelemetry.*`
- `tests/observability/` + `tests/bench/` — new test packages
- `docs/observability.md` — new
- `README.md` — new section
- `docs/benchmarks/*.md` — update reproduction instructions

Total estimated diff: ~1500 lines added, ~50 lines modified.
