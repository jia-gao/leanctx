# OpenTelemetry observability + benchmark CLI for leanctx (v0.3)

## Goal Description

Add OpenTelemetry observability to leanctx as an **API-only, opt-in instrumentation layer** that emits spans and metrics for every compression call — both via the SDK wrappers (`leanctx.Anthropic` / `OpenAI` / `Gemini`) and via direct compressor calls (`Lingua().compress()` etc.) — without taking ownership of OTel SDK or exporter configuration.

Ship a `leanctx bench` CLI organized around **named scenarios** that mirror the existing integration script families (`lingua-local`, `anthropic-e2e`, `selfllm-{anthropic,openai,gemini}`, `agent-structural`). Bench preserves the structural-integrity assertions of the agent benchmark, separates offline (deterministic) from live-provider (network-bound) modes explicitly, and standardizes JSON output (`schema_version: "1"`) so downstream tooling can consume it.

Both threads are bundled into one v0.3 plan because the bench harness is the canonical demonstration of the OTel pipeline working end-to-end on real workloads.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: leanctx is an OTel-API-only library; never owns SDK or exporter configuration.
  - Positive Tests:
    - With `[otel]` extra installed and no app-side OTel SDK setup, `import leanctx` does not register a TracerProvider, MeterProvider, or any exporter; `opentelemetry.trace.get_tracer_provider()` returns the same default object before and after import.
    - With `[otel]` extra installed AND an app-side TracerProvider + MeterProvider configured BEFORE leanctx is imported, every wrapper call produces telemetry visible to the app's configured providers.
    - Setting `OTEL_EXPORTER_OTLP_ENDPOINT` alone (no SDK setup) results in zero exporter creations by leanctx.
  - Negative Tests:
    - leanctx imports must NOT call `set_tracer_provider`, `set_meter_provider`, `OTLPSpanExporter(...)`, `BatchSpanProcessor(...)`, or any exporter-class constructor.
    - Auto-attach behavior described as "if `OTEL_EXPORTER_OTLP_ENDPOINT` is set, leanctx auto-attaches" must NOT exist in v0.3 — the draft's claim is rejected.

- AC-2: Telemetry covers every wrapper request path; one wrapper-rooted span per call; nested middleware/compressor spans are children, not duplicates.
  - **Wrapper paths in scope (12)** — every path must emit exactly one root span named `leanctx.compress` with the documented attribute set, when called once with mode=on:
    1. `leanctx.Anthropic.messages.create()` sync
    2. `leanctx.Anthropic.messages.stream()` sync (returns the upstream's `MessageStreamManager`; span opens at `__enter__`, closes at `__exit__`)
    3. `leanctx.AsyncAnthropic.messages.create()` async
    4. `leanctx.AsyncAnthropic.messages.stream()` async (the existing `_AsyncStreamContextManager`; span opens at `__aenter__`, closes at `__aexit__`)
    5. `leanctx.OpenAI.chat.completions.create()` sync `stream=False`
    6. `leanctx.OpenAI.chat.completions.create()` sync `stream=True` (returns a `Stream` iterator; span lifetime documented in implementation note below)
    7. `leanctx.AsyncOpenAI.chat.completions.create()` async `stream=False`
    8. `leanctx.AsyncOpenAI.chat.completions.create()` async `stream=True` (returns an `AsyncStream`; span lifetime documented below)
    9. `leanctx.Gemini.models.generate_content()` sync
    10. `leanctx.Gemini.aio.models.generate_content()` async
    11. `leanctx.Gemini.models.generate_content_stream()` sync (returns an iterator; span lifetime documented below)
    12. `leanctx.Gemini.aio.models.generate_content_stream()` async (returns a coroutine that resolves to an async iterator; span lifetime documented below)
  - **Stream-path span lifetime contract** (resolves the "exactly one span" ambiguity for streaming):
    - **Upstream-SDK guarantees vs leanctx-wrapper guarantees** — the leanctx-side wrapper is responsible for span lifetime; we cannot assume the upstream SDK exposes a `close()` hook on every stream type.
      - **OpenAI** `Stream` / `AsyncStream` close on iterator exhaustion and expose `.close()` (verified: `openai-python/_streaming.py`). leanctx wraps these in a thin iterator that calls our `__exit__`/`__aexit__` from `__del__` and from `close()`.
      - **Gemini** `generate_content_stream` returns an iterator; the `google-genai` SDK does NOT uniformly expose a `close()` hook. leanctx's wrapper iterator owns its own finalizer (`__del__` calls span exit) for GC-time closure as a backstop.
      - **Anthropic** `messages.stream()` returns context managers (sync `MessageStreamManager`, async `AsyncMessageStreamManager`) — these are paths 2 and 4 below.
    - For context-manager paths (2, 4): span opens at `__(a)enter__`, closes at `__(a)exit__`. `duration_ms` = `__exit__` - `__enter__`.
    - For iterator paths (6, 8, 11, 12): leanctx wraps the upstream's iterator/async-iterator with a span-owning iterator. The span closes at the FIRST of: iterator exhaustion (`StopIteration` / `StopAsyncIteration`), explicit `.close()` call, or `__del__` finalization. `duration_ms` measures full stream lifetime (request submission to last chunk delivered or close).
    - Iteration-time exceptions MUST be captured as span events with `span.set_status(Status(StatusCode.ERROR))` and the span MUST be closed.
  - Positive Tests:
    - Each of the 12 paths verified individually: one root span named `leanctx.compress`, full attribute set populated.
    - Stream-path tests verify the span closes only after the last chunk is yielded.
  - Negative Tests:
    - A wrapper call where `leanctx_config.observability.otel = False` produces zero spans / zero metric updates.
    - A wrapper call must produce **exactly one root span** — see the parentage matrix in AC-6.
    - A streaming wrapper call where iteration is abandoned (iterator dropped without close) must still close the span when the iterator is GC'd.

- AC-3: Method-status taxonomy is exhaustive and observable.
  - Positive Tests: the `leanctx.method` span attribute takes exactly one of these documented values, each independently triggerable in tests:
    - `passthrough` — mode=off
    - `below-threshold` — mode=on but message tokens < trigger threshold
    - `empty` — empty message list
    - `opaque-bailout` — Gemini contents with non-text parts (function_call / image / inline_data)
    - `verbatim` — pipeline ran, all messages routed to Verbatim
    - `lingua` — pipeline ran, Lingua used
    - `selfllm` — pipeline ran, SelfLLM used
    - `hybrid` — pipeline ran, two or more of {Verbatim, Lingua, SelfLLM} used
  - Negative Tests:
    - No span is emitted with `leanctx.method` missing or set to a value outside the documented set.

- AC-4: Provider context propagates from wrappers to spans and metrics.
  - Positive Tests:
    - Every wrapper-originated span carries `leanctx.provider` ∈ {`anthropic`, `openai`, `gemini`}.
    - Every direct-compressor span (Lingua/SelfLLM/Verbatim called outside a wrapper) carries `leanctx.provider = none`.
    - The same provider value is set on the request-scoped metrics labels.
  - Negative Tests:
    - `leanctx.provider` is never an empty string or omitted; unknown sources resolve to `none`.

- AC-5: Cost (`cost_usd`) is preserved end-to-end through middleware aggregation; emitted exactly once per call path; never double-counted.
  - **Emission contract:** the `leanctx.compress.cost_usd` counter is incremented exactly once per top-level call path — at the **outermost** span in the parentage chain (see AC-6 matrix). Inner spans (e.g. middleware-internal or compressor-internal) attach `leanctx.cost_usd` as an attribute for traceability but DO NOT increment the counter again.
  - Positive Tests:
    - A SelfLLM call producing `cost_usd = X > 0` results in the `leanctx.compress.cost_usd` counter increasing by exactly `X` between two snapshots taken before and after the call. The outer span carries `leanctx.cost_usd = X`.
    - A hybrid call where SelfLLM contributes `Y` and another compressor contributes 0 reports the outer span's `leanctx.cost_usd = Y` and increments the counter by `Y`. Tests verify that `_aggregate` in `middleware.py` correctly sums constituent `cost_usd` from per-message stats (the v0.2 regression guard).
    - For Verbatim-only or Lingua-only calls, the outer span's `leanctx.cost_usd = 0.0`. Tests verify the counter's running total does NOT *change* across a Verbatim/Lingua call (no observable increment). Whether the implementation skips the `add(...)` call entirely or invokes `add(0.0)` is an internal detail and not part of this acceptance criterion.
  - Negative Tests:
    - The cost counter must NEVER show negative values.
    - **Regression guard for v0.2 cost-loss bug:** a test calls a Middleware path that uses SelfLLM for one message and Verbatim for another (hybrid). The outer span's `leanctx.cost_usd` MUST equal the SelfLLM's reported `cost_usd`. If `_aggregate` reverts to v0.2 behavior (dropping `cost_usd`), this test must fail.
    - A test that wraps a SelfLLM call in BOTH a wrapper span and a middleware span (the nested case) MUST observe exactly one counter increment, not two.

- AC-6: Span parentage is fully specified for every call shape; one outer span per shape; inner spans are children with parent set explicitly.
  - **Parentage matrix** (this is the source of truth — AC-2 and AC-5 both refer to this). The direct-compressor row covers BOTH sync `compress()` and async `compress_async()` entrypoints.

    | Call shape | Outermost span | Inner spans | Cost-counter increment site |
    |---|---|---|---|
    | Wrapper-routed (`leanctx.{Anthropic,OpenAI,Gemini}.X.create()` or `.stream()`) | `leanctx.compress` opened by wrapper | `leanctx.compressor.compress` per active compressor (child of wrapper span); middleware does NOT open a span (suppressed by depth counter) | wrapper span (outer) |
    | Direct middleware (`Middleware.compress_messages(...)` or `compress_messages_async(...)` called by user) | `leanctx.compress` opened by middleware | `leanctx.compressor.compress` per active compressor (child of middleware span) | middleware span (outer) |
    | Direct compressor — `Lingua()`, `SelfLLM()`, or `Verbatim()` `.compress(...)` / `.compress_async(...)` called by user with no leanctx span on the stack | `leanctx.compressor.compress` opened by compressor | — | compressor span (outer) |
    | Nested user-API mix (user calls e.g. `Lingua().compress(...)` inside a `Middleware.compress_messages(...)` they also invoked directly) | the OUTER `leanctx.compress` from middleware | inner `leanctx.compressor.compress` from the user-invoked Lingua becomes a child via the depth counter | outer middleware span |

    All three compressors (`Lingua`, `SelfLLM`, `Verbatim`) are instrumented uniformly; there is no Verbatim observability gap in v0.3. Span attribute ownership: provider/method/cost are owned by the **outermost** span in any given call shape. Inner compressor children carry only their own per-compressor attributes (`name`, `input_tokens`, `output_tokens`, `ratio`, `cost_usd`); they do NOT re-emit `leanctx.provider`. Cost counter and duration histogram are recorded once per call, on the outermost span's `__exit__`.

  - **Span-emission rule (asymmetric for compression_span vs compressor_span):**
    - A single `contextvars.ContextVar[int]` *depth counter* (default `0`) tracks how many leanctx span frames are currently open in this async/thread context. Both `compression_span` and `compressor_span` increment on enter and decrement on exit (always, in a `try/finally`).
    - `compression_span` (used by wrappers and middleware) emits a span ONLY when entered with depth `== 0`; when depth `> 0` it yields a no-op proxy that still flows stats up to the existing outermost span. This is what guarantees AC-2's "one root per call" and the AC-6 row 1 single-span shape.
    - `compressor_span` (used by direct compressor calls) emits a span on EVERY entry. When depth `== 0` it is a root; when depth `> 0` it is a child of the current OTel span context. This is what gives per-compressor granularity inside wrapper/middleware traces while still keeping direct compressor calls observable.
    - **Async-task propagation:** `contextvars` propagates correctly via `asyncio.create_task`, `asyncio.gather`, and `asyncio.to_thread` (which is the only thread-offload primitive leanctx actually uses today, in `Lingua.compress_async` and `SelfLLM.compress_async`). leanctx does NOT use `concurrent.futures.ThreadPoolExecutor.submit` or `loop.run_in_executor` directly, so we do not claim correctness for those — if a future change adopts them, the depth counter logic must be re-validated.
  - Positive Tests:
    - Direct `Lingua().compress(messages)` (sync): one root span `leanctx.compressor.compress`, attributes include compressor `name`, in/out tokens, ratio, cost; `leanctx.provider = none`.
    - Direct `await Lingua().compress_async(messages)` (async): same shape — root `leanctx.compressor.compress`, attributes populated, `leanctx.provider = none`. The depth counter must reset to 0 after `await` returns.
    - Direct `SelfLLM().compress(messages)` and `await SelfLLM().compress_async(messages)`: same shape as Lingua.
    - Direct `Verbatim().compress(messages)` and `await Verbatim().compress_async(messages)`: same shape; `leanctx.method = verbatim` on the compressor span.
    - Direct `Middleware.compress_messages(...)` (no wrapper): one root `leanctx.compress` span; child `leanctx.compressor.compress` spans for each compressor used.
    - Wrapper-routed call: one root `leanctx.compress` span (from wrapper); NO middleware-level span; child `leanctx.compressor.compress` spans whose `parent_span_id` equals the wrapper root's `span_id`.
    - Nested-depth test: two wrapper-routed calls in sequence (not concurrent) each produce their own complete span tree; the depth counter returns to 0 between them.
    - Concurrent async test: `asyncio.gather(client.messages.create(...), client.messages.create(...))` produces two independent root spans, no parent-child confusion between them.
    - Exception-unwind test: a wrapper-routed call where the upstream raises; the depth counter still returns to 0 (verified by a third sequential call producing a root span, not a child).
  - Negative Tests:
    - Direct-compressor spans never carry `leanctx.provider ∈ {anthropic, openai, gemini}` — only `none`.
    - No orphan: every `leanctx.compressor.compress` span called via wrapper or middleware has the corresponding `leanctx.compress` span as its parent in the same trace.
    - Wrapper-routed calls produce zero `leanctx.compress` spans from middleware (suppressed via depth counter).
    - A streaming wrapper that hasn't finished iterating must NOT have closed its outer span yet — verified by checking the depth counter at iteration mid-point.

- AC-7: `leanctx bench` covers the existing script families as named scenarios.
  - Positive Tests:
    - `leanctx bench list` outputs at least these scenarios: `lingua-local`, `anthropic-e2e`, `selfllm-anthropic`, `selfllm-openai`, `selfllm-gemini`, `agent-structural`.
    - `leanctx bench run lingua-local --workload rag` exits 0 with a single JSON record on stdout.
    - `leanctx bench run agent-structural --workload agent` runs the five structural-integrity invariants (tool_use_id linkage, code verbatim, error verbatim, tool_use input preservation, log compressed) and exits non-zero if any invariant fails.
    - `leanctx bench run selfllm-anthropic --workload rag` produces a valid JSON record when `ANTHROPIC_API_KEY` is set.
  - Negative Tests:
    - `leanctx bench run selfllm-anthropic` without `ANTHROPIC_API_KEY` exits non-zero with a message that includes the missing env var name (no stack trace).
    - A regression that causes Lingua to mutate code blocks must cause `leanctx bench run agent-structural` to exit non-zero with the failed invariant identified.

- AC-8: Bench JSON schema is versioned and dimensionally complete.
  - Positive Tests: every bench JSON record contains all of:
    - `schema_version` (string, currently `"1"`)
    - `leanctx_version` (string)
    - `scenario` (string, one of the registered scenario names)
    - `workload` (string, e.g. `rag`/`chat`/`agent`)
    - `status` (string, `success` | `failure`)
    - `request_provider`, `request_model` (strings; the wrapped SDK target — `null` when scenario doesn't make a wrapped SDK call)
    - `compression_provider`, `compression_model` (strings; the SelfLLM target — `null` when not SelfLLM)
    - `compressor` (string, primary compressor used)
    - `input_tokens`, `output_tokens`, `tokens_saved` (integers)
    - `ratio`, `cost_usd` (floats)
    - `duration_ms` (integer)
    - `warmup` (boolean — false for the run reported; true for an internal warmup pass that is excluded from comparison)
    - `timestamp` (ISO 8601)
    - `lingua_model_revision` (string, present when Lingua is the compressor)
  - Positive Tests:
    - For `agent-structural`, every record additionally includes an `invariants` object: `{tool_linkage: bool, code_verbatim: bool, error_verbatim: bool, tool_input_preserved: bool, log_compressed: bool}`.
  - Negative Tests:
    - A bench record with `status: success` that omits any required field MUST cause the bench CLI itself to fail validation and exit non-zero (even if the underlying scenario produced output).

- AC-9: Multi-run isolation; no cross-run state leakage.
  - Positive Tests:
    - `leanctx bench run <any> --runs 5` produces 5 records.
    - Each run constructs a fresh leanctx client and Middleware (so DedupStrategy state, Lingua model handle, etc. are isolated to the run).
    - 5 runs of identical input on `lingua-local` must NOT show monotonically decreasing `input_tokens` across runs (which would indicate cross-run dedup leak — a v0.2 regression we explicitly fixed and must keep fixed).
  - Negative Tests:
    - A test that injects shared dedup state across runs and asserts the bench CLI either resets it or refuses to run.

- AC-10: Documentation, packaging, and backward compatibility are correct.
  - Positive Tests:
    - `docs/observability.md` exists; covers (a) library-is-API-only contract, (b) supported `leanctx.method` values and meanings, (c) full attribute name list and types, (d) a sample app-side OTel SDK + OTLP exporter setup snippet that the user can copy-paste.
    - `docs/benchmarks/agent-workload.md` and `docs/benchmarks/selfllm-providers.md` are updated to use `leanctx bench` commands; old `python scripts/integration_test_*.py` invocations still work (kept as thin wrappers that delegate to bench scenarios).
    - `pyproject.toml` declares `[otel]` extra (`opentelemetry-api`, `opentelemetry-sdk`) and `[bench]` extra (already includes `respx`; gains any new deps the bench CLI needs).
    - `leanctx bench list` works with NO extras installed: outputs the full list of registered scenarios, marks each scenario's missing extras and missing API keys, exits 0 when listing is purely informational.
  - Negative Tests:
    - `pip install leanctx` (no extras) followed by `leanctx bench run lingua-local` must NOT crash with an opaque `ImportError`; must print a clear diagnostic naming the missing `[lingua]` extra (and `[bench]` if applicable) and exit non-zero.
    - `pip install 'leanctx[bench]'` (without `[lingua]`) followed by `leanctx bench run lingua-local` must print a clear diagnostic naming the missing `[lingua]` extra and exit non-zero.
    - `leanctx bench run selfllm-anthropic` without `ANTHROPIC_API_KEY` must print a clear diagnostic naming the missing env var and exit non-zero (no stack trace).

- AC-11: Existing tests + behavior unchanged; performance acceptable.
  - Positive Tests:
    - The full existing test suite (170 passed + 14 skipped) still passes in CI.
    - `usage.leanctx_method` / `leanctx_ratio` / `leanctx_tokens_saved` / `leanctx_cost_usd` attributes attached to provider responses still work (back-compat preserved; v0.3 spans are additive, do not replace this surface).
    - `ruff check`, `mypy` strict, and `ts-typecheck` CI jobs all green.
    - Cold-import time for `import leanctx` (no extras) regressed by ≤ 10 ms relative to v0.2 baseline (current: ~50 ms; budget: ≤ 60 ms).
  - Negative Tests:
    - Any commit that breaks an existing test must fail CI.
    - Any commit that imports `opentelemetry` from a base path (e.g. `leanctx/__init__.py`) without an inside-try guard must fail CI.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

A complete observability system spanning both layers — wrapper-side spans (one root per `client.X.create()` call), middleware-internal spans (deduplicated when nested), and direct-compressor child spans (`leanctx.compressor.compress` for `Lingua`/`SelfLLM`/`Verbatim`). All eight `leanctx.method` values supported and tested. Five OTel metrics (4 counters + 1 histogram) with provider/method/status labels. Full bench CLI: six scenarios (`lingua-local`, `anthropic-e2e`, `selfllm-{anthropic,openai,gemini}`, `agent-structural`), `--workload` selector, `--runs N` with isolation, `agent-structural` enforces the five structural-integrity invariants. Schema-versioned JSON output (`schema_version: "1"`). `docs/observability.md` covers the API-only contract, attribute reference, and an app-side OTel SDK setup snippet. `docs/benchmarks/*.md` updated to `leanctx bench` commands; existing `scripts/integration_test_*.py` retained as thin compatibility wrappers. README gains an Observability section. ~1500-2000 lines added; cold-import budget preserved at ≤ 60 ms.

### Lower Bound (Minimum Acceptable Scope)

The lower bound is the **same shape as the upper bound** — both layers of instrumentation, all six scenarios, all three workloads, all eleven ACs. v0.3 commits to upper-bound scope. The v1 plan tried to defer M5 (live-provider scenarios) and M6 (direct-compressor instrumentation); Codex correctly flagged that those deferrals contradict AC-6, AC-7, AC-10, and the v0.3 narrative ("bench is the canonical end-to-end OTel demo").

There are no acknowledged scope gaps inside upper-bound. All three compressors (`Lingua`, `SelfLLM`, `Verbatim`) are instrumented uniformly via `compressor_span` (M6/Phase B); all six bench scenarios ship; all eleven ACs are met.

If the v0.3 timeline turns out tighter than expected once the RLCR loop runs, the right move is **not** to partially ship some ACs — it's to **split observability and bench into separate releases** (v0.3 = observability core; v0.3.1 = bench CLI). Both halves stay AC-complete on their own. This option is captured as DEC-4.

### Allowed Choices

- **Can use:** `opentelemetry-api` (`>=1.30,<2.0`) + `opentelemetry-sdk` (same range) in the `[otel]` optional extra. `opentelemetry-api` is the only path used by leanctx code; `opentelemetry-sdk` is a *suggested* install for the user's app, never imported by leanctx.
- **Can use:** Python stdlib `argparse` for the bench CLI (no new dep, consistent with leanctx's existing zero-CLI-framework posture).
- **Can use:** existing internal helpers from `scripts/integration_test_*.py` as the source-of-truth implementation; bench scenario runners may import and call them directly to avoid logic duplication.
- **Can use:** `respx` (already in `[bench]`) for offline scenarios; `httpx` (transitive via `anthropic` / `openai`) elsewhere.
- **Cannot use:** any opentelemetry SDK construction (`TracerProvider(...)`, `MeterProvider(...)`, `BatchSpanProcessor(...)`, exporter classes) inside leanctx itself.
- **Cannot use:** alternative telemetry libraries (no `honeycomb-beeline`, no `datadog-api-client`, no `prometheus-client`). The OTel API is the only protocol.
- **Cannot use:** `click`, `typer`, or any other CLI framework that adds a runtime dep.
- **Cannot use:** `OTEL_*` env-var auto-detection at import time. The library does not own SDK lifecycle.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

```
leanctx/observability/
    __init__.py             # public surface: ObservabilityConfig, set_active_observability
    config.py               # ObservabilityConfig dataclass (off | on)
    api.py                  # internal: get_tracer(), get_meter(), get_counter()
                            # all wrappers around opentelemetry.{trace,metrics} APIs
                            # no-op when otel not available or config.otel=False
    method_status.py        # MethodStatus enum + attribute-set helpers
    middleware_hooks.py     # @contextmanager compression_span(config, provider)
    compressor_hooks.py     # @contextmanager compressor_span(name, parent)  # M6: Lingua + SelfLLM
    metrics.py              # named counter/histogram registration

leanctx/bench/
    __init__.py
    cli.py                  # argparse: leanctx bench {list,run}
    scenarios.py            # ScenarioRegistry + factory pattern
    schema.py               # BenchRecord dataclass; to_dict; validate
    runners/
        lingua_local.py
        anthropic_e2e.py
        selfllm_provider.py     # parameterized for anthropic|openai|gemini
        agent_structural.py     # asserts the five invariants
    workloads/              # bundled fixture data (committed to repo)
        rag.json
        chat.json
        agent.json
```

#### Wrapper-layer instrumentation (illustration)

```python
# leanctx/client.py — additive change to existing wrapper classes
class _Messages:
    def create(self, **kwargs):
        with compression_span(
            self._observability,
            provider="anthropic",
            method="(deferred)",  # filled in by middleware
        ) as span:
            messages = kwargs.get("messages", [])
            compressed, stats = self._middleware.compress_messages(messages)
            kwargs["messages"] = compressed
            response = self._upstream.messages.create(**kwargs)
            span.set_stats(stats)            # populates method, tokens, ratio, cost
            _attach_telemetry(response, stats)  # the existing usage.leanctx_* surface
            return response
```

The `compression_span` context manager:
- **No-ops** entirely when `[otel]` extra not installed or `config.otel = False`
- Calls `opentelemetry.trace.get_tracer("leanctx").start_as_current_span("leanctx.compress")` when active
- Records the four counters + histogram on `__exit__` from the captured stats

#### Cost aggregation fix

The current `_aggregate` helper in `middleware.py` (~line 230) drops `cost_usd`:

```python
# v0.2 — cost lost
def _aggregate(total_in, total_out, methods):
    ...
    return CompressionStats(input_tokens=total_in, ...)
```

v0.3 fix:

```python
def _aggregate(total_in, total_out, total_cost, methods):
    return CompressionStats(
        input_tokens=total_in, output_tokens=total_out,
        ratio=..., method=..., cost_usd=total_cost,
    )
```

Plus `compress_messages` accumulates `total_cost += stats.cost_usd` per inner compressor.

#### Span depth counter (asymmetric emission rule)

A single `contextvars.ContextVar[int]` (default `0`) tracks how many leanctx span frames are currently active in this async/thread context. Both `compression_span` and `compressor_span` increment on enter and decrement on exit (always), but their **emission rules differ**:

```python
# leanctx/observability/middleware_hooks.py — illustrative
_DEPTH: ContextVar[int] = ContextVar("leanctx_span_depth", default=0)

@contextmanager
def compression_span(observability, *, provider, ...):
    if not observability.enabled:
        yield _NoopSpan(); return
    parent_depth = _DEPTH.get()
    token = _DEPTH.set(parent_depth + 1)
    try:
        if parent_depth > 0:
            # Already inside a leanctx span — do not emit another. Stats
            # captured here flow up to the outermost frame via shared state
            # held by the outermost _LeanctxSpan instance.
            yield _PassthroughSpan()
        else:
            tracer = api.get_tracer()  # no-op tracer when API absent
            with tracer.start_as_current_span("leanctx.compress") as span:
                yield _LeanctxSpan(span, provider=provider, ...)
    finally:
        _DEPTH.reset(token)


# leanctx/observability/compressor_hooks.py — illustrative
@contextmanager
def compressor_span(observability, *, name, ...):
    if not observability.enabled:
        yield _NoopSpan(); return
    parent_depth = _DEPTH.get()
    token = _DEPTH.set(parent_depth + 1)
    try:
        # ALWAYS emit. When parent_depth > 0 this becomes a child of the
        # current OTel span context; when 0 it is a root.
        tracer = api.get_tracer()
        with tracer.start_as_current_span("leanctx.compressor.compress") as span:
            yield _LeanctxCompressorSpan(span, name=name, ...)
    finally:
        _DEPTH.reset(token)
```

Why this asymmetry:
- AC-2 mandates "one root span per wrapped call". Wrapper-routed calls go wrapper → middleware → compressor; if `compression_span` emitted a child for middleware, the trace would have two `leanctx.compress` spans for the same conceptual operation. Suppressing the inner `compression_span` keeps the trace clean.
- Per-compressor spans (`leanctx.compressor.compress`) are *additive* granularity, not duplicates. A hybrid call routes one prose message through Lingua and one tool message through SelfLLM; users want both compressor children visible under the wrapper root. Emitting them always gives that granularity.
- The counter (vs boolean) handles arbitrary nesting depth and exception unwind correctly via `try/finally` — and `contextvars` propagates correctly across `asyncio.create_task`, `asyncio.gather`, and `asyncio.to_thread`.

Metric emission is keyed off depth: counters and the histogram are recorded on `__exit__` of the *outermost* span only (depth-0 frame for the originating frame). Inner compressor children participate in the trace and carry their own per-compressor attributes but do NOT independently increment the global counters. This is what makes AC-5's "cost counter increments exactly once per call" mechanically observable.

#### Bench scenario runners

Each runner exposes `run(workload: str, **opts) -> BenchRecord`. The dispatcher in `scenarios.py` is a registry; new scenarios register via decorator.

```python
# leanctx/bench/runners/lingua_local.py
@register_scenario("lingua-local")
def run(workload: str, **opts) -> BenchRecord:
    fixture = load_workload(workload)
    lingua = Lingua(ratio=opts.get("ratio", 0.5))
    out, stats = lingua.compress(fixture.messages)
    return BenchRecord(
        scenario="lingua-local", workload=workload,
        compressor="lingua",
        request_provider=None, request_model=None,
        compression_provider=None, compression_model=None,
        ...stats fields...
        lingua_model_revision=lingua._prompt_compressor.model_name if lingua._prompt_compressor else None,
    )
```

#### `agent-structural` invariants

Reuses the assertion code from `scripts/integration_test_agent_workload.py`. The runner returns `BenchRecord(status="failure", invariants={...false...})` instead of raising — bench's outer driver decides exit code based on `status`. Existing standalone-script behavior preserved by keeping the script as a thin wrapper that runs the bench scenario in-process and propagates the exit code.

### Relevant References

- `leanctx/middleware.py` — central pipeline; site of most-instrumentation-needs
- `leanctx/_aggregate` (around line 230 in middleware.py) — cost-loss bug fix target for AC-5
- `leanctx/compressors/{lingua,selfllm,verbatim}.py` — direct-call surface for AC-6
- `leanctx/_gemini_adapter.py::contents_to_messages` — source of `opaque-bailout` status (AC-3)
- `leanctx/client.py::_Messages._AsyncStreamContextManager` — the async-stream pattern that AC-2 must extend
- `scripts/integration_test_*.py` — source-of-truth scenario logic the runners should reuse
- `docs/benchmarks/agent-workload.md` — structural-integrity contract (5 invariants) preserved by AC-7
- `tests/test_streaming.py` — existing streaming test harness; v0.3 instrumentation must not regress these
- `tests/test_middleware.py` — existing taxonomy tests for `passthrough`, `below-threshold`, `verbatim`, `hybrid`, `empty` (these are 5 of the 8 AC-3 values)

### v0.2 telemetry surface (preserved, additive)

The `usage.leanctx_*` fields attached to provider responses (set by `_attach_telemetry` in `client.py`) are kept as-is for backward compatibility. v0.3 OTel is **additive**: spans + metrics are emitted in addition to the response-attached attributes, not replacing them.

## Dependencies and Sequence

### Milestones

1. **M1: Observability core (foundation, no behavior change)**
   - Phase A: `leanctx/observability/` package skeleton with `ObservabilityConfig` dataclass
   - Phase B: API-only OTel access — `get_tracer()`, `get_meter()`, no-op when API absent
   - Phase C: `MethodStatus` enum (the eight values), attribute-set helpers
   - Phase D: Counter / histogram registration; named, labeled

2. **M2: Wrapper-layer instrumentation (the visible v0.3 feature)**
   - Phase A: `compression_span` context manager (`middleware_hooks.py`)
   - Phase B: Wrap `_Messages.create` / `_AsyncMessages.create` / `_AsyncMessages.stream` (Anthropic) — `provider="anthropic"`
   - Phase C: Wrap `_Completions.create` / `_AsyncCompletions.create` (OpenAI) — handles `stream=True`/`False`
   - Phase D: Wrap `_GeminiModels.generate_content` / `_GeminiAsyncModels.generate_content` and the stream variants — handles opaque-bailout status
   - Phase E: Middleware deduplication via contextvar — middleware-direct callers still observable, wrapper-routed calls not double-instrumented

3. **M3: Cost-aggregation fix (depends on M2)**
   - Phase A: Update `_aggregate` in middleware to thread `cost_usd`
   - Phase B: Tests pinning the AC-5 invariants (SelfLLM cost survives, hybrid sums constituent costs)
   - Phase C: Update `_attach_telemetry` to also attach `leanctx_cost_usd` (already in v0.2 stats but never on response — back-compat addition)

4. **M4: Bench CLI core (depends only on existing public APIs, parallelizable with M2)**
   - Phase A: `leanctx/bench/` package + argparse skeleton (`leanctx bench {list,run}`)
   - Phase B: `BenchRecord` dataclass + JSON `schema_version: "1"` validation
   - Phase C: Scenario registry with decorator-based registration
   - Phase D: Runners — `lingua-local`, `anthropic-e2e` (respx-mocked), `agent-structural` (with all 5 invariants)
   - Phase E: Multi-run isolation (fresh client/middleware per run)
   - Phase F: `--output -|<path>`, `--workload`, `--runs N` CLI flags

5. **M5: Live-provider scenarios (mandatory for v0.3)**
   - Phase A: `selfllm-anthropic` runner (real ANTHROPIC_API_KEY)
   - Phase B: `selfllm-openai` runner
   - Phase C: `selfllm-gemini` runner
   - Phase D: Missing-key error handling (clean error, exit non-zero, no stack trace)

6. **M6: Direct-compressor instrumentation (mandatory for v0.3)**
   - Phase A: `compressor_span` context manager (uses depth counter from M2; emission rule: always emit, child when nested, root when called directly)
   - Phase B: Wire into BOTH sync and async entrypoints of `Lingua`, `SelfLLM`, AND `Verbatim` (`compress` + `compress_async` for each)
   - Phase C: Tests for parentage matrix from AC-6 — all four rows; direct-compressor row covers sync + async × all three compressors; concurrent async, exception unwind, stream lifetime

7. **M7: Documentation + integration glue (depends on M2 + M4)**
   - Phase A: `docs/observability.md`
   - Phase B: README Observability section + supported-providers matrix update
   - Phase C: Update `docs/benchmarks/{agent-workload,selfllm-providers}.md` to use `leanctx bench` commands
   - Phase D: Convert `scripts/integration_test_*.py` to thin wrappers around bench scenarios

8. **M8: Packaging + CI (final)**
   - Phase A: Add `[otel]` extra (`opentelemetry-api`, `opentelemetry-sdk`)
   - Phase B: Update `[bench]` extra if M4/M5 added new deps
   - Phase C: pyproject mypy override for `opentelemetry.*`
   - Phase D: CI: import-time perf benchmark (≤ 60 ms) for AC-11

### Dependency graph

- M1 has no dependencies; produces the foundation everything else uses.
- M2 depends on M1.
- M3 is conceptually independent of M2 but tested through it; sequence after M2 lands.
- M4 depends only on existing public APIs (does not need M1/M2/M3 to start). Can run in parallel with M2.
- M5 depends on M4 (scenario registry) and on real API keys at test time.
- M6 depends on M1 (uses `compressor_hooks`) but should land last — it's the upper-bound add-on.
- M7 depends on M2 (so the docs match shipped behavior) and M4 (so bench commands in docs work).
- M8 depends on all of the above for `[otel]`/`[bench]` final extras and CI perf gate.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task01 | Create `leanctx/observability/` package with `__init__.py`, `config.py` (`ObservabilityConfig`) | AC-1 | coding | - |
| task02 | Implement API-only OTel access (`api.py`) — `get_tracer`, `get_meter`, no-op when API absent | AC-1 | coding | task01 |
| task03 | Add `MethodStatus` enum (8 values) + attribute-set helpers (`method_status.py`) | AC-3 | coding | task01 |
| task04 | Register named counters + histogram (`metrics.py`); labels: provider, method, status | AC-3, AC-4 | coding | task02, task03 |
| task05 | Implement `compression_span` context manager (`middleware_hooks.py`) using a `contextvars.ContextVar[int]` *depth counter* (not a boolean). Increment on `__enter__`, decrement in `try/finally` on `__exit__`. When entered with depth > 0, span builder yields a child of the current OTel context instead of opening a root span | AC-2, AC-6 | coding | task02, task03, task04 |
| task06 | Wire `compression_span` into `_Messages.create`, `_Messages.stream` (sync stream — context-manager based), and `_AsyncMessages.create` (Anthropic) with `provider="anthropic"` | AC-2, AC-4 | coding | task05 |
| task07 | Wire `compression_span` into `_AsyncMessages.stream` async-context-manager path; verify span lifetime per AC-2 stream contract | AC-2 | coding | task06 |
| task08 | Wire `compression_span` into `_Completions.create` and `_AsyncCompletions.create` (OpenAI), handling `stream=True/False` | AC-2, AC-4 | coding | task05 |
| task09 | Wire `compression_span` into `_GeminiModels.generate_content` + async + stream variants; emit `opaque-bailout` status when adapter bails out | AC-2, AC-3, AC-4 | coding | task05 |
| task10 | Update `Middleware.compress_messages` to use `compression_span` from task05; the depth-counter logic ensures it auto-becomes a child when wrapped, root when called directly. No middleware-specific dedup logic needed beyond what task05 provides | AC-2, AC-6 | coding | task05 |
| task11 | Fix `_aggregate` in middleware to thread `cost_usd`; sum constituent costs for hybrid | AC-5 | coding | task10 |
| task12 | Update `_attach_telemetry` to also attach `leanctx_cost_usd` to response.usage (back-compat additive) | AC-5 | coding | task11 |
| task13 | Tests: 12-path wrapper coverage — every wrapper × stream/no-stream × sync/async per AC-2's enumerated path list emits exactly one root span | AC-2 | coding | task07, task08, task09 |
| task14 | Tests: 8-status taxonomy — each `MethodStatus` value triggerable + observable in spans | AC-3 | coding | task09 |
| task15 | Tests: provider-context plumbing — wrapper-routed spans carry `leanctx.provider ∈ {anthropic, openai, gemini}`, direct-compressor spans (from task17) carry `none`, direct-middleware spans carry `none` | AC-4 | coding | task13, task17 |
| task16 | Tests: cost end-to-end per AC-5 — counter increments exactly once per call path (no double-counting wrapper+middleware), SelfLLM cost reaches counter via outermost span, hybrid sums constituent costs via task11's `_aggregate` fix, regression guard against v0.2 cost-loss bug | AC-5 | coding | task12 |
| task17 | `compressor_span` context manager (uses M2's depth counter; emission rule: always emit per `Span depth counter` section); wire into BOTH sync and async entrypoints of all three compressors — `Lingua.compress` + `Lingua.compress_async`, `SelfLLM.compress` + `SelfLLM.compress_async`, `Verbatim.compress` + `Verbatim.compress_async` | AC-4, AC-6 | coding | task05 |
| task18 | Tests: parentage matrix per AC-6 — all four matrix rows; for direct-compressor (row 3), pin BOTH sync (`Lingua().compress(...)`) and async (`await Lingua().compress_async(...)`) cases for each of Lingua/SelfLLM/Verbatim; concurrent async via `asyncio.gather`; exception unwind; stream lifetime per AC-2 stream contract | AC-2, AC-6 | coding | task17 |
| task19 | Create `leanctx/bench/` package skeleton with argparse CLI (`leanctx bench {list,run}`) | AC-7 | coding | - |
| task20 | Implement `BenchRecord` dataclass + `to_dict` + JSON schema (`schema_version: "1"`) + validation | AC-8 | coding | task19 |
| task21 | Scenario registry (decorator-based registration) | AC-7 | coding | task19 |
| task22 | Bundle workload fixtures (`rag.json`, `chat.json`, `agent.json`) — adapt from existing scripts | AC-7 | coding | task21 |
| task23 | Runner: `lingua-local` | AC-7, AC-8 | coding | task21, task22 |
| task24 | Runner: `anthropic-e2e` (respx-mocked, like existing e2e script) | AC-7, AC-8 | coding | task21, task22 |
| task25 | Runner: `agent-structural` — runs the 5 structural-integrity invariants, status=failure on any miss | AC-7, AC-8 | coding | task21, task22 |
| task26 | Runner: `selfllm-anthropic` — live API; clean error when key missing; reports distribution rather than asserting exact values | AC-7 | coding | task21, task22 |
| task27 | Runner: `selfllm-openai` (handles GPT-5/o-series reasoning_effort=minimal already in v0.2 SelfLLM) | AC-7 | coding | task26 |
| task28 | Runner: `selfllm-gemini` (handles 2.5+ thinking_budget=0 already in v0.2 SelfLLM) | AC-7 | coding | task26 |
| task29 | Multi-run isolation: bench `--runs N` constructs fresh client/middleware per run; tests | AC-9 | coding | task23, task24, task25 |
| task30 | Tests: bench CLI exit codes (success/failure), schema validation failure, missing-key handling, no-extras error | AC-7, AC-8, AC-10 | coding | task29 |
| task31 | Convert `scripts/integration_test_*.py` to thin wrappers that invoke `leanctx bench` scenarios in-process | AC-10 | coding | task29 |
| task32 | Write `docs/observability.md` — API-only contract, attribute reference, sample app-side OTel SDK setup | AC-10 | coding | task07, task08, task09 |
| task33 | Update `docs/benchmarks/{agent-workload,selfllm-providers}.md` to use `leanctx bench` commands | AC-10 | coding | task31 |
| task34 | Add Observability section to `README.md` (short, points at `docs/observability.md`) | AC-10 | coding | task32 |
| task35 | `pyproject.toml`: add `[otel]` extra (`opentelemetry-api`, `opentelemetry-sdk`); update `[bench]` if needed; mypy override for `opentelemetry.*` | AC-1, AC-10 | coding | task02 |
| task36 | CI: cold-import perf benchmark (≤ 60 ms) added to test matrix or as a separate job | AC-11 | coding | task02 |
| task37 | Final review: cardinality audit on span/metric attributes — no PII, no message content, no high-cardinality user data | AC-1, AC-3 | analyze | task13, task14, task15, task16 |
| task38 | Final review: `import leanctx` (no extras) does not import `opentelemetry`; verified by static check | AC-1 | analyze | task02, task05, task35 |

## Claude-Codex Deliberation

### Agreements

- API-only OTel instrumentation is the right library posture; leanctx never owns SDK/exporter setup.
- Telemetry method-status taxonomy must include all eight values: `passthrough`, `below-threshold`, `empty`, `opaque-bailout`, `verbatim`, `lingua`, `selfllm`, `hybrid`. The draft only listed three; Codex flagged five missing.
- The agent-structural benchmark's five invariants (tool_use_id linkage, code verbatim, error verbatim, tool_use input preservation, log compressed) must remain enforced and surface in JSON output.
- Bench JSON schema must distinguish **request** provider/model (the wrapped SDK target) from **compression** provider/model (the SelfLLM target). Codex flagged this ambiguity in the draft's example schema.
- Multi-run benchmarks must construct fresh client/middleware state per run — guards against the same DedupStrategy regression we just fixed in v0.2.
- Cost handling needs a middleware-level fix or telemetry will be wrong by construction. Draft incorrectly stated `cost_usd` "already exists in CompressionStats" — it does, but middleware aggregation drops it.

### Resolved Disagreements

- **Topic:** Auto-attaching from `OTEL_EXPORTER_OTLP_ENDPOINT`.
  - Claude (initial draft): "If `OTEL_EXPORTER_OTLP_ENDPOINT` is set when leanctx is imported, leanctx auto-attaches."
  - Codex: This is a bad library behavior — leanctx is not the process owner; auto-creating providers/exporters can conflict with app-level OTel setup, duplicate exports, and let benchmarks accidentally hit prod collectors.
  - **Resolution:** Adopted Codex's position. AC-1 explicitly forbids any SDK or exporter construction by leanctx. App owns SDK lifecycle. Documented in `docs/observability.md`.

- **Topic:** Bench's primary axis (workload vs scenario).
  - Claude (initial draft): "`leanctx bench --workload {rag,chat,agent}`" — workload as the primary axis.
  - Codex: The repo's existing scripts are split by integration mode + compressor (lingua-local, anthropic-e2e, selfllm-anthropic, agent-structural), not by workload. A workload-only CLI loses the dimensional coverage that makes the existing scripts valuable.
  - **Resolution:** Scenarios are the primary axis (`leanctx bench run <scenario>`); workload is a within-scenario selector (`--workload rag`). AC-7 and AC-8 reflect this.

- **Topic:** Single-emission point at middleware.
  - Claude (initial draft): "One emission point: Middleware. No need to touch each Compressor."
  - Codex: `Lingua`, `SelfLLM`, and `Verbatim` are exported public APIs and are called directly by the integration scripts. Middleware-only telemetry leaves direct-call usage uninstrumented. Plus Gemini opaque bailouts skip middleware entirely.
  - **Resolution:** Two-layer instrumentation. Wrapper layer (where most users are) provides root spans with provider context. Direct compressor calls also instrumented (AC-6) with `provider=none`. Middleware is the dedup point — when called from a wrapper, it skips its own span via contextvar to avoid double-counting; when called directly, it emits its own span. Gemini opaque bailout is observed with `method=opaque-bailout` (AC-3) so users can detect when their multimodal traffic is bailing out.

- **Topic:** Determinism claims for bench fixtures.
  - Claude (initial draft): "`test_workloads.py` claiming deterministic results."
  - Codex: Live-provider runs are non-deterministic by construction (no seed, no cached responses). You can test shape and required fields; you cannot promise determinism for `selfllm-*`.
  - **Resolution:** Bench has explicit offline (deterministic) and live-provider modes. Offline scenarios (`lingua-local`, `anthropic-e2e`, `agent-structural`) are deterministic for token counts and structural fields. Live scenarios (`selfllm-*`) report distributions, not exact values. AC-9 multi-run isolation applies to both; determinism guarantees apply only to offline.

- **Topic:** Cost-counter precision (Decimal vs float).
  - Claude (initial draft, open question): Use `Decimal` internally and convert to float at emit time, or accept float-precision drift?
  - Codex: OTel counters expect doubles; using `Decimal` adds complexity for sub-cent precision that doesn't matter for token-cost dashboards.
  - **Resolution:** Float throughout. Document expected precision in `docs/observability.md`.

- **Topic:** Bench as a separate package.
  - Claude (initial draft, open question): Should `bench` be a separate `leanctx-bench` PyPI package?
  - Codex: It's tiny and reuses leanctx's internal code paths; separating adds CI/release overhead without user value.
  - **Resolution:** Keep `bench` as a sub-package inside `leanctx`. Gated by the `[bench]` extra so users who don't run benchmarks don't pay the dependency cost.

- **Topic:** Span naming convention.
  - Claude (initial draft, open question): `leanctx.compress` vs `compress.middleware.compress_messages`?
  - Codex: OTel convention is `<library>.<operation>`. `leanctx.compress` is conventional and short.
  - **Resolution:** `leanctx.compress` for wrapper/middleware spans; `leanctx.compressor.compress` for direct-compressor spans (M6 / AC-6).

### Convergence Round 1 (Codex critique → Claude revision)

Codex's round-1 review found six required changes and three optional improvements. All six required changes are reflected in the plan above:

1. AC-2 wrapper-path count corrected from 11 to 12 (added Anthropic sync `messages.stream()`).
2. AC-2 stream-path span lifetime contract added (covers paths 6, 8, 11, 12 — the iterator/coroutine cases).
3. task05 dedup mechanism rewritten from boolean contextvar to `contextvars.ContextVar[int]` *depth counter* with explicit `try/finally` decrement; tests for nested, exception unwind, concurrent async, stream lifetime.
4. AC-6 added explicit parentage matrix (4 call-shape rows × 4 columns: call shape, outermost span, inner spans, cost-counter increment site); AC-2 references it.
5. AC-5 cost contract rewritten — emitted exactly once per call path at the outermost span; "increment by 0.0" softened to "running total does not change for Verbatim/Lingua"; explicit no-double-counting test.
6. M5 and M6 promoted from "upper-bound only" to mandatory for v0.3, resolving the AC-6/AC-7/AC-10 contradiction. DEC-1 and DEC-2 closed accordingly.

Optional improvements adopted:
- AC-10 `leanctx bench list` now explicitly works without extras (see AC-10 positive test).
- Stream-path `duration_ms` semantics defined explicitly (AC-2 stream contract — full stream lifetime, not setup time).
- Determinism language tightened in AC-9 (offline scenarios deterministic for token counts and structural fields; live scenarios report distributions).

### Convergence Round 2 (Codex round-2 critique → Claude revision)

Codex's round-2 review found six required changes; all six were applied:

1. task13 description aligned to "12-path" (matching AC-2's 12 enumerated wrapper paths).
2. AC-2 stream-lifetime contract rewritten to separate upstream-SDK guarantees from leanctx-wrapper obligations, with per-provider specifics for OpenAI's native `.close()`, Gemini's `__del__` backstop, and Anthropic's context-manager pattern.
3. AC-5 "increment by 0.0" softened to the observable assertion "running total does not change for Verbatim/Lingua".
4. AC-6 parentage matrix expanded to 4 rows (added direct-compressor row covering sync+async + nested user-API mix); corrected mistaken `concurrent.futures` claim to `asyncio.to_thread` (the only thread-offload primitive leanctx actually uses).
5. Lower-bound rewritten same-shape-as-upper-bound, eliminating the M5/M6 deferral contradiction surfaced by round-1.
6. Stale `Middleware-vs-wrapper deduplication` boolean-contextvar pseudocode replaced with the depth-counter design; package-tree comment about `compressor_hooks.py` being deferred fixed.

### Convergence Round 3 (Codex round-3 critique → Claude revision)

Codex's round-3 review surfaced three internal contradictions introduced by the round-2 edits; all three resolved:

1. **Wrapper-vs-middleware span shape:** AC-6 row 1 (single span) contradicted the depth-counter rationale (parent/child pair). Resolved in favor of single-span: `compression_span` emits ONLY at depth 0; nested entries become passthrough proxies that flow stats up. Documented as the "asymmetric emission rule" — `compression_span` suppresses nested emission, `compressor_span` always emits (root or child) for granular per-compressor visibility.
2. **Verbatim observability gap:** the matrix, AC-4, and upper-bound listed Verbatim as instrumented while task17 and the lower-bound deferred it. Resolved by wiring Verbatim too — all three compressors instrumented uniformly, no gap.
3. **Async direct-compressor unpinned:** AC-6 row 3 claimed both sync and async entrypoints, but task17 and the positive tests only covered sync. Resolved by adding async direct-compressor cases to task17 (BOTH `compress` and `compress_async` for all three compressors), task18 (matrix tests pin both), and AC-6 positive tests (explicit `await Lingua().compress_async(...)` case).

Optional improvements adopted: span attribute ownership clarified (provider/method/cost owned by outermost; inner compressor spans carry only per-compressor attributes); convergence-note "4 columns × 3 call shapes" corrected to "4 call-shape rows × 4 columns".

Optional improvements deferred to implementation: GC-abandonment test using explicit `gc.collect()` (CPython-scoped). Reasoning: the AC-2 stream-lifetime contract names this concern; the test author can decide between `__del__`-based and explicit-`gc.collect()`-based assertion at code time.

### Convergence Status

- Final Status: `converged` (after first-pass + round-1 + round-2 + round-3, plus user resolution of DEC-3 / DEC-4 / DEC-5). All Claude-Codex disagreements are resolved; all five product decisions are RESOLVED; plan is ready to enter the implementation loop.

## Pending User Decisions

- DEC-1: ~~Direct-compressor instrumentation in v0.3~~ **RESOLVED** — committed to v0.3 (upper-bound). Codex's round-1 critique correctly flagged that lower-bound (deferring M6) contradicts AC-6, AC-7, AC-10. Plan now mandates M6.

- DEC-2: ~~Live-provider scenarios in v0.3~~ **RESOLVED** — committed to v0.3 (upper-bound). Codex's round-1 critique flagged the same scope contradiction. Plan now mandates M5. Live-provider runs report distributions rather than assert exact values; reflected in AC-7 and AC-9.

- DEC-3: ~~Per-tenant attribution as v0.3 hard requirement~~ **RESOLVED** — deferred to v0.4. v0.3 does not ship a tenant request-context mechanism or cardinality cap. Rationale: the per-tenant motivation in the original draft is real but not load-bearing for any current design partner; v0.4 will design `contextvars.ContextVar[str | None]` for tenant_id + a configurable `max_tenants` cap (with overflow bucketed as `tenant=__overflow__`) properly. v0.3 spans/metrics carry provider/method/status only.

- DEC-4: ~~Bench CLI scope — full v0.3 vs split~~ **RESOLVED** — bundled in v0.3. Observability and bench ship together. The bench CLI is the canonical end-to-end OTel demo; shipping observability alone would leave users without a reproducible way to verify their telemetry shape on known fixtures. The split-release fallback (v0.3 = obs, v0.3.1 = bench) remains the documented contingency in the lower-bound section if the RLCR loop runs hot.

- DEC-5: ~~Workload set at v0.3 (rag + chat + agent vs agent-only)~~ **RESOLVED** — bundle all three workloads. AC-7, AC-8, M4/Phase D, and task22 already reflect this. Total fixture cost is ~50 extra lines; the launch blog needs RAG and chat-style data to land beyond agent traffic.

All five Pending User Decisions are now resolved. No open product questions remain.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

--- Original Design Draft Start ---

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

--- Original Design Draft End ---
