# Observability

leanctx ships an opt-in OpenTelemetry instrumentation layer that emits spans and metrics for every compression call. The library is **API-only**: it never owns the OTel SDK, never registers a `TracerProvider` or `MeterProvider`, and never instantiates an exporter. The application is the sole owner of OTel SDK lifecycle.

## Quick start

Install the `[otel]` extra and configure the OTel SDK in your application before importing leanctx:

```bash
pip install 'leanctx[otel,anthropic]'
```

```python
# application bootstrap (any standard OTel setup works)
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="https://otel-collector.example.com/v1/traces"))
)
metrics.set_meter_provider(
    MeterProvider(metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())])
)

# now import leanctx and enable observability via the client config
import leanctx

client = leanctx.Anthropic(
    leanctx_config={
        "mode": "on",
        "observability": {"otel": True},
    }
)
```

That's it. Every wrapper request path emits a `leanctx.compress` span and increments four counters + one histogram.

## Library posture (AC-1)

leanctx never:
- calls `trace.set_tracer_provider(...)` or `metrics.set_meter_provider(...)`
- instantiates any exporter class (OTLP, console, in-memory, anything)
- creates a `BatchSpanProcessor` or any other span processor

leanctx only:
- calls `trace.get_tracer("leanctx")` and `metrics.get_meter("leanctx")` — these proxy through whatever providers the application has configured
- emits spans and records metric values via the standard OTel API

If the application has not configured an OTel SDK, `get_tracer()` returns OTel's default no-op tracer and emission is a fast no-op. The `[otel]` extra installs `opentelemetry-api>=1.30,<2.0` and `opentelemetry-sdk>=1.30,<2.0`; the SDK is included for convenience but leanctx does not import or use it.

## What gets emitted

### Spans

Two span names are emitted, depending on call shape:

| Span name | When emitted |
|---|---|
| `leanctx.compress` | Once per wrapper-routed call (`leanctx.{Anthropic,OpenAI,Gemini}.X.create()` etc.) OR once per direct `Middleware.compress_messages(...)` call. |
| `leanctx.compressor.compress` | Once per individual compressor invocation (`Lingua`, `SelfLLM`, or `Verbatim`). Becomes a child span of the wrapper/middleware span when nested; becomes a root when called directly. |

The wrapper-routed call always produces **exactly one** `leanctx.compress` span. When the wrapper calls into the middleware, the middleware does NOT open a second `leanctx.compress` span — it would duplicate the root for the same conceptual operation. The depth-counter mechanism in `leanctx.observability.middleware_hooks` enforces this.

### Span attributes

Every `leanctx.compress` span carries:

| Attribute | Type | Meaning |
|---|---|---|
| `leanctx.provider` | string | `anthropic`, `openai`, `gemini` for wrapper-routed calls; `none` for direct middleware calls. |
| `leanctx.method` | string | One of the documented `MethodStatus` values (see below). |
| `leanctx.input_tokens` | int | Tokens observed before compression. |
| `leanctx.output_tokens` | int | Tokens after compression. |
| `leanctx.ratio` | float | `output_tokens / input_tokens` (1.0 means no shrink). |
| `leanctx.cost_usd` | float | Compression cost in USD (best-effort; SelfLLM populates from upstream API pricing, Lingua/Verbatim are 0.0). |
| `leanctx.duration_ms` | float | End-to-end leanctx-pipeline duration. |

`leanctx.compressor.compress` spans carry `leanctx.compressor.name`, the same per-compressor token/ratio/cost attributes, and a `leanctx.method` from the compressor's own taxonomy. Inner compressor children of a wrapper/middleware span do **not** re-emit `leanctx.provider` — provider/method/cost are owned by the outermost span (see AC-6 attribute ownership).

### Method taxonomy (closed set)

`leanctx.method` always takes one of these values:

| Value | Meaning |
|---|---|
| `passthrough` | Mode is off; messages forwarded unchanged. |
| `below-threshold` | Mode is on but message tokens did not exceed the trigger threshold. |
| `empty` | Empty message list; nothing to compress. |
| `opaque-bailout` | Gemini contents contained non-text parts (function_call / image / inline_data) that cannot be safely compressed. |
| `verbatim` | Pipeline ran; all messages routed to Verbatim. |
| `lingua` | Pipeline ran; only Lingua used. |
| `selfllm` | Pipeline ran; only SelfLLM used. |
| `hybrid` | Pipeline ran; two or more of {Verbatim, Lingua, SelfLLM} used. |

### Metrics

| Metric name | Type | Unit |
|---|---|---|
| `leanctx.compress.calls` | counter | requests |
| `leanctx.compress.input_tokens` | counter | tokens |
| `leanctx.compress.output_tokens` | counter | tokens |
| `leanctx.compress.cost_usd` | counter | USD |
| `leanctx.compress.duration_ms` | histogram | ms |

All five carry `provider`, `method`, and `status` attributes. Cost is recorded only when `cost_usd > 0` so a string of Verbatim/Lingua calls leaves the cost-counter running total unchanged.

The cost counter is incremented exactly once per top-level call path — at the **outermost** span. Inner spans (middleware-internal, compressor-internal) do not re-increment.

## Stream-path span lifetime

For streaming wrapper paths (OpenAI `stream=True`, Gemini `generate_content_stream`, Gemini async stream), the leanctx wrapper's span closes at the **first** of:

- iterator exhaustion (`StopIteration` / `StopAsyncIteration`),
- explicit `.close()` / `aclose()` on the wrapper,
- `__del__` finalization (GC backstop, for the abandoned-iterator case).

`duration_ms` covers the full stream lifetime — request submission to last chunk delivered or close. Iteration-time exceptions are recorded on the span via `set_status(Status(StatusCode.ERROR))` before the span is closed, then re-raised.

For Anthropic context-manager stream paths (`messages.stream()` sync or async), the leanctx wrapper's span opens at `__enter__` / `__aenter__` and closes at `__exit__` / `__aexit__`.

## Configuration

Per-client configuration goes in `leanctx_config["observability"]`:

```python
client = leanctx.Anthropic(
    leanctx_config={
        "observability": {
            "otel": True,                          # default: False
            "service_name": "my-app",              # informational; does NOT override OTEL_SERVICE_NAME
            "extra_attributes": {"region": "us-west-2"},  # static attributes added to every span
        }
    }
)
```

Standard OTel environment variables (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, etc.) are honored by the OTel SDK that the application configures. leanctx does not auto-attach to OTel based on env vars — the app must explicitly set up providers before importing leanctx.

## Verifying telemetry shape with bench

The `leanctx bench` CLI runs offline scenarios with deterministic input so you can dry-run your telemetry pipeline before deploying:

```bash
leanctx bench list                                  # see registered scenarios
leanctx bench run lingua-local --workload rag       # offline lingua compression
leanctx bench run agent-structural --workload agent # invariant-enforcement run
```

When OTel is configured at the application level, bench scenarios emit the same `leanctx.compress` and `leanctx.compressor.compress` spans + metrics as production traffic.

## Cardinality

leanctx span and metric attributes are bounded:
- `provider` ∈ `{anthropic, openai, gemini, none}`
- `method` ∈ the eight documented `MethodStatus` values
- `status` ∈ `{success, error}`
- `extra_attributes` from your config — keep these stable and bounded

Per-tenant attribution (e.g. a `tenant_id` attribute) is **out of scope for v0.3**. It requires a request-context mechanism plus an explicit cardinality cap, which is planned for v0.4.

## Backward compatibility

The pre-v0.3 telemetry surface (`usage.leanctx_method`, `usage.leanctx_ratio`, `usage.leanctx_tokens_saved`) is preserved. v0.3 adds `usage.leanctx_cost_usd` and emits OTel spans + metrics in addition to the response-attached attributes — never replacing them.

## Cold-import budget

`import leanctx` (no extras) does not import `opentelemetry`. The `[otel]` extra adds the API package but the import is lazy — paid only on the first call to `compression_span` / `compressor_span`. Cold-import time stays under 60 ms.
