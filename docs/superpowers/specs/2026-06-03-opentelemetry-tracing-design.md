# OpenTelemetry tracing (#372)

## Goal

Make slow-request triage tractable by emitting per-phase spans (prefill, decode,
draft/verify, Flash prefetch, disk-cache I/O, MCP tool calls) for each request,
exportable to a standard OTLP collector (Jaeger/Tempo). Logs alone don't show
the *relative* cost of each phase, and with Flash + speculative combos the
bottleneck shifts request-by-request.

## Acceptance criteria (from #372)

- A single chat request renders a coherent trace in Jaeger/Tempo with
  prefill/decode/verify sub-spans.
- No measurable latency regression when tracing is disabled
  (compile-time/import-time only — no `opentelemetry` import on the off path).

Out of scope (per issue): trace propagation across distributed workers
(follow-up); auto-instrumentation of the HTTP layer (we open a root span
manually instead — see §2).

## Decisions locked during brainstorming

- **Scope:** full issue breadth — inference, speculative/dflash/eagle, flash,
  disk cache, and MCP tool calls.
- **Dependency:** optional `[otel]` extra + **lazy import**. Nothing OTel is
  imported unless `OLMLX_TRACING` is on.
- **Config:** single `OLMLX_TRACING=true` toggle (default off). All
  endpoint/protocol/headers/sampling come from the **native `OTEL_*` env vars**
  the OTLP exporter and SDK already honor (`OTEL_EXPORTER_OTLP_ENDPOINT`,
  `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_TRACES_SAMPLER`, `OTEL_SERVICE_NAME`,
  `OTEL_TRACES_EXPORTER=console` for collector-free local debugging, …).
- **Per-step speculative spans:** literal one-span-per-step as the issue
  specifies; volume bounded by the standard `OTEL_TRACES_SAMPLER`.

## Architecture

### 1. Core wrapper — `utils/tracing.py`

A single module owns *all* contact with `opentelemetry`. No other file imports
`opentelemetry` directly; they import the helpers below. This is what makes the
off path zero-cost and keeps the SDK swappable.

Module state:

```python
_ENABLED: bool = False
_TRACER = None            # opentelemetry Tracer once initialized
```

Public surface:

- `init_tracing(settings) -> None` — called from lifespan startup **only when
  `settings.tracing`**. Lazily `import`s `opentelemetry` + the SDK, builds a
  `TracerProvider` with `Resource(service.name = OTEL_SERVICE_NAME or "olmlx")`,
  attaches a `BatchSpanProcessor` (async export — never blocks the inference
  thread) wrapping an `OTLPSpanExporter` (HTTP/protobuf; reads native `OTEL_*`).
  Sets `_ENABLED = True`, `_TRACER = provider.get_tracer("olmlx")`. Installs the
  logging filter (below). Idempotent.
- `shutdown_tracing() -> None` — `provider.shutdown()` (flushes the batch
  processor) in lifespan shutdown. Resets module state so repeated
  `create_app()` in tests is clean (mirrors the metrics collector
  unregister-on-shutdown pattern).
- `span(name, **attrs)` — **the one call site the whole codebase uses.** Context
  manager.
  - Disabled → returns a shared **no-op span** object that implements
    `set_attribute`, `set_attributes`, `record_exception`, `add_event`,
    `__enter__`/`__exit__` as no-ops. Call sites stay branch-free
    (`with span("prefill", model=m) as sp: ...; sp.set_attribute("eval_count", n)`),
    and the cost when disabled is one attribute-less function call returning a
    singleton — no allocation, no OTel import.
  - Enabled → `_TRACER.start_as_current_span(name)`, applying `**attrs` as span
    attributes. Records exceptions and sets status ERROR on exception.
- `current_context()` / `attach_context(ctx)` — thin wrappers over
  `opentelemetry.context.get_current()` / `attach()`. **Load-bearing:** OTel
  context is `contextvars`-based and does **not** auto-propagate into
  `ThreadPoolExecutor` workers or bare threads. Inference decode runs in the
  `async_mlx_stream` worker thread, Flash prefetch in its own pool, disk-cache
  I/O via `asyncio.to_thread`. Without capturing the context on the calling side
  and re-attaching inside the worker, every span opened in those threads would
  start a *new* trace and orphan from the request. The helpers (no-ops when
  disabled) make the capture/attach explicit at each thread hop.

### 2. Root span — manual, in middleware

The issue puts FastAPI auto-instrumentation out of scope, so we do **not** pull
in `opentelemetry-instrumentation-fastapi`. Instead the existing middleware
stack opens the server/root span. It nests just inside `RequestIDMiddleware`
(so the request id is available as an attribute) and around
`MetricsMiddleware`, giving every downstream span a single parent →
one coherent trace per request.

Root-span attributes: `http.method`, `http.route` (template, not the raw path,
to bound cardinality the same way metrics does), `surface`
(ollama/openai/anthropic/audio), `request_id`, `http.status_code`.

When tracing is disabled the middleware's `span(...)` is the no-op context
manager, so the middleware path is unchanged in cost.

### 3. Span inventory (full breadth)

All spans are children of the request root span (via context propagation).
Attribute keys follow OTel-ish dotted names where a convention exists, otherwise
plain snake_case matching our metrics labels.

**`engine/inference.py`** (the two shared seams the metrics already use):

- Top inference span per public entry — `generate_chat`, `generate_completion`,
  `generate_embeddings`, `generate_transcription`. Attributes: `model`,
  `surface`, `strategy` (speculative strategy or `"none"`),
  `gen.stream` (bool).
- `prefill` child — start of stream creation → first token yield. Attributes:
  `prompt_tokens`, `cache_hit` (prompt-cache reuse), `ttft_ns`.
- `decode` child — the token loop. Attributes: `eval_count`, `decode_tok_s`.
  Opened on the calling side, context captured, re-attached in the
  `async_mlx_stream` worker thread so child decode/step spans nest correctly.

**`engine/speculative.py`, `engine/dflash/decoder.py`, `engine/eagle/decoder.py`**
(shared `prefill`/`step` surface):

- `spec.prefill` — attributes `strategy`, `prompt_tokens`.
- `spec.step` — one span **per decode step** (literal, per the locked decision),
  attributes `strategy`, `proposed`, `accepted`. Volume controlled by the
  operator via `OTEL_TRACES_SAMPLER`. A `spec.verify` sub-span wraps
  `verify_draft_greedy` so the "verify" phase the AC names is visible.

**`engine/flash/`** (prefetch + weight load):

- `flash.prefetch` — wraps predict+submit and `wait_for_layer`, attribute
  `layer_idx`. Created inside the prefetch thread, so it re-attaches the context
  captured when the request opened the decode span.
- `flash.weight_load` — wraps the per-layer neuron/expert load, attributes
  `layer_idx`, `active_neurons`/`experts`.

**`engine/prompt_cache/store.py`** (disk spill):

- `cache.disk_write` — wraps `_save_to_disk`, attributes `cache_id`, `bytes`.
- `cache.disk_read` — wraps `_load_from_disk`, attributes `cache_id`, `bytes`,
  `hit`. Both run under `asyncio.to_thread`; the async wrapper captures context
  before offloading and the worker re-attaches it.

**`chat/`** (terminal chat agent loop):

- `mcp.tool_call` — wraps each MCP tool invocation, attributes `tool.name`,
  `mcp.server`. Parented under a per-turn span so a chat turn's tool calls group
  together.

### 4. Correlation id in logs

`init_tracing` installs a `logging.Filter` (no extra dependency — we do *not*
pull in `opentelemetry-instrumentation-logging`) on the root logger's handlers
that, when a span is active, sets `record.trace_id` and `record.span_id` (hex)
on each log record. The existing `request_id_var`/`RequestIDMiddleware` is
untouched and `request_id` stays available; we add trace/span ids alongside it
so a log line can be pivoted to its trace. The filter reads
`opentelemetry.trace.get_current_span()`; when disabled it is never installed.

### 5. Config & dependencies

- `config.py`: add `tracing: bool = False` (→ `OLMLX_TRACING`). No other tracing
  fields — endpoint/protocol/headers/sampling/service-name are native `OTEL_*`.
- `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  otel = [
    "opentelemetry-sdk>=1.27",
    "opentelemetry-exporter-otlp-proto-http>=1.27",
  ]
  ```
  (`opentelemetry-api` comes transitively via `-sdk`.) Mirror the same two
  packages into the dev dependency group so CI installs them and the tracing
  tests actually run.

### 6. App wiring (`app.py`)

- Lifespan startup: after settings load, `if settings.tracing:
  init_tracing(settings)`. Store nothing extra on `app.state` (module-global
  tracer), matching how the metrics layer keeps its registry module-level.
- Lifespan shutdown: `shutdown_tracing()` after draining in-flight requests, so
  the final spans flush.
- Middleware: add the root-span middleware (or fold the `span(...)` into the
  existing `MetricsMiddleware.dispatch`) just inside `RequestIDMiddleware`.

## Testing

Tests use OTel's `InMemorySpanExporter` to capture spans and assert tree shape +
attributes without a live collector. They `pytest.importorskip("opentelemetry")`
so they skip cleanly when the `[otel]` extra isn't installed (CI installs it).

- **Off-path is truly free (AC #2):** with `OLMLX_TRACING` unset, assert
  `span(...)` returns the no-op singleton and that `opentelemetry` is **not** in
  `sys.modules` after a representative inference call. Guards the
  no-import-when-disabled criterion directly.
- **Coherent chat trace (AC #1):** drive a chat request through the existing
  inference test fixture with an `InMemorySpanExporter` installed; assert the
  span tree has root → inference → {`prefill`, `decode`} and, on a speculative
  config, `spec.prefill`/`spec.step`/`spec.verify` nested under decode, with the
  expected `proposed`/`accepted`/`model`/`strategy` attributes.
- **Cross-thread propagation:** open a parent span on the event loop, run a
  function that opens a child span inside a `ThreadPoolExecutor` worker using
  `attach_context(current_context())`; assert the child's parent is the
  event-loop span (not a new root).
- **Disk-cache & flash spans:** unit-test `_save_to_disk`/`_load_from_disk` and a
  prefetch step under the in-memory exporter, asserting the `cache.disk_*` /
  `flash.*` spans and their attributes.
- **Logging correlation:** with a span active, assert a log record gains
  `trace_id`/`span_id`; with tracing off, assert the filter isn't installed and
  records are unchanged.

## Risks / notes

- **Thread-context propagation is the main correctness risk.** Every span opened
  off the request's coroutine (decode worker, flash pool, disk `to_thread`) must
  go through `attach_context`. The cross-thread test exists specifically to
  catch a regression here; a missed attach shows up as orphaned single-span
  traces, not a crash.
- **Per-step span volume** is intentionally high by default; the spec documents
  `OTEL_TRACES_SAMPLER` as the operator's knob rather than capping in code.
- **`BatchSpanProcessor`** keeps export off the hot path; export failures are
  swallowed by the SDK and never surface to the request.
