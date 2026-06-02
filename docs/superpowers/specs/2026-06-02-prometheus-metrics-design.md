# Prometheus `/metrics` endpoint (#371)

## Goal

Expose the operational metrics olmlx already collects at a Prometheus-scrapable
`GET /metrics` endpoint, so a long-lived instance (especially with experimental
knobs on) can be monitored without re-running `olmlx bench run`.

## Acceptance criteria (from #371)

- Scraped values match `olmlx bench run` reports on the same workload within tolerance.
- No measurable latency regression from collection.
- Cardinality bounded — no per-request labels that explode the series count.

Out of scope: Pushgateway integration, multi-instance aggregation.

## Architecture — three collection layers

Metrics split by *how* they are collected. This keeps per-request cost near
zero and cardinality bounded.

### Layer 1 — Lazy gauges (pull at scrape time)

A custom `prometheus_client` `Collector` (`OlmlxStatsCollector`) is registered
once at app startup, holding a reference to `app.state.model_manager`. On each
`/metrics` scrape it walks `manager.get_loaded()` and reads the
already-accumulating stats objects. No background polling, no per-request cost.

Live point-in-time gauges:

- `olmlx_loaded_models` — count of currently-resident models.
- `olmlx_model_size_bytes{model}` — `LoadedModel.size_bytes`.
- `olmlx_model_active_refs{model}` — `LoadedModel.active_refs`.
- `olmlx_kv_cache_bytes{model,location}` — `location` ∈ {`ram`,`disk`} from
  `CacheMetrics.bytes_in_ram` / `bytes_on_disk`.

Flash-MoE expert-cache *occupancy* is intentionally **not** exposed as a gauge:
it would require reaching into the store's private `LayerLruCache._cache` and
summing per-layer sizes, with no public accessor. The hit/miss/load-call
counters in Layer 1b fully characterize cache effectiveness; add an occupancy
gauge later only if a public accessor is introduced.

### Layer 1b — Process-global cumulative counters folded from per-model stores

Per-model stats objects (`CacheMetrics`, `ExpertCacheStats`, `PrefetchStats`)
live on a `LoadedModel` and **vanish when the model is evicted**. Exposing their
cumulative fields directly as live values would reset counters to zero on
eviction/reload, violating Prometheus' monotonic-counter contract and losing
history.

Fix: a `CounterAccumulator` in `metrics.py` keeps process-lifetime totals.

- The collector, each scrape, reads the live per-model cumulative value and
  computes the delta against the last value it saw for that `(model, metric)`
  key, adding the delta to the process-global total. Counters only ever
  increase.
- When a model is evicted and a *new* store appears for the same name later, its
  live value starts again at 0; the accumulator detects the live value dropping
  **below** the last-seen value and treats the new lower value as a fresh
  baseline (adds the full new value as delta from 0), so no decrease is ever
  emitted.
- Keyed by `(model_name, metric_name)`. The last-seen map persists for the
  process lifetime (bounded by distinct model names × metric count — small).

Counters surfaced this way (all monotonic, `model`-labelled):

- `olmlx_prompt_cache_lookups_total{model,kind,result}` — `kind` ∈
  {`cache_id`,`radix`}, `result` ∈ {`hit`,`miss`} from `CacheMetrics`.
- `olmlx_prompt_cache_evictions_total{model,location}` — `location` ∈
  {`ram`,`disk`}.
- `olmlx_flash_expert_cache_events_total{model,result}` — `result` ∈
  {`hit`,`miss`,`failure`} from `ExpertCacheStats` (`cache_hits`,
  `cache_misses`, `load_failures`); plus `olmlx_flash_expert_load_calls_total{model}`.
- `olmlx_flash_prefetch_events_total{model,result}` — `result` ∈
  {`hit`,`miss`,`failure`}; plus `olmlx_flash_prefetch_submitted_total{model}`.

Note: model-name labelling means a never-reloaded evicted model retains its
last cumulative total in the accumulator (the series stops increasing but does
not vanish), which is the correct Prometheus behaviour for a counter whose
source is gone.

### Layer 2 — HTTP middleware (push, per request)

`MetricsMiddleware` in `app.py`, ordered alongside the existing middleware:

- `olmlx_http_requests_total{path,method,status}` — counter.
- `olmlx_http_request_duration_seconds{path,method}` — histogram.
- `olmlx_http_requests_in_flight` — gauge (inc on entry, dec in `finally`).

`path` is the **matched route template** (`request.scope["route"].path`), not
the raw URL, to bound cardinality. Requests that 404 with no matched route are
bucketed under a literal `"<unmatched>"` label.

The middleware also derives the API surface from the path and sets a
`surface_var` ContextVar (Layer 3 reads it):

- `/v1/messages*` → `anthropic`
- `/v1/*` → `openai`
- `/v1/audio/*` → `audio` (checked before the generic `/v1/` rule)
- everything else (`/api/*`) → `ollama`

### Layer 3 — Engine inference instrumentation (push, per generation)

Recorded from the async generators in `inference.py` where the final
`TimingStats` is known and `model_name` is in scope. These run on the event
loop, so `surface_var` set by the middleware is readable. A single helper:

```
observe_inference(model: str, surface: str, stats: TimingStats, *, error: bool = False)
```

emits:

- `olmlx_inference_requests_total{model,surface}` — counter.
- `olmlx_inference_errors_total{model,surface}` — counter (when `error=True`).
- `olmlx_inference_tokens_total{model,surface,direction}` — counter, `direction`
  ∈ {`prompt`,`completion`} from `prompt_eval_count` / `eval_count`.
- `olmlx_inference_ttft_seconds{model,surface}` — histogram, observed from
  `prompt_eval_duration / 1e9` (prefill time; the same quantity bench reports as
  the prefill phase).
- `olmlx_inference_decode_tokens_per_second{model,surface}` — histogram, observed
  from `eval_count / (eval_duration / 1e9)` when both > 0.
- `olmlx_inference_request_duration_seconds{model,surface}` — histogram, from
  `total_duration / 1e9`.

Because this reads the same finalized `TimingStats` that the HTTP responses (and
therefore the bench harness, which derives its numbers from those response
fields) use, scraped TTFT / decode-tok/s / token counts match bench within
floating-point tolerance — satisfying the first acceptance criterion.

Call sites (all in `inference.py`): the chat streaming-done seam, chat
non-streaming return, completion streaming-done seam, completion non-streaming
return, plus `generate_embeddings` and `generate_transcription`
(prompt-token-only / no-decode cases just leave the decode histogram unobserved).
Errors raised mid-generation are recorded via `error=True` in the existing
exception path before re-raise.

#### Speculative acceptance

Recorded in `speculative_stream.py` at the existing `_log_stats` point (fires on
both the early-finish `return` and normal loop exit):

- `olmlx_speculative_proposed_total{strategy}` — counter, from
  `stats_summary()["proposed"]`.
- `olmlx_speculative_accepted_total{strategy}` — counter, from `accepted_draft`.

`strategy` is derived from the decoder's class via a fixed mapping in
`metrics.py` (`SpeculativeDecoder`→`classic`, `PromptLookupDecoder`→`pld`,
`DFlashDecoder`→`dflash`, `EagleDecoder`→`eagle`, `SelfSpeculativeDecoder`→`self`);
no `strategy_name` attribute exists on the decoders today, and a class→label map
avoids touching five decoder classes. Unknown classes fall back to a literal
`"unknown"` label. The `proposed`/`accepted_draft` from `stats_summary()` are
per-stream cumulative totals; `_log_stats` fires once at stream end, so the
helper records the final summary as the increment for that stream. Acceptance
*rate* is left to PromQL
(`rate(accepted) / rate(proposed)`) rather than exported as a gauge, so it
aggregates correctly across requests.

## New / changed files

- **new** `olmlx/utils/metrics.py` — the registry, all metric objects,
  `OlmlxStatsCollector`, `CounterAccumulator`, and the `record_*` / `observe_*`
  helpers. The only module that imports `prometheus_client`.
- **new** `olmlx/routers/metrics.py` — `GET /metrics` →
  `PlainTextResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)`.
- `olmlx/context.py` — add `surface_var: ContextVar[str]`.
- `olmlx/app.py` — add `MetricsMiddleware`, include the metrics router, register
  `OlmlxStatsCollector(manager)` in `lifespan` after the manager is built (and
  unregister on shutdown so repeated `create_app()` in tests does not raise a
  duplicate-collector error).
- `olmlx/engine/inference.py` — call `observe_inference(...)` at the six
  completion seams + the error path.
- `olmlx/engine/speculative_stream.py` — record proposed/accepted in
  `_log_stats`.
- `pyproject.toml` — add `prometheus-client>=0.20` to core `dependencies`.

## Registry strategy

Use a module-level `CollectorRegistry` in `metrics.py` (not the global default
registry) so tests get a clean, inspectable registry and there is no collision
with any library that touches the default registry. `generate_latest(REGISTRY)`
and the `/metrics` router both reference it. The lazy `OlmlxStatsCollector` is
registered/unregistered against this same registry in `lifespan`.

## Cardinality budget

All labels bounded:

- `model` — currently-loaded set on a single box (typically 1–3); accumulator
  retains evicted names but that set grows only with distinct models ever used.
- `surface` — 4 fixed values.
- `strategy` — ≤5 fixed values.
- `path` — fixed route templates + `<unmatched>`.
- `kind`/`result`/`location`/`direction` — small fixed enums.

No request IDs, no user identifiers, no free-form labels.

## Testing (TDD — write failing tests first)

`tests/test_metrics.py`:

- `OlmlxStatsCollector` emits the expected gauge families from a fake manager
  exposing stub `LoadedModel`s with a `CacheMetrics`, an `ExpertCacheStats`, and
  a `PrefetchStats`.
- `CounterAccumulator` is monotonic across a simulated evict/reload: feed
  increasing live values, then a reset to a lower value (new store), and assert
  the exported total never decreases and equals old_total + new_value.
- `observe_inference` increments request/token counters and observes the
  TTFT / decode-tok/s / duration histograms for a given `TimingStats`; the
  decode-tok/s sample equals `eval_count/(eval_duration/1e9)`.
- `observe_inference(error=True)` increments the error counter.
- Helpers are safe no-ops when a `LoadedModel` lacks a prompt-cache store / flash
  store (the common dense-model case).
- `record_speculative` keys totals by `strategy`.
- "Matches bench" test: given a `TimingStats`, assert `observe_inference`'s
  derived prompt-tok/s and decode-tok/s equal `bench.results.PromptResult`'s
  `prompt_tokens_per_second` / `tokens_per_second` properties computed from the
  same counts/durations.

`tests/test_routers_metrics.py`:

- `GET /metrics` returns 200 with `Content-Type: text/plain; version=0.0.4`.
- The body contains the registered metric names.
- The HTTP request counter increments and the in-flight gauge returns to its
  baseline after a request completes.
- `surface_var` is set to the expected value for representative paths
  (`/api/chat` → `ollama`, `/v1/chat/completions` → `openai`,
  `/v1/messages` → `anthropic`, `/v1/audio/transcriptions` → `audio`).

## Risks / notes

- ContextVar propagation: inference metric recording happens in the async
  generator on the event loop (not the `CancellableStream` background thread), so
  `surface_var` is in scope. The spec does **not** rely on the background thread
  seeing the ContextVar.
- Duplicate-collector registration across repeated `create_app()` in tests is
  avoided by unregistering on shutdown and/or guarding registration.
- `generate_transcription` / `generate_embeddings` have no decode phase; they
  record request + prompt-token counters and skip the decode-tok/s histogram.
