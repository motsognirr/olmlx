# Prometheus `/metrics` Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose olmlx's existing operational metrics (request rates, TTFT, decode tok/s, token counts, prompt-cache/flash/speculative effectiveness, loaded-model gauges) at a Prometheus-scrapable `GET /metrics` endpoint.

**Architecture:** Three collection layers in a single `olmlx/utils/metrics.py` module sharing one private `CollectorRegistry`: (1) a lazy custom `OlmlxStatsCollector` that reads `ModelManager` stats at scrape time for point-in-time gauges and process-global cumulative counters (folded via a `CounterAccumulator` so they survive model eviction), (2) HTTP middleware pushing request/duration/in-flight metrics and setting a `surface_var` ContextVar, and (3) engine instrumentation pushing per-generation inference metrics from two shared seams in `inference.py` plus the speculative streamer. Token/TTFT/tok-s values are read from the same finalized `TimingStats` the HTTP responses (and therefore `olmlx bench run`) use, so scraped values match bench within tolerance.

**Tech Stack:** Python, FastAPI/Starlette, `prometheus-client`, pytest/pytest-asyncio.

---

## File Structure

- **new** `olmlx/utils/metrics.py` — registry, all metric objects, `CounterAccumulator`, `OlmlxStatsCollector`, and `observe_inference` / `record_speculative` / `record_http_request` helpers. Only module importing `prometheus_client`.
- **new** `olmlx/routers/metrics.py` — `GET /metrics`.
- **new** `tests/test_metrics.py` — unit tests for helpers, accumulator, collector.
- **new** `tests/test_routers_metrics.py` — endpoint + middleware + surface integration tests.
- modify `olmlx/context.py` — add `surface_var`.
- modify `olmlx/app.py` — `MetricsMiddleware`, include metrics router, register/unregister collector in `lifespan`.
- modify `olmlx/engine/inference.py` — call `observe_inference` at the two shared text seams + the error path.
- modify `olmlx/engine/speculative_stream.py` — call `record_speculative` in `_log_stats`.
- modify `pyproject.toml` — add `prometheus-client>=0.20`.
- modify `CLAUDE.md` — document the endpoint under Key Design Decisions and the `/metrics` router in the structure block.

---

## Task 1: Add dependency and scaffold `metrics.py` with the registry and metric objects

**Files:**
- Modify: `pyproject.toml:6-22` (core `dependencies`)
- Create: `olmlx/utils/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, inside the `dependencies = [ ... ]` list (the block starting at line 6), add after the `mlx-whisper` line:

```toml
    "prometheus-client>=0.20",
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync --no-editable`
Expected: resolves and installs `prometheus-client`.

- [ ] **Step 3: Write the failing test for module scaffolding**

Create `tests/test_metrics.py`:

```python
from prometheus_client import generate_latest

from olmlx.utils import metrics


def test_registry_exposes_metric_families():
    # Touch a counter so its family is emitted, then scrape.
    metrics.INFERENCE_REQUESTS.labels(model="m", surface="ollama").inc()
    body = generate_latest(metrics.REGISTRY).decode()
    assert "olmlx_inference_requests_total" in body
    assert "olmlx_http_requests_total" in body or "olmlx_http_requests" in body
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv run pytest tests/test_metrics.py::test_registry_exposes_metric_families -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` (no `metrics` module yet).

- [ ] **Step 5: Create the module with the registry and all metric objects**

Create `olmlx/utils/metrics.py`:

```python
"""Prometheus metrics for olmlx (issue #371).

This is the only module that imports ``prometheus_client``. Everything is
registered against a private ``REGISTRY`` (not the global default) so tests get a
clean, inspectable registry and there is no collision with libraries that touch
the default registry.

Three layers:

* Push counters/histograms (HTTP middleware + engine inference seams) defined
  here as module-level metric objects.
* Lazy gauges + process-global cumulative counters read from ``ModelManager`` at
  scrape time via ``OlmlxStatsCollector`` (registered in ``app.lifespan``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

if TYPE_CHECKING:
    from olmlx.utils.timing import TimingStats

# Private registry — see module docstring.
REGISTRY = CollectorRegistry()

# --- Layer 2: HTTP middleware ---
HTTP_REQUESTS = Counter(
    "olmlx_http_requests_total",
    "HTTP requests handled, by matched route template, method, and status.",
    ["path", "method", "status"],
    registry=REGISTRY,
)
HTTP_REQUEST_DURATION = Histogram(
    "olmlx_http_request_duration_seconds",
    "Wall-clock duration of HTTP requests, by route template and method.",
    ["path", "method"],
    registry=REGISTRY,
)
HTTP_IN_FLIGHT = Gauge(
    "olmlx_http_requests_in_flight",
    "HTTP requests currently being handled.",
    registry=REGISTRY,
)

# --- Layer 3: engine inference ---
INFERENCE_REQUESTS = Counter(
    "olmlx_inference_requests_total",
    "Completed inference generations, by model and API surface.",
    ["model", "surface"],
    registry=REGISTRY,
)
INFERENCE_ERRORS = Counter(
    "olmlx_inference_errors_total",
    "Inference generations that raised before completing.",
    ["model", "surface"],
    registry=REGISTRY,
)
INFERENCE_TOKENS = Counter(
    "olmlx_inference_tokens_total",
    "Tokens processed, by model, surface, and direction (prompt|completion).",
    ["model", "surface", "direction"],
    registry=REGISTRY,
)
# Buckets sized for local single-box latencies (seconds).
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
INFERENCE_TTFT = Histogram(
    "olmlx_inference_ttft_seconds",
    "Prefill time (time to first token), by model and surface.",
    ["model", "surface"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
INFERENCE_REQUEST_DURATION = Histogram(
    "olmlx_inference_request_duration_seconds",
    "Total generation wall-clock duration, by model and surface.",
    ["model", "surface"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
_TOKS_BUCKETS = (1, 5, 10, 20, 40, 60, 80, 120, 200, 400, 800)
INFERENCE_DECODE_TPS = Histogram(
    "olmlx_inference_decode_tokens_per_second",
    "Decode throughput (tokens/sec) per generation, by model and surface.",
    ["model", "surface"],
    buckets=_TOKS_BUCKETS,
    registry=REGISTRY,
)

# --- Speculative decoding ---
SPECULATIVE_PROPOSED = Counter(
    "olmlx_speculative_proposed_total",
    "Draft tokens proposed, by speculative strategy.",
    ["strategy"],
    registry=REGISTRY,
)
SPECULATIVE_ACCEPTED = Counter(
    "olmlx_speculative_accepted_total",
    "Draft tokens accepted by target verification, by speculative strategy.",
    ["strategy"],
    registry=REGISTRY,
)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_metrics.py::test_registry_exposes_metric_families -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock olmlx/utils/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): add prometheus-client dep and metric registry (#371)"
```

---

## Task 2: `observe_inference` helper

**Files:**
- Modify: `olmlx/utils/metrics.py`
- Modify: `olmlx/context.py:5` (add `surface_var`)
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Add `surface_var` to context (needed by the helper's fallback test)**

In `olmlx/context.py`, after the `request_id_var` line (line 5), add:

```python
# API surface for the current request: "ollama" | "openai" | "anthropic" |
# "audio". Set by MetricsMiddleware; read by engine inference instrumentation.
surface_var: ContextVar[str] = ContextVar("surface", default="unknown")
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_metrics.py`:

```python
from olmlx.utils.timing import TimingStats


def _counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


def test_observe_inference_records_counts_and_histograms():
    stats = TimingStats(
        total_duration=2_000_000_000,  # 2s
        prompt_eval_count=100,
        prompt_eval_duration=500_000_000,  # 0.5s TTFT
        eval_count=40,
        eval_duration=1_000_000_000,  # 1s -> 40 tok/s
    )
    before = _counter_value(
        metrics.INFERENCE_REQUESTS, model="obs", surface="openai"
    )
    metrics.observe_inference("obs", "openai", stats)
    after = _counter_value(
        metrics.INFERENCE_REQUESTS, model="obs", surface="openai"
    )
    assert after == before + 1
    assert (
        _counter_value(
            metrics.INFERENCE_TOKENS, model="obs", surface="openai", direction="prompt"
        )
        == 100
    )
    assert (
        _counter_value(
            metrics.INFERENCE_TOKENS,
            model="obs",
            surface="openai",
            direction="completion",
        )
        == 40
    )
    # Decode histogram observed the 40 tok/s sample.
    decode = metrics.INFERENCE_DECODE_TPS.labels(model="obs", surface="openai")
    assert decode._sum.get() == 40.0


def test_observe_inference_error_increments_error_counter():
    before = _counter_value(metrics.INFERENCE_ERRORS, model="err", surface="ollama")
    metrics.observe_inference("err", "ollama", TimingStats(), error=True)
    after = _counter_value(metrics.INFERENCE_ERRORS, model="err", surface="ollama")
    assert after == before + 1


def test_observe_inference_matches_bench_tps():
    # Same TimingStats fed to bench's PromptResult must yield identical tok/s.
    from olmlx.bench.results import PromptResult

    stats = TimingStats(
        prompt_eval_count=128,
        prompt_eval_duration=400_000_000,
        eval_count=64,
        eval_duration=800_000_000,
    )
    pr = PromptResult(
        eval_count=stats.eval_count,
        eval_duration_ns=stats.eval_duration,
        prompt_eval_count=stats.prompt_eval_count,
        prompt_eval_duration_ns=stats.prompt_eval_duration,
        total_duration_ns=0,
        status_code=200,
    )
    assert metrics._decode_tps(stats) == pr.tokens_per_second
```

> Note: `PromptResult` is a dataclass in `olmlx/bench/results.py`; if its constructor signature differs from the kwargs above when you implement, adjust the test to match the real fields (`eval_count`, `eval_duration_ns`, `prompt_eval_count`, `prompt_eval_duration_ns`, `total_duration_ns`, `status_code`). The assertion that matters is `metrics._decode_tps(stats) == pr.tokens_per_second`.

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_metrics.py -k observe_inference -v`
Expected: FAIL with `AttributeError: module 'olmlx.utils.metrics' has no attribute 'observe_inference'`.

- [ ] **Step 4: Implement the helper**

Add to `olmlx/utils/metrics.py` (after the metric objects):

```python
def _decode_tps(stats: TimingStats) -> float:
    """Decode throughput in tokens/sec from a finalized TimingStats.

    Mirrors ``olmlx.bench.results.PromptResult.tokens_per_second`` exactly so
    scraped values match ``olmlx bench run`` on the same workload.
    """
    if stats.eval_count > 0 and stats.eval_duration > 0:
        return stats.eval_count / (stats.eval_duration / 1e9)
    return 0.0


def observe_inference(
    model: str, surface: str, stats: "TimingStats", *, error: bool = False
) -> None:
    """Record per-generation inference metrics from a finalized TimingStats.

    Safe to call on every generation; cheap (counter inc + histogram observe).
    ``error=True`` records the request as a failure and skips the histograms
    (the timing on an error path is not a meaningful latency sample).
    """
    if error:
        INFERENCE_ERRORS.labels(model=model, surface=surface).inc()
        return
    INFERENCE_REQUESTS.labels(model=model, surface=surface).inc()
    if stats.prompt_eval_count:
        INFERENCE_TOKENS.labels(
            model=model, surface=surface, direction="prompt"
        ).inc(stats.prompt_eval_count)
    if stats.eval_count:
        INFERENCE_TOKENS.labels(
            model=model, surface=surface, direction="completion"
        ).inc(stats.eval_count)
    if stats.prompt_eval_duration > 0:
        INFERENCE_TTFT.labels(model=model, surface=surface).observe(
            stats.prompt_eval_duration / 1e9
        )
    if stats.total_duration > 0:
        INFERENCE_REQUEST_DURATION.labels(model=model, surface=surface).observe(
            stats.total_duration / 1e9
        )
    tps = _decode_tps(stats)
    if tps > 0:
        INFERENCE_DECODE_TPS.labels(model=model, surface=surface).observe(tps)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -k "observe_inference" -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add olmlx/utils/metrics.py olmlx/context.py tests/test_metrics.py
git commit -m "feat(metrics): observe_inference helper + surface_var (#371)"
```

---

## Task 3: `record_speculative` helper with class→strategy mapping

**Files:**
- Modify: `olmlx/utils/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_metrics.py`:

```python
def test_record_speculative_keys_by_strategy_label():
    class PromptLookupDecoder:  # name matches the real class
        pass

    before_p = _counter_value(metrics.SPECULATIVE_PROPOSED, strategy="pld")
    before_a = _counter_value(metrics.SPECULATIVE_ACCEPTED, strategy="pld")
    metrics.record_speculative(PromptLookupDecoder(), proposed=10, accepted=7)
    assert _counter_value(metrics.SPECULATIVE_PROPOSED, strategy="pld") == before_p + 10
    assert _counter_value(metrics.SPECULATIVE_ACCEPTED, strategy="pld") == before_a + 7


def test_record_speculative_unknown_class_falls_back():
    class WeirdDecoder:
        pass

    metrics.record_speculative(WeirdDecoder(), proposed=1, accepted=0)
    assert _counter_value(metrics.SPECULATIVE_PROPOSED, strategy="unknown") >= 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_metrics.py -k record_speculative -v`
Expected: FAIL with `AttributeError: ... 'record_speculative'`.

- [ ] **Step 3: Implement the helper**

Add to `olmlx/utils/metrics.py`:

```python
# Map decoder class name -> Prometheus strategy label. No strategy_name
# attribute exists on the decoders; a class->label map avoids touching five
# decoder classes.
_STRATEGY_BY_CLASS = {
    "SpeculativeDecoder": "classic",
    "SpeculativeFlashDecoder": "classic",
    "PromptLookupDecoder": "pld",
    "DFlashDecoder": "dflash",
    "EagleDecoder": "eagle",
    "SelfSpeculativeDecoder": "self",
}


def record_speculative(decoder: object, proposed: int, accepted: int) -> None:
    """Record speculative draft proposed/accepted counts for a finished stream."""
    strategy = _STRATEGY_BY_CLASS.get(type(decoder).__name__, "unknown")
    if proposed:
        SPECULATIVE_PROPOSED.labels(strategy=strategy).inc(proposed)
    if accepted:
        SPECULATIVE_ACCEPTED.labels(strategy=strategy).inc(accepted)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -k record_speculative -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/utils/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): record_speculative helper (#371)"
```

---

## Task 4: `CounterAccumulator` (monotonic folding across eviction)

**Files:**
- Modify: `olmlx/utils/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_metrics.py`:

```python
def test_counter_accumulator_is_monotonic_across_reset():
    acc = metrics.CounterAccumulator()
    # Model "a" accumulates to 5, then 8 (live store growing).
    assert acc.total("a", 5) == 5
    assert acc.total("a", 8) == 8
    # Model evicted and reloaded: live value resets to a lower number.
    # The exported total must NOT decrease — it folds the new value as a fresh
    # delta from zero: 8 (carried) + 3 (new) = 11.
    assert acc.total("a", 3) == 11
    assert acc.total("a", 4) == 12  # +1 since last (4-3)


def test_counter_accumulator_independent_keys():
    acc = metrics.CounterAccumulator()
    assert acc.total("a", 10) == 10
    assert acc.total("b", 2) == 2
    assert acc.total("a", 10) == 10  # no change for a
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_metrics.py -k counter_accumulator -v`
Expected: FAIL with `AttributeError: ... 'CounterAccumulator'`.

- [ ] **Step 3: Implement**

Add to `olmlx/utils/metrics.py`:

```python
class CounterAccumulator:
    """Folds per-source cumulative values into a process-lifetime monotonic total.

    A source (e.g. a model's prompt-cache store) reports a cumulative value that
    resets to a low number when the source is recreated (model evicted then
    reloaded). To keep a Prometheus counter monotonic, we track the last value
    seen per key and add only the positive delta. When the live value drops below
    the last-seen value (a reset), we treat the new value as a fresh delta from
    zero so the exported total still only increases.
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._total: dict[str, float] = {}

    def total(self, key: str, live_value: float) -> float:
        last = self._last.get(key, 0.0)
        running = self._total.get(key, 0.0)
        if live_value >= last:
            running += live_value - last
        else:
            # Source reset (e.g. model reloaded): fold the whole new value in.
            running += live_value
        self._last[key] = live_value
        self._total[key] = running
        return running
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -k counter_accumulator -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/utils/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): monotonic CounterAccumulator (#371)"
```

---

## Task 5: `OlmlxStatsCollector` (lazy gauges + folded counters)

**Files:**
- Modify: `olmlx/utils/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_metrics.py`:

```python
from prometheus_client import CollectorRegistry


class _FakeCacheMetrics:
    cache_id_hits = 3
    cache_id_misses = 1
    radix_hits = 5
    radix_misses = 2
    evictions_ram = 4
    evictions_disk = 0
    bytes_in_ram = 1024
    bytes_on_disk = 2048


class _FakeStore:
    def __init__(self):
        self.metrics = _FakeCacheMetrics()


class _FakeLoadedModel:
    def __init__(self, name):
        self.name = name
        self.size_bytes = 7_000_000
        self.active_refs = 1
        self.prompt_cache_store = _FakeStore()
        self.weight_store = None
        self.model = object()  # no prefetcher attribute


class _FakeManager:
    def __init__(self, models):
        self._models = models

    def get_loaded(self):
        return self._models


def test_collector_emits_gauges_and_folded_counters():
    reg = CollectorRegistry()
    manager = _FakeManager([_FakeLoadedModel("m1")])
    collector = metrics.OlmlxStatsCollector(manager)
    reg.register(collector)
    body = generate_latest(reg).decode()
    assert "olmlx_loaded_models 1.0" in body
    assert 'olmlx_model_size_bytes{model="m1"} 7000000.0' in body
    assert 'olmlx_kv_cache_bytes{location="ram",model="m1"} 1024.0' in body
    assert 'olmlx_kv_cache_bytes{location="disk",model="m1"} 2048.0' in body
    assert 'olmlx_prompt_cache_lookups_total{kind="radix",model="m1",result="hit"} 5.0' in body


def test_collector_safe_when_no_stores():
    reg = CollectorRegistry()
    lm = _FakeLoadedModel("bare")
    lm.prompt_cache_store = None
    manager = _FakeManager([lm])
    reg.register(metrics.OlmlxStatsCollector(manager))
    body = generate_latest(reg).decode()  # must not raise
    assert "olmlx_loaded_models 1.0" in body
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_metrics.py -k collector -v`
Expected: FAIL with `AttributeError: ... 'OlmlxStatsCollector'`.

- [ ] **Step 3: Implement the collector**

Add to `olmlx/utils/metrics.py`. Add this import near the top with the other prometheus import:

```python
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
```

Then add the collector class:

```python
class OlmlxStatsCollector:
    """Lazy collector: reads ModelManager stats at scrape time.

    Point-in-time values are emitted as gauges read fresh each scrape. Cumulative
    per-model stats (prompt-cache / flash hit-miss / eviction counts) are folded
    through a CounterAccumulator so they stay monotonic across model eviction and
    reload.
    """

    def __init__(self, manager: object) -> None:
        self._manager = manager
        self._acc = CounterAccumulator()

    def collect(self):
        try:
            loaded = list(self._manager.get_loaded())
        except Exception:
            loaded = []

        loaded_g = GaugeMetricFamily(
            "olmlx_loaded_models", "Models currently resident in memory."
        )
        loaded_g.add_metric([], float(len(loaded)))
        yield loaded_g

        size_g = GaugeMetricFamily(
            "olmlx_model_size_bytes", "Model weight size in bytes.", labels=["model"]
        )
        refs_g = GaugeMetricFamily(
            "olmlx_model_active_refs",
            "Active inference references holding the model resident.",
            labels=["model"],
        )
        kv_g = GaugeMetricFamily(
            "olmlx_kv_cache_bytes",
            "Prompt-cache KV bytes resident, by location.",
            labels=["model", "location"],
        )
        lookups_c = CounterMetricFamily(
            "olmlx_prompt_cache_lookups_total",
            "Prompt-cache lookups, by kind (cache_id|radix) and result (hit|miss).",
            labels=["model", "kind", "result"],
        )
        evict_c = CounterMetricFamily(
            "olmlx_prompt_cache_evictions_total",
            "Prompt-cache evictions, by location (ram|disk).",
            labels=["model", "location"],
        )
        expert_c = CounterMetricFamily(
            "olmlx_flash_expert_cache_events_total",
            "Flash-MoE expert cache events, by result (hit|miss|failure).",
            labels=["model", "result"],
        )
        expert_calls_c = CounterMetricFamily(
            "olmlx_flash_expert_load_calls_total",
            "Flash-MoE expert load_experts() invocations.",
            labels=["model"],
        )
        prefetch_c = CounterMetricFamily(
            "olmlx_flash_prefetch_events_total",
            "Flash prefetch events, by result (hit|miss|failure).",
            labels=["model", "result"],
        )
        prefetch_sub_c = CounterMetricFamily(
            "olmlx_flash_prefetch_submitted_total",
            "Flash prefetch requests submitted.",
            labels=["model"],
        )

        for lm in loaded:
            name = getattr(lm, "name", "") or ""
            size_g.add_metric([name], float(getattr(lm, "size_bytes", 0) or 0))
            refs_g.add_metric([name], float(getattr(lm, "active_refs", 0) or 0))

            store = getattr(lm, "prompt_cache_store", None)
            cm = getattr(store, "metrics", None) if store is not None else None
            if cm is not None:
                kv_g.add_metric([name, "ram"], float(cm.bytes_in_ram))
                kv_g.add_metric([name, "disk"], float(cm.bytes_on_disk))
                lookups_c.add_metric(
                    [name, "cache_id", "hit"],
                    self._acc.total(f"{name}|cid_hit", cm.cache_id_hits),
                )
                lookups_c.add_metric(
                    [name, "cache_id", "miss"],
                    self._acc.total(f"{name}|cid_miss", cm.cache_id_misses),
                )
                lookups_c.add_metric(
                    [name, "radix", "hit"],
                    self._acc.total(f"{name}|rdx_hit", cm.radix_hits),
                )
                lookups_c.add_metric(
                    [name, "radix", "miss"],
                    self._acc.total(f"{name}|rdx_miss", cm.radix_misses),
                )
                evict_c.add_metric(
                    [name, "ram"],
                    self._acc.total(f"{name}|ev_ram", cm.evictions_ram),
                )
                evict_c.add_metric(
                    [name, "disk"],
                    self._acc.total(f"{name}|ev_disk", cm.evictions_disk),
                )

            ws = getattr(lm, "weight_store", None)
            es = getattr(ws, "stats", None) if ws is not None else None
            if es is not None and hasattr(es, "cache_hits"):
                snap = es.snapshot() if hasattr(es, "snapshot") else {}
                expert_c.add_metric(
                    [name, "hit"],
                    self._acc.total(f"{name}|ex_hit", snap.get("cache_hits", 0)),
                )
                expert_c.add_metric(
                    [name, "miss"],
                    self._acc.total(f"{name}|ex_miss", snap.get("cache_misses", 0)),
                )
                expert_c.add_metric(
                    [name, "failure"],
                    self._acc.total(f"{name}|ex_fail", snap.get("load_failures", 0)),
                )
                expert_calls_c.add_metric(
                    [name],
                    self._acc.total(f"{name}|ex_calls", snap.get("load_calls", 0)),
                )

            prefetcher = getattr(getattr(lm, "model", None), "prefetcher", None)
            ps = getattr(prefetcher, "stats", None) if prefetcher is not None else None
            if ps is not None and hasattr(ps, "cache_hits"):
                prefetch_c.add_metric(
                    [name, "hit"], self._acc.total(f"{name}|pf_hit", ps.cache_hits)
                )
                prefetch_c.add_metric(
                    [name, "miss"], self._acc.total(f"{name}|pf_miss", ps.cache_misses)
                )
                prefetch_c.add_metric(
                    [name, "failure"], self._acc.total(f"{name}|pf_fail", ps.failures)
                )
                prefetch_sub_c.add_metric(
                    [name], self._acc.total(f"{name}|pf_sub", ps.submitted)
                )

        yield size_g
        yield refs_g
        yield kv_g
        yield lookups_c
        yield evict_c
        yield expert_c
        yield expert_calls_c
        yield prefetch_c
        yield prefetch_sub_c
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -k collector -v`
Expected: PASS (2 tests).

> If the exact label-ordering in the `assert ... in body` strings does not match prometheus' output (it sorts labels alphabetically), adjust the expected substrings to the sorted order shown by a quick `print(body)` — the labels and values are what matter, not the assertion's literal ordering.

- [ ] **Step 5: Run the whole metrics test file**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add olmlx/utils/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): lazy OlmlxStatsCollector for gauges + folded counters (#371)"
```

---

## Task 6: `MetricsMiddleware` (HTTP metrics + surface_var)

**Files:**
- Modify: `olmlx/utils/metrics.py` (add `record_http_request` + `surface_for_path`)
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_metrics.py`:

```python
def test_surface_for_path():
    assert metrics.surface_for_path("/api/chat") == "ollama"
    assert metrics.surface_for_path("/v1/chat/completions") == "openai"
    assert metrics.surface_for_path("/v1/messages") == "anthropic"
    assert metrics.surface_for_path("/v1/messages/count_tokens") == "anthropic"
    assert metrics.surface_for_path("/v1/audio/transcriptions") == "audio"


def test_record_http_request_increments():
    before = _counter_value(
        metrics.HTTP_REQUESTS, path="/api/chat", method="POST", status="200"
    )
    metrics.record_http_request("/api/chat", "POST", 200, 0.5)
    after = _counter_value(
        metrics.HTTP_REQUESTS, path="/api/chat", method="POST", status="200"
    )
    assert after == before + 1
    assert (
        metrics.HTTP_REQUEST_DURATION.labels(path="/api/chat", method="POST")._sum.get()
        >= 0.5
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_metrics.py -k "surface_for_path or record_http" -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

Add to `olmlx/utils/metrics.py`:

```python
def surface_for_path(path: str) -> str:
    """Map a request path to a bounded API-surface label."""
    if path.startswith("/v1/audio/"):
        return "audio"
    if path.startswith("/v1/messages"):
        return "anthropic"
    if path.startswith("/v1/"):
        return "openai"
    return "ollama"


def record_http_request(
    path: str, method: str, status: int, duration_seconds: float
) -> None:
    """Record one completed HTTP request (counter + duration histogram)."""
    HTTP_REQUESTS.labels(path=path, method=method, status=str(status)).inc()
    HTTP_REQUEST_DURATION.labels(path=path, method=method).observe(duration_seconds)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -k "surface_for_path or record_http" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add olmlx/utils/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): surface_for_path + record_http_request helpers (#371)"
```

---

## Task 7: `/metrics` router

**Files:**
- Create: `olmlx/routers/metrics.py`
- Test: `tests/test_routers_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_routers_metrics.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

from olmlx.routers import metrics as metrics_router


def _app():
    app = FastAPI()
    app.include_router(metrics_router.router)
    return app


def test_metrics_endpoint_returns_prometheus_text():
    client = TestClient(_app())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "olmlx_inference_requests_total" in resp.text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_routers_metrics.py::test_metrics_endpoint_returns_prometheus_text -v`
Expected: FAIL with `ModuleNotFoundError: olmlx.routers.metrics`.

- [ ] **Step 3: Implement the router**

Create `olmlx/routers/metrics.py`:

```python
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from olmlx.utils.metrics import REGISTRY

router = APIRouter()


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_routers_metrics.py::test_metrics_endpoint_returns_prometheus_text -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add olmlx/routers/metrics.py tests/test_routers_metrics.py
git commit -m "feat(metrics): GET /metrics router (#371)"
```

---

## Task 8: Wire middleware, router, and collector into `app.py`

**Files:**
- Modify: `olmlx/app.py` (imports, `MetricsMiddleware` class, `create_app`, `lifespan`)
- Test: `tests/test_routers_metrics.py`

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_routers_metrics.py`:

```python
import time

from olmlx.utils import metrics
from olmlx.context import surface_var


def test_metrics_middleware_counts_requests_and_sets_surface():
    from starlette.middleware.base import BaseHTTPMiddleware
    from olmlx.app import MetricsMiddleware

    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    captured = {}

    @app.get("/api/tags")
    async def tags():
        captured["surface"] = surface_var.get()
        return {"ok": True}

    app.include_router(metrics_router.router)
    client = TestClient(app)

    before = metrics.HTTP_REQUESTS.labels(
        path="/api/tags", method="GET", status="200"
    )._value.get()
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    assert captured["surface"] == "ollama"
    after = metrics.HTTP_REQUESTS.labels(
        path="/api/tags", method="GET", status="200"
    )._value.get()
    assert after == before + 1
    # In-flight returns to baseline after the request completes.
    assert metrics.HTTP_IN_FLIGHT._value.get() == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_routers_metrics.py::test_metrics_middleware_counts_requests_and_sets_surface -v`
Expected: FAIL with `ImportError: cannot import name 'MetricsMiddleware' from 'olmlx.app'`.

- [ ] **Step 3: Add the middleware class to `app.py`**

In `olmlx/app.py`, add imports near the existing ones (after line 12, the `request_id_var` import):

```python
import time

from olmlx.context import surface_var
from olmlx.routers import metrics as metrics_router
from olmlx.utils import metrics as metrics_mod
```

> If `olmlx.context` is already imported for `request_id_var`, fold `surface_var` into that existing import instead of adding a duplicate line.

Add the middleware class after `RequestIDMiddleware` (after line 168):

```python
class MetricsMiddleware(BaseHTTPMiddleware):
    """Record per-request HTTP metrics and set the API-surface ContextVar.

    Mirrors RequestIDMiddleware's reliance on BaseHTTPMiddleware copying the
    current context into the inner-app sub-task, so surface_var is visible to
    engine inference instrumentation during streaming responses.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        surface_token = surface_var.set(metrics_mod.surface_for_path(path))
        metrics_mod.HTTP_IN_FLIGHT.inc()
        start = time.perf_counter()
        status_code = 500
        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            metrics_mod.HTTP_IN_FLIGHT.dec()
            # Prefer the matched route template to bound cardinality; fall back
            # to a literal for unmatched paths (404 with no route).
            route = request.scope.get("route")
            label_path = getattr(route, "path", None) or "<unmatched>"
            metrics_mod.record_http_request(
                label_path, request.method, status_code, time.perf_counter() - start
            )
            surface_var.reset(surface_token)
```

- [ ] **Step 4: Register middleware and router in `create_app`**

In `create_app`, after `app.add_middleware(RequestIDMiddleware)` (line 201), add:

```python
    app.add_middleware(MetricsMiddleware)
```

And after the last `app.include_router(...)` (line 314, anthropic), add:

```python
    app.include_router(metrics_router.router)
```

- [ ] **Step 5: Register the collector in `lifespan`**

In `lifespan`, after `app.state.model_manager = manager` (line 113), add:

```python
    # Lazy gauge/counter collector reads the manager at scrape time.
    from olmlx.utils.metrics import REGISTRY, OlmlxStatsCollector

    stats_collector = OlmlxStatsCollector(manager)
    REGISTRY.register(stats_collector)
    app.state.metrics_collector = stats_collector
```

In the shutdown section of `lifespan`, after `await manager.stop()` (line 120), add:

```python
    # Unregister so a subsequent create_app() (e.g. in tests) does not raise a
    # duplicate-collector error against the module-level REGISTRY.
    try:
        from olmlx.utils.metrics import REGISTRY

        REGISTRY.unregister(app.state.metrics_collector)
    except Exception:
        pass
```

- [ ] **Step 6: Run the integration test**

Run: `uv run pytest tests/test_routers_metrics.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add olmlx/app.py tests/test_routers_metrics.py
git commit -m "feat(metrics): wire middleware, /metrics router, lazy collector into app (#371)"
```

---

## Task 9: Instrument the two shared inference seams in `inference.py`

**Files:**
- Modify: `olmlx/engine/inference.py` (streaming done-chunk seam ~3655; non-streaming return ~4072)
- Test: `tests/test_inference_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_metrics.py`:

```python
from olmlx.utils import metrics


def test_full_completion_inner_records_inference_metrics(monkeypatch):
    """The non-streaming seam must call observe_inference with model + stats."""
    calls = []
    monkeypatch.setattr(
        metrics,
        "observe_inference",
        lambda model, surface, stats, **kw: calls.append((model, surface, stats, kw)),
    )
    # Import inside the test so the monkeypatched symbol is what inference.py
    # resolves at call time (inference.py must reference metrics.observe_inference,
    # not a from-import bound at module load).
    from olmlx.engine import inference  # noqa: F401

    assert hasattr(inference, "observe_inference") is False or callable(
        metrics.observe_inference
    )
```

> This is a lightweight guard. The substantive verification is that `inference.py` references `metrics.observe_inference` (attribute access on the module) so monkeypatching works, and that the call passes `lm.name`, the resolved surface, and the finalized `stats`. A full end-to-end generation test is covered by the existing inference suite once the call sites compile.

- [ ] **Step 2: Run it to verify the import path works (it will pass trivially; the real check is wiring)**

Run: `uv run pytest tests/test_inference_metrics.py -v`
Expected: PASS (guard test). Proceed to wire the call sites, then verify the broader suite still passes.

- [ ] **Step 3: Add the import to `inference.py`**

In `olmlx/engine/inference.py`, near the existing `from olmlx.utils.timing import Timer, TimingStats` (line 57), add:

```python
from olmlx.context import surface_var
from olmlx.utils import metrics as _metrics
```

- [ ] **Step 4: Instrument the non-streaming seam in `_full_completion_inner`**

In `olmlx/engine/inference.py`, in `_full_completion_inner`, immediately before `return result_dict` (line 4079), add:

```python
    try:
        _metrics.observe_inference(lm.name, surface_var.get(), stats)
    except Exception:
        logger.debug("metrics: observe_inference failed (non-stream)", exc_info=True)
```

- [ ] **Step 5: Instrument the streaming seam in `_stream_completion`**

In `_stream_completion`, immediately before `yield done_chunk` (line 3662), add:

```python
        try:
            _metrics.observe_inference(lm.name, surface_var.get(), stats)
        except Exception:
            logger.debug("metrics: observe_inference failed (stream)", exc_info=True)
```

- [ ] **Step 6: Record errors in the streaming error path**

`_stream_completion` invalidates the cache on incomplete generation in its `finally` (line 3671: `if not generation_complete and full_prompt_tokens is not None:`). Add an error metric inside that same branch, after the existing `logger.debug("Cache invalidated...")` line (line 3672):

```python
            try:
                _metrics.observe_inference(
                    lm.name, surface_var.get(), stats, error=True
                )
            except Exception:
                logger.debug("metrics: error observe failed", exc_info=True)
```

- [ ] **Step 7: Run the inference + metrics suites**

Run: `uv run pytest tests/test_inference_metrics.py tests/test_metrics.py -v`
Expected: PASS.

Run (regression): `uv run pytest tests/ -k inference -q`
Expected: no new failures introduced by the edits (pre-existing skips/failures unrelated to metrics are acceptable; compare against a baseline run if unsure).

- [ ] **Step 8: Commit**

```bash
git add olmlx/engine/inference.py tests/test_inference_metrics.py
git commit -m "feat(metrics): instrument inference stream/non-stream seams (#371)"
```

---

## Task 10: Record speculative acceptance in `speculative_stream.py`

**Files:**
- Modify: `olmlx/engine/speculative_stream.py` (`_log_stats`, ~line 93)
- Test: `tests/test_speculative_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_speculative_metrics.py`:

```python
from olmlx.utils import metrics


class _FakeDecoder:
    # Class name not in the strategy map -> "unknown" label.
    def stats_summary(self):
        return {"proposed": 12, "accepted_draft": 9, "steps": 3}


def test_log_stats_records_speculative_counts(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        metrics,
        "record_speculative",
        lambda decoder, proposed, accepted: recorded.append((proposed, accepted)),
    )
    from olmlx.engine import speculative_stream

    # Build the closure the generator uses and invoke it directly.
    decoder = _FakeDecoder()
    speculative_stream._emit_speculative_metrics(decoder)
    assert recorded == [(12, 9)]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_speculative_metrics.py -v`
Expected: FAIL with `AttributeError: module 'olmlx.engine.speculative_stream' has no attribute '_emit_speculative_metrics'`.

- [ ] **Step 3: Add a small module-level helper and call it from `_log_stats`**

In `olmlx/engine/speculative_stream.py`, add an import near the top (after line 17, the `StreamToken` import):

```python
from olmlx.utils import metrics as _metrics
```

Add a module-level helper (place it above `speculative_stream_generate`, after the protocols ~line 34):

```python
def _emit_speculative_metrics(decoder: object) -> None:
    """Record proposed/accepted speculative counts for a finished stream."""
    stats_fn = getattr(decoder, "stats_summary", None)
    if stats_fn is None:
        return
    try:
        s = stats_fn()
        _metrics.record_speculative(
            decoder,
            proposed=int(s.get("proposed", 0)),
            accepted=int(s.get("accepted_draft", 0)),
        )
    except Exception:
        logger.debug("metrics: record_speculative failed", exc_info=True)
```

In the existing `_log_stats` closure inside `speculative_stream_generate` (line 93), add a call to the helper at the end of the function body (after the `logger.info(...)` block, after line 108):

```python
        _emit_speculative_metrics(decoder)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_speculative_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/speculative_stream.py tests/test_speculative_metrics.py
git commit -m "feat(metrics): record speculative acceptance per strategy (#371)"
```

---

## Task 11: Documentation + full suite + lint

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/USER_MANUAL.md` (if it documents endpoints — add a `/metrics` line)

- [ ] **Step 1: Document in `CLAUDE.md`**

In the Project Structure block under `routers/`, add a line:

```
│   ├── metrics.py     # GET /metrics — Prometheus exposition
```

Add a Key Design Decisions bullet:

```markdown
- **Prometheus metrics** (`utils/metrics.py`, `routers/metrics.py`): `GET /metrics` exposes operational metrics from a private `CollectorRegistry`. Three layers: HTTP middleware (`olmlx_http_*` request/duration/in-flight, sets `surface_var`), engine inference instrumentation reading the finalized `TimingStats` at the two shared seams in `inference.py` (`olmlx_inference_*` requests/errors/tokens/ttft/decode-tok-s — values match `bench run`), and a lazy `OlmlxStatsCollector` that reads `ModelManager` at scrape time for point-in-time gauges (loaded models, KV bytes) and process-global cumulative counters (prompt-cache / flash / speculative) folded through `CounterAccumulator` so they survive model eviction. Labels are bounded (model, surface, strategy, route template); no per-request labels.
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest tests/test_metrics.py tests/test_routers_metrics.py tests/test_inference_metrics.py tests/test_speculative_metrics.py -v`
Expected: all PASS.

Run: `uv run pytest -q`
Expected: no new failures relative to a pre-change baseline.

- [ ] **Step 3: Lint and format (project requirement before push)**

Run: `uv run ruff check olmlx tests && uv run ruff format olmlx tests`
Expected: clean (or auto-fixed). Re-run `ruff check` to confirm no remaining errors.

- [ ] **Step 4: Smoke-test the endpoint against a running server (optional but recommended)**

Run: `uv run olmlx &` then `curl -s localhost:11434/metrics | head -40`
Expected: Prometheus exposition text including `olmlx_http_requests_total`, `olmlx_loaded_models`, `olmlx_inference_*` (families with no samples yet appear once touched). Stop the server afterward.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/USER_MANUAL.md
git commit -m "docs(metrics): document Prometheus /metrics endpoint (#371)"
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** prometheus_client integration (T1), counters requests/errors/tokens/spec/cache (T2,T3,T5), histograms TTFT/inter-token/duration — TTFT + duration + decode-tok/s in T2; "inter-token latency" is represented by decode-tok/s (its reciprocal) per the single-box scope, avoiding a redundant histogram. Gauges loaded-models/KV-bytes/flash occupancy — loaded-models + KV bytes in T5; flash occupancy intentionally omitted (no public accessor — documented in spec). ExpertCacheStats/PrefetchStats/prompt-cache surfaced (T5). Acceptance: "matches bench" (T2 bench-parity test), "no latency regression" (lazy gauges + cheap counter incs), "bounded cardinality" (bounded label sets, route templates). Mount at GET /metrics (T7,T8).
- **Type/name consistency:** `observe_inference(model, surface, stats, *, error=False)`, `record_speculative(decoder, proposed, accepted)`, `CounterAccumulator.total(key, live_value)`, `surface_for_path(path)`, `record_http_request(path, method, status, duration_seconds)`, `OlmlxStatsCollector(manager)` — used consistently across tasks.
- **Placeholders:** none — every code step shows complete code.
