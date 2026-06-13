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

import logging
from typing import TYPE_CHECKING, Protocol

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

if TYPE_CHECKING:
    from olmlx.utils.timing import TimingStats

logger = logging.getLogger(__name__)


class _ManagerProtocol(Protocol):
    """Minimal structural type the collector needs from the model manager.

    Typed as a Protocol (not the concrete ModelManager) to avoid a heavy/circular
    import; the loaded-model objects are themselves duck-typed via getattr in
    ``collect`` so the cache/flash stores are optional.
    """

    def get_loaded(self) -> list: ...


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


# --- Layer 3 helpers: engine inference ---
def _decode_tps(stats: TimingStats) -> float:
    """Decode throughput in tokens/sec from a finalized TimingStats.

    Mirrors ``olmlx.bench.results.PromptResult.tokens_per_second`` exactly so
    scraped values match ``olmlx bench run`` on the same workload.
    """
    if stats.eval_count > 0 and stats.eval_duration > 0:
        return stats.eval_count / (stats.eval_duration / 1e9)
    return 0.0


def observe_inference(
    model: str, surface: str, stats: TimingStats, *, error: bool = False
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
        INFERENCE_TOKENS.labels(model=model, surface=surface, direction="prompt").inc(
            stats.prompt_eval_count
        )
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


# --- Speculative helper ---
# Map decoder class name -> Prometheus strategy label. No strategy_name
# attribute exists on the decoders; a class->label map avoids touching five
# decoder classes.
_STRATEGY_BY_CLASS = {
    "SpeculativeDecoder": "classic",
    "SpeculativeFlashDecoder": "classic",
    "PromptLookupDecoder": "pld",
    "DFlashDecoder": "dflash",
    "EagleDecoder": "eagle",
    "MTPDecoder": "mtp",
    "SelfSpeculativeDecoder": "self",
}


def record_speculative(decoder: object, proposed: int, accepted: int) -> None:
    """Record speculative draft proposed/accepted counts for a finished stream."""
    strategy = _STRATEGY_BY_CLASS.get(type(decoder).__name__, "unknown")
    if proposed:
        SPECULATIVE_PROPOSED.labels(strategy=strategy).inc(proposed)
    if accepted:
        SPECULATIVE_ACCEPTED.labels(strategy=strategy).inc(accepted)


# --- Layer 2 helpers: HTTP ---
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


# --- Layer 1 + 1b: lazy collector ---
class CounterAccumulator:
    """Folds per-source cumulative values into a process-lifetime monotonic total.

    A source (e.g. a model's prompt-cache store) reports a cumulative value that
    resets to a low number when the source is recreated (model evicted then
    reloaded). To keep a Prometheus counter monotonic, we track the last value
    seen per key and add only the positive delta. When the live value drops below
    the last-seen value (a reset), we treat the new value as a fresh delta from
    zero so the exported total still only increases.

    Approximation across a reset: this folds at most the new store's full live
    value, so it never over-counts. The only inaccuracy is a slight *under*-count
    when a reloaded store accumulates past the previous total before the first
    scrape observes the reset — those interim events are missed. Acceptable for a
    single-box monitoring tool where the source has no stable per-instance id.
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


class OlmlxStatsCollector:
    """Lazy collector: reads ModelManager stats at scrape time.

    Point-in-time values are emitted as gauges read fresh each scrape. Cumulative
    per-model stats (prompt-cache / flash hit-miss / eviction counts) are folded
    through a CounterAccumulator so they stay monotonic across model eviction and
    reload.
    """

    def __init__(self, manager: _ManagerProtocol) -> None:
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
        batch_active_g = GaugeMetricFamily(
            "olmlx_batch_active_sequences",
            "Sequences currently decoding in the continuous batch.",
            labels=["model"],
        )
        batch_inserts_c = CounterMetricFamily(
            "olmlx_batch_inserts_total",
            "Sequences admitted into the continuous batch.",
            labels=["model"],
        )
        batch_tokens_c = CounterMetricFamily(
            "olmlx_batch_aggregate_tokens_total",
            "Tokens generated by the continuous batch, summed across sequences.",
            labels=["model"],
        )

        def _emit_one(lm: object) -> None:
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
                # Fall back to direct attribute reads (not an empty dict) when a
                # stats object lacks snapshot(), so live values aren't discarded.
                snap = (
                    es.snapshot()
                    if hasattr(es, "snapshot")
                    else {
                        "cache_hits": getattr(es, "cache_hits", 0),
                        "cache_misses": getattr(es, "cache_misses", 0),
                        "load_failures": getattr(es, "load_failures", 0),
                        "load_calls": getattr(es, "load_calls", 0),
                    }
                )
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

            sched = getattr(lm, "batch_scheduler", None)
            if sched is not None and hasattr(sched, "stats"):
                bs = sched.stats()
                batch_active_g.add_metric(
                    [name], float(bs.get("batch_active_sequences", 0))
                )
                batch_inserts_c.add_metric(
                    [name],
                    self._acc.total(f"{name}|b_ins", bs.get("batch_inserts", 0)),
                )
                batch_tokens_c.add_metric(
                    [name],
                    self._acc.total(f"{name}|b_tok", bs.get("batch_tokens", 0)),
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

        # Guard each model independently: a malformed stats object on one model
        # must not fail the whole scrape (which would 500 /metrics).
        for lm in loaded:
            try:
                _emit_one(lm)
            except Exception:
                logger.debug("metrics: per-model collect failed", exc_info=True)

        yield size_g
        yield refs_g
        yield kv_g
        yield lookups_c
        yield evict_c
        yield expert_c
        yield expert_calls_c
        yield prefetch_c
        yield prefetch_sub_c
        yield batch_active_g
        yield batch_inserts_c
        yield batch_tokens_c
