from prometheus_client import CollectorRegistry, generate_latest

from olmlx.utils import metrics
from olmlx.utils.timing import TimingStats


def _counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


# --- Task 1: registry scaffolding ---
def test_registry_exposes_metric_families():
    metrics.INFERENCE_REQUESTS.labels(model="m", surface="ollama").inc()
    body = generate_latest(metrics.REGISTRY).decode()
    assert "olmlx_inference_requests_total" in body
    assert "olmlx_http_requests" in body


# --- Task 2: observe_inference ---
def test_observe_inference_records_counts_and_histograms():
    stats = TimingStats(
        total_duration=2_000_000_000,  # 2s
        prompt_eval_count=100,
        prompt_eval_duration=500_000_000,  # 0.5s TTFT
        eval_count=40,
        eval_duration=1_000_000_000,  # 1s -> 40 tok/s
    )
    before = _counter_value(metrics.INFERENCE_REQUESTS, model="obs", surface="openai")
    metrics.observe_inference("obs", "openai", stats)
    after = _counter_value(metrics.INFERENCE_REQUESTS, model="obs", surface="openai")
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
    decode = metrics.INFERENCE_DECODE_TPS.labels(model="obs", surface="openai")
    assert decode._sum.get() == 40.0


def test_observe_inference_error_increments_error_counter():
    before = _counter_value(metrics.INFERENCE_ERRORS, model="err", surface="ollama")
    metrics.observe_inference("err", "ollama", TimingStats(), error=True)
    after = _counter_value(metrics.INFERENCE_ERRORS, model="err", surface="ollama")
    assert after == before + 1


def test_observe_inference_matches_bench_tps():
    from olmlx.bench.results import PromptResult

    stats = TimingStats(
        prompt_eval_count=128,
        prompt_eval_duration=400_000_000,
        eval_count=64,
        eval_duration=800_000_000,
    )
    pr = PromptResult(
        prompt_name="t",
        category="c",
        output_text="",
        status_code=200,
        eval_count=stats.eval_count,
        eval_duration_ns=stats.eval_duration,
        prompt_eval_count=stats.prompt_eval_count,
        prompt_eval_duration_ns=stats.prompt_eval_duration,
        total_duration_ns=0,
    )
    assert metrics._decode_tps(stats) == pr.tokens_per_second


# --- Task 3: record_speculative ---
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


# --- Task 4: CounterAccumulator ---
def test_counter_accumulator_is_monotonic_across_reset():
    acc = metrics.CounterAccumulator()
    assert acc.total("a", 5) == 5
    assert acc.total("a", 8) == 8
    # Model evicted and reloaded: live value resets to a lower number.
    assert acc.total("a", 3) == 11
    assert acc.total("a", 4) == 12  # +1 since last (4-3)


def test_counter_accumulator_independent_keys():
    acc = metrics.CounterAccumulator()
    assert acc.total("a", 10) == 10
    assert acc.total("b", 2) == 2
    assert acc.total("a", 10) == 10  # no change for a


# --- Task 5: OlmlxStatsCollector ---
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
        self.prompt_cache_store: object = _FakeStore()
        self.weight_store: object = None
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
    assert 'olmlx_model_size_bytes{model="m1"} 7e+06' in body
    assert 'olmlx_kv_cache_bytes{location="ram",model="m1"} 1024.0' in body
    assert 'olmlx_kv_cache_bytes{location="disk",model="m1"} 2048.0' in body
    assert (
        'olmlx_prompt_cache_lookups_total{kind="radix",model="m1",result="hit"} 5.0'
        in body
    )


class _SnapshotlessExpertStats:
    # Has the counter attributes but no snapshot() method.
    cache_hits = 11
    cache_misses = 4
    load_failures = 1
    load_calls = 6


class _FakeWeightStore:
    def __init__(self):
        self.stats = _SnapshotlessExpertStats()


def test_collector_reads_expert_stats_without_snapshot_method():
    reg = CollectorRegistry()
    lm = _FakeLoadedModel("moe")
    lm.weight_store = _FakeWeightStore()
    reg.register(metrics.OlmlxStatsCollector(_FakeManager([lm])))
    body = generate_latest(reg).decode()
    # Live values must be reported, not silently zeroed by the {} fallback.
    assert (
        'olmlx_flash_expert_cache_events_total{model="moe",result="hit"} 11.0' in body
    )
    assert 'olmlx_flash_expert_load_calls_total{model="moe"} 6.0' in body


def test_collector_safe_when_no_stores():
    reg = CollectorRegistry()
    lm = _FakeLoadedModel("bare")
    lm.prompt_cache_store = None
    manager = _FakeManager([lm])
    reg.register(metrics.OlmlxStatsCollector(manager))
    body = generate_latest(reg).decode()  # must not raise
    assert "olmlx_loaded_models 1.0" in body


# --- Task 6: HTTP helpers ---
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
