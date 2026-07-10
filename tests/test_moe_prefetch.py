"""Tests for olmlx.engine.flash.moe_prefetch — MoE expert prefetcher."""

from __future__ import annotations

import mlx.core as mx
import pytest

from tests.test_flash_moe_bundler import _make_synthetic_moe_weights


HIDDEN, INTER, EXPERTS = 64, 32, 8


@pytest.fixture()
def store(tmp_path):
    # 2 MoE layers (indices 1, 2 — layer 0 is dense) over 8 experts
    model_dir = _make_synthetic_moe_weights(HIDDEN, INTER, EXPERTS, 2, 1, tmp_path)
    output_dir = tmp_path / "flash_moe"

    from olmlx.engine.flash.moe_bundler import bundle_moe_experts

    bundle_moe_experts(model_dir, output_dir)

    from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

    s = FlashMoeWeightStore(output_dir, num_io_threads=4, cache_budget_experts=8)
    yield s
    s.close()


@pytest.fixture()
def bank():
    from olmlx.engine.flash.moe_predictor import MoeLookaheadBank

    return MoeLookaheadBank(
        [1, 2],
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        rank=4,
        num_experts_per_tok=2,
    )


def _make_prefetcher(bank, store, **kwargs):
    from olmlx.engine.flash.moe_prefetch import MoePrefetcher

    return MoePrefetcher(bank, store, **kwargs)


class TestMoePrefetcher:
    def test_submit_then_wait_warms_cache(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:
            hidden = mx.random.normal((1, 1, HIDDEN))
            pf.submit(1, hidden)
            pf.wait(2)
            assert pf.stats.submitted == 1
            # ceil(1.5 * 2) = 3 experts predicted; all landed in the cache
            assert pf.stats.cache_misses == 3
            cached, missing = store._cache.get_cached_indices(2, list(range(EXPERTS)))
            assert len(cached) == 3
        finally:
            pf.close()

    def test_wait_without_submit_returns(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:
            pf.wait(2)  # no pending entry — must not block or raise
        finally:
            pf.close()

    def test_submit_last_moe_layer_is_noop(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:
            pf.submit(2, mx.random.normal((1, 1, HIDDEN)))  # layer 2 has no successor
            pf.wait(2)
            assert pf.stats.submitted == 0
        finally:
            pf.close()

    def test_prefill_guard_skips_large_batches(self, bank, store):
        pf = _make_prefetcher(bank, store, max_positions=4)
        try:
            pf.submit(1, mx.random.normal((1, 5, HIDDEN)))  # 5 positions > 4
            pf.wait(2)
            assert pf.stats.submitted == 0
        finally:
            pf.close()

    def test_duplicate_submit_is_noop(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:
            hidden = mx.random.normal((1, 1, HIDDEN))
            pf.submit(1, hidden)
            pf.submit(1, hidden)  # second submit while first pending: no-op
            pf.wait(2)
            assert pf.stats.submitted == 1  # never double-registers pending
        finally:
            pf.close()

    def test_prediction_failure_unblocks_wait(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:

            def _boom(*a, **k):
                raise RuntimeError("synthetic predictor failure")

            bank.predict_next = _boom
            pf.submit(1, mx.random.normal((1, 1, HIDDEN)))
            pf.wait(2)  # must not hang
            assert pf.stats.failures == 1
        finally:
            pf.close()

    def test_score_push_failure_unblocks_wait(self, bank, store):
        """A raising set_layer_scores must not leave wait() hanging."""
        pf = _make_prefetcher(bank, store, scored_eviction=True)
        try:

            def _boom(*a, **k):
                raise RuntimeError("synthetic score-push failure")

            store.set_layer_scores = _boom
            pf.submit(1, mx.random.normal((1, 1, HIDDEN)))
            with pf._lock:
                state = pf._pending.get(2)
            assert state is not None
            assert state.done.wait(timeout=5.0), "wait(2) would hang forever"
            assert pf.stats.failures == 1
        finally:
            pf.close()

    def test_scored_eviction_pushes_scores(self, bank, store):
        pf = _make_prefetcher(bank, store, scored_eviction=True)
        try:
            pf.submit(1, mx.random.normal((1, 1, HIDDEN)))
            pf.wait(2)
            with store._cache._lock:
                assert 2 in store._cache._scores
                assert len(store._cache._scores[2]) == EXPERTS
        finally:
            pf.close()

    def test_scored_eviction_disabled_pushes_nothing(self, bank, store):
        pf = _make_prefetcher(bank, store, scored_eviction=False)
        try:
            pf.submit(1, mx.random.normal((1, 1, HIDDEN)))
            pf.wait(2)
            with store._cache._lock:
                assert 2 not in store._cache._scores
        finally:
            pf.close()

    def test_close_is_idempotent(self, bank, store):
        pf = _make_prefetcher(bank, store)
        pf.close()
        pf.close()


class TestPrefetchConfig:
    def test_flash_moe_config_prefetch_defaults(self):
        from olmlx.config import FlashMoeConfig

        cfg = FlashMoeConfig(enabled=True, cache_budget_experts=48, io_threads=32)
        assert cfg.prefetch is True
        assert cfg.lookahead_margin == 1.5
        assert cfg.prefetch_max_positions == 8
        assert cfg.scored_eviction is True

    def test_settings_prefetch_defaults(self, monkeypatch):
        from olmlx.config import Settings

        for var in (
            "OLMLX_FLASH_MOE_PREFETCH",
            "OLMLX_FLASH_MOE_LOOKAHEAD_MARGIN",
            "OLMLX_FLASH_MOE_PREFETCH_MAX_POSITIONS",
            "OLMLX_FLASH_MOE_SCORED_EVICTION",
        ):
            monkeypatch.delenv(var, raising=False)
        s = Settings(_env_file=None)
        assert s.flash_moe_prefetch is True
        assert s.flash_moe_lookahead_margin == 1.5
        assert s.flash_moe_prefetch_max_positions == 8
        assert s.flash_moe_scored_eviction is True
