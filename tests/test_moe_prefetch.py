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
        """A submit while the previous prefetch I/O is still in flight is a
        no-op. The I/O is gated on an event so 'in flight' is deterministic —
        with self-clearing pending entries, real I/O on tiny test experts can
        finish between the two submits."""
        import threading

        pf = _make_prefetcher(bank, store)
        gate = threading.Event()
        real_prefetch = store.prefetch_experts

        def _gated(layer_idx, indices):
            gate.wait(timeout=5.0)
            return real_prefetch(layer_idx, indices)

        store.prefetch_experts = _gated
        try:
            hidden = mx.random.normal((1, 1, HIDDEN))
            pf.submit(1, hidden)
            pf.submit(1, hidden)  # second submit while first pending: no-op
            gate.set()
            pf.wait(2)
            assert pf.stats.submitted == 1  # never double-registers pending
        finally:
            gate.set()
            pf.close()

    def test_pending_self_clears_after_io(self, bank, store):
        """Non-blocking prefetch: nothing calls wait() on the hot path, so a
        completed prefetch must clear its own pending entry — otherwise every
        later submit for the same layer dedups against the stale entry and
        prefetch silently stops after one pass."""
        import time

        pf = _make_prefetcher(bank, store)
        try:
            hidden = mx.random.normal((1, 1, HIDDEN))
            pf.submit(1, hidden)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with pf._lock:
                    if 2 not in pf._pending:
                        break
                time.sleep(0.005)
            else:
                pytest.fail("pending entry for layer 2 never self-cleared")
            pf.submit(1, hidden)  # must register again, not dedup
            pf.wait(2)
            assert pf.stats.submitted == 2
        finally:
            pf.close()

    def test_prediction_failure_unblocks_wait(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:

            def _boom(*a, **k):
                raise RuntimeError("synthetic predictor failure")

            bank.predict_next_np = _boom
            pf.submit(1, mx.random.normal((1, 1, HIDDEN)))
            pf.wait(2)  # must not hang
            assert pf.stats.failures == 1
        finally:
            pf.close()

    def test_prediction_failure_registers_no_pending(self, bank, store):
        """Inline prediction: a failing predictor must leave no pending entry."""
        pf = _make_prefetcher(bank, store)
        try:

            def _boom(*a, **k):
                raise RuntimeError("synthetic predictor failure")

            bank.predict_next_np = _boom
            pf.submit(1, mx.random.normal((1, 1, HIDDEN)))
            with pf._lock:
                assert 2 not in pf._pending
            assert pf.stats.submitted == 0
        finally:
            pf.close()

    def test_prediction_is_inline_no_background_eval(self, bank, store):
        """Prediction runs synchronously in submit() on the calling thread —
        scores are pushed before submit() returns, and no thread other than
        the caller ever touches mx (eval-avoidance contract)."""
        pf = _make_prefetcher(bank, store, scored_eviction=True)
        try:
            pf.submit(1, mx.random.normal((1, 1, HIDDEN)))
            # No wait(): scores must already be pushed by the inline prediction.
            with store._cache._lock:
                assert 2 in store._cache._scores
            assert pf.stats.submitted == 1
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
        assert cfg.prefetch is False
        assert cfg.lookahead_margin == 1.5
        assert cfg.prefetch_max_positions == 8
        assert cfg.scored_eviction is True
        assert cfg.prefetch_min_recall == 0.0

    def test_settings_prefetch_defaults(self, monkeypatch):
        from olmlx.config import Settings

        for var in (
            "OLMLX_FLASH_MOE_PREFETCH",
            "OLMLX_FLASH_MOE_LOOKAHEAD_MARGIN",
            "OLMLX_FLASH_MOE_PREFETCH_MAX_POSITIONS",
            "OLMLX_FLASH_MOE_SCORED_EVICTION",
            "OLMLX_FLASH_MOE_PREFETCH_MIN_RECALL",
        ):
            monkeypatch.delenv(var, raising=False)
        s = Settings(_env_file=None)
        assert s.flash_moe_prefetch is False
        assert s.flash_moe_lookahead_margin == 1.5
        assert s.flash_moe_prefetch_max_positions == 8
        assert s.flash_moe_scored_eviction is True
        assert s.flash_moe_prefetch_min_recall == 0.0
