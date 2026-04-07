"""Tests for speculative prefetching (cross-layer, draft-informed, lookahead)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from olmlx.engine.flash.flash_mlp import FlashMLP, WindowManager
from olmlx.engine.flash.prefetch import Prefetcher, PrefetchStats
from olmlx.engine.flash.predictor import (
    LookaheadBank,
    LookaheadPredictor,
    PredictorBank,
    SparsityPredictor,
)
from olmlx.engine.flash.weight_store import FlashWeightStore, NeuronCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundled_model(tmp_path, hidden=16, inter=8, num_layers=2):
    """Create synthetic safetensors and bundle them for flash inference."""
    from safetensors.numpy import save_file

    from olmlx.engine.flash.bundler import bundle_ffn_weights

    tensors = {}
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}.mlp"
        tensors[f"{prefix}.gate_proj.weight"] = np.random.randn(inter, hidden).astype(
            np.float16
        )
        tensors[f"{prefix}.up_proj.weight"] = np.random.randn(inter, hidden).astype(
            np.float16
        )
        tensors[f"{prefix}.down_proj.weight"] = np.random.randn(hidden, inter).astype(
            np.float16
        )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(tensors, str(model_dir / "model.safetensors"))

    flash_dir = tmp_path / "flash"
    bundle_ffn_weights(model_dir, flash_dir)
    return flash_dir, model_dir, tensors


# ---------------------------------------------------------------------------
# NeuronCache.get_cached_indices tests
# ---------------------------------------------------------------------------


class TestNeuronCacheGetCachedIndices:
    def test_empty_cache(self):
        cache = NeuronCache(max_neurons_per_layer=64)
        cached, missing = cache.get_cached_indices(0, [0, 1, 2])
        assert cached == []
        assert missing == [0, 1, 2]

    def test_all_cached(self):
        cache = NeuronCache(max_neurons_per_layer=64)
        for i in range(3):
            cache.put(0, i, (mx.zeros(4), mx.zeros(4), mx.zeros(4)))
        cached, missing = cache.get_cached_indices(0, [0, 1, 2])
        assert cached == [0, 1, 2]
        assert missing == []

    def test_partial_cache(self):
        cache = NeuronCache(max_neurons_per_layer=64)
        cache.put(0, 0, (mx.zeros(4), mx.zeros(4), mx.zeros(4)))
        cache.put(0, 2, (mx.zeros(4), mx.zeros(4), mx.zeros(4)))
        cached, missing = cache.get_cached_indices(0, [0, 1, 2])
        assert cached == [0, 2]
        assert missing == [1]

    def test_wrong_layer(self):
        cache = NeuronCache(max_neurons_per_layer=64)
        cache.put(0, 0, (mx.zeros(4), mx.zeros(4), mx.zeros(4)))
        cached, missing = cache.get_cached_indices(1, [0])
        assert cached == []
        assert missing == [0]


# ---------------------------------------------------------------------------
# FlashWeightStore.prefetch_neurons tests
# ---------------------------------------------------------------------------


class TestPrefetchNeurons:
    @pytest.fixture()
    def store_setup(self, tmp_path):
        hidden, inter, num_layers = 16, 8, 2
        flash_dir, _, _ = _make_bundled_model(tmp_path, hidden, inter, num_layers)
        store = FlashWeightStore(flash_dir, num_io_threads=4, cache_budget_neurons=64)
        return store, hidden, inter

    def test_prefetch_warms_cache(self, store_setup):
        """After prefetch, load_neurons should find everything cached."""
        store, hidden, inter = store_setup
        neuron_indices = list(range(inter))

        # Prefetch neurons
        store.prefetch_neurons(0, neuron_indices)

        # Verify they're now cached by checking load_neurons is instant
        gate, up, down = store.load_neurons(0, neuron_indices)
        mx.eval(gate, up, down)
        assert gate.shape == (hidden, inter)
        assert up.shape == (hidden, inter)
        assert down.shape == (inter, hidden)

    def test_prefetch_idempotent(self, store_setup):
        """Prefetching the same neurons twice should not error."""
        store, _, inter = store_setup
        neuron_indices = list(range(inter))
        store.prefetch_neurons(0, neuron_indices)
        store.prefetch_neurons(0, neuron_indices)  # should be a no-op

    def test_prefetch_empty_list(self, store_setup):
        """Prefetching empty list should be a no-op."""
        store, _, _ = store_setup
        store.prefetch_neurons(0, [])  # should not error

    def test_prefetch_preallocated_buffer(self, tmp_path):
        """prefetch_neurons should work with preallocated buffer path."""
        hidden, inter, num_layers = 16, 8, 2
        flash_dir, _, _ = _make_bundled_model(tmp_path, hidden, inter, num_layers)
        store = FlashWeightStore(
            flash_dir,
            num_io_threads=4,
            cache_budget_neurons=64,
            use_preallocated_buffer=True,
        )
        neuron_indices = list(range(inter))
        store.prefetch_neurons(0, neuron_indices)

        # Verify they're cached
        gate, up, down = store.load_neurons(0, neuron_indices)
        mx.eval(gate, up, down)
        assert gate.shape == (hidden, inter)

    def test_prefetch_preallocated_does_not_hold_lock_during_io(self, tmp_path):
        """buf.lock must not be held while waiting on I/O futures."""
        import threading
        import time

        hidden, inter, num_layers = 16, 8, 2
        flash_dir, _, _ = _make_bundled_model(tmp_path, hidden, inter, num_layers)
        store = FlashWeightStore(
            flash_dir,
            num_io_threads=4,
            cache_budget_neurons=64,
            use_preallocated_buffer=True,
        )

        buf = store._buffers[0]
        lock_held_during_io = threading.Event()

        original_read = store._read_neuron_raw

        def slow_read(layer_idx, neuron_idx):
            time.sleep(0.05)  # simulate slow I/O
            return original_read(layer_idx, neuron_idx)

        store._read_neuron_raw = slow_read

        # Start prefetch in background, then try to acquire lock during I/O
        t = threading.Thread(
            target=store.prefetch_neurons, args=(0, list(range(inter)))
        )
        t.start()
        time.sleep(0.02)  # let I/O start

        # If lock is free during I/O, we can acquire it immediately
        if not buf.lock.acquire(timeout=0.2):
            lock_held_during_io.set()
        else:
            buf.lock.release()

        t.join(timeout=5.0)

        assert not lock_held_during_io.is_set(), (
            "buf.lock was held while I/O futures were in progress"
        )


# ---------------------------------------------------------------------------
# Prefetcher tests (Phase 1 — cross-layer)
# ---------------------------------------------------------------------------


class TestPrefetcher:
    @pytest.fixture()
    def prefetch_setup(self, tmp_path):
        hidden, inter, num_layers = 16, 8, 2
        flash_dir, _, _ = _make_bundled_model(tmp_path, hidden, inter, num_layers)
        store = FlashWeightStore(flash_dir, num_io_threads=4, cache_budget_neurons=64)
        bank = PredictorBank(num_layers, hidden, inter, rank=4)
        prefetcher = Prefetcher(
            predictor_bank=bank,
            weight_store=store,
            num_layers=num_layers,
            confidence_threshold=0.3,
            min_neurons=2,
            io_threads=4,
        )
        return prefetcher, store, bank, hidden, inter, num_layers

    def test_submit_and_wait(self, prefetch_setup):
        """submit() for layer 0 should prefetch layer 1 neurons."""
        prefetcher, store, _, hidden, _, _ = prefetch_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        prefetcher.submit(0, x)
        prefetcher.wait(1)  # exercises the blocking wait() path
        prefetcher.close()  # drain any residual background work

        # Stats should show submission
        assert prefetcher.stats.submitted >= 1

    def test_submit_last_layer_noop(self, prefetch_setup):
        """submit() for the last layer should be a no-op."""
        prefetcher, _, _, hidden, _, num_layers = prefetch_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        initial_submitted = prefetcher.stats.submitted
        prefetcher.submit(num_layers - 1, x)
        assert prefetcher.stats.submitted == initial_submitted

    def test_wait_without_submit(self, prefetch_setup):
        """wait() with no pending prefetch should return immediately."""
        prefetcher, _, _, _, _, _ = prefetch_setup
        prefetcher.wait(0)  # should not hang

    def test_cancel_clears_pending(self, prefetch_setup):
        """cancel() should clear all pending prefetches."""
        prefetcher, _, _, hidden, _, _ = prefetch_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        prefetcher.submit(0, x)
        prefetcher.cancel()
        # wait should not find anything
        prefetcher.wait(1)

    def test_submit_bulk(self, prefetch_setup):
        """submit_bulk should prefetch neurons for multiple layers."""
        prefetcher, _, _, hidden, _, num_layers = prefetch_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        layer_states = {i: x for i in range(num_layers)}
        prefetcher.submit_bulk(layer_states)

        for i in range(num_layers):
            prefetcher.wait(i)
        prefetcher.close()

        assert prefetcher.stats.submitted >= 1

    def test_io_failure_increments_counter(self, prefetch_setup):
        """Prefetch I/O failure should increment stats.failures."""
        prefetcher, store, _, hidden, _, _ = prefetch_setup
        # Monkey-patch weight_store to raise on prefetch
        original = store.prefetch_neurons
        store.prefetch_neurons = lambda *a, **kw: (_ for _ in ()).throw(
            IOError("disk read failed")
        )

        x = mx.random.normal((1, hidden)).astype(mx.float16)
        prefetcher.submit(0, x)
        prefetcher.wait(1)
        prefetcher.close()

        assert prefetcher.stats.failures >= 1
        store.prefetch_neurons = original

    def test_submit_after_close_does_not_hang(self, prefetch_setup):
        """If executor.submit() fails, wait() must not hang."""
        prefetcher, _, _, hidden, _, _ = prefetch_setup
        prefetcher.close()  # shut down executor so submit() will fail

        x = mx.random.normal((1, hidden)).astype(mx.float16)
        prefetcher.submit(0, x)

        # wait() must return promptly, not block forever
        import threading

        completed = threading.Event()

        def _wait():
            prefetcher.wait(1)
            completed.set()

        t = threading.Thread(target=_wait)
        t.start()
        t.join(timeout=2.0)
        assert completed.is_set(), "wait() hung after executor.submit() failure"

    def test_close(self, prefetch_setup):
        """close() should shut down executor without error."""
        prefetcher, _, _, _, _, _ = prefetch_setup
        prefetcher.close()


# ---------------------------------------------------------------------------
# Async prediction tests (#148)
# ---------------------------------------------------------------------------


class TestAsyncPrediction:
    """Tests for moving prediction off the forward-pass thread."""

    @pytest.fixture()
    def prefetch_setup(self, tmp_path):
        hidden, inter, num_layers = 16, 8, 2
        flash_dir, _, _ = _make_bundled_model(tmp_path, hidden, inter, num_layers)
        store = FlashWeightStore(flash_dir, num_io_threads=4, cache_budget_neurons=64)
        bank = PredictorBank(num_layers, hidden, inter, rank=4)
        prefetcher = Prefetcher(
            predictor_bank=bank,
            weight_store=store,
            num_layers=num_layers,
            confidence_threshold=0.3,
            min_neurons=2,
            io_threads=4,
        )
        return prefetcher, store, bank, hidden, inter, num_layers

    def test_submit_returns_immediately(self, prefetch_setup):
        """submit() should return before prediction completes."""
        import time
        from unittest.mock import patch

        prefetcher, _, _, hidden, _, _ = prefetch_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        original_predict = prefetcher._predict

        def slow_predict(*args, **kwargs):
            time.sleep(0.5)
            return original_predict(*args, **kwargs)

        with patch.object(prefetcher, "_predict", side_effect=slow_predict):
            start = time.monotonic()
            prefetcher.submit(0, x)
            elapsed = time.monotonic() - start

        assert elapsed < 0.1, f"submit() took {elapsed:.3f}s, should return immediately"
        # Clean up: wait for background work to finish
        prefetcher.close()

    def test_prediction_runs_on_background_thread(self, prefetch_setup):
        """Prediction should run on the prediction thread, not the calling thread."""
        import threading
        from unittest.mock import patch

        prefetcher, _, _, hidden, _, _ = prefetch_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        predict_thread_name = []
        original_predict = prefetcher._predict

        def capture_thread(*args, **kwargs):
            predict_thread_name.append(threading.current_thread().name)
            return original_predict(*args, **kwargs)

        with patch.object(prefetcher, "_predict", side_effect=capture_thread):
            prefetcher.submit(0, x)
            prefetcher.close()  # wait for prediction to complete

        assert len(predict_thread_name) == 1
        assert "prefetch-predict" in predict_thread_name[0], (
            f"Prediction ran on {predict_thread_name[0]!r}, expected prefetch-predict thread"
        )

    def test_submit_and_wait_async(self, prefetch_setup):
        """submit + close should ensure prediction and I/O complete."""
        prefetcher, _, _, hidden, _, _ = prefetch_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        prefetcher.submit(0, x)
        # close() drains both prediction and I/O executors
        prefetcher.close()

        assert prefetcher.stats.submitted >= 1

    def test_submit_bulk_synchronous(self, prefetch_setup):
        """submit_bulk runs predictions synchronously (all I/O queued before return)."""
        prefetcher, _, _, hidden, _, num_layers = prefetch_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        layer_states = {i: x for i in range(num_layers)}
        prefetcher.submit_bulk(layer_states)

        # All I/O should be queued already — wait() should not hang
        for i in range(num_layers):
            prefetcher.wait(i)
        prefetcher.close()

        assert prefetcher.stats.submitted >= 1

    def test_prediction_failure_no_hang(self, prefetch_setup):
        """If prediction raises, wait() should not hang."""
        import threading
        from unittest.mock import patch

        prefetcher, _, _, hidden, _, _ = prefetch_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        entered = threading.Event()

        def failing_predict(*args, **kwargs):
            entered.set()
            raise RuntimeError("prediction boom")

        with patch.object(prefetcher, "_predict", side_effect=failing_predict):
            prefetcher.submit(0, x)
            entered.wait(timeout=2.0)  # ensure mock ran before patch is restored

        # wait() should return promptly (done was set by error handler)
        completed = threading.Event()

        def _wait():
            prefetcher.wait(1)
            completed.set()

        t = threading.Thread(target=_wait)
        t.start()
        t.join(timeout=2.0)
        assert completed.is_set(), "wait() hung after prediction failure"
        prefetcher.close()

    def test_close_drains_prediction_queue(self, prefetch_setup):
        """close() should block until in-flight predictions complete."""
        import time
        from unittest.mock import patch

        prefetcher, _, _, hidden, _, _ = prefetch_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        original_predict = prefetcher._predict
        predict_completed = False

        def slow_predict(*args, **kwargs):
            nonlocal predict_completed
            time.sleep(0.3)
            result = original_predict(*args, **kwargs)
            predict_completed = True
            return result

        with patch.object(prefetcher, "_predict", side_effect=slow_predict):
            prefetcher.submit(0, x)
            prefetcher.close()

        assert predict_completed, "close() returned before prediction finished"

    def test_submit_after_close_no_error(self, prefetch_setup):
        """submit() after close() should silently drop, not raise."""
        prefetcher, _, _, hidden, _, _ = prefetch_setup
        prefetcher.close()

        x = mx.random.normal((1, hidden)).astype(mx.float16)
        # Should not raise
        prefetcher.submit(0, x)


# ---------------------------------------------------------------------------
# FlashMLP + Prefetcher integration tests
# ---------------------------------------------------------------------------


class TestFlashMLPWithPrefetcher:
    @pytest.fixture()
    def mlp_setup(self, tmp_path):
        hidden, inter, num_layers = 16, 8, 2
        flash_dir, _, _ = _make_bundled_model(tmp_path, hidden, inter, num_layers)
        store = FlashWeightStore(flash_dir, num_io_threads=4, cache_budget_neurons=64)
        bank = PredictorBank(num_layers, hidden, inter, rank=4)
        wm = WindowManager(num_layers=num_layers, window_size=3)
        prefetcher = Prefetcher(
            predictor_bank=bank,
            weight_store=store,
            num_layers=num_layers,
            confidence_threshold=0.3,
            min_neurons=2,
            io_threads=4,
        )

        mlps = []
        for i in range(num_layers):
            mlp = FlashMLP(
                layer_idx=i,
                hidden_size=hidden,
                intermediate_size=inter,
                predictor=bank.predictors[i],
                weight_store=store,
                window_manager=wm,
                sparsity_threshold=0.5,
                min_active_neurons=inter,
                prefetcher=prefetcher,
            )
            mlps.append(mlp)
        return mlps, prefetcher, hidden

    def test_forward_with_prefetcher(self, mlp_setup):
        """FlashMLP should produce valid output when prefetcher is attached."""
        mlps, prefetcher, hidden = mlp_setup
        x = mx.random.normal((1, 1, hidden)).astype(mx.float16)

        # Run both layers without mx.eval between them — prediction runs
        # on a background thread and mx.eval is not safe for concurrent calls.
        out0 = mlps[0](x)
        out1 = mlps[1](out0)
        mx.eval(out1)
        assert out0.shape == (1, 1, hidden)
        assert out1.shape == (1, 1, hidden)

        # Prefetcher should have submitted at least 1 prefetch
        prefetcher.close()
        assert prefetcher.stats.submitted >= 1

    def test_forward_without_prefetcher(self, tmp_path):
        """FlashMLP should work normally when prefetcher is None."""
        hidden, inter, num_layers = 16, 8, 1
        flash_dir, _, _ = _make_bundled_model(tmp_path, hidden, inter, num_layers)
        store = FlashWeightStore(flash_dir, num_io_threads=2, cache_budget_neurons=64)
        pred = SparsityPredictor(hidden, inter, rank=4)
        wm = WindowManager(num_layers=1, window_size=3)

        mlp = FlashMLP(
            layer_idx=0,
            hidden_size=hidden,
            intermediate_size=inter,
            predictor=pred,
            weight_store=store,
            window_manager=wm,
            sparsity_threshold=0.5,
            min_active_neurons=inter,
            prefetcher=None,
        )

        x = mx.random.normal((1, 1, hidden)).astype(mx.float16)
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (1, 1, hidden)


# ---------------------------------------------------------------------------
# LookaheadPredictor tests (Phase 3)
# ---------------------------------------------------------------------------


class TestLookaheadPredictor:
    def test_output_shape(self):
        hidden, inter = 16, 8
        pred = LookaheadPredictor(hidden, inter, rank=4)
        x = mx.random.normal((2, hidden))
        scores = pred(x)
        mx.eval(scores)
        assert scores.shape == (2, inter)
        # Scores should be in [0, 1] (sigmoid)
        assert float(mx.min(scores).item()) >= 0
        assert float(mx.max(scores).item()) <= 1

    def test_predict_active_returns_sorted_indices(self):
        hidden, inter = 16, 8
        pred = LookaheadPredictor(hidden, inter, rank=4)
        x = mx.random.normal((2, hidden))
        indices = pred.predict_active(x, threshold=0.3, min_neurons=2)
        mx.eval(indices)
        assert indices.ndim == 1
        assert len(indices) >= 2
        # Verify sorted
        idx_list = indices.tolist()
        assert idx_list == sorted(idx_list)

    def test_predict_active_min_neurons(self):
        hidden, inter = 16, 8
        pred = LookaheadPredictor(hidden, inter, rank=4)
        x = mx.random.normal((1, hidden))
        indices = pred.predict_active(x, threshold=0.99, min_neurons=4)
        mx.eval(indices)
        assert len(indices) >= 4

    def test_predict_active_max_neurons(self):
        hidden, inter = 16, 8
        pred = LookaheadPredictor(hidden, inter, rank=4)
        x = mx.random.normal((1, hidden))
        indices = pred.predict_active(x, threshold=0.01, min_neurons=1, max_neurons=3)
        mx.eval(indices)
        assert len(indices) <= 3


class TestLookaheadBank:
    def test_save_load_roundtrip(self, tmp_path):
        hidden, inter, num_layers = 16, 8, 3
        bank = LookaheadBank(num_layers, hidden, inter, rank=4)

        # Set known weights
        for i, pred in enumerate(bank.predictors):
            pred.down.weight = mx.ones_like(pred.down.weight) * (i + 1)
            pred.up.weight = mx.ones_like(pred.up.weight) * (i + 1) * 0.1

        save_dir = tmp_path / "lookahead"
        bank.save(save_dir)

        loaded = LookaheadBank.load(save_dir)
        assert loaded.num_layers == num_layers
        assert len(loaded.predictors) == num_layers - 1

        for i in range(num_layers - 1):
            mx.eval(loaded.predictors[i].parameters())
            assert mx.allclose(
                loaded.predictors[i].down.weight,
                bank.predictors[i].down.weight,
                atol=1e-5,
            )

    def test_load_preserves_all_init_attributes(self, tmp_path):
        """LookaheadBank.load() must set all attributes that __init__ sets."""
        hidden, inter, num_layers = 16, 8, 3
        bank = LookaheadBank(num_layers, hidden, inter, rank=4)
        save_dir = tmp_path / "lookahead_attrs"
        bank.save(save_dir)

        loaded = LookaheadBank.load(save_dir)

        # All attributes from __init__ must exist on the loaded bank
        init_bank = LookaheadBank(num_layers, hidden, inter, rank=4)
        for attr in vars(init_bank):
            assert hasattr(loaded, attr), f"Loaded bank missing attribute: {attr}"

    def test_predict_next_layer(self):
        hidden, inter, num_layers = 16, 8, 3
        bank = LookaheadBank(num_layers, hidden, inter, rank=4)
        x = mx.random.normal((1, hidden))

        indices = bank.predict_next_layer(0, x, threshold=0.3, min_neurons=2)
        mx.eval(indices)
        assert len(indices) >= 2

    def test_predict_last_layer_returns_empty(self):
        hidden, inter, num_layers = 16, 8, 3
        bank = LookaheadBank(num_layers, hidden, inter, rank=4)
        x = mx.random.normal((1, hidden))

        # Layer 2 is the last (index num_layers-1), its predictor index = 2
        # which is >= len(predictors) = 2
        indices = bank.predict_next_layer(num_layers - 1, x, min_neurons=1)
        mx.eval(indices)
        assert len(indices) == 0

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LookaheadBank.load(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Prefetcher with LookaheadBank tests (Phase 3)
# ---------------------------------------------------------------------------


class TestPrefetcherWithLookahead:
    @pytest.fixture()
    def prefetch_la_setup(self, tmp_path):
        hidden, inter, num_layers = 16, 8, 3
        flash_dir, _, _ = _make_bundled_model(tmp_path, hidden, inter, num_layers)
        store = FlashWeightStore(flash_dir, num_io_threads=4, cache_budget_neurons=64)
        sparsity_bank = PredictorBank(num_layers, hidden, inter, rank=4)
        lookahead_bank = LookaheadBank(num_layers, hidden, inter, rank=4)
        prefetcher = Prefetcher(
            predictor_bank=sparsity_bank,
            weight_store=store,
            num_layers=num_layers,
            lookahead_bank=lookahead_bank,
            confidence_threshold=0.3,
            min_neurons=2,
            io_threads=4,
        )
        return prefetcher, store, hidden, num_layers

    def test_submit_uses_lookahead(self, prefetch_la_setup):
        """When lookahead_bank is set, submit should use it for cross-layer."""
        prefetcher, _, hidden, _ = prefetch_la_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        prefetcher.submit(0, x)
        prefetcher.wait(1)
        assert prefetcher.stats.submitted >= 1

    def test_submit_bulk_uses_sparsity(self, prefetch_la_setup):
        """submit_bulk always uses the sparsity predictor (not lookahead)."""
        prefetcher, _, hidden, num_layers = prefetch_la_setup
        x = mx.random.normal((1, hidden)).astype(mx.float16)

        layer_states = {i: x for i in range(num_layers)}
        prefetcher.submit_bulk(layer_states)

        for i in range(num_layers):
            prefetcher.wait(i)
        assert prefetcher.stats.submitted >= 1


# ---------------------------------------------------------------------------
# Lookahead predictor training tests (Phase 3 — prepare.py)
# ---------------------------------------------------------------------------


class TestTrainLookaheadPredictors:
    def test_train_basic(self):
        from olmlx.engine.flash.prepare import _train_lookahead_predictors

        hidden, inter = 16, 8
        num_layers = 3
        num_samples = 10

        # Create fake recordings: (inputs, targets) per layer
        recordings = {}
        for layer_idx in range(num_layers):
            inputs = [mx.random.normal((hidden,)) for _ in range(num_samples)]
            targets = [
                (mx.random.uniform(shape=(inter,)) > 0.5).astype(mx.float32)
                for _ in range(num_samples)
            ]
            recordings[layer_idx] = (inputs, targets)

        bank = _train_lookahead_predictors(
            recordings,
            hidden,
            inter,
            rank=4,
            epochs=2,
        )

        assert bank is not None
        assert len(bank.predictors) == num_layers - 1

    def test_train_single_layer_returns_none(self):
        from olmlx.engine.flash.prepare import _train_lookahead_predictors

        recordings = {
            0: (
                [mx.random.normal((16,))],
                [(mx.random.uniform(shape=(8,)) > 0.5).astype(mx.float32)],
            )
        }
        bank = _train_lookahead_predictors(recordings, 16, 8, rank=4, epochs=1)
        assert bank is None

    def test_train_with_progress(self):
        from olmlx.engine.flash.prepare import _train_lookahead_predictors

        hidden, inter, num_layers = 16, 8, 2
        recordings = {}
        for i in range(num_layers):
            recordings[i] = (
                [mx.random.normal((hidden,)) for _ in range(5)],
                [
                    (mx.random.uniform(shape=(inter,)) > 0.5).astype(mx.float32)
                    for _ in range(5)
                ],
            )

        progress_calls = []

        def callback(desc, frac):
            progress_calls.append((desc, frac))

        bank = _train_lookahead_predictors(
            recordings, hidden, inter, rank=4, epochs=2, progress_callback=callback
        )
        assert bank is not None
        assert len(progress_calls) > 0

    def test_train_warns_on_recording_count_mismatch(self, caplog):
        """Should warn when input/target recording counts differ across layers."""
        from olmlx.engine.flash.prepare import _train_lookahead_predictors

        hidden, inter = 16, 8
        recordings = {
            0: (
                [mx.random.normal((hidden,)) for _ in range(10)],
                [
                    (mx.random.uniform(shape=(inter,)) > 0.5).astype(mx.float32)
                    for _ in range(10)
                ],
            ),
            1: (
                [mx.random.normal((hidden,)) for _ in range(5)],
                [
                    (mx.random.uniform(shape=(inter,)) > 0.5).astype(mx.float32)
                    for _ in range(3)  # fewer targets than inputs in layer 0
                ],
            ),
        }

        with caplog.at_level("WARNING", logger="olmlx.engine.flash.prepare"):
            bank = _train_lookahead_predictors(
                recordings, hidden, inter, rank=4, epochs=1
            )
        assert bank is not None
        assert any("Recording count mismatch" in r.message for r in caplog.records)

    def test_train_predictors_progress_per_epoch(self):
        """_train_predictors should report progress per epoch, not per layer."""
        from olmlx.engine.flash.prepare import _train_predictors

        hidden, inter, num_layers = 16, 8, 3
        recordings = {}
        for i in range(num_layers):
            recordings[i] = (
                [mx.random.normal((hidden,)) for _ in range(5)],
                [
                    (mx.random.uniform(shape=(inter,)) > 0.5).astype(mx.float32)
                    for _ in range(5)
                ],
            )

        progress_calls = []

        def callback(desc, frac):
            progress_calls.append((desc, frac))

        epochs = 3
        _train_predictors(
            recordings,
            hidden,
            inter,
            rank=4,
            epochs=epochs,
            progress_callback=callback,
        )

        # Should have num_layers * epochs callbacks (one per epoch),
        # not just num_layers callbacks (one per layer)
        assert len(progress_calls) == num_layers * epochs


# ---------------------------------------------------------------------------
# FlashModelWrapper + Prefetcher integration tests
# ---------------------------------------------------------------------------


class TestFlashModelWrapperPrefetch:
    def test_wrapper_creates_prefetcher(self, tmp_path):
        from olmlx.engine.flash.flash_model import FlashConfig, FlashModelWrapper

        dim, num_layers, inter = 16, 2, 8
        flash_dir, _, _ = _make_bundled_model(tmp_path, dim, inter, num_layers)
        store = FlashWeightStore(flash_dir, num_io_threads=2, cache_budget_neurons=64)

        from types import SimpleNamespace

        pred_bank = SimpleNamespace(
            predictors=[
                SparsityPredictor(dim, inter, rank=4) for _ in range(num_layers)
            ]
        )
        config = FlashConfig(
            hidden_size=dim,
            intermediate_size=inter,
            num_layers=num_layers,
            prefetch=True,
            prefetch_confidence_threshold=0.3,
            prefetch_min_neurons=2,
            prefetch_io_threads=4,
        )

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [
                    SimpleNamespace(mlp=nn.Linear(dim, dim)) for _ in range(num_layers)
                ]

            def __call__(self, x, cache=None, **kwargs):
                return x

        model = _FakeModel()
        wrapper = FlashModelWrapper(model, pred_bank, store, config)
        assert wrapper.prefetcher is not None

        # Each FlashMLP should have the prefetcher set
        for layer in model.layers:
            assert isinstance(layer.mlp, FlashMLP)
            assert layer.mlp.prefetcher is wrapper.prefetcher

    def test_wrapper_no_prefetcher_by_default(self, tmp_path):
        from olmlx.engine.flash.flash_model import FlashConfig, FlashModelWrapper

        dim, num_layers, inter = 16, 2, 8
        flash_dir, _, _ = _make_bundled_model(tmp_path, dim, inter, num_layers)
        store = FlashWeightStore(flash_dir, num_io_threads=2, cache_budget_neurons=64)

        from types import SimpleNamespace

        pred_bank = SimpleNamespace(
            predictors=[
                SparsityPredictor(dim, inter, rank=4) for _ in range(num_layers)
            ]
        )
        config = FlashConfig(
            hidden_size=dim,
            intermediate_size=inter,
            num_layers=num_layers,
            prefetch=False,
        )

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [
                    SimpleNamespace(mlp=nn.Linear(dim, dim)) for _ in range(num_layers)
                ]

            def __call__(self, x, cache=None, **kwargs):
                return x

        model = _FakeModel()
        wrapper = FlashModelWrapper(model, pred_bank, store, config)
        assert wrapper.prefetcher is None

    def test_wrapper_with_lookahead_bank(self, tmp_path):
        from olmlx.engine.flash.flash_model import FlashConfig, FlashModelWrapper

        dim, num_layers, inter = 16, 2, 8
        flash_dir, _, _ = _make_bundled_model(tmp_path, dim, inter, num_layers)
        store = FlashWeightStore(flash_dir, num_io_threads=2, cache_budget_neurons=64)

        from types import SimpleNamespace

        pred_bank = SimpleNamespace(
            predictors=[
                SparsityPredictor(dim, inter, rank=4) for _ in range(num_layers)
            ]
        )
        la_bank = LookaheadBank(num_layers, dim, inter, rank=4)
        config = FlashConfig(
            hidden_size=dim,
            intermediate_size=inter,
            num_layers=num_layers,
            prefetch=True,
            prefetch_min_neurons=2,
            prefetch_io_threads=4,
        )

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [
                    SimpleNamespace(mlp=nn.Linear(dim, dim)) for _ in range(num_layers)
                ]

            def __call__(self, x, cache=None, **kwargs):
                return x

        model = _FakeModel()
        wrapper = FlashModelWrapper(model, pred_bank, store, config, la_bank)
        assert wrapper.prefetcher is not None
        assert wrapper.prefetcher._lookahead_bank is la_bank


# ---------------------------------------------------------------------------
# SpeculativeFlashDecoder + Prefetcher (Phase 2) tests
# ---------------------------------------------------------------------------


class TestSpeculativeDecoderPrefetch:
    def test_decoder_accepts_prefetcher(self):
        from olmlx.engine.flash.speculative import SpeculativeFlashDecoder

        class _FakeModel(nn.Module):
            def __init__(self, vocab=32):
                super().__init__()
                self.lm_head = nn.Linear(16, vocab, bias=False)

            def __call__(self, x, cache=None, **kwargs):
                return mx.random.normal((x.shape[0], x.shape[1], 32))

        draft = _FakeModel()
        target = _FakeModel()

        # Should accept prefetcher kwarg without error
        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=2,
            prefetcher=None,
        )
        assert decoder._prefetcher is None

    def test_decoder_draft_captures_hidden_states_with_prefetcher(self):
        """When prefetcher is attached, draft generation should capture hidden states."""
        from unittest.mock import MagicMock

        from olmlx.engine.flash.speculative import SpeculativeFlashDecoder

        hidden = 16
        vocab = 32

        class _FakeInnerModel(nn.Module):
            def __call__(self, x, cache=None, **kwargs):
                return mx.random.normal((x.shape[0], x.shape[1], hidden))

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _FakeInnerModel()
                self.lm_head = nn.Linear(hidden, vocab, bias=False)

            def __call__(self, x, cache=None, **kwargs):
                h = self.model(x, cache=cache)
                return self.lm_head(h)

        draft = _FakeModel()
        target = _FakeModel()
        prefetcher = MagicMock()
        prefetcher.num_layers = 2

        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=2,
            prefetcher=prefetcher,
        )

        # Simulate cached mode
        decoder._draft_cache = []
        decoder._target_cache = []
        tokens, captured_hidden = decoder._draft_generate_cached(0, 2)

        assert len(tokens) == 2
        assert len(captured_hidden) == 2
        # Each captured tensor should be (1, hidden_size), not (1, vocab_size)
        for h in captured_hidden:
            assert h.shape[-1] == hidden

    def test_decoder_draft_fallback_no_inner_model(self):
        """Draft model without .model attribute: no capture, tokens still generated."""
        from unittest.mock import MagicMock

        from olmlx.engine.flash.speculative import SpeculativeFlashDecoder

        vocab = 32

        class _FlatModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(16, vocab, bias=False)

            def __call__(self, x, cache=None, **kwargs):
                return mx.random.normal((x.shape[0], x.shape[1], vocab))

        draft = _FlatModel()
        target = _FlatModel()
        prefetcher = MagicMock()
        prefetcher.num_layers = 2

        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=2,
            prefetcher=prefetcher,
        )

        decoder._draft_cache = []
        decoder._target_cache = []
        tokens, captured_hidden = decoder._draft_generate_cached(0, 2)

        assert len(tokens) == 2
        assert len(captured_hidden) == 0  # gracefully skipped

    def test_decoder_draft_no_capture_without_prefetcher(self):
        """Without prefetcher, draft generation should return empty list."""
        from olmlx.engine.flash.speculative import SpeculativeFlashDecoder

        vocab = 32

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(16, vocab, bias=False)

            def __call__(self, x, cache=None, **kwargs):
                return mx.random.normal((x.shape[0], x.shape[1], vocab))

        draft = _FakeModel()
        target = _FakeModel()

        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=2,
            prefetcher=None,
        )

        decoder._draft_cache = []
        decoder._target_cache = []
        tokens, captured = decoder._draft_generate_cached(0, 2)

        assert len(tokens) == 2
        assert len(captured) == 0

    def test_submit_draft_prefetch_dimension_mismatch(self):
        """When draft hidden_size != target hidden_size, submit_bulk is not called."""
        from unittest.mock import MagicMock

        from olmlx.engine.flash.speculative import SpeculativeFlashDecoder

        class _FakeModel(nn.Module):
            def __call__(self, x, cache=None, **kwargs):
                return mx.random.normal((x.shape[0], x.shape[1], 32))

        draft = _FakeModel()
        target = _FakeModel()
        prefetcher = MagicMock()
        prefetcher.num_layers = 2
        prefetcher.hidden_size = 64  # different from draft hidden dim of 16

        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=2,
            prefetcher=prefetcher,
        )

        # Call with hidden states of dim 16 (mismatches prefetcher's expected 64)
        draft_hidden = [mx.random.normal((1, 16)) for _ in range(2)]
        decoder._submit_draft_prefetch(draft_hidden)

        prefetcher.submit_bulk.assert_not_called()

    def test_submit_draft_prefetch_calls_submit_bulk(self):
        """When dimensions match, submit_bulk should be called."""
        from unittest.mock import MagicMock

        from olmlx.engine.flash.speculative import SpeculativeFlashDecoder

        class _FakeModel(nn.Module):
            def __call__(self, x, cache=None, **kwargs):
                return mx.random.normal((x.shape[0], x.shape[1], 32))

        draft = _FakeModel()
        target = _FakeModel()
        prefetcher = MagicMock()
        prefetcher.num_layers = 2
        prefetcher.hidden_size = 16  # matches draft hidden dim

        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=2,
            prefetcher=prefetcher,
        )

        draft_hidden = [mx.random.normal((1, 16)) for _ in range(2)]
        decoder._submit_draft_prefetch(draft_hidden)

        prefetcher.submit_bulk.assert_called_once()

    def test_submit_draft_prefetch_depth_mapping(self):
        """Draft positions should map to target layers by depth ratio."""
        from unittest.mock import MagicMock

        from olmlx.engine.flash.speculative import SpeculativeFlashDecoder

        class _FakeModel(nn.Module):
            def __call__(self, x, cache=None, **kwargs):
                return mx.random.normal((x.shape[0], x.shape[1], 32))

        draft = _FakeModel()
        target = _FakeModel()
        prefetcher = MagicMock()
        prefetcher.num_layers = 6
        prefetcher.hidden_size = 8

        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
            prefetcher=prefetcher,
        )

        # 3 draft positions, each with a distinct hidden state
        h0 = mx.ones((1, 8)) * 1.0
        h1 = mx.ones((1, 8)) * 2.0
        h2 = mx.ones((1, 8)) * 3.0
        decoder._submit_draft_prefetch([h0, h1, h2])

        prefetcher.submit_bulk.assert_called_once()
        layer_states = prefetcher.submit_bulk.call_args[0][0]

        # Different layers should get different signals (not all the same mean)
        # Early layers should get signal from early draft positions,
        # late layers from late draft positions
        assert len(layer_states) == 6
        val_layer0 = layer_states[0].tolist()[0][0]
        val_layer5 = layer_states[5].tolist()[0][0]
        assert val_layer0 != val_layer5


# ---------------------------------------------------------------------------
# PrefetchStats tests
# ---------------------------------------------------------------------------


class TestPrefetchStats:
    def test_hit_rate_zero(self):
        stats = PrefetchStats()
        assert stats.hit_rate() == 0.0

    def test_hit_rate_all_hits(self):
        stats = PrefetchStats(cache_hits=10, cache_misses=0)
        assert stats.hit_rate() == 1.0

    def test_hit_rate_mixed(self):
        stats = PrefetchStats(cache_hits=3, cache_misses=7)
        assert abs(stats.hit_rate() - 0.3) < 1e-6

    def test_failures_default_zero(self):
        stats = PrefetchStats()
        assert stats.failures == 0
