"""Tests for olmlx.engine.flash.moe_lookahead_train."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from olmlx.engine.flash.moe_lookahead_train import (
    build_multi_hot,
    recall_at_m,
    record_moe_router_traces,
    record_moe_router_traces_decode,
    train_from_traces,
    train_moe_lookahead,
)


class TestBuildMultiHot:
    def test_basic(self):
        inds = np.array([[0, 2], [1, 3]])
        out = build_multi_hot(inds, num_experts=4)
        expected = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)
        assert out.dtype == np.float32


class TestRecallAtM:
    def test_perfect_prediction(self):
        scores = np.array([[0.9, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.9]])
        true_inds = np.array([[0, 1], [2, 3]])
        assert recall_at_m(scores, true_inds, m=2) == 1.0

    def test_half_recall(self):
        scores = np.array([[0.9, 0.0, 0.1, 0.8]])  # top-2 = {0, 3}
        true_inds = np.array([[0, 1]])  # only 0 found
        assert recall_at_m(scores, true_inds, m=2) == 0.5


class TestTrainFromTraces:
    def test_learns_deterministic_mapping(self):
        """A learnable hidden->expert rule must reach high holdout recall."""
        rng = np.random.default_rng(42)
        num_experts, hidden_size, positions = 4, 16, 400
        hid = rng.standard_normal((positions, hidden_size)).astype(np.float32)
        # Rule: expert pair determined by the sign of feature 0
        next_inds = np.where(
            hid[:, :1] > 0,
            np.array([[0, 1]]),
            np.array([[2, 3]]),
        ).astype(np.int32)
        traces = {
            1: (hid, next_inds),  # inds at layer 1 unused for pair 1->2 input
            2: (hid, next_inds),  # targets for the 1->2 pair
        }
        bank, recalls = train_from_traces(
            traces,
            [1, 2],
            hidden_size,
            num_experts,
            num_experts_per_tok=2,
            rank=8,
            epochs=200,
            holdout_fraction=0.1,
        )
        assert "1→2" in recalls
        assert recalls["1→2"] > 0.9  # trivially learnable rule
        assert bank.trained_pairs == {0}
        # Recall must also land on the bank itself (pair-indexed) so save()
        # persists it and serve-time recall gating can use it.
        assert bank.pair_recalls == {0: recalls["1→2"]}

    def test_missing_layer_trace_skipped(self):
        hid = np.zeros((10, 16), dtype=np.float32)
        inds = np.zeros((10, 2), dtype=np.int32)
        bank, recalls = train_from_traces(
            {1: (hid, inds)},  # layer 4 trace missing
            [1, 4],
            16,
            8,
            num_experts_per_tok=2,
            rank=4,
            epochs=1,
        )
        assert recalls == {}
        # Finding 2: a skipped pair must not be marked trained — its head
        # stays randomly initialized and must not be saved/served as usable.
        assert bank.trained_pairs == set()

    def test_partial_pairs_only_actually_trained_marked(self):
        """3 MoE layers, pair 0 (1->2) trains, pair 1 (2->4) has a missing
        trace and must stay untrained even though pair 0 succeeded."""
        rng = np.random.default_rng(7)
        hid = rng.standard_normal((50, 16)).astype(np.float32)
        inds = rng.integers(0, 4, size=(50, 2)).astype(np.int32)
        bank, recalls = train_from_traces(
            {1: (hid, inds), 2: (hid, inds)},  # layer 4 trace missing
            [1, 2, 4],
            16,
            4,
            num_experts_per_tok=2,
            rank=4,
            epochs=1,
        )
        assert bank.trained_pairs == {0}
        assert "1→2" in recalls
        assert "2→4" not in recalls

    def test_small_n_trained_not_evaluated_logs(self, caplog):
        """n <= 10 trains (no holdout) — must still be marked trained, and
        logged distinctly from a skipped pair so it's not confused with one."""
        import logging

        hid = np.zeros((5, 16), dtype=np.float32)
        inds = np.zeros((5, 2), dtype=np.int32)
        with caplog.at_level(
            logging.INFO, logger="olmlx.engine.flash.moe_lookahead_train"
        ):
            bank, recalls = train_from_traces(
                {1: (hid, inds), 2: (hid, inds)},
                [1, 2],
                16,
                4,
                num_experts_per_tok=2,
                rank=4,
                epochs=1,
            )
        assert bank.trained_pairs == {0}
        assert "1→2" not in recalls  # no holdout at n<=10
        assert "not evaluated" in caplog.text
        assert "n=5" in caplog.text


class _FakeRoutedMoE(nn.Module):
    """Stands in for a _FlashMoEBase replacement: has _route, returns x."""

    def __init__(self, num_experts: int, k: int):
        super().__init__()
        self._num_experts = num_experts
        self._k = k

    def _route(self, x: mx.array):
        flat = x.reshape(-1, x.shape[-1])
        logits = mx.zeros((flat.shape[0], self._num_experts))
        inds = mx.argpartition(logits, kth=-self._k, axis=-1)[..., -self._k :]
        return inds.reshape(*x.shape[:-1], self._k), None

    def __call__(self, x: mx.array) -> mx.array:
        return x


class _FakeLayer(nn.Module):
    def __init__(self, moe: nn.Module | None):
        super().__init__()
        if moe is not None:
            self.mlp = moe

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        return getattr(self, "mlp", lambda v: v)(x) + 0.1


class _FakeModel(nn.Module):
    def __init__(self, hidden: int, num_experts: int, k: int):
        super().__init__()
        self._embed = nn.Embedding(32, hidden)
        self.layers = [
            _FakeLayer(None),  # dense layer 0
            _FakeLayer(_FakeRoutedMoE(num_experts, k)),  # MoE layer 1
            _FakeLayer(_FakeRoutedMoE(num_experts, k)),  # MoE layer 2
        ]

    def __call__(self, inputs: mx.array) -> mx.array:
        h = self._embed(inputs)
        for layer in self.layers:
            h = layer(h)
        return h


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text][:16]


class TestRecordTraces:
    def test_records_hidden_and_inds_per_moe_layer(self):
        model = _FakeModel(hidden=8, num_experts=4, k=2)
        traces = record_moe_router_traces(
            model,
            _FakeTokenizer(),
            ["hello world", "foo bar"],
            [1, 2],
            max_positions_per_layer=100,
        )
        assert set(traces.keys()) == {1, 2}
        hid, inds = traces[1]
        assert hid.shape[1] == 8
        assert inds.shape[1] == 2
        assert hid.shape[0] == inds.shape[0]
        # Both texts' positions recorded, aligned across layers
        assert traces[1][0].shape[0] == traces[2][0].shape[0]

    def test_position_cap_respected(self):
        model = _FakeModel(hidden=8, num_experts=4, k=2)
        traces = record_moe_router_traces(
            model,
            _FakeTokenizer(),
            ["a much longer text that produces many tokens"] * 4,
            [1, 2],
            max_positions_per_layer=5,
        )
        assert traces[1][0].shape[0] <= 5

    def test_originals_restored_after_recording(self):
        model = _FakeModel(hidden=8, num_experts=4, k=2)
        original_1 = model.layers[1].mlp
        original_2 = model.layers[2].mlp
        record_moe_router_traces(model, _FakeTokenizer(), ["hello"], [1, 2])
        assert model.layers[1].mlp is original_1
        assert model.layers[2].mlp is original_2

    def test_layer_without_route_skipped(self):
        model = _FakeModel(hidden=8, num_experts=4, k=2)
        del model.layers[2].mlp  # layer 2 has no MoE module at all
        traces = record_moe_router_traces(model, _FakeTokenizer(), ["hello"], [1, 2])
        assert 1 in traces
        assert 2 not in traces


class TestDecodeTraceRecording:
    """Decode-state tracing (--self-generate): record hidden/routing during
    the model's own greedy generation, not teacher-forced prefill — the
    serve-time prediction runs on decode states, and prefill-trained heads
    are off-distribution there (same lesson as DFlash --self-generate)."""

    @staticmethod
    def _gen_fn(prefill_len: int, decode_steps: int):
        def _gen(model, tokenizer, prompt_ids, max_new):
            model(mx.array([prompt_ids[:prefill_len]]))  # prefill: skipped
            for _ in range(min(decode_steps, max_new)):
                model(mx.array([[1]]))  # decode steps: recorded

        return _gen

    def test_records_only_decode_positions(self):
        model = _FakeModel(hidden=8, num_experts=4, k=2)
        traces = record_moe_router_traces_decode(
            model,
            _FakeTokenizer(),
            [1, 2],
            max_new_tokens=4,
            max_prompts=1,
            prompt_source=iter([[1, 2, 3, 4, 5]]),
            _generate_fn=self._gen_fn(prefill_len=5, decode_steps=4),
        )
        assert set(traces.keys()) == {1, 2}
        # Only the 4 decode steps recorded — the 5-position prefill skipped.
        assert traces[1][0].shape[0] == 4
        assert traces[2][1].shape == (4, 2)
        assert traces[1][0].shape[0] == traces[2][0].shape[0]

    def test_decode_position_cap(self):
        model = _FakeModel(hidden=8, num_experts=4, k=2)
        traces = record_moe_router_traces_decode(
            model,
            _FakeTokenizer(),
            [1, 2],
            max_positions_per_layer=3,
            max_new_tokens=10,
            max_prompts=5,
            prompt_source=iter([[1, 2, 3]] * 5),
            _generate_fn=self._gen_fn(prefill_len=3, decode_steps=10),
        )
        assert traces[1][0].shape[0] == 3

    def test_originals_restored(self):
        model = _FakeModel(hidden=8, num_experts=4, k=2)
        original_1 = model.layers[1].mlp
        record_moe_router_traces_decode(
            model,
            _FakeTokenizer(),
            [1, 2],
            max_new_tokens=2,
            max_prompts=1,
            prompt_source=iter([[1, 2]]),
            _generate_fn=self._gen_fn(prefill_len=2, decode_steps=2),
        )
        assert model.layers[1].mlp is original_1


class TestTrainMoeLookaheadCalibrationDatasetValidation:
    """Finding 3: an unrecognized ``calibration_dataset`` string used to
    silently fall back to c4 (only "synthetic" was special-cased). Must be
    validated against {None, "c4", "synthetic"} before anything else runs —
    checked here without needing a real model/flash_moe_dir by asserting the
    ValueError fires before any file I/O."""

    def test_invalid_calibration_dataset_raises(self, tmp_path):
        with pytest.raises(ValueError, match="calibration_dataset"):
            train_moe_lookahead(
                "unused/model",
                tmp_path / "does-not-exist",
                calibration_dataset="not-a-real-dataset",
            )

    @pytest.mark.parametrize("value", [None, "c4", "synthetic"])
    def test_valid_calibration_dataset_does_not_raise_validation_error(
        self, value, tmp_path
    ):
        # No flash_moe_config.json present — must fail on the *next* check
        # (missing file), never on calibration_dataset validation.
        with pytest.raises(FileNotFoundError):
            train_moe_lookahead(
                "unused/model",
                tmp_path / "does-not-exist",
                calibration_dataset=value,
            )


class TestTrainMoeLookaheadSidecarProvenance:
    """Finding 4: the sidecar should record the training config that
    produced it (epochs, lr, num_samples, holdout_fraction), written by the
    pipeline function rather than MoeLookaheadBank itself (which stays
    provenance-agnostic)."""

    def _write_flash_moe_config(self, flash_moe_dir):
        import json

        flash_moe_dir.mkdir(parents=True, exist_ok=True)
        (flash_moe_dir / "flash_moe_config.json").write_text(
            json.dumps(
                {
                    "moe_layer_indices": [1, 2],
                    "hidden_size": 16,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                }
            )
        )

    def test_sidecar_gets_training_config(self, monkeypatch, tmp_path):
        import json

        flash_moe_dir = tmp_path / "flash_moe"
        self._write_flash_moe_config(flash_moe_dir)

        rng = np.random.default_rng(3)
        hid = rng.standard_normal((30, 16)).astype(np.float32)
        inds = rng.integers(0, 4, size=(30, 2)).astype(np.int32)
        fake_traces = {1: (hid, inds), 2: (hid, inds)}

        fake_store = type("S", (), {"close": lambda self: None})()
        monkeypatch.setattr(
            "olmlx.engine.flash.flash_moe_model.load_flash_moe_model",
            lambda *a, **kw: (object(), object(), fake_store),
        )
        monkeypatch.setattr(
            "olmlx.engine.flash.prepare._get_calibration_data",
            lambda n: ["synthetic text"] * n,
        )
        monkeypatch.setattr(
            "olmlx.engine.flash.prepare._get_c4_calibration_data",
            lambda n: ["c4 text"] * n,
        )
        monkeypatch.setattr(
            "olmlx.engine.flash.moe_lookahead_train.record_moe_router_traces",
            lambda *a, **kw: fake_traces,
        )

        out_dir = train_moe_lookahead(
            "unused/model",
            flash_moe_dir,
            epochs=2,
            lr=5e-4,
            num_samples=7,
            calibration_dataset="synthetic",
            holdout_fraction=0.2,
        )

        from olmlx.engine.flash.moe_predictor import SIDECAR_NAME

        sidecar = json.loads((out_dir / SIDECAR_NAME).read_text())
        assert sidecar["training_config"] == {
            "epochs": 2,
            "lr": 5e-4,
            "num_samples": 7,
            "holdout_fraction": 0.2,
            "self_generate": False,
        }
        # load() must ignore the unknown key rather than choking on it.
        from olmlx.engine.flash.moe_predictor import MoeLookaheadBank

        MoeLookaheadBank.load(out_dir)
