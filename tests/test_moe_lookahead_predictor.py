"""Tests for olmlx.engine.flash.moe_predictor — expert lookahead heads."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.flash.moe_predictor import SIDECAR_NAME, MoeLookaheadBank


class TestMoeLookaheadBank:
    def test_head_count_and_pair_map(self):
        # Non-contiguous MoE layers (dense layers at 0 and 3)
        bank = MoeLookaheadBank(
            [1, 2, 4, 5],
            hidden_size=16,
            num_experts=8,
            rank=4,
            num_experts_per_tok=2,
        )
        assert len(bank.heads) == 3
        assert bank.next_moe_layer(1) == 2
        assert bank.next_moe_layer(2) == 4  # skips dense layer 3
        assert bank.next_moe_layer(4) == 5
        assert bank.next_moe_layer(5) is None  # last MoE layer
        assert bank.next_moe_layer(0) is None  # not an MoE layer

    def test_predict_next_shape_and_count(self):
        bank = MoeLookaheadBank(
            [0, 1], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        hidden = mx.random.normal((1, 3, 16))  # (B, L, H)
        result = bank.predict_next(0, hidden, margin=1.5)
        assert result is not None
        indices, scores = result
        assert len(indices) == 3  # ceil(1.5 * 2)
        assert indices == sorted(indices)
        assert all(0 <= i < 8 for i in indices)
        assert scores.shape == (8,)
        assert scores.dtype == np.float32

    def test_predict_next_margin_capped_at_num_experts(self):
        bank = MoeLookaheadBank(
            [0, 1], hidden_size=16, num_experts=4, rank=4, num_experts_per_tok=4
        )
        result = bank.predict_next(0, mx.random.normal((1, 1, 16)), margin=2.0)
        assert result is not None
        indices, _ = result
        assert len(indices) == 4  # capped

    def test_predict_next_returns_none_for_last_layer(self):
        bank = MoeLookaheadBank(
            [0, 1], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        assert bank.predict_next(1, mx.random.normal((1, 1, 16))) is None

    def test_save_load_round_trip(self, tmp_path):
        bank = MoeLookaheadBank(
            [1, 2, 4], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        out = tmp_path / "moe_lookahead"
        bank.save(out)

        sidecar = json.loads((out / SIDECAR_NAME).read_text())
        assert sidecar["hidden_size"] == 16
        assert sidecar["num_experts"] == 8
        assert sidecar["moe_layer_indices"] == [1, 2, 4]

        loaded = MoeLookaheadBank.load(out)
        assert loaded.moe_layer_indices == [1, 2, 4]
        assert len(loaded.heads) == 2

        hidden = mx.random.normal((1, 2, 16))
        orig = bank.predict_next(1, hidden)
        rt = loaded.predict_next(1, hidden)
        assert orig is not None and rt is not None
        np.testing.assert_allclose(orig[1], rt[1], rtol=1e-5)

    def test_load_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MoeLookaheadBank.load(tmp_path / "nonexistent")

    def test_requires_two_moe_layers(self):
        with pytest.raises(ValueError):
            MoeLookaheadBank([3], hidden_size=16, num_experts=8, rank=4)

    def test_direct_construction_defaults_all_pairs_trained(self):
        """Directly constructing a bank (not via train_from_traces) marks
        every pair trained — preserves the pre-existing usable-by-default
        behaviour for hand-built or legacy banks."""
        bank = MoeLookaheadBank(
            [1, 2, 4], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        assert bank.trained_pairs == {0, 1}

    def test_next_moe_layer_returns_none_for_untrained_pair(self):
        bank = MoeLookaheadBank(
            [1, 2, 4], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        bank.trained_pairs = {1}  # only pair 1 (layer 2 -> 4) is trained

        assert bank.next_moe_layer(1) is None  # pair 0 (1 -> 2) untrained
        assert bank.next_moe_layer(2) == 4  # pair 1 trained

    def test_predict_next_returns_none_for_untrained_pair(self):
        bank = MoeLookaheadBank(
            [0, 1], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        bank.trained_pairs = set()  # nothing trained

        assert bank.predict_next(0, mx.random.normal((1, 1, 16))) is None

    def test_save_load_round_trips_trained_pairs(self, tmp_path):
        bank = MoeLookaheadBank(
            [1, 2, 4], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        bank.trained_pairs = {1}
        out = tmp_path / "moe_lookahead"
        bank.save(out)

        sidecar = json.loads((out / SIDECAR_NAME).read_text())
        assert sidecar["trained_pairs"] == [1]

        loaded = MoeLookaheadBank.load(out)
        assert loaded.trained_pairs == {1}

    def test_load_missing_trained_pairs_key_defaults_to_all_trained(self, tmp_path):
        """No released banks predate this field, but the sidecar-missing-key
        case should still resolve to all-trained for constructor-default
        consistency."""
        bank = MoeLookaheadBank(
            [1, 2, 4], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        out = tmp_path / "moe_lookahead"
        bank.save(out)
        sidecar_path = out / SIDECAR_NAME
        sidecar = json.loads(sidecar_path.read_text())
        del sidecar["trained_pairs"]
        sidecar_path.write_text(json.dumps(sidecar))

        loaded = MoeLookaheadBank.load(out)
        assert loaded.trained_pairs == {0, 1}
