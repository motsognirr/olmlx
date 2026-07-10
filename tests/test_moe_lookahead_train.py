"""Tests for olmlx.engine.flash.moe_lookahead_train."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from olmlx.engine.flash.moe_lookahead_train import (
    build_multi_hot,
    recall_at_m,
    record_moe_router_traces,
    train_from_traces,
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
