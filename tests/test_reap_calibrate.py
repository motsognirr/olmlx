from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.reap.calibrate import (
    SaliencyAccumulator,
    collect_saliency,
    load_saliency,
    save_saliency,
)
from tests.reap_factories import (
    make_tiny_deepseek_v3,
    make_tiny_gpt_oss,
    make_tiny_qwen3_moe,
)


class FakeTokenizer:
    """Maps each character to a token id; enough for tiny-vocab models."""

    def encode(self, text):
        return [ord(c) % 100 + 1 for c in text]


def _tagged(n_per_source=3):
    return [("english", "hello world " * 8), ("code", "def f(x): return x " * 6)][
        :2
    ] * n_per_source


class TestAccumulator:
    def test_add_and_mean(self):
        acc = SaliencyAccumulator(2, 4, ["a", "b"])
        acc.add(0, 0, np.array([1, 1, 3]), np.array([2.0, 4.0, 6.0]))
        sal = acc.saliency(["a"])
        assert sal.shape == (2, 4)
        assert sal[0, 1] == pytest.approx(3.0)  # mean of 2,4
        assert sal[0, 3] == pytest.approx(6.0)
        assert sal[0, 0] == 0.0 and sal[1, 1] == 0.0

    def test_source_combination(self):
        acc = SaliencyAccumulator(1, 4, ["a", "b"])
        acc.add(0, 0, np.array([0]), np.array([1.0]))
        acc.add(0, 1, np.array([0]), np.array([3.0]))
        assert acc.saliency(["a"])[0, 0] == pytest.approx(1.0)
        assert acc.saliency(None)[0, 0] == pytest.approx(2.0)  # pooled mean

    def test_npz_roundtrip(self, tmp_path):
        acc = SaliencyAccumulator(1, 4, ["a"])
        acc.add(0, 0, np.array([2]), np.array([5.0]))
        meta = {
            "model_type": "qwen3_moe",
            "num_experts": 4,
            "top_k": 2,
            "moe_layer_indices": [0],
            "sources": ["a"],
        }
        save_saliency(tmp_path / "s.npz", acc, meta)
        acc2, meta2 = load_saliency(tmp_path / "s.npz")
        np.testing.assert_array_equal(acc.sum, acc2.sum)
        np.testing.assert_array_equal(acc.count, acc2.count)
        assert acc2.sources == ["a"] and meta2["model_type"] == "qwen3_moe"


@pytest.mark.parametrize(
    "factory", [make_tiny_qwen3_moe, make_tiny_gpt_oss, make_tiny_deepseek_v3]
)
class TestCollectSaliency:
    def test_counts_and_meta(self, factory):
        mx.random.seed(0)
        model, args = factory()
        acc, meta = collect_saliency(model, FakeTokenizer(), _tagged(), max_tokens=16)
        n_moe = len(meta["moe_layer_indices"])
        assert acc.sum.shape == (n_moe, 2, 8)
        # every routed token contributes exactly top_k slots per MoE layer
        total_tokens = sum(
            min(len(FakeTokenizer().encode(t)), 16) for _, t in _tagged()
        )
        assert acc.count.sum() == n_moe * total_tokens * meta["top_k"]
        assert (acc.sum >= 0).all()
        assert meta["num_experts"] == 8 and meta["top_k"] == 2

    def test_taps_removed_after_collect(self, factory):
        model, _ = factory()
        layers = model.model.layers
        before = [type(getattr(layer, "mlp", None)).__name__ for layer in layers]
        collect_saliency(model, FakeTokenizer(), _tagged(1), max_tokens=8)
        after = [type(getattr(layer, "mlp", None)).__name__ for layer in layers]
        assert before == after

    def test_forward_unchanged_by_tap(self, factory):
        mx.random.seed(1)
        model, _ = factory()
        ids = mx.array([[1, 2, 3, 4, 5]])
        ref = np.array(model(ids), copy=True)
        collect_saliency(model, FakeTokenizer(), _tagged(1), max_tokens=8)
        out = np.array(model(ids))
        np.testing.assert_allclose(out, ref, atol=1e-5)


class TestSaliencyMatchesManualReference:
    def test_qwen3_hand_computed(self):
        """Independent reimplementation of REAP S_j for one batch pins the tap."""
        mx.random.seed(2)
        model, _ = make_tiny_qwen3_moe(num_layers=1)
        text = "abcdefghij"
        acc, meta = collect_saliency(
            model, FakeTokenizer(), [("src", text)], max_tokens=32
        )

        moe = model.model.layers[0].mlp
        ids = mx.array([FakeTokenizer().encode(text)])
        # reference: run the pre-MoE part by calling the layer's attention manually is
        # overkill; instead recompute routing on the *hidden states the MoE saw* by
        # re-running the model with a capture of the MoE input.
        seen = {}
        orig_call = type(moe).__call__

        def capture(self_, x):
            seen["x"] = x
            return orig_call(self_, x)

        type(moe).__call__ = capture
        try:
            model(ids)
        finally:
            type(moe).__call__ = orig_call
        x = seen["x"].reshape(-1, seen["x"].shape[-1])
        probs = mx.softmax(moe.gate(x).astype(mx.float32), axis=-1, precise=True)
        inds = mx.argpartition(-probs, kth=1, axis=-1)[..., :2]
        scores = mx.take_along_axis(probs, inds, axis=-1)
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        ys = moe.switch_mlp(x, inds)
        norms = mx.linalg.norm(ys.astype(mx.float32), axis=-1)
        expected = np.zeros(8)
        counts = np.zeros(8)
        np.add.at(
            expected,
            np.array(inds).reshape(-1),
            np.array(scores * norms, dtype=np.float64).reshape(-1),
        )
        np.add.at(counts, np.array(inds).reshape(-1), 1)
        np.testing.assert_allclose(acc.sum[0, 0], expected, rtol=1e-4)
        np.testing.assert_array_equal(acc.count[0, 0], counts)
