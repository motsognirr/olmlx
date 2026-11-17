"""Tests for olmlx.engine.edge0.latent_moe: latent expert store + forward."""

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.edge0 import ggml_quants as gq
from olmlx.engine.edge0 import latent_moe as lm
from olmlx.engine.edge0.pack import Edge0Pack
from tests.edge0_factories import make_pack, random_blocks


@pytest.fixture(autouse=True)
def cpu_device():
    """Pin these parity tests to the CPU.

    They compare against float64 references at float32 tolerances, and a
    multi-row float32 matmul on Metal is only accurate to ~1e-3.
    """
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _reference_forward(truth, layer, x, inds, scores):
    """Per-route float64 latent MoE: ``sum_k w_k * U_g (Down_e silu(G_e xl) * (Up_e xl))``."""
    batch, tokens, _ = x.shape
    out = np.zeros((batch, tokens, truth.hidden))
    inverse = 1.0 / truth.scales[layer].astype(np.float64)
    for b in range(batch):
        for t in range(tokens):
            for k in range(inds.shape[-1]):
                expert = int(inds[b, t, k])
                group = int(truth.labels[layer, expert])
                gate, up, down = (
                    w.astype(np.float64) for w in truth.expert_weights[(layer, expert)]
                )
                xl = (
                    truth.down_proj[layer][group].astype(np.float64) @ x[b, t] * inverse
                )
                hidden = _silu(gate @ xl) * (up @ xl)
                out[b, t] += scores[b, t, k] * (
                    truth.up_proj[layer][group].astype(np.float64) @ (down @ hidden)
                )
    return out


class TestDecode:
    def test_q3_k_matches_numpy(self):
        raw = random_blocks(np.random.default_rng(0), gq.GGML_Q3_K, 3 * 4 * 512)
        got = lm.decode_q3_k(mx.array(raw.reshape(3, 4, 2, 110)))
        assert got.shape == (3, 4, 512) and got.dtype == mx.float32
        np.testing.assert_allclose(
            np.array(got).reshape(-1), gq.dequantize_q3_k(raw), rtol=1e-6
        )

    def test_iq4_nl_matches_numpy(self):
        raw = random_blocks(np.random.default_rng(1), gq.GGML_IQ4_NL, 5 * 64)
        got = lm.decode_iq4_nl(mx.array(raw.reshape(5, 2, 18)))
        assert got.shape == (5, 64)
        np.testing.assert_allclose(
            np.array(got).reshape(-1), gq.dequantize_iq4_nl(raw), rtol=1e-6
        )

    def test_decode_records(self, tmp_path):
        truth = make_pack(tmp_path)
        pack = Edge0Pack(truth.root)
        with lm.LatentExpertStore(pack) as store:
            raw = store.load(1, [5, 2])
        gate, up, down = lm.decode_records(raw, pack.geometry)
        for position, expert in enumerate((5, 2)):
            ref_gate, ref_up, ref_down = truth.expert_weights[(1, expert)]
            np.testing.assert_allclose(np.array(gate[position]), ref_gate, rtol=1e-6)
            np.testing.assert_allclose(np.array(up[position]), ref_up, rtol=1e-6)
            np.testing.assert_allclose(np.array(down[position]), ref_down, rtol=1e-6)


class TestStore:
    def test_returns_records_in_request_order(self, tmp_path):
        truth = make_pack(tmp_path)
        pack = Edge0Pack(truth.root)
        blob = (truth.root / "experts.bin").read_bytes()
        with lm.LatentExpertStore(pack) as store:
            raw = np.array(store.load(1, [7, 0, 3]))
        assert raw.shape == (3, truth.record_bytes) and raw.dtype == np.uint8
        for row, expert in zip(raw, (7, 0, 3)):
            start = pack.expert_offset(1, expert)
            assert row.tobytes() == blob[start : start + truth.record_bytes]

    def test_cache_counts_hits_and_misses(self, tmp_path):
        pack = Edge0Pack(make_pack(tmp_path).root)
        with lm.LatentExpertStore(pack, cache_budget_experts=4) as store:
            store.load(0, [1, 2])
            store.load(0, [2, 3])
            assert store.stats.snapshot() == {
                "load_calls": 2,
                "cache_hits": 1,
                "cache_misses": 3,
                "load_failures": 0,
            }

    def test_zero_budget_disables_the_cache(self, tmp_path):
        pack = Edge0Pack(make_pack(tmp_path).root)
        with lm.LatentExpertStore(pack, cache_budget_experts=0) as store:
            store.load(0, [1])
            store.load(0, [1])
            assert store.stats.snapshot()["cache_hits"] == 0

    def test_cached_records_do_not_alias_the_read_buffer(self, tmp_path):
        # mx.array must copy host bytes: the pread buffer is freed while the
        # cached array lives on.
        pack = Edge0Pack(make_pack(tmp_path).root)
        with lm.LatentExpertStore(pack) as store:
            first = np.array(store.load(0, [4]))
            again = np.array(store.load(0, [4]))
        np.testing.assert_array_equal(first, again)

    def test_empty_request_is_rejected(self, tmp_path):
        pack = Edge0Pack(make_pack(tmp_path).root)
        with lm.LatentExpertStore(pack) as store:
            with pytest.raises(ValueError, match="empty"):
                store.load(0, [])


class TestProjections:
    @pytest.mark.parametrize("dense_fmt", [8, 4])
    def test_packed_projections_reproduce_the_blocks(self, tmp_path, dense_fmt):
        truth = make_pack(tmp_path, dense_fmt=dense_fmt)
        pack = Edge0Pack(truth.root)
        tensors = lm.pack_projections(pack, 1)
        bits = tensors["bits"]
        assert bits == dense_fmt
        down = mx.dequantize(
            tensors["down.weight"],
            tensors["down.scales"],
            tensors["down.biases"],
            group_size=32,
            bits=bits,
        )
        up = mx.dequantize(
            tensors["up.weight"],
            tensors["up.scales"],
            tensors["up.biases"],
            group_size=32,
            bits=bits,
        )
        # down: [G*latent, hidden]; up: [hidden, G*latent] (groups side by side).
        np.testing.assert_allclose(
            np.array(down).reshape(2, 256, 64), truth.down_proj[1], rtol=1e-5, atol=1e-7
        )
        np.testing.assert_allclose(
            np.array(up).reshape(64, 2, 256).transpose(1, 0, 2),
            truth.up_proj[1],
            rtol=1e-5,
            atol=1e-7,
        )


def _build(truth, layer, **kwargs):
    pack = Edge0Pack(truth.root)
    store = lm.LatentExpertStore(pack)
    tensors = lm.pack_projections(pack, layer)
    tensors["inverse_scale"] = mx.array(pack.inverse_column_scale(layer))
    moe = lm.LatentMoE(
        layer_idx=layer,
        geometry=pack.geometry,
        labels=pack.labels[layer],
        tensors=tensors,
        store=store,
        **kwargs,
    )
    return moe, store


class TestLatentMoE:
    @pytest.mark.parametrize("dense_fmt", [8, 4])
    def test_matches_per_route_reference(self, tmp_path, dense_fmt):
        truth = make_pack(tmp_path, dense_fmt=dense_fmt)
        rng = np.random.default_rng(3)
        x = rng.standard_normal((1, 5, truth.hidden))
        inds = np.stack(
            [rng.choice(truth.experts, size=3, replace=False) for _ in range(5)]
        )[None]
        scores = rng.uniform(0.1, 1.0, size=(1, 5, 3))
        moe, store = _build(truth, 1)
        try:
            got = moe(
                mx.array(x.astype(np.float32)),
                mx.array(inds.astype(np.int32)),
                mx.array(scores.astype(np.float32)),
            )
            expected = _reference_forward(truth, 1, x, inds, scores)
            assert got.shape == (1, 5, truth.hidden)
            np.testing.assert_allclose(
                np.array(got), expected, rtol=1e-4, atol=2e-6 * np.abs(expected).max()
            )
        finally:
            store.close()

    def test_expert_batches_match_a_single_batch(self, tmp_path):
        # A long prefill touches most experts; they are streamed through in
        # bounded batches, each expert serving all of its routes at once.
        truth = make_pack(tmp_path)
        rng = np.random.default_rng(4)
        x = mx.array(rng.standard_normal((1, 9, truth.hidden)).astype(np.float32))
        inds = mx.array(
            np.stack(
                [rng.choice(truth.experts, size=2, replace=False) for _ in range(9)]
            )[None].astype(np.int32)
        )
        scores = mx.array(rng.uniform(0.1, 1.0, size=(1, 9, 2)).astype(np.float32))
        whole, store_a = _build(truth, 0, max_decode_experts=64)
        sliced, store_b = _build(truth, 0, max_decode_experts=3)
        try:
            np.testing.assert_allclose(
                np.array(sliced(x, inds, scores)),
                np.array(whole(x, inds, scores)),
                rtol=1e-5,
                atol=1e-6,
            )
        finally:
            store_a.close()
            store_b.close()

    def test_output_dtype_follows_input(self, tmp_path):
        truth = make_pack(tmp_path)
        moe, store = _build(truth, 0)
        try:
            x = mx.zeros((1, 1, truth.hidden), dtype=mx.float16)
            out = moe(x, mx.array([[[0, 1]]]), mx.ones((1, 1, 2), dtype=mx.float16))
            assert out.dtype == mx.float16
        finally:
            store.close()
