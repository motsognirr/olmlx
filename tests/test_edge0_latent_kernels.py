"""Parity of the fused Metal latent kernels with the dense reference path."""

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.edge0 import latent_kernels
from olmlx.engine.edge0.pack import Edge0Pack
from tests.edge0_factories import make_pack
from tests.test_edge0_latent_moe import _build, _reference_forward

pytestmark = pytest.mark.usefixtures("metal_default_device")


def _inputs(truth, tokens, top_k, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, tokens, truth.hidden))
    inds = np.stack(
        [rng.choice(truth.experts, size=top_k, replace=False) for _ in range(tokens)]
    )[None]
    scores = rng.uniform(0.1, 1.0, size=(1, tokens, top_k))
    return x, inds, scores


def test_supported_on_metal(tmp_path):
    assert latent_kernels.kernels_supported(
        Edge0Pack(make_pack(tmp_path).root).geometry
    )


def test_not_supported_on_cpu(tmp_path):
    geometry = Edge0Pack(make_pack(tmp_path).root).geometry
    mx.set_default_device(mx.cpu)  # restored by metal_default_device
    assert not latent_kernels.kernels_supported(geometry)


@pytest.mark.parametrize("tokens,top_k", [(1, 3), (7, 2)])
def test_kernel_path_matches_float64_reference(tmp_path, tokens, top_k):
    truth = make_pack(tmp_path)
    x, inds, scores = _inputs(truth, tokens, top_k, seed=5)
    moe, store = _build(truth, 1)
    try:
        got = moe(
            mx.array(x.astype(np.float32)),
            mx.array(inds.astype(np.int32)),
            mx.array(scores.astype(np.float32)),
        )
        assert moe._use_kernels
        expected = _reference_forward(truth, 1, x, inds, scores)
        np.testing.assert_allclose(
            np.array(got), expected, rtol=1e-4, atol=2e-6 * np.abs(expected).max()
        )
    finally:
        store.close()


def test_kernel_path_matches_dense_path(tmp_path):
    truth = make_pack(tmp_path, latent=512, ff=64)
    x, inds, scores = _inputs(truth, 5, 3, seed=6)
    args = (
        mx.array(x.astype(np.float32)),
        mx.array(inds.astype(np.int32)),
        mx.array(scores.astype(np.float32)),
    )
    fused, store_a = _build(truth, 0)
    dense, store_b = _build(truth, 0, use_kernels=False)
    try:
        a = np.array(fused(*args))
        b = np.array(dense(*args))
        assert fused._use_kernels and not dense._use_kernels
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=2e-6 * np.abs(b).max())
    finally:
        store_a.close()
        store_b.close()


def test_batched_kernel_path_matches_reference(tmp_path):
    # More distinct experts than the batch cap: several kernel batches whose
    # outputs accumulate into the same (token, group) rows.
    truth = make_pack(tmp_path)
    x, inds, scores = _inputs(truth, 9, 3, seed=7)
    moe, store = _build(truth, 1, max_kernel_experts=3)
    try:
        got = moe(
            mx.array(x.astype(np.float32)),
            mx.array(inds.astype(np.int32)),
            mx.array(scores.astype(np.float32)),
        )
        assert moe._use_kernels
        assert store.stats.snapshot()["cache_misses"] == len(np.unique(inds))
        expected = _reference_forward(truth, 1, x, inds, scores)
        np.testing.assert_allclose(
            np.array(got), expected, rtol=1e-4, atol=2e-6 * np.abs(expected).max()
        )
    finally:
        store.close()


@pytest.mark.parametrize("separate_record_limit", [0, 16])
def test_stacked_and_separate_record_kernels_agree(tmp_path, separate_record_limit):
    # separate_record_limit=0 forces the stacked-buffer kernels; 16 hands each
    # expert record to the kernel as its own input (the decode fast path).
    truth = make_pack(tmp_path)
    x, inds, scores = _inputs(truth, 4, 3, seed=8)
    moe, store = _build(truth, 0, separate_record_limit=separate_record_limit)
    try:
        got = moe(
            mx.array(x.astype(np.float32)),
            mx.array(inds.astype(np.int32)),
            mx.array(scores.astype(np.float32)),
        )
        assert moe._use_kernels
        expected = _reference_forward(truth, 0, x, inds, scores)
        np.testing.assert_allclose(
            np.array(got), expected, rtol=1e-4, atol=2e-6 * np.abs(expected).max()
        )
    finally:
        store.close()


def test_too_many_separate_records_is_rejected(tmp_path):
    geometry = Edge0Pack(make_pack(tmp_path).root).geometry
    records = [mx.zeros(geometry.record_bytes, dtype=mx.uint8)] * 17
    with pytest.raises(ValueError, match="separate records"):
        latent_kernels.routed_latent(
            records,
            mx.zeros((1, geometry.latent)),
            mx.zeros(1, dtype=mx.uint32),
            mx.ones(1),
            geometry,
        )
