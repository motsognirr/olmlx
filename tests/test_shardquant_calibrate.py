"""Tests for the shard calibration pipeline (#377 Tier 1).

Mirrors the mocked-model pattern of test_spectralquant_calibrate_coverage.py.
"""

from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import KVCache

import olmlx.engine.shardquant_calibrate as shc


# ---------------------------------------------------------------------------
# save / load round trip
# ---------------------------------------------------------------------------


def _entry(D=8, H=2, bits=4, rope=True):
    rng = np.random.RandomState(0)
    g = 8 // bits

    def orth(seed):
        q, _ = np.linalg.qr(rng.randn(D, D).astype(np.float32))
        return q

    return {
        "k_basis": mx.array(np.stack([orth(h) for h in range(H)])),
        "k_rank": 5,
        "k_codebook": mx.array(
            np.sort(rng.randn(H, 1 << bits), axis=-1).astype(np.float32)
        ),
        "k_mean": mx.array(rng.randn(H, D).astype(np.float32) * 0.1),
        "v_rotation": mx.array(orth(99)),
        "v_codebooks": mx.array(rng.randn(D // g, 256, g).astype(np.float32)),
        "rope_freqs": mx.array(rng.rand(D // 2).astype(np.float32)) if rope else None,
        "rope_dims": D if rope else None,
        "rope_traditional": False,
    }


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        calibration = {0: _entry(rope=True), 2: _entry(rope=False)}
        meta = {"bits": 4, "head_dim": 8, "n_kv_heads": 2, "num_layers": 3}
        shc.save_shard_calibration(calibration, meta, tmp_path / "shard")

        assert (tmp_path / "shard" / "shard_config.json").exists()
        assert (tmp_path / "shard" / "calibration.safetensors").exists()

        loaded, loaded_meta = shc.load_shard_calibration(tmp_path / "shard")
        assert set(loaded.keys()) == {0, 2}
        assert loaded_meta["bits"] == 4
        e = loaded[0]
        np.testing.assert_allclose(
            np.array(e["k_basis"]), np.array(calibration[0]["k_basis"]), atol=1e-6
        )
        assert e["k_rank"] == 5
        np.testing.assert_allclose(
            np.array(e["k_mean"]), np.array(calibration[0]["k_mean"]), atol=1e-7
        )
        assert e["k_codebook"].shape == (2, 16)
        assert e["v_codebooks"].shape == (4, 256, 2)
        np.testing.assert_allclose(
            np.array(e["rope_freqs"]),
            np.array(calibration[0]["rope_freqs"]),
            atol=1e-7,
        )
        assert loaded[2]["rope_freqs"] is None


# ---------------------------------------------------------------------------
# Full pipeline with a mocked model (mirrors spectral coverage tests)
# ---------------------------------------------------------------------------


class _FakeAttn:
    def __init__(self, head_dim):
        self.rope = nn.RoPE(head_dim, traditional=False, base=10000.0)


class _FakeLayer:
    def __init__(self, head_dim):
        self.self_attn = _FakeAttn(head_dim)


class _FakeBackbone:
    def __init__(self, num_layers, n_kv_heads, head_dim):
        self.layers = [_FakeLayer(head_dim) for _ in range(num_layers)]
        args = MagicMock()
        args.num_key_value_heads = n_kv_heads
        args.num_attention_heads = n_kv_heads
        self.args = args


class _FakeModel:
    def __init__(self, backbone, head_dim):
        self._backbone = backbone
        self._head_dim = head_dim
        self.layers = backbone.layers

    def __call__(self, input_ids, cache=None):
        seq = input_ids.shape[1]
        n_kv = self._backbone.args.num_key_value_heads
        rng = np.random.RandomState(seq)
        for entry in cache:
            keys = mx.array(rng.randn(1, n_kv, seq, self._head_dim).astype(np.float32))
            values = mx.array(
                rng.randn(1, n_kv, seq, self._head_dim).astype(np.float32)
            )
            entry.update_and_fetch(keys, values)


def _run_calibration(tmp_path, head_dim=8, num_layers=2, n_kv_heads=2, bits=4):
    backbone = _FakeBackbone(num_layers, n_kv_heads, head_dim)
    model = _FakeModel(backbone, head_dim)
    texts = ["one two three four five six seven eight nine ten"] * 4

    patches = [
        patch(
            "olmlx.engine.flash.prepare.load_model_with_strict_fallback",
            return_value=(model, MagicMock()),
        ),
        patch("olmlx.engine.flash.prepare._get_backbone", return_value=backbone),
        patch(
            "olmlx.engine.flash.prepare._get_c4_calibration_data", return_value=texts
        ),
        patch("olmlx.engine.flash.prepare._get_calibration_data", return_value=texts),
        patch(
            "olmlx.engine.flash.prepare._encode_tokens",
            side_effect=lambda tok, text: list(range(len(text.split()))),
        ),
        patch("olmlx.engine.turboquant_cache._detect_head_dim", return_value=head_dim),
        patch(
            "mlx_lm.models.cache.make_prompt_cache",
            side_effect=lambda owner: [KVCache() for _ in range(num_layers)],
        ),
    ]
    for p in patches:
        p.start()
    try:
        return shc.calibrate_model_shard(
            "fake/model",
            output_dir=tmp_path / "shard",
            num_samples=4,
            bits=bits,
        )
    finally:
        for p in patches:
            p.stop()


class TestCalibrateModelShard:
    def test_end_to_end_artifacts(self, tmp_path):
        out = _run_calibration(tmp_path)
        assert out == tmp_path / "shard"
        calibration, meta = shc.load_shard_calibration(out)
        assert set(calibration.keys()) == {0, 1}
        assert meta["bits"] == 4
        assert meta["head_dim"] == 8
        assert meta["n_kv_heads"] == 2

        e = calibration[0]
        # Per-head bases (the review-comment correction): (H, D, D)
        assert e["k_basis"].shape == (2, 8, 8)
        assert 1 <= e["k_rank"] <= 8
        assert e["k_codebook"].shape == (2, 16)  # per-head, 2^4 each
        assert e["k_mean"].shape == (2, 8)  # per-head mean of unit keys
        assert e["v_rotation"].shape == (8, 8)
        assert e["v_codebooks"].shape == (4, 256, 2)  # g = 8//4 = 2
        # RoPE detected from the fake attn and stored
        assert e["rope_freqs"] is not None
        assert e["rope_dims"] == 8

    def test_bases_orthonormal(self, tmp_path):
        out = _run_calibration(tmp_path)
        calibration, _ = shc.load_shard_calibration(out)
        for e in calibration.values():
            for h in range(e["k_basis"].shape[0]):
                B = np.array(e["k_basis"][h])
                np.testing.assert_allclose(B @ B.T, np.eye(8), atol=1e-3)

    def test_bits_2_changes_group_size(self, tmp_path):
        out = _run_calibration(tmp_path, bits=2)
        calibration, meta = shc.load_shard_calibration(out)
        assert meta["bits"] == 2
        assert calibration[0]["v_codebooks"].shape == (2, 256, 4)  # g = 4
        assert calibration[0]["k_codebook"].shape == (2, 4)  # per-head, 2^2

    def test_rank_by_cumulative_energy(self):
        """Low-rank data picks a small rank that still captures the energy
        threshold; the participation ratio collapsed to ~2 under a dominant
        eigenvalue and discarded real signal (the K-quality gap measured by
        scripts/shard_ab_bench.py)."""
        # 3 strong directions + faint isotropic noise: a 0.99 threshold
        # keeps just the strong directions.
        ev = np.concatenate([[10.0, 5.0, 2.0], np.full(13, 0.01)])
        assert shc._rank_from_eigenvalues(mx.array(ev.astype(np.float32)), 0.99) <= 6
        # Heavy-tailed spectrum: one dominant + meaningful tail. The
        # participation ratio would say ~2; the energy rule keeps the tail.
        ev2 = np.concatenate([[100.0], np.full(15, 1.0)])
        r2 = shc._rank_from_eigenvalues(mx.array(ev2.astype(np.float32)))
        assert r2 >= 12, f"energy rank {r2} discarded a meaningful tail"

    def test_k_energy_knob_threads_through(self, tmp_path):
        """A looser k_energy must produce a lower rank than the default."""
        out_default = _run_calibration(tmp_path / "a")
        cal_default, _ = shc.load_shard_calibration(out_default)

        # Re-run with a very loose threshold via the public kwarg.
        backbone = _FakeBackbone(2, 2, 8)
        model = _FakeModel(backbone, 8)
        texts = ["one two three four five six seven eight nine ten"] * 4
        patches = [
            patch(
                "olmlx.engine.flash.prepare.load_model_with_strict_fallback",
                return_value=(model, MagicMock()),
            ),
            patch("olmlx.engine.flash.prepare._get_backbone", return_value=backbone),
            patch(
                "olmlx.engine.flash.prepare._get_c4_calibration_data",
                return_value=texts,
            ),
            patch(
                "olmlx.engine.flash.prepare._get_calibration_data", return_value=texts
            ),
            patch(
                "olmlx.engine.flash.prepare._encode_tokens",
                side_effect=lambda tok, text: list(range(len(text.split()))),
            ),
            patch("olmlx.engine.turboquant_cache._detect_head_dim", return_value=8),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                side_effect=lambda owner: [KVCache() for _ in range(2)],
            ),
        ]
        for p in patches:
            p.start()
        try:
            out_loose = shc.calibrate_model_shard(
                "fake/model",
                output_dir=tmp_path / "b" / "shard",
                num_samples=4,
                bits=4,
                k_energy=0.5,
            )
        finally:
            for p in patches:
                p.stop()
        cal_loose, _ = shc.load_shard_calibration(out_loose)
        assert cal_loose[0]["k_rank"] < cal_default[0]["k_rank"]

    def test_load_tier1_artifacts_backward_compatible(self, tmp_path):
        """Tier-1 calibrations (shared 1-D k_codebook, no k_mean) must load
        with k_mean=None so the runtime falls back to the un-centered path
        instead of KeyError-ing on every model calibrated before this."""
        import safetensors.numpy

        import json as _json

        d = tmp_path / "shard"
        d.mkdir()
        rng = np.random.RandomState(0)
        tensors = {
            "layer_0_k_basis": rng.randn(2, 8, 8).astype(np.float32),
            "layer_0_k_codebook": np.sort(rng.randn(16)).astype(np.float32),
            "layer_0_v_rotation": rng.randn(8, 8).astype(np.float32),
            "layer_0_v_codebooks": rng.randn(4, 256, 2).astype(np.float32),
        }
        safetensors.numpy.save_file(tensors, str(d / "calibration.safetensors"))
        (d / "shard_config.json").write_text(
            _json.dumps(
                {
                    "meta": {"bits": 4},
                    "layers": {
                        "0": {
                            "k_rank": 5,
                            "rope_dims": None,
                            "rope_traditional": False,
                        }
                    },
                }
            )
        )
        loaded, _ = shc.load_shard_calibration(d)
        assert loaded[0]["k_mean"] is None
        assert loaded[0]["k_codebook"].shape == (16,)

    def test_compress_decompress_with_real_artifacts(self, tmp_path):
        """The calibration output drives an actual cache round trip."""
        from olmlx.engine.shardquant_cache import make_shard_cache

        out = _run_calibration(tmp_path)

        class Model:
            layers = [object(), object()]

            def make_cache(self):
                return [KVCache(), KVCache()]

        caches = make_shard_cache(Model(), out, bits=4)
        rng = np.random.RandomState(5)
        k = mx.array(rng.randn(1, 2, 100, 8).astype(np.float32))
        v = mx.array(rng.randn(1, 2, 100, 8).astype(np.float32))
        ko, vo = caches[0].update_and_fetch(k, v)
        assert ko.shape == (1, 2, 100, 8)
        assert vo.shape == (1, 2, 100, 8)
        cos = np.sum(np.array(ko) * np.array(k), -1) / (
            np.linalg.norm(np.array(ko), axis=-1) * np.linalg.norm(np.array(k), axis=-1)
            + 1e-9
        )
        assert cos.mean() > 0.7  # random data has no low-rank structure; loose bound
