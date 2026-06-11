# Shard KV Cache Quantizer — Tier 1 Implementation Plan (#377)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `OLMLX_KV_CACHE_QUANT=shard:{2,4,8}` — an asymmetric K/V KV-cache quantizer (RoPE-undo + per-head PCA + rank truncation for K; rotation + product-VQ for V; FP16 sink/window) with a `olmlx shard prepare <model>` calibration command, per issue #377 Tier 1 and the maintainer's review comment on that issue.

**Architecture:** Three new engine modules mirroring the spectralquant trio: `shardquant.py` (primitives: RoPE invert/reapply, rotations, product-VQ, K compress/decompress), `shardquant_cache.py` (`ShardKVCache`, a drop-in `_BaseCache` with FP16 sink + FP16 sliding window + compressed middle, **transient** dequantize-on-read — no persistent dequant side buffer, per the review comment's memory analysis), and `shardquant_calibrate.py` (per-head eigendecomposition of *de-roped* keys, k-means PQ codebooks for values; reuses a KV-collection helper extracted from `spectralquant_calibrate.py`). Dispatch/config/CLI wiring mirrors spectral exactly.

**Key design decisions (locked, from issue + review comment):**
- **Transient dequant** (spectral-style): resident memory = quantized + sink/window only. The latency cost (full-middle dequant + RoPE reapply per layer per step) is the documented trade-off.
- **Per-head K bases** (the review comment's correction #2): basis tensor `(n_kv_heads, D, D)`; a **uniform per-layer rank** (max of per-head participation ratios) keeps storage rectangular.
- **RoPE-undo before PCA** with exact reapply at dequant. Frequencies are *stored in the calibration artifacts* so calibration and runtime are guaranteed consistent. Unknown RoPE variants fall back to identity (still exactness-preserving through the undo/redo pair — just less rank collapse).
- **V path:** Hadamard rotation when `head_dim` is a power of 2, seeded random orthogonal otherwise — either way the matrix is stored in the calibration file so the runtime has one uniform code path. Product VQ with 256-entry codebooks per subvector position; subvector size `g = 8 // bits` → exactly `bits` bits/dim.
- **Sink 4 + window 64** FP16 (constructor params, module constants for defaults).
- **Exact `trim()`** — both middle and window are per-token, so trim is well-defined (slice); `is_trimmable() → True` keeps prompt-cache prefix reuse working.
- **No disk spill** (`_is_serializable_cache` gains `ShardKVCache`); **deepcopy via default walk** (no `mx.Dtype` attributes; all mutable arrays exposed via `state` so the snapshot path's `flatten_cache_state` + `mx.eval` materializes them — the #284 requirement).
- **Plain MLX ops, no `mx.compile`** in v1 — correctness first; compiled kernels are a follow-up perf PR.

**Tech Stack:** MLX, numpy (k-means in numpy — no sklearn dependency), safetensors, pytest. TDD throughout per CLAUDE.md.

**Branch:** create `feat/shard-kv-quant` from `main` before Task 1.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `olmlx/engine/shardquant.py` | Primitives: `RopeSpec` + `detect_rope_spec` + `rope_transform`, `make_v_rotation` (Hadamard/QR), `fit_vq_codebooks` (numpy k-means), `vq_assign`/`vq_gather`, `scalar_assign`, `shard_compress_keys`/`shard_decompress_keys`, `shard_compress_values`/`shard_decompress_values` |
| Create `olmlx/engine/shardquant_cache.py` | `ShardKVCache` (`_BaseCache` subclass) + `make_shard_cache` factory |
| Create `olmlx/engine/shardquant_calibrate.py` | `calibrate_model_shard`, `save_shard_calibration`/`load_shard_calibration` |
| Modify `olmlx/engine/spectralquant_calibrate.py` | Extract `_load_calibration_model` + `collect_kv_vectors` from `calibrate_model` (shared with shard; spectral behavior unchanged) |
| Modify `olmlx/config.py:53-72` | `validate_kv_cache_quant_format`: add `shard` with bits `{2,4,8}` |
| Modify `olmlx/engine/inference.py:1306-1343` | `_make_shard_prompt_cache` + dispatch branch |
| Modify `olmlx/engine/model_manager.py` | `ShardCalibrationMissingError`, `_find_shard_dir`, `_auto_calibrate_shard`, `LoadedModel.shard_calibration_dir`, `_is_serializable_cache` |
| Modify `olmlx/cli.py` | `olmlx shard prepare` subcommand |
| Modify `CLAUDE.md`, `README.md` | Document the new mode |
| Test `tests/test_shardquant.py` | Primitives + config validation |
| Test `tests/test_shardquant_cache.py` | Cache behavior |
| Test `tests/test_shardquant_calibrate.py` | Calibration pipeline (mocked model, mirrors `test_spectralquant_calibrate_coverage.py`) |
| Test `tests/test_shardquant_integration.py` | Dispatch, `_find_shard_dir`, serializable guard |
| Test `tests/live/test_shard_quant_real.py` | real_model end-to-end (calibrate tiny model, generate) |

Conventions to follow: `pack_indices`/`unpack_indices` are **reused from `olmlx.engine.spectralquant`** (1–8 bit support already exists). `compute_covariance`, `eigendecompose`, `compute_d_eff`, `fit_codebook` are **reused from `olmlx.engine.spectralquant_calibrate`/`spectralquant`**.

---

### Task 0: Branch

- [ ] **Step 0.1:** `git checkout -b feat/shard-kv-quant origin/main`

---

### Task 1: RoPE spec — detect, apply, invert

**Files:**
- Create: `olmlx/engine/shardquant.py`
- Test: `tests/test_shardquant.py`

The load-bearing property: `rope_transform(rope_transform(x, spec, off), spec, off, inverse=True) ≈ x`, and `rope_transform(x, spec, off)` matches `mlx.nn.RoPE` output for the standard case (so de-roping cached keys with this function actually lands in the no-RoPE basis).

- [ ] **Step 1.1: Write the failing tests**

```python
"""Tests for Shard KV-cache quantization primitives (#377 Tier 1)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


class TestRopeSpec:
    def _spec(self, dims=64, base=10000.0, traditional=False, scale=1.0):
        from olmlx.engine.shardquant import RopeSpec

        freqs = scale * mx.power(
            mx.array(base, dtype=mx.float32),
            -mx.arange(0, dims, 2, dtype=mx.float32) / dims,
        )
        return RopeSpec(dims=dims, freqs=freqs, traditional=traditional)

    def test_roundtrip_inverse(self):
        """inverse(apply(x)) reconstructs x for any offset."""
        from olmlx.engine.shardquant import rope_transform

        spec = self._spec()
        mx.random.seed(0)
        x = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        for offset in (0, 7, 1000):
            y = rope_transform(x, spec, offset)
            back = rope_transform(y, spec, offset, inverse=True)
            np.testing.assert_allclose(
                np.array(back, dtype=np.float32),
                np.array(x, dtype=np.float32),
                atol=2e-3,
            )

    def test_matches_nn_rope(self):
        """Forward application matches mlx.nn.RoPE (non-traditional)."""
        from olmlx.engine.shardquant import rope_transform

        dims = 32
        rope = nn.RoPE(dims, traditional=False, base=10000.0)
        spec = self._spec(dims=dims)
        mx.random.seed(1)
        x = mx.random.normal((1, 2, 6, dims))
        for offset in (0, 5):
            expected = rope(x, offset=offset)
            got = rope_transform(x, spec, offset)
            np.testing.assert_allclose(
                np.array(got), np.array(expected), atol=1e-4
            )

    def test_matches_nn_rope_traditional(self):
        from olmlx.engine.shardquant import rope_transform

        dims = 32
        rope = nn.RoPE(dims, traditional=True, base=10000.0)
        spec = self._spec(dims=dims, traditional=True)
        mx.random.seed(2)
        x = mx.random.normal((1, 2, 6, dims))
        expected = rope(x, offset=3)
        got = rope_transform(x, spec, 3)
        np.testing.assert_allclose(np.array(got), np.array(expected), atol=1e-4)

    def test_partial_dims_passthrough(self):
        """Dims beyond spec.dims are untouched (partial rotary models)."""
        from olmlx.engine.shardquant import rope_transform

        spec = self._spec(dims=16)
        mx.random.seed(3)
        x = mx.random.normal((1, 1, 4, 64))
        y = rope_transform(x, spec, 11)
        np.testing.assert_allclose(
            np.array(y[..., 16:]), np.array(x[..., 16:]), atol=1e-6
        )

    def test_rotation_preserves_norms(self):
        from olmlx.engine.shardquant import rope_transform

        spec = self._spec()
        mx.random.seed(4)
        x = mx.random.normal((1, 2, 8, 64))
        y = rope_transform(x, spec, 42)
        np.testing.assert_allclose(
            np.array(mx.linalg.norm(y, axis=-1)),
            np.array(mx.linalg.norm(x, axis=-1)),
            rtol=1e-4,
        )


class TestDetectRopeSpec:
    def test_detects_nn_rope(self):
        from olmlx.engine.shardquant import detect_rope_spec

        class Attn:
            rope = nn.RoPE(64, traditional=False, base=500000.0, scale=1.0)

        spec = detect_rope_spec(Attn())
        assert spec is not None
        assert spec.dims == 64
        assert spec.traditional is False
        # freqs[0] = scale * base^0 = 1.0
        assert abs(float(spec.freqs[0]) - 1.0) < 1e-6
        # freqs decay with base
        assert float(spec.freqs[-1]) < float(spec.freqs[0])

    def test_detects_freqs_carrying_rope(self):
        """mlx-lm custom ropes (Llama3RoPE etc.) carry wavelength-like _freqs
        and call mx.fast.rope(..., base=None, freqs=self._freqs); the spec's
        angular freqs are their reciprocal."""
        from olmlx.engine.shardquant import detect_rope_spec

        class FakeLlama3RoPE:
            dims = 32
            traditional = False
            _freqs = mx.power(
                mx.array(10000.0), mx.arange(0, 32, 2, dtype=mx.float32) / 32
            )

        class Attn:
            rope = FakeLlama3RoPE()

        spec = detect_rope_spec(Attn())
        assert spec is not None
        assert spec.dims == 32
        np.testing.assert_allclose(
            np.array(spec.freqs),
            1.0 / np.array(FakeLlama3RoPE._freqs),
            rtol=1e-5,
        )

    def test_no_rope_returns_none(self):
        from olmlx.engine.shardquant import detect_rope_spec

        class Attn:
            pass

        assert detect_rope_spec(Attn()) is None

    def test_unknown_rope_returns_none(self):
        from olmlx.engine.shardquant import detect_rope_spec

        class WeirdRope:
            pass  # no dims, no _freqs

        class Attn:
            rope = WeirdRope()

        assert detect_rope_spec(Attn()) is None
```

- [ ] **Step 1.2: Run to verify failure**

Run: `uv run pytest tests/test_shardquant.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.engine.shardquant'`

- [ ] **Step 1.3: Implement**

Create `olmlx/engine/shardquant.py`:

```python
"""ShardQuant primitives: asymmetric K/V KV-cache compression (#377 Tier 1).

Reference design: https://github.com/krish1905/shard
- K: undo RoPE -> per-head PCA basis -> rank truncation -> scalar Lloyd-Max.
- V: orthogonal rotation (Hadamard) -> per-position product VQ (256 centroids).
Both sides normalize to the unit sphere and store float32 norms, matching
the TurboQuant/SpectralQuant convention in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from olmlx.engine.spectralquant import pack_indices, unpack_indices

__all__ = [
    "RopeSpec",
    "detect_rope_spec",
    "rope_transform",
    "make_v_rotation",
    "fit_vq_codebooks",
    "vq_assign",
    "vq_gather",
    "scalar_assign",
    "shard_compress_keys",
    "shard_decompress_keys",
    "shard_compress_values",
    "shard_decompress_values",
    "pack_indices",
    "unpack_indices",
]

# Number of centroids per product-VQ codebook. uint8 indices, one byte per
# subvector; subvector size g = 8 // bits gives exactly `bits` bits/dim.
VQ_CENTROIDS = 256

# Sequence-chunk size for the broadcast argmin in vq_assign / scalar_assign.
# Bounds the transient (..., chunk, P, 256) distance tensor (~256 MB at
# H=8, D/g=64 — same budget rationale as spectralquant's broadcast cap).
_ASSIGN_SEQ_CHUNK = 512


@dataclass
class RopeSpec:
    """Angular-frequency description of a layer's rotary embedding.

    ``freqs`` (dims//2,) are *angular* frequencies: the rotation angle for
    pair ``i`` at position ``p`` is ``p * freqs[i]``.  Stored in the shard
    calibration artifacts so calibration-time de-rope and runtime re-rope
    are guaranteed to use identical transforms.
    """

    dims: int
    freqs: mx.array
    traditional: bool


def detect_rope_spec(attn: Any) -> RopeSpec | None:
    """Extract a RopeSpec from an attention module, or None if unsupported.

    Handles ``mlx.nn.RoPE`` (base/scale parameterization) and mlx-lm's
    custom rope modules (Llama3RoPE, ...) that precompute wavelength-like
    ``_freqs`` and call ``mx.fast.rope(..., base=None, freqs=self._freqs)``
    — for those the angular frequency is ``1 / _freqs``.  Unknown rope
    types return None; the caller falls back to identity (no de-rope),
    which stays correct (undo/redo of identity) at reduced rank collapse.
    """
    rope = getattr(attn, "rope", None)
    if rope is None:
        return None
    import mlx.nn as nn

    if isinstance(rope, nn.RoPE):
        dims = rope.dims
        base = float(rope.base)
        scale = float(getattr(rope, "scale", 1.0))
        freqs = scale * mx.power(
            mx.array(base, dtype=mx.float32),
            -mx.arange(0, dims, 2, dtype=mx.float32) / dims,
        )
        return RopeSpec(dims=dims, freqs=freqs, traditional=bool(rope.traditional))

    inv_freqs = getattr(rope, "_freqs", None)
    dims = getattr(rope, "dims", None)
    if isinstance(inv_freqs, mx.array) and isinstance(dims, int):
        if inv_freqs.ndim == 1 and inv_freqs.shape[0] == dims // 2:
            return RopeSpec(
                dims=dims,
                freqs=(1.0 / inv_freqs.astype(mx.float32)),
                traditional=bool(getattr(rope, "traditional", False)),
            )
    return None


def rope_transform(
    x: mx.array, spec: RopeSpec, offset: int, *, inverse: bool = False
) -> mx.array:
    """Apply (or invert) rotary embedding for contiguous positions.

    Args:
        x: (..., S, D) tensor; the last ``D - spec.dims`` dims pass through.
        spec: RopeSpec with angular freqs.
        offset: absolute position of x[..., 0, :].
        inverse: apply the inverse rotation (de-rope).
    """
    S = x.shape[-2]
    pos = mx.arange(offset, offset + S, dtype=mx.float32)
    angles = pos[:, None] * spec.freqs[None, :]  # (S, dims//2)
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    if inverse:
        sin = -sin

    half = spec.dims // 2
    x32 = x.astype(mx.float32)
    head = x32[..., : spec.dims]
    if spec.traditional:
        xr = head.reshape(*head.shape[:-1], half, 2)
        x1, x2 = xr[..., 0], xr[..., 1]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        rotated = mx.stack([y1, y2], axis=-1).reshape(*head.shape)
    else:
        x1, x2 = head[..., :half], head[..., half:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        rotated = mx.concatenate([y1, y2], axis=-1)

    if spec.dims < x.shape[-1]:
        rotated = mx.concatenate([rotated, x32[..., spec.dims :]], axis=-1)
    return rotated.astype(x.dtype)
```

- [ ] **Step 1.4: Run to verify pass**

Run: `uv run pytest tests/test_shardquant.py -x -q`
Expected: PASS. If `test_matches_nn_rope` fails on the freqs convention, fix `rope_transform`/freq derivation until it passes — the nn.RoPE parity test is the source of truth.

- [ ] **Step 1.5: Commit**

```bash
git add olmlx/engine/shardquant.py tests/test_shardquant.py
git commit -m "feat(shardquant): RoPE spec detection + invertible rope transform (#377)"
```

---

### Task 2: V rotation + product VQ + scalar assign

**Files:**
- Modify: `olmlx/engine/shardquant.py`
- Test: `tests/test_shardquant.py`

- [ ] **Step 2.1: Write the failing tests** (append to `tests/test_shardquant.py`)

```python
class TestVRotation:
    def test_hadamard_for_power_of_two(self):
        from olmlx.engine.shardquant import make_v_rotation

        R = make_v_rotation(64)
        Rn = np.array(R)
        # Orthonormal
        np.testing.assert_allclose(Rn @ Rn.T, np.eye(64), atol=1e-5)
        # Hadamard: all entries ±1/sqrt(64)
        np.testing.assert_allclose(np.abs(Rn), 1.0 / 8.0, atol=1e-6)

    def test_qr_fallback_for_non_power_of_two(self):
        from olmlx.engine.shardquant import make_v_rotation

        R = make_v_rotation(80, seed=3)
        Rn = np.array(R)
        np.testing.assert_allclose(Rn @ Rn.T, np.eye(80), atol=1e-4)
        # Deterministic per seed
        np.testing.assert_array_equal(np.array(make_v_rotation(80, seed=3)), Rn)


class TestProductVQ:
    def test_fit_assign_gather_roundtrip_exact_on_clustered_data(self):
        """Data drawn exactly from K distinct subvectors reconstructs exactly."""
        from olmlx.engine.shardquant import fit_vq_codebooks, vq_assign, vq_gather

        rng = np.random.RandomState(0)
        P, g, K = 4, 2, 16
        true_centroids = rng.randn(P, K, g).astype(np.float32) * 5
        picks = rng.randint(0, K, size=(1000, P))
        data = np.stack(
            [true_centroids[p, picks[:, p]] for p in range(P)], axis=1
        )  # (N, P, g)
        cbs = fit_vq_codebooks(data.reshape(1000, P * g), group_size=g, seed=0)
        assert cbs.shape == (P, 256, g)
        x = mx.array(data.reshape(1, 1, 1000, P * g))
        idx = vq_assign(x.reshape(1, 1, 1000, P, g), cbs)
        assert idx.shape == (1, 1, 1000, P)
        assert idx.dtype == mx.uint8
        recon = vq_gather(idx, cbs)
        np.testing.assert_allclose(
            np.array(recon).reshape(1000, P, g), data, atol=1e-4
        )

    def test_quality_on_gaussian_data(self):
        """256 centroids on unit-sphere-ish 2-dim subvectors: high cosine sim."""
        from olmlx.engine.shardquant import fit_vq_codebooks, vq_assign, vq_gather

        rng = np.random.RandomState(1)
        N, D, g = 4000, 16, 2
        data = rng.randn(N, D).astype(np.float32)
        data /= np.linalg.norm(data, axis=-1, keepdims=True)
        cbs = fit_vq_codebooks(data, group_size=g, seed=1)
        x = mx.array(data.reshape(1, 1, N, D // g, g))
        recon = np.array(vq_gather(vq_assign(x, cbs), cbs)).reshape(N, D)
        cos = np.sum(recon * data, axis=-1) / (
            np.linalg.norm(recon, axis=-1) * np.linalg.norm(data, axis=-1)
        )
        assert cos.mean() > 0.95

    def test_fewer_points_than_centroids(self):
        from olmlx.engine.shardquant import fit_vq_codebooks

        rng = np.random.RandomState(2)
        cbs = fit_vq_codebooks(rng.randn(50, 8).astype(np.float32), group_size=2)
        assert cbs.shape == (4, 256, 2)
        assert np.isfinite(np.array(cbs)).all()


class TestScalarAssign:
    def test_matches_naive_argmin(self):
        from olmlx.engine.shardquant import scalar_assign

        rng = np.random.RandomState(3)
        codebook = mx.array(np.sort(rng.randn(16)).astype(np.float32))
        y = mx.array(rng.randn(1, 2, 700, 5).astype(np.float32))
        idx = scalar_assign(y, codebook)
        assert idx.dtype == mx.uint8
        expected = np.argmin(
            np.abs(np.array(y)[..., None] - np.array(codebook)), axis=-1
        )
        np.testing.assert_array_equal(np.array(idx), expected)
```

- [ ] **Step 2.2: Run to verify failure**

Run: `uv run pytest tests/test_shardquant.py -x -q -k "VRotation or ProductVQ or ScalarAssign"`
Expected: FAIL with `ImportError: cannot import name 'make_v_rotation'`

- [ ] **Step 2.3: Implement** (append to `olmlx/engine/shardquant.py`)

```python
def make_v_rotation(dim: int, seed: int = 0) -> mx.array:
    """Orthonormal rotation for the V path.

    Sylvester-Hadamard (scaled 1/sqrt(dim)) when ``dim`` is a power of two,
    otherwise a seeded random orthogonal matrix via QR.  The matrix is
    persisted in the calibration artifacts, so the runtime never needs to
    re-derive it — one uniform code path either way.
    """
    if dim > 0 and (dim & (dim - 1)) == 0:
        h = np.array([[1.0]], dtype=np.float64)
        while h.shape[0] < dim:
            h = np.block([[h, h], [h, -h]])
        return mx.array((h / np.sqrt(dim)).astype(np.float32))
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.randn(dim, dim).astype(np.float32))
    return mx.array(q.astype(np.float32))


def fit_vq_codebooks(
    data: np.ndarray,
    group_size: int,
    n_centroids: int = VQ_CENTROIDS,
    max_iter: int = 25,
    seed: int = 0,
) -> mx.array:
    """Fit per-position product-VQ codebooks via plain numpy k-means.

    Args:
        data: (N, D) rotated value vectors; D % group_size == 0.
        group_size: subvector length g.

    Returns:
        (D // group_size, n_centroids, group_size) float32 codebooks.
    """
    n, d = data.shape
    if d % group_size != 0:
        raise ValueError(f"dim {d} not divisible by group_size {group_size}")
    p = d // group_size
    sub = data.reshape(n, p, group_size).astype(np.float32)
    rng = np.random.RandomState(seed)
    codebooks = np.empty((p, n_centroids, group_size), dtype=np.float32)

    for j in range(p):
        pts = sub[:, j, :]  # (N, g)
        init_idx = rng.randint(0, n, size=n_centroids)
        cent = pts[init_idx].copy()
        for _ in range(max_iter):
            # (N, K) squared distances via expansion; K*g is small.
            d2 = (
                (pts**2).sum(-1, keepdims=True)
                - 2.0 * pts @ cent.T
                + (cent**2).sum(-1)[None, :]
            )
            assign = d2.argmin(axis=1)
            new_cent = cent.copy()
            moved = 0.0
            for k in range(n_centroids):
                mask = assign == k
                if mask.any():
                    nc = pts[mask].mean(axis=0)
                    moved = max(moved, float(np.abs(nc - new_cent[k]).max()))
                    new_cent[k] = nc
                else:
                    # Re-seed empty clusters to a random point.
                    new_cent[k] = pts[rng.randint(0, n)]
            cent = new_cent
            if moved < 1e-6:
                break
        codebooks[j] = cent
    return mx.array(codebooks)


def vq_assign(x: mx.array, codebooks: mx.array) -> mx.array:
    """Assign each subvector to its nearest centroid.

    Args:
        x: (..., S, P, g) subvectors.
        codebooks: (P, K, g).

    Returns:
        (..., S, P) uint8 indices.
    """
    c_sq = mx.sum(codebooks * codebooks, axis=-1)  # (P, K)
    chunks = []
    S = x.shape[-3]
    for s0 in range(0, S, _ASSIGN_SEQ_CHUNK):
        xc = x[..., s0 : s0 + _ASSIGN_SEQ_CHUNK, :, :].astype(mx.float32)
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; ||x||^2 constant in argmin.
        scores = c_sq - 2.0 * mx.einsum("...pg,pkg->...pk", xc, codebooks)
        chunks.append(mx.argmin(scores, axis=-1).astype(mx.uint8))
    return mx.concatenate(chunks, axis=-2) if len(chunks) > 1 else chunks[0]


def vq_gather(idx: mx.array, codebooks: mx.array) -> mx.array:
    """Inverse of vq_assign: (..., S, P) uint8 -> (..., S, P, g) float32."""
    p, k, g = codebooks.shape
    flat = codebooks.reshape(p * k, g)
    flat_idx = idx.astype(mx.uint32) + mx.arange(p, dtype=mx.uint32) * k
    return flat[flat_idx]


def scalar_assign(y: mx.array, codebook: mx.array) -> mx.array:
    """Nearest-centroid index per element, chunked over the seq axis.

    Args:
        y: (..., S, R) coefficients.
        codebook: (K,) sorted centroids.

    Returns:
        (..., S, R) uint8 indices.
    """
    chunks = []
    S = y.shape[-2]
    for s0 in range(0, S, _ASSIGN_SEQ_CHUNK):
        yc = y[..., s0 : s0 + _ASSIGN_SEQ_CHUNK, :].astype(mx.float32)
        dists = mx.abs(yc[..., None] - codebook)
        chunks.append(mx.argmin(dists, axis=-1).astype(mx.uint8))
    return mx.concatenate(chunks, axis=-2) if len(chunks) > 1 else chunks[0]
```

- [ ] **Step 2.4: Run to verify pass**

Run: `uv run pytest tests/test_shardquant.py -x -q`
Expected: PASS

- [ ] **Step 2.5: Commit**

```bash
git add olmlx/engine/shardquant.py tests/test_shardquant.py
git commit -m "feat(shardquant): V rotation (Hadamard/QR) + product VQ + scalar assign (#377)"
```

---

### Task 3: K and V compress/decompress

**Files:**
- Modify: `olmlx/engine/shardquant.py`
- Test: `tests/test_shardquant.py`

Conventions: per-head K basis `(H, D, D)` with **rows = eigenvectors** (project `y = x @ basis^T` via broadcasting matmul; reconstruct `x̂ = y @ basis`). Both paths normalize to the unit sphere first and return float32 norms `(B, H, S, 1)`.

- [ ] **Step 3.1: Write the failing tests** (append to `tests/test_shardquant.py`)

```python
def _random_orthogonal(dim, seed):
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.randn(dim, dim).astype(np.float32))
    return q


class TestKeyCompress:
    def test_roundtrip_high_quality_full_rank_8bit(self):
        """Full rank + 8-bit: reconstruction within scalar-quant error."""
        from olmlx.engine.shardquant import (
            shard_compress_keys,
            shard_decompress_keys,
        )
        from olmlx.engine.spectralquant import fit_codebook

        B, H, S, D = 1, 2, 50, 32
        basis = mx.array(
            np.stack([_random_orthogonal(D, h) for h in range(H)])
        )  # (H, D, D)
        mx.random.seed(5)
        x = mx.random.normal((B, H, S, D))
        # Codebook fit on the actual rotated coefficients.
        x32 = x.astype(mx.float32)
        xn = x32 / mx.linalg.norm(x32, axis=-1, keepdims=True)
        y = mx.matmul(xn, mx.swapaxes(basis, -1, -2))
        cb = fit_codebook(y.reshape(-1), bits=8)

        packed, norms = shard_compress_keys(x, basis, rank=D, codebook=cb, bits=8)
        assert norms.shape == (B, H, S, 1)
        recon = shard_decompress_keys(
            packed, norms, basis, rank=D, codebook=cb, bits=8, dtype=x.dtype
        )
        assert recon.shape == x.shape
        cos = np.sum(np.array(recon) * np.array(x), -1) / (
            np.linalg.norm(np.array(recon), axis=-1)
            * np.linalg.norm(np.array(x), axis=-1)
        )
        assert cos.mean() > 0.99

    def test_rank_truncation_on_low_rank_data(self):
        """Data living in an r-dim subspace survives rank-r truncation."""
        from olmlx.engine.shardquant import (
            shard_compress_keys,
            shard_decompress_keys,
        )
        from olmlx.engine.spectralquant import fit_codebook

        B, H, S, D, R = 1, 1, 200, 16, 4
        rng = np.random.RandomState(7)
        Q = _random_orthogonal(D, 9)
        coeffs = rng.randn(S, R).astype(np.float32)
        data = coeffs @ Q[:R, :]  # rows of Q span the subspace
        x = mx.array(data.reshape(B, H, S, D))
        basis = mx.array(Q.reshape(1, D, D))  # rows = basis vectors
        xn = np.array(x) / np.linalg.norm(np.array(x), axis=-1, keepdims=True)
        y = mx.array(xn) @ mx.array(Q.T)
        cb = fit_codebook(mx.array(np.array(y)[..., :R].reshape(-1)), bits=4)

        packed, norms = shard_compress_keys(x, basis, rank=R, codebook=cb, bits=4)
        # Packed coefficient payload is rank-sized, not head_dim-sized.
        assert packed.shape[-1] == R // 2  # 4-bit pack: 2 per byte
        recon = shard_decompress_keys(
            packed, norms, basis, rank=R, codebook=cb, bits=4, dtype=x.dtype
        )
        cos = np.sum(np.array(recon) * np.array(x), -1) / (
            np.linalg.norm(np.array(recon), axis=-1)
            * np.linalg.norm(np.array(x), axis=-1)
        )
        assert cos.mean() > 0.9

    def test_per_head_bases_are_independent(self):
        """Head h must be projected with basis[h], not basis[0]."""
        from olmlx.engine.shardquant import (
            shard_compress_keys,
            shard_decompress_keys,
        )
        from olmlx.engine.spectralquant import fit_codebook

        D = 8
        basis = mx.array(
            np.stack([np.eye(D, dtype=np.float32), _random_orthogonal(D, 1)])
        )
        mx.random.seed(8)
        x = mx.random.normal((1, 2, 30, D))
        cb = fit_codebook(mx.random.normal((4000,)) * 0.35, bits=8)
        packed, norms = shard_compress_keys(x, basis, rank=D, codebook=cb, bits=8)
        recon = shard_decompress_keys(
            packed, norms, basis, rank=D, codebook=cb, bits=8, dtype=x.dtype
        )
        for h in range(2):
            cos = np.sum(np.array(recon)[0, h] * np.array(x)[0, h], -1) / (
                np.linalg.norm(np.array(recon)[0, h], axis=-1)
                * np.linalg.norm(np.array(x)[0, h], axis=-1)
            )
            assert cos.mean() > 0.95, f"head {h} mis-projected"


class TestValueCompress:
    def test_roundtrip_quality(self):
        from olmlx.engine.shardquant import (
            fit_vq_codebooks,
            make_v_rotation,
            shard_compress_values,
            shard_decompress_values,
        )

        B, H, S, D, g = 1, 2, 300, 16, 2
        R = make_v_rotation(D)
        mx.random.seed(9)
        x = mx.random.normal((B, H, S, D)).astype(mx.float16)
        x32 = np.array(x, dtype=np.float32).reshape(-1, D)
        xn = x32 / np.linalg.norm(x32, axis=-1, keepdims=True)
        rotated = xn @ np.array(R).T
        cbs = fit_vq_codebooks(rotated, group_size=g, seed=0)

        idx, norms = shard_compress_values(x, R, cbs)
        assert idx.shape == (B, H, S, D // g)
        assert idx.dtype == mx.uint8
        recon = shard_decompress_values(idx, norms, R, cbs, dtype=x.dtype)
        assert recon.shape == x.shape
        assert recon.dtype == x.dtype
        cos = np.sum(
            np.array(recon, dtype=np.float32) * x32.reshape(B, H, S, D), -1
        ) / (
            np.linalg.norm(np.array(recon, dtype=np.float32), axis=-1)
            * np.linalg.norm(x32.reshape(B, H, S, D), axis=-1)
        )
        assert cos.mean() > 0.9
```

- [ ] **Step 3.2: Run to verify failure**

Run: `uv run pytest tests/test_shardquant.py -x -q -k "KeyCompress or ValueCompress"`
Expected: FAIL with ImportError

- [ ] **Step 3.3: Implement** (append to `olmlx/engine/shardquant.py`)

```python
_NORM_EPS = 1e-8


def _normalize(x: mx.array) -> tuple[mx.array, mx.array]:
    """Unit-sphere normalize; returns (normalized float32, float32 norms)."""
    x32 = x.astype(mx.float32)
    norms = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
    eps = mx.array(_NORM_EPS, dtype=mx.float32)
    return x32 / mx.maximum(norms, eps), norms


def shard_compress_keys(
    keys: mx.array,
    basis: mx.array,
    rank: int,
    codebook: mx.array,
    bits: int,
) -> tuple[mx.array, mx.array]:
    """Compress (already de-roped) keys: normalize -> per-head project ->
    rank-truncate -> scalar quantize -> bit-pack.

    Args:
        keys: (B, H, S, D) de-roped keys.
        basis: (H, D, D), rows = eigenvectors.
        rank: kept leading coefficients.
        codebook: (2**bits,) Lloyd-Max centroids for the kept coefficients.

    Returns:
        (packed, norms): packed uint8 indices (B, H, S, packed_rank) +
        float32 norms (B, H, S, 1).
    """
    xn, norms = _normalize(keys)
    y = mx.matmul(xn, mx.swapaxes(basis, -1, -2))  # (B,H,S,D) @ (H,D,D)^T
    idx = scalar_assign(y[..., :rank], codebook)
    return pack_indices(idx, bits), norms


def shard_decompress_keys(
    packed: mx.array,
    norms: mx.array,
    basis: mx.array,
    rank: int,
    codebook: mx.array,
    bits: int,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Inverse of shard_compress_keys (still in the no-RoPE basis)."""
    idx = unpack_indices(packed, bits, rank)
    y = codebook[idx.astype(mx.uint32)]
    d = basis.shape[-1]
    if rank < d:
        pad = mx.zeros((*y.shape[:-1], d - rank), dtype=y.dtype)
        y = mx.concatenate([y, pad], axis=-1)
    x = mx.matmul(y, basis) * norms
    return x.astype(dtype) if dtype is not None else x


def shard_compress_values(
    values: mx.array,
    rotation: mx.array,
    codebooks: mx.array,
) -> tuple[mx.array, mx.array]:
    """Compress values: normalize -> rotate -> product VQ.

    Args:
        values: (B, H, S, D).
        rotation: (D, D) orthonormal (rows = basis).
        codebooks: (P, K, g) with P * g == D.

    Returns:
        (idx, norms): uint8 indices (B, H, S, P) + float32 norms (B, H, S, 1).
    """
    p, _, g = codebooks.shape
    xn, norms = _normalize(values)
    rotated = xn @ rotation.T
    sub = rotated.reshape(*rotated.shape[:-1], p, g)
    return vq_assign(sub, codebooks), norms


def shard_decompress_values(
    idx: mx.array,
    norms: mx.array,
    rotation: mx.array,
    codebooks: mx.array,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Inverse of shard_compress_values."""
    sub = vq_gather(idx, codebooks)
    d = rotation.shape[0]
    rotated = sub.reshape(*sub.shape[:-2], d)
    x = (rotated @ rotation) * norms
    return x.astype(dtype) if dtype is not None else x
```

- [ ] **Step 3.4: Run to verify pass**

Run: `uv run pytest tests/test_shardquant.py -x -q`
Expected: PASS

- [ ] **Step 3.5: Commit**

```bash
git add olmlx/engine/shardquant.py tests/test_shardquant.py
git commit -m "feat(shardquant): asymmetric K/V compress-decompress primitives (#377)"
```

---

### Task 4: ShardKVCache

**Files:**
- Create: `olmlx/engine/shardquant_cache.py`
- Test: `tests/test_shardquant_cache.py`

Layout invariant: a token stream of `offset` tokens is partitioned as `[sink | middle | window]` with `sink_len + mid_len + win_len == offset`. Sink and window hold exact post-RoPE K and exact V in the input dtype. The middle holds compressed K (de-roped) and compressed V. Middle token `j` has absolute position `sink_len + j` (needed for re-rope). New tokens append to the window; overflow beyond `window_size` is compressed from the window's *front* into the middle. Sink fills from the very first tokens seen.

`update_and_fetch` per call: append → spill overflow to middle → return `concat([sink, decompress(middle), window])`. **Transient dequant** — no persistent dequantized buffer (the review comment's resident-memory requirement).

`trim(n)`: remove exactly n tokens from the end — first from window, then middle, then sink. Exact, so `is_trimmable() → True`.

Deepcopy/snapshot safety (#284 family): no `mx.Dtype` attributes; every mutable `mx.array` is reachable from `state` so `snapshot_cache_for_persistence`'s `flatten_cache_state` + `mx.eval` materializes in-place-write graphs before the cross-thread crossing; default deepcopy walk then suffices (the spectral pattern, not the turboquant custom-`__deepcopy__` pattern).

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_shardquant_cache.py`:

```python
"""Tests for ShardKVCache (#377 Tier 1)."""

import copy

import mlx.core as mx
import numpy as np
import pytest


def _random_orthogonal(dim, seed):
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.randn(dim, dim).astype(np.float32))
    return q


def _make_cache(D=16, H=2, bits=4, rank=None, sink=4, window=8, rope=True):
    """Build a ShardKVCache with synthetic (but valid) calibration tensors."""
    from olmlx.engine.shardquant import RopeSpec, fit_vq_codebooks, make_v_rotation
    from olmlx.engine.shardquant_cache import ShardKVCache
    from olmlx.engine.spectralquant import fit_codebook

    rank = rank if rank is not None else D
    basis = mx.array(np.stack([_random_orthogonal(D, 10 + h) for h in range(H)]))
    rng = np.random.RandomState(0)
    k_codebook = fit_codebook(mx.array(rng.randn(4096).astype(np.float32) * 0.3), bits=bits)
    v_rot = make_v_rotation(D)
    g = 8 // bits
    sample = rng.randn(4096, D).astype(np.float32)
    sample /= np.linalg.norm(sample, axis=-1, keepdims=True)
    v_codebooks = fit_vq_codebooks(sample @ np.array(v_rot).T, group_size=g)
    spec = None
    if rope:
        freqs = mx.power(
            mx.array(10000.0, dtype=mx.float32),
            -mx.arange(0, D, 2, dtype=mx.float32) / D,
        )
        spec = RopeSpec(dims=D, freqs=freqs, traditional=False)
    return ShardKVCache(
        rope_spec=spec,
        k_basis=basis,
        k_rank=rank,
        k_codebook=k_codebook,
        k_bits=bits,
        v_rotation=v_rot,
        v_codebooks=v_codebooks,
        sink_size=sink,
        window_size=window,
    )


def _feed(cache, total, D=16, H=2, step=1, seed=0):
    """Feed `total` tokens, returning the exact K/V that were fed."""
    mx.random.seed(seed)
    ks = mx.random.normal((1, H, total, D)).astype(mx.float16)
    vs = mx.random.normal((1, H, total, D)).astype(mx.float16)
    out = None
    for s0 in range(0, total, step):
        out = cache.update_and_fetch(
            ks[..., s0 : s0 + step, :], vs[..., s0 : s0 + step, :]
        )
    return ks, vs, out


class TestShardKVCacheBasics:
    def test_output_shape_and_dtype(self):
        cache = _make_cache()
        ks, vs, (k, v) = _feed(cache, 20)
        assert k.shape == (1, 2, 20, 16)
        assert v.shape == (1, 2, 20, 16)
        assert k.dtype == mx.float16
        assert cache.offset == 20

    def test_small_prompt_stays_exact(self):
        """Until sink+window overflows, everything is FP16-exact."""
        cache = _make_cache(sink=4, window=8)
        ks, vs, (k, v) = _feed(cache, 10)  # 10 <= 4 + 8
        np.testing.assert_array_equal(np.array(k), np.array(ks))
        np.testing.assert_array_equal(np.array(v), np.array(vs))

    def test_sink_and_window_exact_after_overflow(self):
        cache = _make_cache(sink=4, window=8)
        ks, vs, (k, v) = _feed(cache, 40)
        # First sink tokens exact
        np.testing.assert_array_equal(
            np.array(k[..., :4, :]), np.array(ks[..., :4, :])
        )
        np.testing.assert_array_equal(
            np.array(v[..., :4, :]), np.array(vs[..., :4, :])
        )
        # Last window tokens exact
        np.testing.assert_array_equal(
            np.array(k[..., -8:, :]), np.array(ks[..., -8:, :])
        )
        np.testing.assert_array_equal(
            np.array(v[..., -8:, :]), np.array(vs[..., -8:, :])
        )

    def test_middle_reconstruction_quality(self):
        cache = _make_cache(sink=4, window=8, bits=8)
        ks, vs, (k, v) = _feed(cache, 64)
        mid_k = np.array(k[..., 4:-8, :], dtype=np.float32)
        ref_k = np.array(ks[..., 4:-8, :], dtype=np.float32)
        cos = np.sum(mid_k * ref_k, -1) / (
            np.linalg.norm(mid_k, axis=-1) * np.linalg.norm(ref_k, axis=-1) + 1e-9
        )
        assert cos.mean() > 0.9, f"K middle cosine {cos.mean()}"
        mid_v = np.array(v[..., 4:-8, :], dtype=np.float32)
        ref_v = np.array(vs[..., 4:-8, :], dtype=np.float32)
        cos_v = np.sum(mid_v * ref_v, -1) / (
            np.linalg.norm(mid_v, axis=-1) * np.linalg.norm(ref_v, axis=-1) + 1e-9
        )
        assert cos_v.mean() > 0.85, f"V middle cosine {cos_v.mean()}"

    def test_prefill_then_decode_matches_pure_decode(self):
        """One 30-token prefill + decode steps ≈ 1-token-at-a-time feed."""
        c1 = _make_cache()
        c2 = _make_cache()
        mx.random.seed(3)
        ks = mx.random.normal((1, 2, 34, 16)).astype(mx.float16)
        vs = mx.random.normal((1, 2, 34, 16)).astype(mx.float16)
        c1.update_and_fetch(ks[..., :30, :], vs[..., :30, :])
        for i in range(30, 34):
            k1, v1 = c1.update_and_fetch(
                ks[..., i : i + 1, :], vs[..., i : i + 1, :]
            )
        out = None
        for i in range(34):
            out = c2.update_and_fetch(
                ks[..., i : i + 1, :], vs[..., i : i + 1, :]
            )
        k2, v2 = out
        np.testing.assert_allclose(
            np.array(k1, dtype=np.float32),
            np.array(k2, dtype=np.float32),
            atol=5e-2,
        )

    def test_no_rope_spec_still_works(self):
        cache = _make_cache(rope=False)
        ks, vs, (k, v) = _feed(cache, 30)
        assert k.shape == (1, 2, 30, 16)

    def test_empty(self):
        cache = _make_cache()
        assert cache.empty()
        _feed(cache, 3)
        assert not cache.empty()


class TestShardKVCacheTrim:
    def test_is_trimmable(self):
        assert _make_cache().is_trimmable()

    def test_trim_within_window(self):
        cache = _make_cache(sink=4, window=8)
        ks, vs, _ = _feed(cache, 10)
        assert cache.trim(3) == 3
        assert cache.offset == 7
        k, v = cache.update_and_fetch(
            ks[..., 7:8, :], vs[..., 7:8, :]
        )
        assert k.shape[2] == 8
        np.testing.assert_array_equal(
            np.array(k[..., :8, :]), np.array(ks[..., :8, :])
        )

    def test_trim_into_middle(self):
        cache = _make_cache(sink=4, window=8)
        ks, vs, _ = _feed(cache, 40)
        # 40 = 4 sink + 28 middle + 8 window; trim 20 reaches the middle.
        assert cache.trim(20) == 20
        assert cache.offset == 20
        k, v = cache.update_and_fetch(ks[..., 20:21, :], vs[..., 20:21, :])
        assert k.shape[2] == 21
        # Sink still exact
        np.testing.assert_array_equal(
            np.array(k[..., :4, :]), np.array(ks[..., :4, :])
        )

    def test_trim_to_empty(self):
        cache = _make_cache()
        ks, vs, _ = _feed(cache, 30)
        assert cache.trim(30) == 30
        assert cache.offset == 0
        assert cache.empty()
        # Reusable after full trim
        _feed(cache, 5)
        assert cache.offset == 5

    def test_trim_clamps_to_offset(self):
        cache = _make_cache()
        _feed(cache, 10)
        assert cache.trim(99) == 10
        assert cache.offset == 0


class TestShardKVCacheStateAndCopy:
    def test_state_exposes_arrays_and_setter_raises(self):
        cache = _make_cache()
        _feed(cache, 40)
        st = cache.state
        assert len(st) > 0
        assert all(isinstance(a, mx.array) for a in st)
        with pytest.raises(NotImplementedError):
            cache.state = st

    def test_state_empty_when_fresh(self):
        assert _make_cache().state == []

    def test_deepcopy_then_diverge(self):
        """Snapshot path: deepcopy, then mutating the original must not
        affect the copy (and vice versa)."""
        cache = _make_cache()
        ks, vs, _ = _feed(cache, 40)
        snap = copy.deepcopy(cache)
        mx.eval([a for a in snap.state])
        k_before = np.array(snap.update_and_fetch(
            ks[..., :1, :] * 0, vs[..., :1, :] * 0
        )[0][..., :40, :])
        # Continue the original with different tokens
        mx.random.seed(99)
        more_k = mx.random.normal((1, 2, 5, 16)).astype(mx.float16)
        cache.update_and_fetch(more_k, more_k)
        assert cache.offset == 45
        assert snap.offset == 41

    def test_no_dtype_attribute(self):
        """No mx.Dtype attrs — keeps the default deepcopy walk safe
        (the TurboQuant pickle failure mode)."""
        cache = _make_cache()
        _feed(cache, 40)
        assert not any(
            isinstance(v, mx.Dtype) for v in cache.__dict__.values()
        )


class TestMakeShardCache:
    def _calib_entry(self, D=16, H=2, bits=4):
        from olmlx.engine.shardquant import fit_vq_codebooks, make_v_rotation
        from olmlx.engine.spectralquant import fit_codebook

        rng = np.random.RandomState(0)
        sample = rng.randn(1024, D).astype(np.float32)
        sample /= np.linalg.norm(sample, axis=-1, keepdims=True)
        v_rot = make_v_rotation(D)
        return {
            "k_basis": mx.array(
                np.stack([_random_orthogonal(D, h) for h in range(H)])
            ),
            "k_rank": D // 2,
            "k_codebook": fit_codebook(
                mx.array(rng.randn(2048).astype(np.float32) * 0.3), bits=bits
            ),
            "v_rotation": v_rot,
            "v_codebooks": fit_vq_codebooks(
                sample @ np.array(v_rot).T, group_size=8 // bits
            ),
            "rope_freqs": None,
            "rope_dims": None,
            "rope_traditional": False,
        }

    def test_make_shard_cache_quantizes_attention_layers(self, tmp_path, monkeypatch):
        from mlx_lm.models.cache import KVCache

        from olmlx.engine import shardquant_cache as scm
        from olmlx.engine.shardquant_cache import ShardKVCache, make_shard_cache

        calibration = {0: self._calib_entry(), 1: self._calib_entry()}
        monkeypatch.setattr(
            "olmlx.engine.shardquant_calibrate.load_shard_calibration",
            lambda d: (calibration, {"bits": 4, "head_dim": 16, "n_kv_heads": 2}),
        )

        class Model:
            layers = [object(), object()]

            def make_cache(self):
                return [KVCache(), KVCache()]

        caches = make_shard_cache(Model(), tmp_path, bits=4)
        assert len(caches) == 2
        assert all(isinstance(c, ShardKVCache) for c in caches)

    def test_make_shard_cache_preserves_non_attention(self, tmp_path, monkeypatch):
        from mlx_lm.models.cache import ArraysCache, KVCache

        from olmlx.engine.shardquant_cache import ShardKVCache, make_shard_cache

        calibration = {0: self._calib_entry(), 1: self._calib_entry()}
        monkeypatch.setattr(
            "olmlx.engine.shardquant_calibrate.load_shard_calibration",
            lambda d: (calibration, {"bits": 4, "head_dim": 16, "n_kv_heads": 2}),
        )

        ssm = ArraysCache(size=2)

        class Model:
            layers = [object(), object()]

            def make_cache(self):
                return [ssm, KVCache()]

        caches = make_shard_cache(Model(), tmp_path, bits=4)
        assert caches[0] is ssm
        assert isinstance(caches[1], ShardKVCache)

    def test_missing_layer_calibration_falls_back(self, tmp_path, monkeypatch):
        from mlx_lm.models.cache import KVCache

        from olmlx.engine.shardquant_cache import ShardKVCache, make_shard_cache

        calibration = {0: self._calib_entry()}  # layer 1 missing
        monkeypatch.setattr(
            "olmlx.engine.shardquant_calibrate.load_shard_calibration",
            lambda d: (calibration, {"bits": 4, "head_dim": 16, "n_kv_heads": 2}),
        )

        class Model:
            layers = [object(), object()]

            def make_cache(self):
                return [KVCache(), KVCache()]

        caches = make_shard_cache(Model(), tmp_path, bits=4)
        assert isinstance(caches[0], ShardKVCache)
        assert isinstance(caches[1], KVCache)
```

- [ ] **Step 4.2: Run to verify failure**

Run: `uv run pytest tests/test_shardquant_cache.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.engine.shardquant_cache'`

- [ ] **Step 4.3: Implement**

Create `olmlx/engine/shardquant_cache.py`:

```python
"""Shard KV cache: drop-in replacement for mlx-lm's KVCache (#377 Tier 1).

Asymmetric K/V compression with an FP16 sink + FP16 sliding window and a
quantized middle region, dequantized transiently on every fetch:

    [ sink (exact) | middle (compressed) | window (exact) ]

K middle is stored *de-roped* (RoPE inverted at compress time) in a per-head
PCA basis with rank truncation; RoPE is re-applied at fetch using the
absolute positions sink_len..sink_len+mid_len.  V middle is rotation +
product-VQ.  No persistent dequantized side buffer — resident memory is
quantized + sink/window only (the Tier-1 memory contract from #377).

Deepcopy/snapshot safety: no ``mx.Dtype`` attributes (default deepcopy walk
works — the SpectralQuant pattern), and every mutable array is exposed via
``state`` so ``snapshot_cache_for_persistence``'s flatten + ``mx.eval``
materializes in-place-write graphs before any cross-thread reuse (#284).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_lm.models.cache import KVCache, _BaseCache, create_attention_mask

from olmlx.engine.shardquant import (
    RopeSpec,
    rope_transform,
    shard_compress_keys,
    shard_compress_values,
    shard_decompress_keys,
    shard_decompress_values,
)

logger = logging.getLogger(__name__)

#: Default sink/window sizes from the Shard reference design.
SINK_TOKENS = 4
WINDOW_TOKENS = 64


class ShardKVCache(_BaseCache):
    """KV cache with Shard-style asymmetric K/V compression."""

    step = 256

    def __init__(
        self,
        rope_spec: RopeSpec | None,
        k_basis: mx.array,
        k_rank: int,
        k_codebook: mx.array,
        k_bits: int,
        v_rotation: mx.array,
        v_codebooks: mx.array,
        sink_size: int = SINK_TOKENS,
        window_size: int = WINDOW_TOKENS,
    ):
        self.rope_spec = rope_spec
        self.k_basis = k_basis
        self.k_rank = k_rank
        self.k_codebook = k_codebook
        self.k_bits = k_bits
        self.v_rotation = v_rotation
        self.v_codebooks = v_codebooks
        self.sink_size = sink_size
        self.window_size = window_size

        # Exact regions (input dtype). Window buffers are small (<= window
        # + one prefill batch transiently); plain concat/slice is fine.
        self._k_sink: mx.array | None = None
        self._v_sink: mx.array | None = None
        self._k_win: mx.array | None = None
        self._v_win: mx.array | None = None
        # Compressed middle, grown in `step` increments like the other
        # quant caches. Only [..., :_mid_len, :] is valid.
        self._k_mid: mx.array | None = None
        self._k_mid_norms: mx.array | None = None
        self._v_mid: mx.array | None = None
        self._v_mid_norms: mx.array | None = None
        self._mid_len = 0
        self.offset = 0

    # -- internal helpers ---------------------------------------------------

    def _sink_len(self) -> int:
        return 0 if self._k_sink is None else self._k_sink.shape[2]

    def _win_len(self) -> int:
        return 0 if self._k_win is None else self._k_win.shape[2]

    def _append_middle(self, k_packed, k_norms, v_idx, v_norms) -> None:
        n = k_packed.shape[2]
        need = self._mid_len + n
        if self._k_mid is None or need > self._k_mid.shape[2]:
            new_steps = (need + self.step - 1) // self.step * self.step
            def _grow(buf, last_dim, dtype):
                shape = (*k_packed.shape[:2], new_steps, last_dim)
                fresh = mx.zeros(shape, dtype=dtype)
                if buf is not None:
                    fresh[..., : self._mid_len, :] = buf[..., : self._mid_len, :]
                return fresh

            self._k_mid = _grow(self._k_mid, k_packed.shape[-1], mx.uint8)
            self._k_mid_norms = _grow(self._k_mid_norms, 1, mx.float32)
            self._v_mid = _grow(self._v_mid, v_idx.shape[-1], mx.uint8)
            self._v_mid_norms = _grow(self._v_mid_norms, 1, mx.float32)
        assert (
            self._k_mid is not None
            and self._k_mid_norms is not None
            and self._v_mid is not None
            and self._v_mid_norms is not None
        )
        self._k_mid[..., self._mid_len : need, :] = k_packed
        self._k_mid_norms[..., self._mid_len : need, :] = k_norms
        self._v_mid[..., self._mid_len : need, :] = v_idx
        self._v_mid_norms[..., self._mid_len : need, :] = v_norms
        self._mid_len = need

    def _compress_into_middle(self, k: mx.array, v: mx.array) -> None:
        """Compress a window-front slice. Its absolute start position is
        sink_len + mid_len (the next middle slot)."""
        start = self._sink_len() + self._mid_len
        if self.rope_spec is not None:
            k = rope_transform(k, self.rope_spec, start, inverse=True)
        k_packed, k_norms = shard_compress_keys(
            k, self.k_basis, self.k_rank, self.k_codebook, self.k_bits
        )
        v_idx, v_norms = shard_compress_values(v, self.v_rotation, self.v_codebooks)
        self._append_middle(k_packed, k_norms, v_idx, v_norms)

    def _decompress_middle(self, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        assert (
            self._k_mid is not None
            and self._k_mid_norms is not None
            and self._v_mid is not None
            and self._v_mid_norms is not None
        )
        m = self._mid_len
        k = shard_decompress_keys(
            self._k_mid[..., :m, :],
            self._k_mid_norms[..., :m, :],
            self.k_basis,
            self.k_rank,
            self.k_codebook,
            self.k_bits,
        )
        if self.rope_spec is not None:
            k = rope_transform(k, self.rope_spec, self._sink_len())
        v = shard_decompress_values(
            self._v_mid[..., :m, :],
            self._v_mid_norms[..., :m, :],
            self.v_rotation,
            self.v_codebooks,
        )
        return k.astype(dtype), v.astype(dtype)

    # -- _BaseCache interface -----------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        num_steps = keys.shape[2]
        dtype = keys.dtype

        # 1. Fill the sink from the front of the stream.
        if self._sink_len() < self.sink_size:
            take = min(self.sink_size - self._sink_len(), num_steps)
            k_head, v_head = keys[..., :take, :], values[..., :take, :]
            if self._k_sink is None:
                self._k_sink, self._v_sink = k_head, v_head
            else:
                self._k_sink = mx.concatenate([self._k_sink, k_head], axis=2)
                self._v_sink = mx.concatenate([self._v_sink, v_head], axis=2)
            keys, values = keys[..., take:, :], values[..., take:, :]

        # 2. Append the rest to the window.
        if keys.shape[2] > 0:
            if self._k_win is None:
                self._k_win, self._v_win = keys, values
            else:
                self._k_win = mx.concatenate([self._k_win, keys], axis=2)
                self._v_win = mx.concatenate([self._v_win, values], axis=2)

        # 3. Spill window overflow into the compressed middle.
        overflow = self._win_len() - self.window_size
        if overflow > 0:
            assert self._k_win is not None and self._v_win is not None
            self._compress_into_middle(
                self._k_win[..., :overflow, :], self._v_win[..., :overflow, :]
            )
            self._k_win = self._k_win[..., overflow:, :]
            self._v_win = self._v_win[..., overflow:, :]

        self.offset += num_steps

        # 4. Assemble the full view (transient middle dequant).
        parts_k, parts_v = [], []
        if self._k_sink is not None:
            parts_k.append(self._k_sink)
            parts_v.append(self._v_sink)
        if self._mid_len > 0:
            mk, mv = self._decompress_middle(dtype)
            parts_k.append(mk)
            parts_v.append(mv)
        if self._k_win is not None and self._win_len() > 0:
            parts_k.append(self._k_win)
            parts_v.append(self._v_win)
        if len(parts_k) == 1:
            return parts_k[0], parts_v[0]
        return mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)

    @property
    def state(self):
        if self.offset == 0:
            return []
        arrays = []
        for a in (self._k_sink, self._v_sink, self._k_win, self._v_win):
            if a is not None and a.shape[2] > 0:
                arrays.append(a)
        if self._mid_len > 0:
            assert (
                self._k_mid is not None
                and self._k_mid_norms is not None
                and self._v_mid is not None
                and self._v_mid_norms is not None
            )
            arrays.extend(
                [
                    self._k_mid[..., : self._mid_len, :],
                    self._k_mid_norms[..., : self._mid_len, :],
                    self._v_mid[..., : self._mid_len, :],
                    self._v_mid_norms[..., : self._mid_len, :],
                ]
            )
        return arrays

    @state.setter
    def state(self, v):
        raise NotImplementedError(
            "ShardKVCache does not support state restoration. "
            "Disable disk cache offload when using shard quantization."
        )

    def is_trimmable(self):
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        remaining = n
        # Window first (newest tokens)
        take = min(remaining, self._win_len())
        if take > 0:
            assert self._k_win is not None and self._v_win is not None
            keep = self._win_len() - take
            self._k_win = self._k_win[..., :keep, :] if keep else None
            self._v_win = self._v_win[..., :keep, :] if keep else None
            remaining -= take
        # Then middle (per-token storage: trim = shrink the valid length)
        take = min(remaining, self._mid_len)
        if take > 0:
            self._mid_len -= take
            remaining -= take
        # Then sink
        take = min(remaining, self._sink_len())
        if take > 0:
            assert self._k_sink is not None and self._v_sink is not None
            keep = self._sink_len() - take
            self._k_sink = self._k_sink[..., :keep, :] if keep else None
            self._v_sink = self._v_sink[..., :keep, :] if keep else None
            remaining -= take
        self.offset -= n
        if self.offset == 0:
            self._k_mid = None
            self._k_mid_norms = None
            self._v_mid = None
            self._v_mid_norms = None
            self._mid_len = 0
        return n

    def make_mask(self, *args, **kwargs):
        kwargs["offset"] = self.offset
        return create_attention_mask(*args, **kwargs)

    def empty(self):
        return self.offset == 0


def make_shard_cache(model: Any, calibration_dir: Path, bits: int) -> list:
    """Create a cache list with ShardKVCache for attention layers.

    Mirrors ``make_spectral_cache``: non-attention caches (ArraysCache for
    SSM layers) are preserved; layers without calibration data fall back to
    the default cache with a warning.
    """
    from olmlx.engine.shardquant_calibrate import load_shard_calibration

    calibration, meta = load_shard_calibration(Path(calibration_dir))

    num_layers = len(model.layers)
    if hasattr(model, "make_cache"):
        default_caches = model.make_cache()
        if not isinstance(default_caches, list):
            default_caches = [None] * num_layers
    else:
        default_caches = [None] * num_layers

    caches = []
    quantized = 0
    for i, default in enumerate(default_caches):
        if default is not None and not isinstance(default, KVCache):
            caches.append(default)
            continue
        entry = calibration.get(i)
        if entry is None:
            logger.warning("No shard calibration for layer %d, using default", i)
            caches.append(default if default is not None else KVCache())
            continue
        rope_spec = None
        if entry.get("rope_freqs") is not None:
            rope_spec = RopeSpec(
                dims=entry["rope_dims"],
                freqs=entry["rope_freqs"],
                traditional=entry["rope_traditional"],
            )
        caches.append(
            ShardKVCache(
                rope_spec=rope_spec,
                k_basis=entry["k_basis"],
                k_rank=entry["k_rank"],
                k_codebook=entry["k_codebook"],
                k_bits=bits,
                v_rotation=entry["v_rotation"],
                v_codebooks=entry["v_codebooks"],
            )
        )
        quantized += 1

    logger.info(
        "Created Shard KV cache: %d/%d layers quantized, %d-bit",
        quantized,
        len(caches),
        bits,
    )
    return caches
```

Note: `_grow` writes the old contents into a fresh zeros buffer instead of the concat-pattern the other caches use — functionally identical; if `fresh[..., :n, :] = buf[...]` proves awkward, use the spectral concat-and-truncate pattern instead. Keep whichever passes the tests.

`ArraysCache(size=2)` constructor signature in the test: check `mlx_lm.models.cache.ArraysCache.__init__` and adjust the test if it differs (it takes `size` in current mlx-lm; if not, construct however the installed version requires — the test only needs *any* non-KVCache cache instance).

- [ ] **Step 4.4: Run to verify pass**

Run: `uv run pytest tests/test_shardquant_cache.py -x -q`
Expected: PASS (the `make_shard_cache` tests will still fail on the missing `shardquant_calibrate` import — that's expected; implement a minimal `load_shard_calibration` stub raising `NotImplementedError` is NOT acceptable; instead reorder: if these two tests block, mark them with `pytest.importorskip("olmlx.engine.shardquant_calibrate")` temporarily and remove the skip in Task 6. Simpler: the monkeypatch target requires the module to exist — create `olmlx/engine/shardquant_calibrate.py` now with just the module docstring and an empty `load_shard_calibration` raising on call (the tests monkeypatch it, so it's never called).)

```python
"""Shard calibration: per-head no-RoPE PCA for K, product-VQ for V (#377)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_shard_calibration(calibration_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Load shard calibration artifacts. Implemented in Task 6."""
    raise NotImplementedError
```

- [ ] **Step 4.5: Commit**

```bash
git add olmlx/engine/shardquant_cache.py olmlx/engine/shardquant_calibrate.py tests/test_shardquant_cache.py
git commit -m "feat(shardquant): ShardKVCache with sink/window + compressed middle (#377)"
```

---

### Task 5: Extract shared KV-collection helper from spectral calibration

**Files:**
- Modify: `olmlx/engine/spectralquant_calibrate.py:323-494`
- Test: existing `tests/test_spectralquant_calibrate_coverage.py` + `tests/test_spectralquant.py` must stay green (this is a pure refactor)

Split `calibrate_model` into:
1. `_load_calibration_model(model_path) -> tuple` — the model-loading block (lines 369-391): returns `(model, tokenizer, inner, head_dim, n_kv_heads, num_layers)`.
2. `collect_kv_vectors(model, tokenizer, inner, *, num_layers, n_kv_heads, head_dim, texts, max_tokens_per_head, progress_callback=None, progress_lo=0.1, progress_hi=0.5) -> dict` — the collection loop (lines 405-494 including the empty-collection guard): returns `kv_collectors` (the per-layer/per-head dict of chunk lists).

**Critical:** keep all currently-lazy imports (`olmlx.engine.flash.prepare.*`, `mlx_lm.models.cache.make_prompt_cache`, `olmlx.engine.turboquant_cache._detect_head_dim`) *inside* the new functions, at call time — the existing coverage tests patch those module attributes and rely on the late import.

- [ ] **Step 5.1: Refactor**

In `olmlx/engine/spectralquant_calibrate.py`, add above `calibrate_model`:

```python
def _load_calibration_model(model_path: str):
    """Load a model for calibration; returns model + architecture facts.

    Shared by the spectral and shard calibration pipelines.
    """
    from olmlx.engine.flash.prepare import (
        _get_backbone,
        load_model_with_strict_fallback,
    )
    from olmlx.engine.turboquant_cache import _detect_head_dim

    try:
        model, tokenizer = load_model_with_strict_fallback(model_path, lazy=False)
    except ValueError:
        import mlx_vlm

        model, processor = mlx_vlm.load(model_path, lazy=False)
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

    inner = _get_backbone(model)
    num_layers = len(inner.layers)
    cfg_holder = _resolve_config_holder(inner, model)
    cfg_ns = _config_namespace(cfg_holder)
    head_dim = _detect_head_dim(cfg_holder, layers_hint=inner)
    logger.debug("calibration: resolved head_dim=%d", head_dim)

    n_kv_heads = getattr(cfg_ns, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(cfg_ns, "num_attention_heads", 1)

    return model, tokenizer, inner, head_dim, n_kv_heads, num_layers


def collect_kv_vectors(
    model: Any,
    tokenizer: Any,
    inner: Any,
    *,
    num_layers: int,
    n_kv_heads: int,
    head_dim: int,
    texts: list[str],
    max_tokens_per_head: int,
    progress_callback: Any | None = None,
    progress_lo: float = 0.1,
    progress_hi: float = 0.5,
) -> dict[int, dict[int, dict[str, list[mx.array]]]]:
    """Collect post-RoPE K/V vectors per (layer, head) from forward passes.

    Returns kv_collectors[layer][head]["key"|"value"] = list of (seq, head_dim)
    chunks. Each chunk starts at position 0 of its sample (relevant for
    de-roping in the shard pipeline). Raises if nothing was collected.
    """
    from olmlx.engine.flash.prepare import _encode_tokens
    from mlx_lm.models.cache import make_prompt_cache

    kv_collectors: dict[int, dict[int, dict[str, list[mx.array]]]] = {}
    for i in range(num_layers):
        kv_collectors[i] = {}
        for h in range(n_kv_heads):
            kv_collectors[i][h] = {"key": [], "value": []}

    tokens_collected = {
        (i, h, k): 0
        for i in range(num_layers)
        for h in range(n_kv_heads)
        for k in ("key", "value")
    }

    cache_model = _resolve_cache_owner(inner, model)
    first_exc: Exception | None = None
    for sample_idx, text in enumerate(texts):
        tokens = _encode_tokens(tokenizer, text)
        if len(tokens) < 2:
            logger.debug(
                "Skipping sample %d: too short (%d tokens)", sample_idx, len(tokens)
            )
            continue
        if len(tokens) > 512:
            tokens = tokens[:512]
        input_ids = mx.array([tokens])

        prompt_cache = make_prompt_cache(cache_model)
        try:
            model(input_ids, cache=prompt_cache)
        except Exception as exc:
            if first_exc is None:
                first_exc = exc
            logger.debug("Skipping sample %d: %s", sample_idx, exc)
            del prompt_cache
            continue
        mx.eval([c.state for c in prompt_cache if hasattr(c, "state")])

        for layer_idx in range(min(num_layers, len(prompt_cache))):
            cache_entry = prompt_cache[layer_idx]
            if not _is_attention_cache(cache_entry, head_dim):
                continue
            state = cache_entry.state
            cached_keys = state[0]
            cached_values = state[1]

            for h in range(min(n_kv_heads, cached_keys.shape[1])):
                if tokens_collected[(layer_idx, h, "key")] < max_tokens_per_head:
                    k_h = cached_keys[0, h, :, :]
                    remaining = (
                        max_tokens_per_head - tokens_collected[(layer_idx, h, "key")]
                    )
                    k_h = k_h[:remaining]
                    kv_collectors[layer_idx][h]["key"].append(k_h)
                    tokens_collected[(layer_idx, h, "key")] += k_h.shape[0]

                if tokens_collected[(layer_idx, h, "value")] < max_tokens_per_head:
                    v_h = cached_values[0, h, :, :]
                    remaining = (
                        max_tokens_per_head
                        - tokens_collected[(layer_idx, h, "value")]
                    )
                    v_h = v_h[:remaining]
                    kv_collectors[layer_idx][h]["value"].append(v_h)
                    tokens_collected[(layer_idx, h, "value")] += v_h.shape[0]

        del prompt_cache
        if progress_callback:
            frac = progress_lo + (sample_idx + 1) / len(texts) * (
                progress_hi - progress_lo
            )
            progress_callback(f"Collected {sample_idx + 1}/{len(texts)} samples", frac)

    if sum(tokens_collected.values()) == 0:
        raise _build_empty_collection_error(first_exc)

    return kv_collectors
```

Then rewrite the corresponding body of `calibrate_model` to call them (everything from "Load model" through the empty-collection guard is replaced; the eigenspectral analysis, save, and metadata sections are unchanged):

```python
    if progress_callback:
        progress_callback("Loading model", 0.0)

    model, tokenizer, inner, head_dim, n_kv_heads, num_layers = (
        _load_calibration_model(model_path)
    )

    if progress_callback:
        progress_callback("Generating calibration data", 0.05)

    from olmlx.engine.flash.prepare import (
        _get_c4_calibration_data,
        _get_calibration_data,
    )

    if calibration_dataset == "synthetic":
        texts = _get_calibration_data(num_samples)
    else:
        texts = _get_c4_calibration_data(num_samples)

    if progress_callback:
        progress_callback("Collecting KV vectors", 0.1)

    kv_collectors = collect_kv_vectors(
        model,
        tokenizer,
        inner,
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        texts=texts,
        max_tokens_per_head=max_tokens_per_head,
        progress_callback=progress_callback,
    )
```

(Drop the now-unused locals/imports from `calibrate_model`; keep `import gc`/`time` where still used.)

- [ ] **Step 5.2: Run the spectral suites to verify the refactor is invisible**

Run: `uv run pytest tests/test_spectralquant.py tests/test_spectralquant_calibrate_coverage.py -q`
Expected: ALL PASS, zero test edits.

- [ ] **Step 5.3: Commit**

```bash
git add olmlx/engine/spectralquant_calibrate.py
git commit -m "refactor(spectralquant): extract shared KV-collection helpers for shard calibration (#377)"
```

---

### Task 6: Shard calibration pipeline + save/load

**Files:**
- Modify: `olmlx/engine/shardquant_calibrate.py` (replace the Task-4 stub)
- Test: `tests/test_shardquant_calibrate.py`

Per layer: detect `rope_spec` from `inner.layers[i].self_attn` (fall back to `model.layers[i]` attribute walk; `None` → identity). De-rope each collected K chunk at offset 0 (each chunk starts at sample position 0 — guaranteed by the collector). Per-head: normalize → covariance → eigendecompose (reuse spectral helpers). Layer rank = `max(compute_d_eff(eigenvalues_h))` over heads. K codebook: pool all heads' truncated rotated coefficients → `fit_codebook(bits)`. V: pool heads, normalize, rotate by `make_v_rotation(head_dim, seed=layer)`, fit PQ codebooks with `g = 8 // bits`. Skip a layer (no artifacts) when `head_dim % g != 0` or no data was collected.

- [ ] **Step 6.1: Write the failing tests**

Create `tests/test_shardquant_calibrate.py`:

```python
"""Tests for the shard calibration pipeline (#377 Tier 1).

Mirrors the mocked-model pattern of test_spectralquant_calibrate_coverage.py.
"""

import json
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
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
        "k_codebook": mx.array(np.sort(rng.randn(1 << bits)).astype(np.float32)),
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
            keys = mx.array(
                rng.randn(1, n_kv, seq, self._head_dim).astype(np.float32)
            )
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
        patch("olmlx.engine.flash.prepare._get_c4_calibration_data", return_value=texts),
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
        assert e["k_codebook"].shape == (16,)  # 2^4
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
        assert calibration[0]["k_codebook"].shape == (4,)  # 2^2

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
        cos = np.sum(np.array(ko) * np.array(k), -1) / (
            np.linalg.norm(np.array(ko), axis=-1)
            * np.linalg.norm(np.array(k), axis=-1)
            + 1e-9
        )
        assert cos.mean() > 0.7  # random data has no low-rank structure; loose bound
```

- [ ] **Step 6.2: Run to verify failure**

Run: `uv run pytest tests/test_shardquant_calibrate.py -x -q`
Expected: FAIL (`save_shard_calibration` does not exist)

- [ ] **Step 6.3: Implement** — replace `olmlx/engine/shardquant_calibrate.py`:

```python
"""Shard calibration: per-head no-RoPE PCA for K, product-VQ for V (#377).

Reuses the KV-collection scaffolding from spectralquant_calibrate (the
collected chunks each start at sample position 0, which is what makes the
chunk-wise de-rope below exact) and spectral's covariance / eigh / d_eff /
Lloyd-Max helpers. The shard-specific analysis is:

- K: de-rope each chunk -> per-head covariance + eigendecomposition ->
  per-layer rank = max participation ratio over heads -> one pooled
  Lloyd-Max codebook over the kept rotated coefficients.
- V: pooled across heads -> normalize -> fixed orthonormal rotation
  (Hadamard / seeded QR) -> per-position k-means PQ codebooks (256 entries,
  group size 8 // bits).

RoPE frequencies are persisted so runtime re-rope is guaranteed identical
to calibration-time de-rope.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from olmlx.engine.shardquant import (
    detect_rope_spec,
    fit_vq_codebooks,
    make_v_rotation,
    rope_transform,
)
from olmlx.engine.spectralquant import fit_codebook
from olmlx.engine.spectralquant_calibrate import (
    _load_calibration_model,
    collect_kv_vectors,
    compute_covariance,
    compute_d_eff,
    eigendecompose,
)

logger = logging.getLogger(__name__)

_SHARD_DEFAULT_MAX_TOKENS_PER_HEAD = 8192
_SHARD_DEFAULT_NUM_SAMPLES = 256

#: layer_idx -> calibration entry
ShardCalibration = dict[int, dict[str, Any]]


def save_shard_calibration(
    calibration: ShardCalibration, meta: dict[str, Any], output_dir: Path
) -> None:
    """Write shard_config.json + calibration.safetensors."""
    import safetensors.numpy

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {"meta": meta, "layers": {}}
    tensors: dict[str, np.ndarray] = {}

    for layer, entry in calibration.items():
        prefix = f"layer_{layer}"
        layer_cfg: dict[str, Any] = {
            "k_rank": entry["k_rank"],
            "rope_dims": entry["rope_dims"],
            "rope_traditional": entry["rope_traditional"],
        }
        config["layers"][str(layer)] = layer_cfg
        tensors[f"{prefix}_k_basis"] = np.array(entry["k_basis"])
        tensors[f"{prefix}_k_codebook"] = np.array(entry["k_codebook"])
        tensors[f"{prefix}_v_rotation"] = np.array(entry["v_rotation"])
        tensors[f"{prefix}_v_codebooks"] = np.array(entry["v_codebooks"])
        if entry["rope_freqs"] is not None:
            tensors[f"{prefix}_rope_freqs"] = np.array(entry["rope_freqs"])

    (output_dir / "shard_config.json").write_text(json.dumps(config, indent=2))
    safetensors.numpy.save_file(tensors, str(output_dir / "calibration.safetensors"))


def load_shard_calibration(
    calibration_dir: Path,
) -> tuple[ShardCalibration, dict[str, Any]]:
    """Load shard calibration; returns (per-layer entries, meta)."""
    import safetensors.numpy

    calibration_dir = Path(calibration_dir)
    config = json.loads((calibration_dir / "shard_config.json").read_text())
    tensors = safetensors.numpy.load_file(
        str(calibration_dir / "calibration.safetensors")
    )

    result: ShardCalibration = {}
    for layer_str, layer_cfg in config["layers"].items():
        layer = int(layer_str)
        prefix = f"layer_{layer}"
        freqs_np = tensors.get(f"{prefix}_rope_freqs")
        result[layer] = {
            "k_basis": mx.array(tensors[f"{prefix}_k_basis"]),
            "k_rank": layer_cfg["k_rank"],
            "k_codebook": mx.array(tensors[f"{prefix}_k_codebook"]),
            "v_rotation": mx.array(tensors[f"{prefix}_v_rotation"]),
            "v_codebooks": mx.array(tensors[f"{prefix}_v_codebooks"]),
            "rope_freqs": mx.array(freqs_np) if freqs_np is not None else None,
            "rope_dims": layer_cfg["rope_dims"],
            "rope_traditional": layer_cfg["rope_traditional"],
        }
    return result, config.get("meta", {})


def _find_attention_module(model: Any, inner: Any, layer_idx: int) -> Any | None:
    """Best-effort lookup of layer layer_idx's attention module."""
    for src in (inner, model):
        layers = getattr(src, "layers", None)
        if layers is None or layer_idx >= len(layers):
            continue
        attn = getattr(layers[layer_idx], "self_attn", None)
        if attn is not None:
            return attn
    return None


def _derope_chunks(chunks: list[mx.array], rope_spec) -> mx.array:
    """De-rope per-sample chunks (each starts at position 0) and concat."""
    if rope_spec is None:
        return mx.concatenate(chunks, axis=0)
    out = []
    for c in chunks:
        # (seq, D) -> rope_transform works on (..., S, D)
        out.append(rope_transform(c, rope_spec, 0, inverse=True))
    return mx.concatenate(out, axis=0)


def calibrate_model_shard(
    model_path: str,
    output_dir: Path | None = None,
    num_samples: int = _SHARD_DEFAULT_NUM_SAMPLES,
    calibration_dataset: str | None = None,
    bits: int = 4,
    max_tokens_per_head: int = _SHARD_DEFAULT_MAX_TOKENS_PER_HEAD,
    progress_callback: Any | None = None,
) -> Path:
    """Run shard calibration on a model. Returns the shard directory."""
    import gc
    import time

    if bits not in (2, 4, 8):
        raise ValueError(f"shard calibration supports bits in {{2,4,8}}, got {bits}")
    group_size = 8 // bits

    if output_dir is None:
        output_dir = Path(model_path) / "shard"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Loading model", 0.0)

    model, tokenizer, inner, head_dim, n_kv_heads, num_layers = (
        _load_calibration_model(model_path)
    )

    if progress_callback:
        progress_callback("Generating calibration data", 0.05)

    from olmlx.engine.flash.prepare import (
        _get_c4_calibration_data,
        _get_calibration_data,
    )

    if calibration_dataset == "synthetic":
        texts = _get_calibration_data(num_samples)
    else:
        texts = _get_c4_calibration_data(num_samples)

    if progress_callback:
        progress_callback("Collecting KV vectors", 0.1)

    kv_collectors = collect_kv_vectors(
        model,
        tokenizer,
        inner,
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        texts=texts,
        max_tokens_per_head=max_tokens_per_head,
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback("Analyzing", 0.5)

    if head_dim % group_size != 0:
        raise ValueError(
            f"head_dim={head_dim} not divisible by VQ group size {group_size} "
            f"(bits={bits}); shard quantization unsupported for this model."
        )

    calibration: ShardCalibration = {}
    for layer_idx in range(num_layers):
        head_chunks = kv_collectors[layer_idx]
        if not any(head_chunks[h]["key"] for h in range(n_kv_heads)):
            logger.debug("No KV data for layer %d, skipping", layer_idx)
            continue

        attn = _find_attention_module(model, inner, layer_idx)
        rope_spec = detect_rope_spec(attn) if attn is not None else None
        if rope_spec is None:
            logger.debug(
                "Layer %d: no recognizable RoPE; calibrating in roped basis",
                layer_idx,
            )

        # --- K: per-head no-RoPE PCA -----------------------------------
        bases = []
        ranks = []
        coeff_pool = []
        for h in range(n_kv_heads):
            chunks = head_chunks[h]["key"]
            if not chunks:
                break
            k_nope = _derope_chunks(chunks, rope_spec).astype(mx.float32)
            norms = mx.maximum(
                mx.linalg.norm(k_nope, axis=-1, keepdims=True), mx.array(1e-8)
            )
            k_unit = k_nope / norms
            eigenvalues, eigenvectors = eigendecompose(compute_covariance(k_unit))
            basis = eigenvectors.T  # rows = eigenvectors, descending variance
            bases.append(basis)
            ranks.append(compute_d_eff(eigenvalues))
            coeff_pool.append(k_unit @ basis.T)
        if len(bases) < n_kv_heads:
            logger.debug("Layer %d: incomplete per-head K data, skipping", layer_idx)
            continue
        k_rank = max(ranks)
        kept = mx.concatenate([c[:, :k_rank] for c in coeff_pool], axis=0)
        k_codebook = fit_codebook(kept.reshape(-1), bits=bits)

        # --- V: pooled rotation + PQ ------------------------------------
        v_chunks = []
        for h in range(n_kv_heads):
            v_chunks.extend(head_chunks[h]["value"])
        v_data = mx.concatenate(v_chunks, axis=0).astype(mx.float32)
        v_norms = mx.maximum(
            mx.linalg.norm(v_data, axis=-1, keepdims=True), mx.array(1e-8)
        )
        v_rotation = make_v_rotation(head_dim, seed=layer_idx)
        v_rotated = (v_data / v_norms) @ v_rotation.T
        v_codebooks = fit_vq_codebooks(
            np.array(v_rotated), group_size=group_size, seed=layer_idx
        )

        calibration[layer_idx] = {
            "k_basis": mx.stack(bases),  # (H, D, D)
            "k_rank": int(k_rank),
            "k_codebook": k_codebook,
            "v_rotation": v_rotation,
            "v_codebooks": v_codebooks,
            "rope_freqs": rope_spec.freqs if rope_spec is not None else None,
            "rope_dims": rope_spec.dims if rope_spec is not None else None,
            "rope_traditional": (
                rope_spec.traditional if rope_spec is not None else False
            ),
        }
        if progress_callback:
            frac = 0.5 + (layer_idx + 1) / num_layers * 0.4
            progress_callback(f"Calibrated layer {layer_idx + 1}/{num_layers}", frac)

    del kv_collectors
    gc.collect()
    mx.clear_cache()

    if progress_callback:
        progress_callback("Saving calibration", 0.9)

    meta = {
        "num_layers": num_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "bits": bits,
        "v_group_size": group_size,
        "num_samples": num_samples,
        "max_tokens_per_head": max_tokens_per_head,
        "calibration_dataset": calibration_dataset or "c4",
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_shard_calibration(calibration, meta, output_dir)

    if progress_callback:
        progress_callback("Done", 1.0)

    logger.info("Shard calibration complete: %s", output_dir)
    return output_dir
```

Note `mx.linalg.norm` usage: if the installed MLX version lacks it for this shape, use `mx.sqrt(mx.sum(x*x, axis=-1, keepdims=True))` (the codebase convention).

- [ ] **Step 6.4: Run to verify pass**

Run: `uv run pytest tests/test_shardquant_calibrate.py tests/test_shardquant_cache.py -q`
Expected: PASS (including the Task-4 `make_shard_cache` tests now that `load_shard_calibration` is real)

- [ ] **Step 6.5: Commit**

```bash
git add olmlx/engine/shardquant_calibrate.py tests/test_shardquant_calibrate.py
git commit -m "feat(shardquant): calibration pipeline with per-head no-RoPE PCA + PQ fitting (#377)"
```

---

### Task 7: Config validation — `shard:{2,4,8}`

**Files:**
- Modify: `olmlx/config.py:53-72` (`validate_kv_cache_quant_format`) and the comment at `olmlx/config.py:171-175`
- Test: `tests/test_shardquant.py`

- [ ] **Step 7.1: Write the failing tests** (append to `tests/test_shardquant.py`)

```python
class TestConfigValidation:
    def test_shard_4_accepted(self):
        from olmlx.config import validate_kv_cache_quant_format

        assert validate_kv_cache_quant_format("shard:4") == "shard:4"

    def test_shard_2_accepted(self):
        from olmlx.config import validate_kv_cache_quant_format

        assert validate_kv_cache_quant_format("shard:2") == "shard:2"

    def test_shard_8_accepted(self):
        from olmlx.config import validate_kv_cache_quant_format

        assert validate_kv_cache_quant_format("shard:8") == "shard:8"

    def test_shard_3_rejected(self):
        from olmlx.config import validate_kv_cache_quant_format

        with pytest.raises(ValueError):
            validate_kv_cache_quant_format("shard:3")

    def test_spectral_8_still_rejected(self):
        """8-bit is shard-only; spectral/turboquant stay {2,4}."""
        from olmlx.config import validate_kv_cache_quant_format

        with pytest.raises(ValueError):
            validate_kv_cache_quant_format("spectral:8")
        with pytest.raises(ValueError):
            validate_kv_cache_quant_format("turboquant:8")
```

- [ ] **Step 7.2: Run to verify failure**

Run: `uv run pytest tests/test_shardquant.py -q -k ConfigValidation`
Expected: FAIL on shard values

- [ ] **Step 7.3: Implement** — in `olmlx/config.py` replace the body of `validate_kv_cache_quant_format`:

```python
    if v is None:
        return v
    _VALID_BITS_BY_METHOD = {
        "turboquant": {"2", "4"},
        "spectral": {"2", "4"},
        "shard": {"2", "4", "8"},
    }
    parts = v.split(":", 1)
    if (
        len(parts) != 2
        or parts[0] not in _VALID_BITS_BY_METHOD
        or parts[1] not in _VALID_BITS_BY_METHOD[parts[0]]
    ):
        raise ValueError(
            f"Invalid kv_cache_quant={v!r}. "
            f"Expected '<method>:<bits>' where method:bits is one of "
            f"turboquant:{{2,4}}, spectral:{{2,4}}, shard:{{2,4,8}}."
        )
    return v
```

Also update the field comment at line ~171 to mention shard.

- [ ] **Step 7.4: Run to verify pass + no spectral regressions**

Run: `uv run pytest tests/test_shardquant.py tests/test_spectralquant.py -q`
Expected: PASS (the spectral config tests assert accept/reject behavior that is unchanged)

- [ ] **Step 7.5: Commit**

```bash
git add olmlx/config.py tests/test_shardquant.py
git commit -m "feat(config): accept OLMLX_KV_CACHE_QUANT=shard:{2,4,8} (#377)"
```

---

### Task 8: Engine + model-manager wiring

**Files:**
- Modify: `olmlx/engine/inference.py:1306-1343` (factory + dispatch)
- Modify: `olmlx/engine/model_manager.py` (exception, `_find_shard_dir`, `_auto_calibrate_shard`, `LoadedModel` field at ~line 529, `ensure_loaded` at ~line 1453-1477, `_is_serializable_cache` at 241-248)
- Test: `tests/test_shardquant_integration.py`

- [ ] **Step 8.1: Write the failing tests**

Create `tests/test_shardquant_integration.py`:

```python
"""Wiring tests: dispatch, calibration discovery, serialization guard (#377)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestPromptCacheDispatch:
    def test_shard_method_dispatches_to_make_shard_cache(self):
        from olmlx.engine import inference

        lm = MagicMock()
        lm.kv_cache_quant = "shard:4"
        lm.is_vlm = False
        lm.shard_calibration_dir = Path("/tmp/fake-shard")
        with patch(
            "olmlx.engine.shardquant_cache.make_shard_cache",
            return_value=["sentinel"],
        ) as mk:
            result = inference._make_prompt_cache_for_lm(lm)
        assert result == ["sentinel"]
        mk.assert_called_once_with(lm.model, Path("/tmp/fake-shard"), bits=4)

    def test_spectral_dispatch_unchanged(self):
        from olmlx.engine import inference

        lm = MagicMock()
        lm.kv_cache_quant = "spectral:4"
        lm.is_vlm = False
        with patch(
            "olmlx.engine.spectralquant_cache.make_spectral_cache",
            return_value=["spectral-sentinel"],
        ):
            assert inference._make_prompt_cache_for_lm(lm) == ["spectral-sentinel"]


class TestSerializableGuard:
    def test_shard_cache_blocks_disk_save(self):
        from olmlx.engine.model_manager import _is_serializable_cache
        from olmlx.engine.shardquant_cache import ShardKVCache

        cache = ShardKVCache.__new__(ShardKVCache)  # no init needed for isinstance
        assert _is_serializable_cache([cache]) is False

    def test_plain_cache_still_serializable(self):
        from mlx_lm.models.cache import KVCache

        from olmlx.engine.model_manager import _is_serializable_cache

        assert _is_serializable_cache([KVCache()]) is True


def _manager_with_store(tmp_path):
    from olmlx.engine.model_manager import ModelManager

    mgr = ModelManager.__new__(ModelManager)
    store = MagicMock()
    store.local_path.return_value = tmp_path
    mgr.store = store
    return mgr


class TestFindShardDir:
    def test_none_when_not_shard(self, tmp_path):
        mgr = _manager_with_store(tmp_path)
        assert mgr._find_shard_dir("m", None) is None
        assert mgr._find_shard_dir("m", "spectral:4") is None

    def test_missing_calibration_raises_with_command(self, tmp_path):
        from olmlx.engine.model_manager import ShardCalibrationMissingError

        mgr = _manager_with_store(tmp_path)
        with pytest.raises(ShardCalibrationMissingError) as exc:
            mgr._find_shard_dir("org/model", "shard:4")
        assert "olmlx shard prepare org/model" in str(exc.value)

    def test_found_when_calibration_exists(self, tmp_path):
        shard = tmp_path / "shard"
        shard.mkdir()
        (shard / "shard_config.json").write_text(json.dumps({"meta": {"bits": 4}}))
        mgr = _manager_with_store(tmp_path)
        assert mgr._find_shard_dir("org/model", "shard:4") == shard

    def test_bits_mismatch_raises(self, tmp_path):
        from olmlx.engine.model_manager import ShardCalibrationMissingError

        shard = tmp_path / "shard"
        shard.mkdir()
        (shard / "shard_config.json").write_text(json.dumps({"meta": {"bits": 2}}))
        mgr = _manager_with_store(tmp_path)
        with pytest.raises(ShardCalibrationMissingError) as exc:
            mgr._find_shard_dir("org/model", "shard:4")
        assert "--bits 4" in str(exc.value)

    def test_shard_error_is_spectral_subclass(self):
        """Starlette resolves handlers by walking type(exc).__mro__, so the
        existing 400 handler for SpectralCalibrationMissingError catches the
        shard error too — no app.py change needed."""
        from olmlx.engine.model_manager import (
            ShardCalibrationMissingError,
            SpectralCalibrationMissingError,
        )

        assert issubclass(ShardCalibrationMissingError, SpectralCalibrationMissingError)
```

- [ ] **Step 8.2: Run to verify failure**

Run: `uv run pytest tests/test_shardquant_integration.py -x -q`
Expected: FAIL

- [ ] **Step 8.3: Implement — inference.py**

After `_make_spectral_prompt_cache` (line ~1321) add:

```python
def _make_shard_prompt_cache(
    model: Any, bits: int, calibration_dir: Any, is_vlm: bool = False
) -> list:
    """Create a Shard-compressed prompt cache for the model (#377)."""
    from olmlx.engine.shardquant_cache import make_shard_cache

    cache_model = _get_model_for_cache(model, is_vlm)
    return make_shard_cache(cache_model, calibration_dir, bits=bits)
```

In `_make_prompt_cache_for_lm` (line ~1331), add the branch before the turboquant fallthrough:

```python
    if lm.kv_cache_quant is not None:
        method, bits = _parse_kv_cache_quant(lm.kv_cache_quant)
        if method == "spectral":
            return _make_spectral_prompt_cache(
                lm.model, bits, lm.spectral_calibration_dir, is_vlm=lm.is_vlm
            )
        if method == "shard":
            return _make_shard_prompt_cache(
                lm.model, bits, lm.shard_calibration_dir, is_vlm=lm.is_vlm
            )
        return _make_turboquant_prompt_cache(lm.model, bits, is_vlm=lm.is_vlm)
```

- [ ] **Step 8.4: Implement — model_manager.py**

1. Below `SpectralCalibrationMissingError` (line ~464):

```python
class ShardCalibrationMissingError(SpectralCalibrationMissingError):
    """Shard quant configured but calibration artifacts missing/mismatched.

    Subclasses SpectralCalibrationMissingError so the existing app.py
    exception handler (HTTP 400) catches it via Starlette's MRO walk.
    """
```

2. In `LoadedModel` (line ~529), next to `spectral_calibration_dir`:

```python
    shard_calibration_dir: Any = None  # Path | None, typed as Any to avoid import
```

3. In `ensure_loaded` next to the `_spectral_dir` lookup (line ~1453):

```python
                    _shard_dir = await asyncio.to_thread(
                        self._find_shard_dir, hf_path, kv_cache_quant
                    )
```

and pass `shard_calibration_dir=_shard_dir,` in the `LoadedModel(...)` construction.

4. `_is_serializable_cache` (line 241):

```python
def _is_serializable_cache(cache: list) -> bool:
    """Check if a cache list can be serialized with mlx-lm's save_prompt_cache."""
    from olmlx.engine.shardquant_cache import ShardKVCache
    from olmlx.engine.spectralquant_cache import SpectralQuantKVCache
    from olmlx.engine.turboquant_cache import TurboQuantKVCache

    return not any(
        isinstance(c, (TurboQuantKVCache, SpectralQuantKVCache, ShardKVCache))
        for c in cache
    )
```

5. Add `_find_shard_dir` + `_auto_calibrate_shard` after `_auto_calibrate_spectral` (line ~2428), mirroring the spectral pair:

```python
    def _find_shard_dir(
        self, hf_path: str, kv_cache_quant: str | None
    ) -> Path | None:
        """Return the shard calibration directory if shard quant is configured."""
        if kv_cache_quant is None or not kv_cache_quant.startswith("shard:"):
            return None
        if self.store is None:
            return None

        try:
            configured_bits = int(kv_cache_quant.split(":", 1)[1])
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid shard quant config {kv_cache_quant!r}; "
                f"expected shard:2, shard:4 or shard:8"
            )
        if configured_bits not in (2, 4, 8):
            raise ValueError(
                f"Invalid shard bit width {kv_cache_quant!r}; expected 2, 4 or 8"
            )

        shard_path = self.store.local_path(hf_path) / "shard"
        if shard_path.exists() and (shard_path / "shard_config.json").exists():
            try:
                config = json.loads((shard_path / "shard_config.json").read_text())
            except (json.JSONDecodeError, OSError) as exc:
                recalibrate_cmd = f"olmlx shard prepare {hf_path}"
                if configured_bits != 4:
                    recalibrate_cmd += f" --bits {configured_bits}"
                raise ShardCalibrationMissingError(
                    f"Shard quant configured ({kv_cache_quant}) but calibration "
                    f"file at {shard_path}/shard_config.json is unreadable "
                    f"({exc}). Re-run '{recalibrate_cmd}'."
                )
            cal_bits = config.get("meta", {}).get("bits")
            if cal_bits is not None and cal_bits != configured_bits:
                calibrate_cmd = (
                    f"olmlx shard prepare {hf_path} --bits {configured_bits}"
                )
                raise ShardCalibrationMissingError(
                    f"Shard quant configured ({kv_cache_quant}) but calibration "
                    f"data at {shard_path} was generated with --bits {cal_bits}. "
                    f"Run '{calibrate_cmd}' to re-calibrate at "
                    f"{configured_bits}-bit, or set "
                    f"OLMLX_KV_CACHE_QUANT=shard:{cal_bits} to use the "
                    f"existing calibration."
                )
            return shard_path

        if settings.kv_cache_auto_calibrate:
            return self._auto_calibrate_shard(hf_path, kv_cache_quant)

        calibrate_cmd = f"olmlx shard prepare {hf_path}"
        if configured_bits != 4:
            calibrate_cmd += f" --bits {configured_bits}"
        raise ShardCalibrationMissingError(
            f"Shard quant configured ({kv_cache_quant}) but no calibration data "
            f"found at {shard_path}. Run '{calibrate_cmd}' to calibrate. "
            f"Calibration collects KV vectors from text samples (C4 by "
            f"default), fits per-head no-RoPE PCA bases for keys and "
            f"product-VQ codebooks for values. "
            f"Or set OLMLX_KV_CACHE_AUTO_CALIBRATE=true to auto-calibrate."
        )

    def _auto_calibrate_shard(self, hf_path: str, kv_cache_quant: str) -> Path:
        """Run shard calibration automatically with default settings."""
        import logging

        from olmlx.engine.shardquant_calibrate import calibrate_model_shard

        method, bits_str = kv_cache_quant.split(":")
        assert method == "shard", (
            f"_auto_calibrate_shard called with non-shard quant: "
            f"{kv_cache_quant!r}"
        )
        bits = int(bits_str)
        local_dir = self.store.local_path(hf_path)
        logger = logging.getLogger(__name__)
        logger.info(
            "Auto-calibrating shard quant (%s-bit) for %s "
            "(this may take several minutes)...",
            bits,
            hf_path,
        )
        try:
            output_dir = calibrate_model_shard(
                model_path=str(local_dir),
                num_samples=64,
                calibration_dataset="c4",
                bits=bits,
                max_tokens_per_head=2048,
            )
        except Exception as exc:
            raise ShardCalibrationMissingError(
                f"Auto-calibration failed for {hf_path}: {exc}. "
                f"Run 'olmlx shard prepare {hf_path}' manually."
            ) from exc
        shard_path = Path(output_dir)
        if shard_path.exists() and (shard_path / "shard_config.json").exists():
            logger.info("Auto-calibration complete for %s", hf_path)
            return shard_path
        raise ShardCalibrationMissingError(
            f"Auto-calibration completed but shard data not found at "
            f"{shard_path}. Run 'olmlx shard prepare {hf_path}' manually."
        )
```

6. Update the `kv_cache_auto_calibrate` comment in `config.py` (~line 177) and the `Settings.kv_cache_quant` comment to mention shard, and relax the spectral-only check at `config.py:327-332` **only if it rejects shard** — read that block first; it warns when `kv_cache_auto_calibrate` is set without spectral. Extend its condition to also accept `shard:` prefixes:

```python
        if self.kv_cache_auto_calibrate and (
            self.kv_cache_quant is None
            or not self.kv_cache_quant.startswith(("spectral:", "shard:"))
        ):
```

(Adapt to the actual code at that line — the intent: auto-calibrate is meaningful for both spectral and shard now. Update the message string accordingly, and check `tests/test_spectralquant.py::test_auto_calibrate_*` for assertions on that message.)

- [ ] **Step 8.5: Run to verify pass**

Run: `uv run pytest tests/test_shardquant_integration.py tests/test_spectralquant.py tests/test_inference.py -q`
Expected: PASS

- [ ] **Step 8.6: Commit**

```bash
git add olmlx/engine/inference.py olmlx/engine/model_manager.py olmlx/config.py tests/test_shardquant_integration.py
git commit -m "feat(engine): wire shard KV quant into cache dispatch + model loading (#377)"
```

---

### Task 9: CLI — `olmlx shard prepare`

**Files:**
- Modify: `olmlx/cli.py` — command function near `cmd_spectral_prepare` (line ~2550), parser near the spectral parser (line ~3583), dispatch table entry (line ~3732)
- Test: `tests/test_shardquant_integration.py`

- [ ] **Step 9.1: Write the failing test** (append to `tests/test_shardquant_integration.py`)

```python
class TestCliShardPrepare:
    def test_shard_prepare_invokes_calibration(self, tmp_path):
        from olmlx import cli

        args = SimpleNamespace(
            model="org/model",
            samples=8,
            bits=4,
            calibration_dataset=None,
            max_tokens=1024,
        )
        store = MagicMock()
        store.registry.resolve.return_value = None
        store.ensure_downloaded.return_value = tmp_path
        with (
            patch.object(cli, "_create_store", return_value=store),
            patch.object(cli, "_configure_logging"),
            patch(
                "olmlx.engine.shardquant_calibrate.calibrate_model_shard",
                return_value=tmp_path / "shard",
            ) as cal,
        ):
            cli.cmd_shard_prepare(args)
        cal.assert_called_once_with(
            model_path=str(tmp_path),
            num_samples=8,
            calibration_dataset=None,
            bits=4,
            max_tokens_per_head=1024,
            progress_callback=cli._flash_progress,
        )

    def test_parser_accepts_shard_prepare(self):
        from olmlx.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["shard", "prepare", "org/model", "--bits", "8"])
        assert args.bits == 8
        assert args.model == "org/model"
```

Check the actual parser-builder function name in `olmlx/cli.py` (search `def build_parser` / `def main`); adjust `build_parser` in the test to the real name.

- [ ] **Step 9.2: Run to verify failure**

Run: `uv run pytest tests/test_shardquant_integration.py -q -k CliShard`
Expected: FAIL

- [ ] **Step 9.3: Implement** — in `olmlx/cli.py`:

After `cmd_spectral_prepare` (line ~2584):

```python
def cmd_shard_prepare(args):
    """Prepare a model for shard quant (per-head PCA-K + VQ-V calibration)."""
    _configure_logging()

    store = _create_store()
    _resolved = store.registry.resolve(args.model)
    hf_path = _resolved.hf_path if _resolved is not None else args.model
    local_dir = store.ensure_downloaded(hf_path)
    model_path = str(local_dir)

    print(f"Running shard calibration for {args.model}...")
    print(f"  Model path: {model_path}")
    print(f"  Bits: {args.bits}")
    dataset_label = args.calibration_dataset or "c4"
    print(f"  Calibration dataset: {dataset_label}")
    print(f"  Calibration samples: {args.samples}")
    print(f"  Max tokens per head: {args.max_tokens}")
    print()

    from olmlx.engine.shardquant_calibrate import calibrate_model_shard

    output_dir = calibrate_model_shard(
        model_path=model_path,
        num_samples=args.samples,
        calibration_dataset=args.calibration_dataset,
        bits=args.bits,
        max_tokens_per_head=args.max_tokens,
        progress_callback=_flash_progress,
    )

    print("\nShard calibration complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use shard quant:")
    print(f"  OLMLX_KV_CACHE_QUANT=shard:{args.bits} olmlx serve")
```

After the spectral parser block (line ~3614):

```python
    # Shard quant calibration (#377)
    shard = sub.add_parser("shard", help="Shard KV cache compression")
    shard_sub = shard.add_subparsers(dest="shard_command")

    shard_prepare_p = shard_sub.add_parser(
        "prepare", help="Run shard calibration for a model"
    )
    shard_prepare_p.add_argument("model", help="Model name or HF path")
    shard_prepare_p.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)",
    )
    shard_prepare_p.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 4, 8],
        help="Bits per dimension for K and V (default: 4)",
    )
    shard_prepare_p.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Calibration dataset: 'c4' (default) or 'synthetic'",
    )
    shard_prepare_p.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens to collect per head (default: 8192)",
    )
```

Dispatch table (line ~3732): add `("shard", "prepare"): "cmd_shard_prepare",`. Also check how the table handles a bare `olmlx shard` with no subcommand (mirror whatever `spectral` does).

Also update the `--kv-cache-quant` help string at `cli.py:3077` to mention `shard:4`.

- [ ] **Step 9.4: Run to verify pass**

Run: `uv run pytest tests/test_shardquant_integration.py -q`
Expected: PASS

- [ ] **Step 9.5: Commit**

```bash
git add olmlx/cli.py tests/test_shardquant_integration.py
git commit -m "feat(cli): olmlx shard prepare subcommand (#377)"
```

---

### Task 10: Live test (real_model)

**Files:**
- Create: `tests/live/test_shard_quant_real.py`

Look at an existing live test (e.g. `tests/live/test_vlm_cache_grammar.py` or `tests/live/test_rerank_real.py`) for the `real_model` marker/skip conventions and copy them exactly.

- [ ] **Step 10.1: Write the test**

```python
"""Live end-to-end shard-quant test: calibrate a tiny model, generate (#377).

Run: uv run pytest tests/live/test_shard_quant_real.py -m real_model -v
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.real_model

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@pytest.fixture(scope="module")
def calibrated_model_path(tmp_path_factory):
    from mlx_lm.utils import get_model_path

    from olmlx.engine.shardquant_calibrate import calibrate_model_shard

    model_path = Path(get_model_path(MODEL)[0])
    out = tmp_path_factory.mktemp("shard-calib")
    calibrate_model_shard(
        model_path=str(model_path),
        output_dir=out,
        num_samples=8,
        calibration_dataset="synthetic",
        bits=4,
        max_tokens_per_head=1024,
    )
    return model_path, out


def test_generation_stays_coherent_with_shard_cache(calibrated_model_path):
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    from olmlx.engine.shardquant_cache import make_shard_cache

    model_path, calib_dir = calibrated_model_path
    model, tokenizer = load(str(model_path))
    cache = make_shard_cache(model, calib_dir, bits=4)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from one to ten in words."}],
        add_generation_prompt=True,
    )
    text = ""
    for resp in stream_generate(
        model, tokenizer, prompt, max_tokens=64, prompt_cache=cache
    ):
        text += resp.text
    lowered = text.lower()
    # Loose coherence check: the model should produce several number words.
    hits = sum(w in lowered for w in ["one", "two", "three", "four", "five"])
    assert hits >= 3, f"incoherent output under shard quant: {text!r}"
```

Adjust `get_model_path` usage to whatever the other live tests use to resolve the local model dir (check first — some use the store API).

- [ ] **Step 10.2: Run it once locally**

Run: `uv run pytest tests/live/test_shard_quant_real.py -m real_model -v`
Expected: PASS (downloads the 0.5B model if absent; calibration with 8 synthetic samples takes ~1-2 min)

- [ ] **Step 10.3: Commit**

```bash
git add tests/live/test_shard_quant_real.py
git commit -m "test(shardquant): live end-to-end calibrate + generate coverage (#377)"
```

---

### Task 11: Docs + full verification + PR

**Files:**
- Modify: `CLAUDE.md` (KV quantization bullet list + project structure tree), `README.md` (if it documents spectral/turboquant — check first)
- Modify: `olmlx/engine/model_manager.py:251-263` comment (`_KV_QUANT_PREFIXES_BLOCKING_SNAPSHOT` — document why shard does NOT need an entry: no `mx.Dtype` attrs, all arrays in `state`)

- [ ] **Step 11.1: Update CLAUDE.md**

In the **KV quantization** section add:

```markdown
  - `shard:{2,4,8}` — Shard-style asymmetric K/V (#377 Tier 1, ref github.com/krish1905/shard): K = RoPE-undo → per-head PCA basis → rank truncation → Lloyd-Max scalar quant (RoPE re-applied on read; frequencies persisted in the calibration artifacts so undo/redo match exactly; unknown RoPE variants fall back to identity — still exactness-preserving, less rank collapse). V = orthonormal rotation (Hadamard when head_dim is a power of 2, seeded QR otherwise — matrix persisted) → per-position product VQ (256 centroids, group `8//bits` → exactly `bits` bits/dim). First 4 (sink) + last 64 (window) tokens stay FP16-exact in `ShardKVCache`; the middle dequantizes **transiently** on every fetch (no persistent side buffer — resident memory is quantized + sink/window, the Tier-1 memory contract; the cost is full-middle dequant + re-rope per layer per step). Requires `olmlx shard prepare <model>` (per-head eigendecomposition of de-roped keys + k-means PQ for values); missing/mismatched calibration → HTTP 400 with the exact command (`ShardCalibrationMissingError`, a `SpectralCalibrationMissingError` subclass so the existing handler applies); `OLMLX_KV_CACHE_AUTO_CALIBRATE=true` covers shard too. Trim is exact (window → middle → sink; both compressed regions are per-token), so prompt-cache prefix reuse works. No disk spill (`_is_serializable_cache`); deepcopy-safe via the default walk (no `mx.Dtype` attrs, all mutable arrays exposed through `state` for the #284 eager-eval pass).
```

Add `shardquant*.py` to the project-structure tree next to the other quant entries, and `shard` to the cli.py subcommand list in the tree.

- [ ] **Step 11.2: Check README**

Run: `grep -n "spectral\|turboquant\|KV_CACHE_QUANT" README.md` — if spectral/turboquant are documented there, add a matching shard entry (same level of detail); if not, skip.

- [ ] **Step 11.3: Lint + full test suite**

```bash
uv run ruff check olmlx tests && uv run ruff format --check olmlx tests
uv run pytest -q
```

Expected: ruff clean; test suite passes (note the known ~35% local full-suite SIGABRT flake — if it SIGABRTs, rerun targeted suites: `uv run pytest tests/test_shardquant*.py tests/test_spectralquant*.py tests/test_turboquant.py tests/test_inference.py tests/test_config*.py -q` and trust CI for the rest).

- [ ] **Step 11.4: Commit docs, push, open PR**

```bash
git add CLAUDE.md README.md olmlx/engine/model_manager.py
git commit -m "docs: document shard KV quant mode (#377)"
git push -u origin feat/shard-kv-quant
gh pr create --title "feat(engine): shard KV cache quantizer — Tier 1 (#377)" --body "..."
```

PR body should cover: what Tier 1 implements (per the issue + review comment), the locked design decisions (transient dequant + the latency trade-off, per-head K bases, persisted RoPE freqs/rotations, sink/window, exact trim), what's out of scope (Tier 2 int4 Q·K Metal kernel, disk spill, distributed, speculative composition — speculative decoders own their caches and never see shard), and the suggested follow-up A/B bench vs `spectral:4`. End with the Claude Code attribution per repo convention.

---

## Self-Review Notes

- **Spec coverage:** issue Tier 1 = PCA-K + VQ-V + sink/window + dequantize-on-read (`Tasks 1-4, 6`), calibration CLI (`Task 9`), config validation (`Task 7`), suggested files all created. Review-comment amendments: per-head K fitting/storage/runtime (Tasks 3/6), transient-dequant choice made explicit (Task 4 + docs), deepcopy/snapshot composition (Task 4 design + docs), trimmability decided and implemented exactly (Task 4), what-over-spectral deltas documented (Task 11). 8-bit support comes via spectral's 1-8-bit packers + `{2,4,8}` validation.
- **Known simplifications (intentional, document in PR):** single Lloyd-Max codebook for kept K coefficients (not per-coordinate); per-layer (not per-head) V codebooks; rank uniform per layer (max over heads); no `mx.compile` on the hot path yet; window buffers use concat (≤ window+batch tokens).
- **Type consistency check:** `make_shard_cache(model, calibration_dir, bits=...)` matches the Task-8 dispatch call; `load_shard_calibration -> (dict, meta)` matches Task-4 monkeypatches; `calibrate_model_shard(model_path=..., num_samples=..., calibration_dataset=..., bits=..., max_tokens_per_head=..., progress_callback=...)` matches the Task-9 CLI test; `RopeSpec(dims, freqs, traditional)` is used identically in Tasks 1/4/6.
- **Execution caveats for the implementer:** (1) the nn.RoPE parity test is the source of truth for freq conventions — if it fails, fix the implementation, not the test; (2) `mx.linalg.norm` may need the sqrt-sum spelling; (3) check `ArraysCache` constructor signature; (4) check the CLI parser-builder function name; (5) check `config.py:327` auto-calibrate guard wording and the spectral tests asserting it; (6) verify live-test model-path resolution against existing live tests.
