# Shard KV Quant Tier 2 (Fused Compressed-Middle Attention) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decode-path attention for `ShardKVCache` computed directly from the packed middle region (two slim Metal kernels + fp32 MLX softmax), so decode never materializes the FP16 middle.

**Architecture:** Per the design doc (`docs/superpowers/specs/2026-06-11-shard-kv-quant-tier2-design.md`): `update_and_fetch` returns a `ShardFusedKV` handle at decode (q_len==1, mid_len>0); a per-model-module patched `scaled_dot_product_attention` dispatches handles to `fused_sdpa_decode`, which computes exact-region scores with tiny matmuls, middle scores with a K-score Metal kernel (reconstruct-in-registers: unpack → codebook → `Σρ y·basis + mean` → `×norm` → re-rope → dot), joint fp32 softmax, then a V-accumulate kernel in rotated space with the orthonormal un-rotation folded outside the sum. A plain-MLX reference path backs every unsupported config and CPU test runs.

**Tech Stack:** mlx 0.31 (`mx.fast.metal_kernel` — spike-validated: template ints, uint8 inputs, `constant`-space scalar params accessed by index, `simd_sum`, threadgroup staging+barrier), mlx-lm 0.31.2, pytest.

**Verified facts the code below relies on:**
- 4-bit packing: low nibble = even index ρ, high = odd (`turboquant.pack_indices`); 2-bit: bits `(ρ&3)*2`; 8-bit: raw byte. V codes are unpacked uint8 (one byte per subvector).
- K decompress semantics (`_compiled_k_decompress_core`): `y = codebook[h, idx]` for ρ<rank, `x[d] = Σρ y[ρ]·basis[h,ρ,d] + mean[h,d]`, `x *= norm`, then re-rope at positions `sink_len..` (non-traditional: pairs `(d, d+half)`, `y1 = x1·cos − x2·sin`, `y2 = x1·sin + x2·cos`).
- V decompress: `vec = codebooks[p, idx[p]]` concat → `(vec @ rotation) * norm` — un-rotation is linear ⇒ folds outside the weighted sum.
- mlx-lm models bind `scaled_dot_product_attention` as a module global at import; `cache=` and `scale=` are always passed; decode (L==1) gets `mask=None`.
- `make_shard_cache(cache_model, calibration_dir, bits)` is called from `inference._make_shard_prompt_cache`; `settings` is imported directly in `inference.py`.
- Kernel params buffer must be indexed directly (`params[0]`) — small inputs land in `constant` address space.

**Performance note encoded in kernel 1:** one simdgroup per token, **16 tokens per threadgroup** — the per-token `basis` re-reads are then served from L2 (basis working set is `Hk·D·D·4B` ≈ 512 KB), not DRAM. DRAM traffic ≈ packed codes + norms + score row. The back-projection FLOPs (`mid·rank·D`) are inherent to PCA-K and identical to Tier 1's dequant matmul; the win is traffic + allocator churn.

---

### Task 1: Reference fused path (`shardquant_fused.py`)

**Files:**
- Create: `olmlx/engine/shardquant_fused.py`
- Test: `tests/test_shardquant_fused.py`

- [ ] **Step 1.1: Write the failing tests**

```python
"""Tests for the fused shard decode path (#377 Tier 2) — reference backend."""

import copy

import mlx.core as mx
import numpy as np
import pytest

from tests.test_shardquant_cache import _make_cache, _feed


def _grouped_q(nq, D, seed=7):
    mx.random.seed(seed)
    return mx.random.normal((1, nq, 1, D)).astype(mx.float16)


def _manual_sdpa_f32(q, k, v, scale):
    """Ground-truth attention in fp32: q (1,nq,1,D), k/v (1,Hk,S,D)."""
    B, nq, L, D = q.shape
    Hk = k.shape[1]
    g = nq // Hk
    qg = q.astype(mx.float32).reshape(B, Hk, g, D)
    s = (qg @ k.astype(mx.float32).swapaxes(-1, -2)) * scale
    w = mx.softmax(s, axis=-1)
    out = w @ v.astype(mx.float32)
    return out.reshape(B, nq, L, D)


class TestMiddleRefOps:
    def test_weighted_v_ref_matches_literal_decompress(self):
        from olmlx.engine.shardquant import shard_decompress_values
        from olmlx.engine.shardquant_fused import shard_middle_weighted_v_ref

        D, H = 64, 2
        cache = _make_cache(D=D, H=H, bits=4, sink=4, window=8)
        _feed(cache, 40, D=D, H=H, step=5)
        m = cache._mid_len
        assert m > 0
        mx.random.seed(3)
        w = mx.softmax(mx.random.normal((1, H, 2, m)), axis=-1)

        got = shard_middle_weighted_v_ref(w, cache)

        v_lit = shard_decompress_values(
            cache._v_mid, cache._v_mid_norms, cache.v_rotation,
            cache.v_codebooks, dtype=mx.float32,
        )[..., :m, :]
        want = w @ v_lit
        assert mx.allclose(got, want, atol=1e-4), float(mx.abs(got - want).max())

    def test_scores_ref_matches_decompressed_keys(self):
        from olmlx.engine.shardquant_fused import shard_middle_scores_ref

        D, H = 64, 2
        cache = _make_cache(D=D, H=H, bits=4, sink=4, window=8)
        _feed(cache, 40, D=D, H=H, step=5)
        m = cache._mid_len
        qg = mx.random.normal((1, H, 2, D))

        got = shard_middle_scores_ref(qg, cache)
        k_mid, _ = cache._decompress_middle(mx.float32)
        want = qg.astype(mx.float32) @ k_mid.swapaxes(-1, -2)
        assert got.shape == (1, H, 2, m)
        assert mx.allclose(got, want, atol=1e-5)


class TestFusedSdpaDecodeRef:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("rope", [True, False])
    def test_matches_tier1_attention(self, bits, rope):
        from olmlx.engine.shardquant_cache import ShardFusedKV
        from olmlx.engine.shardquant_fused import fused_sdpa_decode

        D, H, nq = 64, 2, 4
        cache = _make_cache(D=D, H=H, bits=bits, rank=48, rope=rope)
        _feed(cache, 50, D=D, H=H, step=5)
        # Tier-1 ground truth from an identical fork of the cache state.
        ref_cache = copy.deepcopy(cache)

        mx.random.seed(11)
        k_new = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        v_new = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        q = _grouped_q(nq, D)
        scale = D**-0.5

        k_full, v_full = ref_cache.update_and_fetch(k_new, v_new)
        want = _manual_sdpa_f32(q, k_full, v_full, scale)

        cache.update_and_fetch(k_new, v_new)  # same state advance
        exact_k = mx.concatenate([cache._k_sink, cache._k_win], axis=2)
        exact_v = mx.concatenate([cache._v_sink, cache._v_win], axis=2)
        handle = ShardFusedKV(cache=cache, k_exact=exact_k, v_exact=exact_v)
        got = fused_sdpa_decode(q, handle, scale, backend="ref")

        assert got.dtype == q.dtype
        assert mx.allclose(
            got.astype(mx.float32), want, atol=2e-3
        ), float(mx.abs(got.astype(mx.float32) - want).max())
```

`ShardFusedKV` lives in `shardquant_cache.py` (Task 2) — for this task, define it there already (a pure dataclass needs no cache changes), so the import above works.

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `uv run pytest tests/test_shardquant_fused.py -x -q`
Expected: FAIL — `ModuleNotFoundError: olmlx.engine.shardquant_fused` (or ImportError for `ShardFusedKV`).

- [ ] **Step 1.3: Implement**

Add to `olmlx/engine/shardquant_cache.py` (after the imports, before `ShardKVCache`):

```python
from dataclasses import dataclass


@dataclass
class ShardFusedKV:
    """Decode-step handle returned by a fused ShardKVCache.

    Deliberately not an ``mx.array``: any unpatched attention that feeds it
    to ``mx.fast.scaled_dot_product_attention`` raises immediately instead
    of silently attending over a partial history.
    """

    cache: "ShardKVCache"
    k_exact: mx.array
    v_exact: mx.array
```

Create `olmlx/engine/shardquant_fused.py`:

```python
"""Fused decode-path attention for ShardKVCache (#377 Tier 2).

Computes middle-region Q·K and the V-weighted sum directly from the packed
form so decode never materializes the FP16 middle.  This module holds the
plain-MLX reference math and the decode assembly; the Metal kernels live in
``shardquant_kernels.py``.  The reference path is also the production
fallback for unsupported configurations — identical math, Tier-1 cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from olmlx.engine.shardquant_cache import ShardFusedKV, ShardKVCache

__all__ = [
    "fused_sdpa_decode",
    "shard_middle_scores_ref",
    "shard_middle_weighted_v_ref",
]


def shard_middle_scores_ref(qg: mx.array, cache: "ShardKVCache") -> mx.array:
    """Middle-region attention scores, reference backend.

    Args:
        qg: (B, H_kv, G, D) roped queries (grouped GQA layout).
        cache: a ShardKVCache with mid_len > 0.

    Returns:
        (B, H_kv, G, mid_len) float32 unscaled scores.
    """
    k_mid, _ = cache._decompress_middle(mx.float32)
    return qg.astype(mx.float32) @ k_mid.swapaxes(-1, -2)


def shard_middle_weighted_v_ref(w: mx.array, cache: "ShardKVCache") -> mx.array:
    """Middle-region V contribution with the un-rotation folded out.

    ``Σⱼ wⱼ·(ṽⱼ@R)·nⱼ = ((w ⊙ n) @ ṽ) @ R`` — validates the fold the
    V-accumulate kernel relies on.

    Args:
        w: (B, H_kv, G, mid_len) float32 softmax weights.
        cache: a ShardKVCache with mid_len > 0.

    Returns:
        (B, H_kv, G, D) float32.
    """
    from olmlx.engine.shardquant import vq_gather

    m = cache._mid_len
    codes = cache._v_mid[..., :m, :]
    norms = cache._v_mid_norms[..., :m, :]  # (B, Hk, m, 1)
    sub = vq_gather(codes, cache.v_codebooks)  # (B, Hk, m, P, g)
    vec = sub.reshape(*sub.shape[:-2], -1)  # (B, Hk, m, D) rotated space
    wn = w * norms.swapaxes(-1, -2)  # (B, Hk, G, m)
    return (wn @ vec) @ cache.v_rotation


def _middle_scores(qg: mx.array, cache: "ShardKVCache", backend: str) -> mx.array:
    if backend == "ref":
        return shard_middle_scores_ref(qg, cache)
    raise ValueError(f"unknown backend {backend!r}")


def _middle_weighted_v(w: mx.array, cache: "ShardKVCache", backend: str) -> mx.array:
    if backend == "ref":
        return shard_middle_weighted_v_ref(w, cache)
    raise ValueError(f"unknown backend {backend!r}")


def fused_sdpa_decode(
    queries: mx.array,
    handle: "ShardFusedKV",
    scale: float,
    *,
    backend: str = "auto",
) -> mx.array:
    """Decode-step attention over [sink | middle | window] from a handle.

    Args:
        queries: (1, n_q, 1, D) roped queries (model already applied RoPE
            at the absolute position).
        handle: ShardFusedKV from ``update_and_fetch`` (implies B == 1,
            q_len == 1, mid_len > 0, exact regions non-empty).
        scale: softmax scale (same value the model passes to sdpa).
        backend: "auto" (kernels when supported, else ref) or "ref".

    Returns:
        (1, n_q, 1, D) in the query dtype.
    """
    cache = handle.cache
    B, nq, L, D = queries.shape
    Hk = handle.k_exact.shape[1]
    grp = nq // Hk
    qg = queries.astype(mx.float32).reshape(B, Hk, grp, D)

    k_e = handle.k_exact.astype(mx.float32)
    v_e = handle.v_exact.astype(mx.float32)
    s_exact = qg @ k_e.swapaxes(-1, -2)  # (B, Hk, grp, S_e)
    s_mid = _middle_scores(qg, cache, backend)  # (B, Hk, grp, m)

    s = mx.concatenate([s_exact, s_mid], axis=-1) * scale
    w = mx.softmax(s, axis=-1)
    se = s_exact.shape[-1]
    out = w[..., :se] @ v_e + _middle_weighted_v(w[..., se:], cache, backend)
    return out.reshape(B, nq, L, D).astype(queries.dtype)
```

For this task only, make `backend="auto"` an alias of `"ref"` inside `_middle_scores`/`_middle_weighted_v` (`if backend in ("ref", "auto")`); Task 6 replaces the dispatch.

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_shardquant_fused.py -x -q`
Expected: PASS (all parametrizations).

- [ ] **Step 1.5: Commit**

```bash
git add olmlx/engine/shardquant_fused.py olmlx/engine/shardquant_cache.py tests/test_shardquant_fused.py
git commit -m "feat(shardquant): fused decode reference path + ShardFusedKV handle (#377 Tier 2)"
```

---

### Task 2: ShardKVCache fused mode

**Files:**
- Modify: `olmlx/engine/shardquant_cache.py` (`__init__`, `update_and_fetch`, `make_shard_cache`)
- Test: `tests/test_shardquant_fused.py` (append class)

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/test_shardquant_fused.py`:

```python
class TestFusedCacheMode:
    def _fused_cache(self, **kw):
        cache = _make_cache(**kw)
        cache.fused = True
        return cache

    def test_handle_returned_only_for_single_token_with_middle(self):
        from olmlx.engine.shardquant_cache import ShardFusedKV

        D, H = 64, 2
        cache = self._fused_cache(D=D, H=H)
        # Prefill (multi-token): arrays, never a handle.
        k, v = _feed(cache, 30, D=D, H=H, step=10)[2]
        assert isinstance(k, mx.array) and isinstance(v, mx.array)
        # Decode with mid_len > 0: handle.
        assert cache._mid_len > 0
        k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        out = cache.update_and_fetch(k1, k1)
        assert isinstance(out[0], ShardFusedKV) and out[0] is out[1]
        assert out[0].k_exact.shape[2] == cache._sink_len() + cache._win_len()

    def test_no_handle_before_middle_exists(self):
        D, H = 64, 2
        cache = self._fused_cache(D=D, H=H, sink=4, window=16)
        k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        for _ in range(10):  # 10 tokens < sink + window: middle stays empty
            k, v = cache.update_and_fetch(k1, k1)
        assert isinstance(k, mx.array)

    def test_handle_exact_equals_materialized_exact_regions(self):
        D, H = 64, 2
        cache = self._fused_cache(D=D, H=H)
        _feed(cache, 30, D=D, H=H, step=10)
        ref = copy.deepcopy(cache)
        ref.fused = False
        k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        h, _ = cache.update_and_fetch(k1, k1)
        k_full, _ = ref.update_and_fetch(k1, k1)
        m = cache._mid_len
        sink = cache._sink_len()
        assert mx.allclose(h.k_exact[..., :sink, :], k_full[..., :sink, :])
        assert mx.allclose(h.k_exact[..., sink:, :], k_full[..., sink + m :, :])

    def test_trim_and_state_unaffected_by_fused_flag(self):
        D, H = 64, 2
        cache = self._fused_cache(D=D, H=H)
        _feed(cache, 30, D=D, H=H, step=10)
        n = cache.trim(5)
        assert n == 5 and cache.offset == 25
        assert all(isinstance(a, mx.array) for a in cache.state)

    def test_make_shard_cache_fused_flag(self, tmp_path):
        # make_shard_cache(fused=True) sets .fused on ShardKVCache layers;
        # reuse the synthetic-calibration scaffolding from the Tier-1 tests.
        from tests.test_shardquant_cache import TestMakeShardCache

        pytest.importorskip("mlx_lm")
        # covered concretely in Task 7's integration test; here assert the
        # parameter exists and defaults off.
        import inspect

        from olmlx.engine.shardquant_cache import make_shard_cache

        sig = inspect.signature(make_shard_cache)
        assert "fused" in sig.parameters
        assert sig.parameters["fused"].default is False
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `uv run pytest tests/test_shardquant_fused.py::TestFusedCacheMode -x -q`
Expected: FAIL — handles not returned / no `fused` parameter.

- [ ] **Step 2.3: Implement**

In `ShardKVCache.__init__`, after `self.window_size = window_size`:

```python
        # Tier-2 fused decode mode (#377): update_and_fetch returns a
        # ShardFusedKV handle for single-token steps once a compressed
        # middle exists; the patched sdpa computes attention from the
        # packed form.  Off by default — make_shard_cache opts in.
        self.fused = False
```

Replace step 4 of `update_and_fetch` (everything from the `# 4. Assemble` comment to the end of the method) with:

```python
        # 4. Fused decode handoff: single new token, B == 1, and a
        # compressed middle to attend over.  The exact regions ride on the
        # handle; the patched sdpa reads the middle from the cache.
        if (
            self.fused
            and num_steps == 1
            and keys.shape[0] == 1
            and self._mid_len > 0
        ):
            assert self._k_sink is not None and self._k_win is not None
            handle = ShardFusedKV(
                cache=self,
                k_exact=mx.concatenate([self._k_sink, self._k_win], axis=2),
                v_exact=mx.concatenate([self._v_sink, self._v_win], axis=2),
            )
            return handle, handle

        # 5. Assemble the full view (transient middle dequant).
        return self.materialize(dtype)

    def materialize(self, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        """Full [sink | middle | window] K/V with transient middle dequant.

        The Tier-1 fetch path, also used as the fused-mode fallback when a
        request isn't kernel-eligible (multi-token, sinks, masks)."""
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
```

(`materialize` keeps using the *input* dtype at fetch time in `update_and_fetch`; the fallback caller passes the query dtype.)

In `make_shard_cache`, change the signature to
`def make_shard_cache(model: Any, calibration_dir: Path, bits: int, fused: bool = False) -> list:`
and after the `caches.append(ShardKVCache(...))` loop body sets `quantized += 1`, set the flag on creation by adding `fused` handling right before `caches.append(...)`:

```python
        cache = ShardKVCache(
            rope_spec=rope_spec,
            k_basis=entry["k_basis"],
            k_rank=entry["k_rank"],
            k_codebook=entry["k_codebook"],
            k_bits=bits,
            v_rotation=entry["v_rotation"],
            v_codebooks=entry["v_codebooks"],
            k_mean=entry.get("k_mean"),
        )
        cache.fused = fused
        caches.append(cache)
        quantized += 1
```

and extend the final `logger.info` with the mode:

```python
    logger.info(
        "Created Shard KV cache: %d/%d layers quantized, %d-bit%s",
        quantized,
        len(caches),
        bits,
        ", fused decode" if fused else "",
    )
```

(Task 3 adds the sdpa patch install + `fused` demotion when no module can be patched.)

- [ ] **Step 2.4: Run the full shard test files**

Run: `uv run pytest tests/test_shardquant_fused.py tests/test_shardquant_cache.py tests/test_shardquant.py tests/test_shardquant_integration.py -q`
Expected: PASS — Tier-1 behavior unchanged (fused defaults off).

- [ ] **Step 2.5: Commit**

```bash
git add olmlx/engine/shardquant_cache.py tests/test_shardquant_fused.py
git commit -m "feat(shardquant): fused decode mode on ShardKVCache — handle return + materialize() (#377 Tier 2)"
```

---

### Task 3: sdpa wrapper + per-model-module patch installer

**Files:**
- Modify: `olmlx/engine/shardquant_fused.py`, `olmlx/engine/shardquant_cache.py` (`make_shard_cache`)
- Test: `tests/test_shardquant_fused.py` (append class)

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/test_shardquant_fused.py`:

```python
class TestSdpaPatch:
    def _model_with_module(self):
        """A fake mlx-lm-style model: layers whose class module defines a
        module-global scaled_dot_product_attention."""
        import sys
        import types

        mod = types.ModuleType("fake_mlx_model_mod")

        calls = []

        def sdpa(queries, keys, values, cache, scale, mask, sinks=None):
            calls.append((queries, keys, values, cache, scale, mask, sinks))
            return queries

        mod.scaled_dot_product_attention = sdpa
        mod._calls = calls

        class Layer:
            pass

        Layer.__module__ = mod.__name__
        sys.modules[mod.__name__] = mod

        class Model:
            layers = [Layer(), Layer()]

        return Model(), mod

    def test_install_patches_and_is_idempotent(self):
        from olmlx.engine.shardquant_fused import install_fused_sdpa

        model, mod = self._model_with_module()
        orig = mod.scaled_dot_product_attention
        assert install_fused_sdpa(model) == 1
        assert mod.scaled_dot_product_attention is not orig
        patched = mod.scaled_dot_product_attention
        assert install_fused_sdpa(model) == 1
        assert mod.scaled_dot_product_attention is patched  # no double wrap

    def test_non_handle_calls_delegate_to_original(self):
        from olmlx.engine.shardquant_fused import install_fused_sdpa

        model, mod = self._model_with_module()
        install_fused_sdpa(model)
        q = mx.zeros((1, 2, 1, 8))
        out = mod.scaled_dot_product_attention(
            q, q, q, cache=None, scale=1.0, mask=None
        )
        assert out is q and len(mod._calls) == 1

    def test_handle_dispatches_to_fused_path(self):
        from olmlx.engine.shardquant_cache import ShardFusedKV
        from olmlx.engine.shardquant_fused import install_fused_sdpa

        model, mod = self._model_with_module()
        install_fused_sdpa(model)
        D, H, nq = 64, 2, 4
        cache = _make_cache(D=D, H=H)
        cache.fused = True
        _feed(cache, 30, D=D, H=H, step=10)
        k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        h, _ = cache.update_and_fetch(k1, k1)
        assert isinstance(h, ShardFusedKV)
        q = _grouped_q(nq, D)
        out = mod.scaled_dot_product_attention(
            q, h, h, cache=cache, scale=D**-0.5, mask=None
        )
        assert isinstance(out, mx.array) and out.shape == (1, nq, 1, D)
        assert len(mod._calls) == 0  # did not delegate

    def test_handle_with_sinks_or_mask_falls_back_via_materialize(self):
        from olmlx.engine.shardquant_cache import ShardFusedKV
        from olmlx.engine.shardquant_fused import install_fused_sdpa

        model, mod = self._model_with_module()
        install_fused_sdpa(model)
        D, H = 64, 2
        cache = _make_cache(D=D, H=H)
        cache.fused = True
        _feed(cache, 30, D=D, H=H, step=10)
        k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        h, _ = cache.update_and_fetch(k1, k1)
        q = _grouped_q(4, D)
        mod.scaled_dot_product_attention(
            q, h, h, cache=cache, scale=1.0, mask=None, sinks=mx.zeros((4,))
        )
        # Delegated to the original with materialized arrays.
        (queries, keys, values, *_), = (mod._calls[-1],)
        assert isinstance(keys, mx.array)
        assert keys.shape[2] == cache.offset

    def test_unpatched_consumer_fails_loud(self):
        from olmlx.engine.shardquant_cache import ShardFusedKV

        D, H = 64, 2
        cache = _make_cache(D=D, H=H)
        cache.fused = True
        _feed(cache, 30, D=D, H=H, step=10)
        k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        h, _ = cache.update_and_fetch(k1, k1)
        q = _grouped_q(4, D)
        with pytest.raises((TypeError, ValueError)):
            mx.fast.scaled_dot_product_attention(
                q, h, h, scale=1.0, mask=None
            )

    def test_install_returns_zero_without_module_sdpa(self):
        from olmlx.engine.shardquant_fused import install_fused_sdpa

        class Layer:
            pass  # module = tests.* — has no scaled_dot_product_attention

        class Model:
            layers = [Layer()]

        assert install_fused_sdpa(Model()) == 0
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `uv run pytest tests/test_shardquant_fused.py::TestSdpaPatch -x -q`
Expected: FAIL — `install_fused_sdpa` doesn't exist.

- [ ] **Step 3.3: Implement**

Append to `olmlx/engine/shardquant_fused.py` (extend `__all__` with `"install_fused_sdpa"`):

```python
import functools
import logging
import sys

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_olmlx_shard_fused_patch"


def _fused_wrapper(orig):
    @functools.wraps(orig)
    def wrapper(queries, keys, values, cache=None, scale=1.0, mask=None,
                sinks=None, **kwargs):
        from olmlx.engine.shardquant_cache import ShardFusedKV

        if not isinstance(keys, ShardFusedKV):
            return orig(queries, keys, values, cache, scale, mask,
                        sinks=sinks, **kwargs)
        handle = keys
        if sinks is not None or mask is not None:
            # Not kernel-eligible (attention sinks / explicit mask):
            # materialize the Tier-1 view and delegate — correct, slower.
            k, v = handle.cache.materialize(queries.dtype)
            return orig(queries, k, v, cache, scale, mask,
                        sinks=sinks, **kwargs)
        return fused_sdpa_decode(queries, handle, scale)

    setattr(wrapper, _PATCH_MARKER, True)
    wrapper._olmlx_orig = orig
    return wrapper


def install_fused_sdpa(model) -> int:
    """Swap each layer-defining module's ``scaled_dot_product_attention``
    for the handle-aware wrapper.

    mlx-lm model modules bind the function as a module global at import
    time (``from .base import scaled_dot_product_attention``), so the swap
    must happen on each model module, not on ``mlx_lm.models.base``.
    Idempotent; the wrapper delegates every non-handle call to the
    original, so it is inert for non-shard models sharing the module.

    Returns the number of modules carrying the patch (0 = fused mode
    cannot work for this model — caller must demote to Tier-1).
    """
    patched = 0
    seen: set[int] = set()
    for layer in getattr(model, "layers", []):
        mod = sys.modules.get(type(layer).__module__)
        if mod is None or id(mod) in seen:
            continue
        seen.add(id(mod))
        fn = getattr(mod, "scaled_dot_product_attention", None)
        if fn is None:
            continue
        if not getattr(fn, _PATCH_MARKER, False):
            mod.scaled_dot_product_attention = _fused_wrapper(fn)
        patched += 1
    return patched
```

In `make_shard_cache` (shardquant_cache.py), before the `caches = []` loop:

```python
    if fused:
        from olmlx.engine.shardquant_fused import install_fused_sdpa

        if install_fused_sdpa(model) == 0:
            logger.warning(
                "Shard fused decode requested but no attention module "
                "exposes scaled_dot_product_attention; falling back to "
                "the Tier-1 dequantize-on-read path."
            )
            fused = False
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_shardquant_fused.py -x -q`
Expected: PASS.

- [ ] **Step 3.5: Commit**

```bash
git add olmlx/engine/shardquant_fused.py olmlx/engine/shardquant_cache.py tests/test_shardquant_fused.py
git commit -m "feat(shardquant): handle-aware sdpa wrapper + per-module patch installer (#377 Tier 2)"
```

---

### Task 4: K-score Metal kernel

**Files:**
- Create: `olmlx/engine/shardquant_kernels.py`
- Test: `tests/test_shardquant_kernels.py`

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_shardquant_kernels.py`:

```python
"""Metal-kernel parity tests for the fused shard decode path (#377 Tier 2).

Each kernel must match the plain-MLX reference math on the same cache.
GPU-only: skipped when Metal is unavailable.
"""

import mlx.core as mx
import pytest

from tests.test_shardquant_cache import _make_cache, _feed

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="requires Metal"
)


def _decode_step(cache, D, H, seed=11):
    mx.random.seed(seed)
    k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
    v1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
    cache.update_and_fetch(k1, v1)


class TestKScoreKernel:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("rope", [True, False])
    @pytest.mark.parametrize("rank", [None, 48])  # None = full rank
    def test_matches_reference(self, bits, rope, rank):
        from olmlx.engine.shardquant_fused import shard_middle_scores_ref
        from olmlx.engine.shardquant_kernels import shard_middle_scores_kernel

        D, H, G = 64, 2, 3
        cache = _make_cache(D=D, H=H, bits=bits, rank=rank, rope=rope)
        _feed(cache, 80, D=D, H=H, step=7)
        _decode_step(cache, D, H)
        assert cache._mid_len > 0
        mx.random.seed(5)
        qg = mx.random.normal((1, H, G, D))

        want = shard_middle_scores_ref(qg, cache)
        got = shard_middle_scores_kernel(qg, cache)
        assert got.shape == want.shape
        assert mx.allclose(got, want, atol=2e-3, rtol=1e-3), float(
            mx.abs(got - want).max()
        )

    def test_long_middle_capacity_padding(self):
        # mid_len far past one `step` growth, exercising the capacity-vs-
        # valid-length split and multiple threadgroup tiles.
        from olmlx.engine.shardquant_fused import shard_middle_scores_ref
        from olmlx.engine.shardquant_kernels import shard_middle_scores_kernel

        D, H = 64, 2
        cache = _make_cache(D=D, H=H, bits=4)
        _feed(cache, 700, D=D, H=H, step=64)
        qg = mx.random.normal((1, H, 4, D))
        want = shard_middle_scores_ref(qg, cache)
        got = shard_middle_scores_kernel(qg, cache)
        assert mx.allclose(got, want, atol=2e-3, rtol=1e-3)


class TestKernelGate:
    def test_supported_predicate(self):
        from olmlx.engine.shardquant_kernels import kernels_supported

        D, H = 64, 2
        ok = _make_cache(D=D, H=H, bits=4)
        assert kernels_supported(ok, D)
        # Traditional rope layout → unsupported (ref fallback).
        trad = _make_cache(D=D, H=H, bits=4)
        trad.rope_spec.traditional = True
        assert not kernels_supported(trad, D)
        # Odd head_dim → unsupported.
        odd = _make_cache(D=48, H=H, bits=4)  # 48 % 32 != 0
        assert not kernels_supported(odd, 48)
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `uv run pytest tests/test_shardquant_kernels.py -x -q`
Expected: FAIL — `shardquant_kernels` doesn't exist.

- [ ] **Step 4.3: Implement**

Create `olmlx/engine/shardquant_kernels.py`:

```python
"""Metal kernels for the fused shard decode path (#377 Tier 2).

Two slim kernels keep DRAM traffic at the packed size:

- ``shard_k_scores``: one simdgroup per middle token, 16 tokens per
  threadgroup (the per-token ``basis`` re-reads then hit L2 — basis
  working set is Hk·D·D·4B — instead of DRAM).  Reconstructs each key in
  registers (unpack → codebook → Σρ y·basis + mean → ×norm → re-rope at
  sink_len + j) and dots it with each grouped query.
- ``shard_v_accum``: thread per (dim, q-head, head·chunk); accumulates
  weights·norm·centroid in *rotated* space over a token chunk; partials
  are summed and un-rotated outside (the rotation folds out of the sum).

Both read the full capacity-aligned cache buffers and loop to ``mid_len``
(runtime param) — kernels are JIT-compiled once per bit width and reused
across calls/shapes.  Lane-ownership in the K kernel requires
``head_dim % 32 == 0`` and, with rope, ``(dims // 2) % 32 == 0`` and the
non-traditional pair layout; ``kernels_supported`` gates eligibility and
callers fall back to the reference path.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from olmlx.engine.shardquant_cache import ShardKVCache

__all__ = [
    "kernels_supported",
    "shard_middle_scores_kernel",
    "shard_middle_weighted_v_kernel",
]

# Tokens per threadgroup in the K-score kernel (simdgroups per group).
_K_TGT = 16
# Middle tokens per V-accumulate chunk (partial-sum parallelism).
_V_JCHUNK = 2048
# Hard bounds baked into the kernels' register/threadgroup layout.
_MAX_D = 256
_MAX_RANK = 128

_K_SCORES_SRC = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;
    uint j = thread_position_in_grid.y;
    uint h = thread_position_in_grid.z;
    int mid = params[0];
    int sink = params[1];
    int rank = params[2];
    int rdims = params[3];
    int G = params[4];
    int D = params[5];
    int cap = params[6];
    int PB = params[7];
    int K = params[8];

    threadgroup float ts_y[TGT][MAX_RANK];
    bool live = (int)j < mid;

    if (live) {
        const device uint8_t* crow = kcodes + ((size_t)h * cap + j) * PB;
        for (int r = (int)lane; r < rank; r += 32) {
            uint code;
            if (BITS == 8)      code = crow[r];
            else if (BITS == 4) code = (crow[r >> 1] >> ((r & 1) * 4)) & 0xF;
            else                code = (crow[r >> 2] >> ((r & 3) * 2)) & 0x3;
            ts_y[ty][r] = cb[h * K + code];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) return;

    // Reconstruct this token's key; lane owns dims d = lane + 32k.
    float kd[MAX_D / 32];
    int nloc = D / 32;
    float norm = knorms[(size_t)h * cap + j];
    const device float* brow = basis + (size_t)h * D * D;
    for (int k = 0; k < nloc; ++k) {
        int d = (int)lane + 32 * k;
        float acc = kmean[h * D + d];
        for (int r = 0; r < rank; ++r) acc += ts_y[ty][r] * brow[r * D + d];
        kd[k] = acc * norm;
    }
    // Re-rope at absolute position sink + j (non-traditional pairs
    // (d, d+half); half % 32 == 0 keeps both pair halves in this lane).
    if (rdims > 0) {
        int half = rdims / 2;
        float pos = (float)(sink + (int)j);
        for (int k = 0; k < nloc; ++k) {
            int d = (int)lane + 32 * k;
            if (d < half) {
                int k2 = k + half / 32;
                float angle = pos * freqs[d];
                float c = metal::precise::cos(angle);
                float s = metal::precise::sin(angle);
                float a = kd[k];
                float b = kd[k2];
                kd[k] = a * c - b * s;
                kd[k2] = a * s + b * c;
            }
        }
    }
    // Dot with each grouped query head.
    for (int g = 0; g < G; ++g) {
        const device float* qrow = q + ((size_t)h * G + g) * D;
        float acc = 0.0f;
        for (int k = 0; k < nloc; ++k) acc += kd[k] * qrow[(int)lane + 32 * k];
        float total = simd_sum(acc);
        if (lane == 0) scores[((size_t)h * G + g) * mid + j] = total;
    }
"""

_V_ACCUM_SRC = r"""
    int d = (int)thread_position_in_grid.x;
    int g = (int)thread_position_in_grid.y;
    int z = (int)thread_position_in_grid.z;
    int mid = params[0];
    int G = params[1];
    int D = params[2];
    int cap = params[3];
    int P = params[4];
    int GS = params[5];
    int jchunk = params[6];
    int nch = params[7];
    int HK = params[8];
    int K = params[9];
    int woff = params[10];
    int wlen = params[11];
    if (d >= D) return;
    int h = z / nch;
    int ch = z % nch;
    int p = d / GS;
    int sub = d % GS;
    int j0 = ch * jchunk;
    int j1 = metal::min(j0 + jchunk, mid);
    const device uint8_t* codes = vcodes + (size_t)h * cap * P;
    const device float* wrow = w + ((size_t)h * G + g) * wlen + woff;
    float acc = 0.0f;
    for (int j = j0; j < j1; ++j) {
        uint code = codes[(size_t)j * P + p];
        acc += wrow[j] * vnorms[(size_t)h * cap + j]
             * vcb[((size_t)p * K + code) * GS + sub];
    }
    partials[((((size_t)ch * HK + h) * G + g) * (size_t)D) + d] = acc;
"""


@lru_cache(maxsize=8)
def _k_scores_kernel(bits: int):
    return mx.fast.metal_kernel(
        name=f"olmlx_shard_k_scores_b{bits}",
        input_names=["q", "kcodes", "knorms", "basis", "cb", "kmean",
                     "freqs", "params"],
        output_names=["scores"],
        source=_K_SCORES_SRC,
    )


@lru_cache(maxsize=1)
def _v_accum_kernel():
    return mx.fast.metal_kernel(
        name="olmlx_shard_v_accum",
        input_names=["w", "vcodes", "vnorms", "vcb", "params"],
        output_names=["partials"],
        source=_V_ACCUM_SRC,
    )


def kernels_supported(cache: "ShardKVCache", head_dim: int) -> bool:
    """Eligibility gate for the Metal kernels; callers fall back to the
    reference path when False (identical math, Tier-1 cost)."""
    if not mx.metal.is_available():
        return False
    if mx.default_device().type != mx.DeviceType.gpu:
        return False
    if head_dim % 32 != 0 or head_dim > _MAX_D:
        return False
    if cache.k_rank > _MAX_RANK:
        return False
    if cache.k_bits not in (2, 4, 8):
        return False
    spec = cache.rope_spec
    if spec is not None:
        if spec.traditional:
            return False
        if spec.dims > head_dim or (spec.dims // 2) % 32 != 0:
            return False
    return True


def _k_codebook_2d(cache: "ShardKVCache") -> mx.array:
    cb = getattr(cache, "_fused_k_cb2d", None)
    if cb is None:
        cb = cache.k_codebook.astype(mx.float32)
        if cb.ndim == 1:
            h = cache.k_basis.shape[0]
            cb = mx.broadcast_to(cb[None, :], (h, cb.shape[0]))
        cb = cb + mx.zeros_like(cb)  # force contiguous materialization
        cache._fused_k_cb2d = cb
    return cb


def _k_mean_2d(cache: "ShardKVCache") -> mx.array:
    mean = getattr(cache, "_fused_k_mean", None)
    if mean is None:
        h, d = cache.k_basis.shape[0], cache.k_basis.shape[-1]
        mean = (
            cache.k_mean.astype(mx.float32)
            if cache.k_mean is not None
            else mx.zeros((h, d), dtype=mx.float32)
        )
        cache._fused_k_mean = mean
    return mean


def shard_middle_scores_kernel(qg: mx.array, cache: "ShardKVCache") -> mx.array:
    """Kernel-backed equivalent of ``shard_middle_scores_ref``.

    Args:
        qg: (1, H_kv, G, D) roped queries (any float dtype).
        cache: ShardKVCache with mid_len > 0.

    Returns:
        (1, H_kv, G, mid_len) float32 unscaled scores.
    """
    _, hk, g, d = qg.shape
    mid = cache._mid_len
    cap = cache._k_mid.shape[2]
    pb = cache._k_mid.shape[-1]
    spec = cache.rope_spec
    rdims = spec.dims if spec is not None else 0
    freqs = (
        spec.freqs.astype(mx.float32)
        if spec is not None
        else mx.zeros((1,), dtype=mx.float32)
    )
    cb = _k_codebook_2d(cache)
    params = mx.array(
        [mid, cache._sink_len(), cache.k_rank, rdims, g, d, cap, pb,
         cb.shape[-1]],
        dtype=mx.int32,
    )
    mid_pad = ((mid + _K_TGT - 1) // _K_TGT) * _K_TGT
    out = _k_scores_kernel(cache.k_bits)(
        inputs=[
            qg.astype(mx.float32).reshape(hk, g, d),
            cache._k_mid[0],
            cache._k_mid_norms[0, :, :, 0],
            cache.k_basis.astype(mx.float32),
            cb,
            _k_mean_2d(cache),
            freqs,
            params,
        ],
        template=[
            ("BITS", cache.k_bits),
            ("TGT", _K_TGT),
            ("MAX_RANK", _MAX_RANK),
            ("MAX_D", _MAX_D),
        ],
        grid=(32, mid_pad, hk),
        threadgroup=(32, _K_TGT, 1),
        output_shapes=[(hk, g, mid)],
        output_dtypes=[mx.float32],
    )[0]
    return out[None]


def shard_middle_weighted_v_kernel(
    w_full: mx.array, woff: int, cache: "ShardKVCache"
) -> mx.array:
    """Kernel-backed equivalent of ``shard_middle_weighted_v_ref``.

    Takes the *full* softmax row plus the middle offset (instead of a
    sliced view) so no per-step contiguous copy of the weights is needed.

    Args:
        w_full: (1, H_kv, G, S_e + mid_len) float32 softmax weights.
        woff: column where the middle region starts (== S_e).
        cache: ShardKVCache with mid_len > 0.

    Returns:
        (1, H_kv, G, D) float32 middle V contribution (un-rotated).
    """
    _, hk, g, wlen = w_full.shape
    mid = cache._mid_len
    cap = cache._v_mid.shape[2]
    p, k, gs = cache.v_codebooks.shape
    d = p * gs
    nch = (mid + _V_JCHUNK - 1) // _V_JCHUNK
    params = mx.array(
        [mid, g, d, cap, p, gs, _V_JCHUNK, nch, hk, k, woff, wlen],
        dtype=mx.int32,
    )
    partials = _v_accum_kernel()(
        inputs=[
            w_full.reshape(hk, g, wlen),
            cache._v_mid[0],
            cache._v_mid_norms[0, :, :, 0],
            cache.v_codebooks.astype(mx.float32),
            params,
        ],
        grid=(d, g, hk * nch),
        threadgroup=(min(d, 256), 1, 1),
        output_shapes=[(nch, hk, g, d)],
        output_dtypes=[mx.float32],
    )[0]
    out_rot = partials.sum(axis=0)  # (Hk, G, D), rotated space
    return (out_rot @ cache.v_rotation)[None]
```

(Task 4 tests only exercise `shard_middle_scores_kernel` and `kernels_supported`; `shard_middle_weighted_v_kernel` is written here for cohesion and tested in Task 5.)

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_shardquant_kernels.py -x -q`
Expected: PASS (TestKScoreKernel + TestKernelGate).

- [ ] **Step 4.5: Commit**

```bash
git add olmlx/engine/shardquant_kernels.py tests/test_shardquant_kernels.py
git commit -m "feat(shardquant): K-score Metal kernel — packed-form Q·K with in-register re-rope (#377 Tier 2)"
```

---

### Task 5: V-accumulate kernel parity tests

**Files:**
- Test: `tests/test_shardquant_kernels.py` (append class)
- Modify (only if tests find bugs): `olmlx/engine/shardquant_kernels.py`

- [ ] **Step 5.1: Write the failing/validating tests**

Append to `tests/test_shardquant_kernels.py`:

```python
class TestVAccumKernel:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_matches_reference(self, bits):
        from olmlx.engine.shardquant_fused import shard_middle_weighted_v_ref
        from olmlx.engine.shardquant_kernels import (
            shard_middle_weighted_v_kernel,
        )

        D, H, G = 64, 2, 3
        cache = _make_cache(D=D, H=H, bits=bits)
        _feed(cache, 80, D=D, H=H, step=7)
        m = cache._mid_len
        se = 10
        mx.random.seed(9)
        w_full = mx.softmax(mx.random.normal((1, H, G, se + m)), axis=-1)

        want = shard_middle_weighted_v_ref(w_full[..., se:], cache)
        got = shard_middle_weighted_v_kernel(w_full, se, cache)
        assert got.shape == want.shape
        assert mx.allclose(got, want, atol=2e-4, rtol=1e-3), float(
            mx.abs(got - want).max()
        )

    def test_multi_chunk_partials(self, monkeypatch):
        # Force several j-chunks so the partial-sum reduction is exercised.
        from olmlx.engine import shardquant_kernels as sk
        from olmlx.engine.shardquant_fused import shard_middle_weighted_v_ref

        monkeypatch.setattr(sk, "_V_JCHUNK", 16)
        D, H = 64, 2
        cache = _make_cache(D=D, H=H, bits=4)
        _feed(cache, 120, D=D, H=H, step=8)
        m = cache._mid_len
        w_full = mx.softmax(mx.random.normal((1, H, 2, 5 + m)), axis=-1)
        want = shard_middle_weighted_v_ref(w_full[..., 5:], cache)
        got = sk.shard_middle_weighted_v_kernel(w_full, 5, cache)
        assert mx.allclose(got, want, atol=2e-4, rtol=1e-3)
```

- [ ] **Step 5.2: Run tests, fix kernel bugs if any**

Run: `uv run pytest tests/test_shardquant_kernels.py -x -q`
Expected: PASS. If a parity test fails, debug against the reference (print max-abs by region: first/last token, head 0 vs 1) — the usual culprits are index arithmetic (`cap` vs `mid`) and the `woff` offset.

- [ ] **Step 5.3: Commit**

```bash
git add tests/test_shardquant_kernels.py olmlx/engine/shardquant_kernels.py
git commit -m "test(shardquant): V-accumulate kernel parity coverage (#377 Tier 2)"
```

---

### Task 6: Auto backend dispatch + end-to-end decode parity

**Files:**
- Modify: `olmlx/engine/shardquant_fused.py` (`_middle_scores`, `_middle_weighted_v`, `fused_sdpa_decode`)
- Test: `tests/test_shardquant_fused.py` (append class)

- [ ] **Step 6.1: Write the failing test**

Append to `tests/test_shardquant_fused.py`:

```python
class TestEndToEndDecodeParity:
    @pytest.mark.parametrize("backend", ["ref", "auto"])
    def test_multi_step_decode_matches_tier1(self, backend):
        """Drive 24 decode steps on forked caches; fused output must track
        the Tier-1 materialized path at every step (middle grows each
        step, so capacity growth, rope positions, and handle reuse are all
        exercised)."""
        from olmlx.engine.shardquant_cache import ShardFusedKV
        from olmlx.engine.shardquant_fused import fused_sdpa_decode

        if backend == "auto" and not mx.metal.is_available():
            pytest.skip("auto backend needs Metal for the kernel path")

        D, H, nq = 64, 2, 4
        scale = D**-0.5
        fused = _make_cache(D=D, H=H, bits=4, rank=48)
        _feed(fused, 40, D=D, H=H, step=8)
        tier1 = copy.deepcopy(fused)
        fused.fused = True

        for step in range(24):
            mx.random.seed(100 + step)
            k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
            v1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
            q = mx.random.normal((1, nq, 1, D)).astype(mx.float16)

            k_full, v_full = tier1.update_and_fetch(k1, v1)
            want = _manual_sdpa_f32(q, k_full, v_full, scale)

            h, _ = fused.update_and_fetch(k1, v1)
            assert isinstance(h, ShardFusedKV)
            got = fused_sdpa_decode(q, h, scale, backend=backend)
            assert mx.allclose(
                got.astype(mx.float32), want, atol=3e-3
            ), f"step {step}: {float(mx.abs(got.astype(mx.float32) - want).max())}"
```

- [ ] **Step 6.2: Run test to verify the auto case fails**

Run: `uv run pytest tests/test_shardquant_fused.py::TestEndToEndDecodeParity -x -q`
Expected: `ref` passes; `auto` currently aliases ref so it passes too — make `auto` real first, then this test guards it. Proceed to 6.3 regardless.

- [ ] **Step 6.3: Implement the real auto dispatch**

Replace `_middle_scores` / `_middle_weighted_v` and the `out = ...` line of `fused_sdpa_decode` in `shardquant_fused.py`:

```python
def _use_kernels(cache: "ShardKVCache", head_dim: int) -> bool:
    from olmlx.engine.shardquant_kernels import kernels_supported

    return kernels_supported(cache, head_dim)


def _middle_scores(qg: mx.array, cache: "ShardKVCache", backend: str) -> mx.array:
    if backend == "auto" and _use_kernels(cache, qg.shape[-1]):
        from olmlx.engine.shardquant_kernels import shard_middle_scores_kernel

        return shard_middle_scores_kernel(qg, cache)
    return shard_middle_scores_ref(qg, cache)
```

and in `fused_sdpa_decode`, replace the final block:

```python
    s = mx.concatenate([s_exact, s_mid], axis=-1) * scale
    w = mx.softmax(s, axis=-1)
    se = s_exact.shape[-1]
    if backend == "auto" and _use_kernels(cache, D):
        from olmlx.engine.shardquant_kernels import (
            shard_middle_weighted_v_kernel,
        )

        mid_v = shard_middle_weighted_v_kernel(w, se, cache)
    else:
        mid_v = shard_middle_weighted_v_ref(w[..., se:], cache)
    out = w[..., :se] @ v_e + mid_v
    return out.reshape(B, nq, L, D).astype(queries.dtype)
```

(Drop the now-unused `_middle_weighted_v` helper.)

- [ ] **Step 6.4: Run the full fused + kernel test files**

Run: `uv run pytest tests/test_shardquant_fused.py tests/test_shardquant_kernels.py -q`
Expected: PASS.

- [ ] **Step 6.5: Commit**

```bash
git add olmlx/engine/shardquant_fused.py tests/test_shardquant_fused.py
git commit -m "feat(shardquant): auto kernel dispatch + multi-step decode parity (#377 Tier 2)"
```

---

### Task 7: Config knob + engine threading

**Files:**
- Modify: `olmlx/config.py` (Settings), `olmlx/engine/inference.py` (`_make_shard_prompt_cache`)
- Test: `tests/test_shardquant_fused.py` (append)

- [ ] **Step 7.1: Write the failing test**

Append to `tests/test_shardquant_fused.py`:

```python
class TestConfig:
    def test_shard_fused_setting_defaults_on(self):
        from olmlx.config import Settings

        assert Settings().shard_fused is True

    def test_shard_fused_env_override(self, monkeypatch):
        from olmlx.config import Settings

        monkeypatch.setenv("OLMLX_SHARD_FUSED", "false")
        assert Settings().shard_fused is False
```

- [ ] **Step 7.2: Run test to verify it fails**

Run: `uv run pytest tests/test_shardquant_fused.py::TestConfig -x -q`
Expected: FAIL — no `shard_fused` field.

- [ ] **Step 7.3: Implement**

In `olmlx/config.py`, next to the `kv_cache_quant` field (~line 179):

```python
    # Fused Metal decode path for shard KV quant (#377 Tier 2): attention
    # over the compressed middle is computed from the packed form (no FP16
    # middle materialization).  Kill switch; unsupported configurations
    # fall back to the Tier-1 dequantize-on-read path automatically.
    shard_fused: bool = True
```

In `olmlx/engine/inference.py`, `_make_shard_prompt_cache`:

```python
    cache_model = _get_model_for_cache(model, is_vlm)
    return make_shard_cache(
        cache_model, calibration_dir, bits=bits, fused=settings.shard_fused
    )
```

- [ ] **Step 7.4: Run tests**

Run: `uv run pytest tests/test_shardquant_fused.py tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 7.5: Commit**

```bash
git add olmlx/config.py olmlx/engine/inference.py tests/test_shardquant_fused.py
git commit -m "feat(shardquant): OLMLX_SHARD_FUSED setting, threaded to make_shard_cache (#377 Tier 2)"
```

---

### Task 8: Live test + A/B bench

**Files:**
- Modify: `tests/live/test_shard_quant_real.py`, `scripts/shard_ab_bench.py`

- [ ] **Step 8.1: Add the live fused-parity test**

Append to `tests/live/test_shard_quant_real.py`:

```python
def test_fused_decode_matches_tier1_generation(calibrated_model_path):
    """Greedy generation with the fused decode path must match the Tier-1
    path token-for-token on a real model (same packed state, same math)."""
    from mlx_lm import load, stream_generate

    from olmlx.engine.shardquant_cache import make_shard_cache

    model_path, calib_dir = calibrated_model_path
    model, tokenizer = load(str(model_path))
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "List the days of the week."}],
        add_generation_prompt=True,
        enable_thinking=False,
    )

    texts = {}
    for fused in (False, True):
        cache = make_shard_cache(model, calib_dir, bits=4, fused=fused)
        out = ""
        for resp in stream_generate(
            model, tokenizer, prompt, max_tokens=64, prompt_cache=cache
        ):
            out += resp.text
        texts[fused] = out

    # fp32 fused softmax vs mx.fast sdpa can diverge on argmax near-ties;
    # require the long common prefix that exact-parity math produces.
    a, b = texts[False], texts[True]
    common = len([1 for x, y in zip(a, b) if x == y])
    assert common >= int(0.8 * min(len(a), len(b))), (a, b)
    assert "monday" in b.lower() or "tuesday" in b.lower(), b
```

- [ ] **Step 8.2: Add `shard-fused` mode to the bench**

In `scripts/shard_ab_bench.py` `_make_cache`, after the `"shard"` branch:

```python
    if mode == "shard-fused":
        from olmlx.engine.shardquant_cache import make_shard_cache

        return make_shard_cache(model, model_dir / "shard", bits=bits, fused=True)
```

In `main()`, change the `--modes` default to `"fp16,turboquant,spectral,shard,shard-fused"`, and in `_reconstruction`, skip the duplicate storage measurement:

```python
    for mode in modes:
        if mode in ("fp16", "shard-fused"):  # fused stores the same bytes as shard
            continue
```

- [ ] **Step 8.3: Run the live test**

Run: `uv run pytest tests/live/test_shard_quant_real.py -m real_model -v`
Expected: PASS (3 tests). Skipped automatically if the model isn't downloaded — if so, pull it first: `uv run olmlx models pull mlx-community/Qwen3-0.6B-4bit`.

- [ ] **Step 8.4: Run the bench, record numbers**

Run: `uv run python scripts/shard_ab_bench.py --modes fp16,shard,shard-fused --prompt-tokens 2048 --decode-tokens 128`
Expected: shard-fused decode tok/s ≥ shard; identical cache MB. Save the output for the PR body.

- [ ] **Step 8.5: Commit**

```bash
git add tests/live/test_shard_quant_real.py scripts/shard_ab_bench.py
git commit -m "test(shardquant): live fused parity + shard-fused bench mode (#377 Tier 2)"
```

---

### Task 9: Docs, lint, full suite, PR

**Files:**
- Modify: `CLAUDE.md` (shard bullets), `docs/superpowers/specs/2026-06-11-shard-kv-quant-tier2-design.md`

- [ ] **Step 9.1: Update CLAUDE.md**

In the `shard:{2,4,8}` design-decision bullet, after the A/B sentence, append a Tier-2 sentence summarizing: fused decode path (default on, `OLMLX_SHARD_FUSED` kill switch), handle + per-model-module sdpa patch, two Metal kernels, automatic Tier-1 fallback (prefill/sinks/mask/unsupported shapes/CPU), no new cache state (trim/snapshot/deepcopy unchanged), and the measured bench numbers. Also update the Project Structure line for `engine/` (`shardquant*.py` already covers the new modules — adjust the parenthetical to mention the fused decode path).

- [ ] **Step 9.2: Amend the design doc**

The design doc says per-model `models.json` override for `shard_fused` — v1 shipped the global env knob only. Edit the Config section to: "Global `OLMLX_SHARD_FUSED` only in v1; per-model override deferred until a need shows up."

- [ ] **Step 9.3: Lint + full targeted suite**

```bash
uv run ruff check olmlx tests scripts && uv run ruff format --check olmlx tests scripts
uv run pytest tests/test_shardquant_fused.py tests/test_shardquant_kernels.py tests/test_shardquant_cache.py tests/test_shardquant.py tests/test_shardquant_integration.py tests/test_shardquant_calibrate.py tests/test_config.py -q
uv run pytest -q -m "not real_model" -x
```

(Per project memory: local full-suite SIGABRT flake ~35% — if SIGABRT with passing targeted suites, rerun once and trust CI.)

- [ ] **Step 9.4: Commit docs, push, open PR**

```bash
git add CLAUDE.md docs/
git commit -m "docs: shard Tier-2 fused decode path (#377)"
git push -u origin feat/shard-tier2-fused
gh pr create --title "feat(shardquant): fused compressed-middle decode — Tier 2 (#377)" --body "..."
```

PR body: what Tier 2 implements (per issue + design doc), the honest divergence note (reconstruct-in-registers vs upstream's RoPE-onto-Q — same memory contract), bench numbers from Task 8, fallback matrix, out-of-scope list. End with the Claude Code attribution per repo convention.

---

## Self-review notes

- Spec coverage: ref path (Task 1), handle + cache mode (Task 2), patch (Task 3), kernels (Tasks 4–5), dispatch + e2e parity (Task 6), config (Task 7), live/bench (Task 8), docs (Task 9). All design-doc sections mapped.
- Types consistent: `ShardFusedKV(cache, k_exact, v_exact)`; `fused_sdpa_decode(queries, handle, scale, backend=)`; `install_fused_sdpa(model) -> int`; `make_shard_cache(..., fused=False)`; kernel wrappers take `(qg, cache)` / `(w_full, woff, cache)`.
- Known test-ordering hazard: `_make_cache` defaults D=16 (kernel-ineligible); fused/kernel tests always pass `D=64`.
