# Qwen3-Coder-Next Throughput Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift decode throughput for Flash-MoE + TurboQuant KV inference (target: Qwen3-Coder-Next-4bit) without changing output bits, by eliminating per-token CPU round-trips, serial I/O waits, and redundant full-cache KV dequantization.

**Architecture:** Four surgical edits across four files. All changes are mathematically identity-preserving. Each task produces a committable change guarded by existing tests, with new unit tests added first (TDD).

**Tech Stack:** Python 3.11, MLX (`mx.core`, `mx.nn`), pytest, `concurrent.futures`, `uv` for runtime. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-17-qwen3-coder-throughput-design.md`

**Baseline:** `~/.olmlx/bench/api_bench_20260417T191157Z.json` — decode p50 ~9-11 tok/s (ollama) / 13-17 tok/s (openai) for coding+reasoning prompts.

---

## File Structure

| Path | Role | Change |
|---|---|---|
| `olmlx/engine/flash/moe_weight_store.py` | SSD expert loader | Add `remap_lut` field to `LoadedExperts`; switch to `as_completed` for parallel I/O |
| `olmlx/engine/flash/flash_moe.py` | Runtime MoE dispatch | Replace Python-list remap with `mx.take` using device-side LUT |
| `olmlx/engine/turboquant_cache.py` | KV cache | Add dequantized side buffer; dequantize only new tokens per step |
| `tests/test_flash_moe_weight_store.py` | LoadedExperts tests | New test for `remap_lut` correctness + `as_completed` ordering |
| `tests/test_flash_moe.py` | FlashMoE tests | New test asserting `mx.take` remap matches reference Python path |
| `tests/test_turboquant.py` | KV cache tests | New tests for incremental dequant equivalence across N, bits, resize, trim |

## Tasks

### Task 1: Add `remap_lut` field to `LoadedExperts`

**Files:**
- Modify: `olmlx/engine/flash/moe_weight_store.py` — `LoadedExperts` dataclass + `load_experts()`
- Test: `tests/test_flash_moe_weight_store.py` — `TestFlashMoeWeightStore`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_flash_moe_weight_store.py`, inside `class TestFlashMoeWeightStore` (you'll find the setup fixture in the existing class — reuse it):

```python
    def test_load_experts_provides_remap_lut(self, tmp_path):
        """LoadedExperts.remap_lut maps global expert indices to local stack positions."""
        from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore
        from tests.test_flash_moe_bundler import _make_synthetic_moe_weights
        from olmlx.engine.flash.moe_bundler import bundle_moe_experts
        import mlx.core as mx

        model_dir = _make_synthetic_moe_weights(
            hidden=32, inter=16, experts=8, num_layers=1, layer_offset=0, tmp_path=tmp_path
        )
        output_dir = tmp_path / "flash_moe"
        bundle_moe_experts(model_dir, output_dir)

        store = FlashMoeWeightStore(output_dir, num_io_threads=2, cache_budget_experts=16)
        try:
            loaded = store.load_experts(layer_idx=0, expert_indices=[5, 2, 7])

            assert loaded.remap_lut is not None
            assert loaded.remap_lut.shape == (8,)
            assert loaded.remap_lut.dtype == mx.uint32

            lut = loaded.remap_lut.tolist()
            assert lut[5] == 0  # first requested global maps to stack pos 0
            assert lut[2] == 1
            assert lut[7] == 2
            # Unrequested entries carry the sentinel
            for i in (0, 1, 3, 4, 6):
                assert lut[i] == 0xFFFFFFFF
        finally:
            store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_flash_moe_weight_store.py::TestFlashMoeWeightStore::test_load_experts_provides_remap_lut -v`

Expected: FAIL with `AttributeError: 'LoadedExperts' object has no attribute 'remap_lut'` (or `TypeError` from the dataclass).

- [ ] **Step 3: Extend the `LoadedExperts` dataclass**

In `olmlx/engine/flash/moe_weight_store.py`, add one field to the `LoadedExperts` dataclass (currently ends with `expert_index_map: dict[int, int] = field(default_factory=dict)`):

```python
    remap_lut: mx.array | None = None
```

- [ ] **Step 4: Populate `remap_lut` inside `load_experts`**

In `olmlx/engine/flash/moe_weight_store.py`, function `load_experts`, replace the return-statement block (the `return LoadedExperts(...)` call near the end) with a version that builds the LUT from `expert_index_map` right before the return:

```python
        # Build device-side expert remap LUT: global expert idx -> local stack pos.
        # Sentinel 0xFFFFFFFF marks unused entries (never dereferenced in practice
        # because `remap` uses only indices present in `expert_index_map`).
        lut = np.full(layout.num_experts, 0xFFFFFFFF, dtype=np.uint32)
        for eidx, pos in expert_index_map.items():
            lut[eidx] = pos
        remap_lut = mx.array(lut)

        return LoadedExperts(
            gate_weight=_stack_or_none(components["gate_weight"]),
            gate_scales=_stack_or_none(components["gate_scales"]),
            gate_biases=_stack_or_none(components["gate_biases"]),
            gate_bias=_stack_or_none(components["gate_bias"]),
            up_weight=mx.stack(components["up_weight"]),
            up_scales=_stack_or_none(components["up_scales"]),
            up_biases=_stack_or_none(components["up_biases"]),
            up_bias=_stack_or_none(components["up_bias"]),
            down_weight=mx.stack(components["down_weight"]),
            down_scales=_stack_or_none(components["down_scales"]),
            down_biases=_stack_or_none(components["down_biases"]),
            down_bias=_stack_or_none(components["down_bias"]),
            is_quantized=layout.is_quantized,
            bits=layout.bits,
            group_size=layout.group_size,
            quant_mode=layout.quant_mode,
            expert_index_map=expert_index_map,
            remap_lut=remap_lut,
        )
```

- [ ] **Step 5: Run the new test and all existing weight-store tests**

Run: `uv run pytest tests/test_flash_moe_weight_store.py -v`

Expected: PASS (all tests in file, including the new one).

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/flash/moe_weight_store.py tests/test_flash_moe_weight_store.py
git commit -m "perf(flash-moe): add remap_lut to LoadedExperts

Device-side lookup table mapping global expert index to local stack
position. Populated alongside expert_index_map in load_experts. Enables
a follow-up change to vectorize the per-token remap in FlashMoE.__call__
via mx.take instead of a Python list comprehension."
```

---

### Task 2: Vectorize `FlashMoE.__call__` remap with `mx.take`

**Files:**
- Modify: `olmlx/engine/flash/flash_moe.py` — `FlashMoE.__call__` lines 77-90
- Test: `tests/test_flash_moe.py` — add bit-equal test vs reference path

- [ ] **Step 1: Write the failing test**

Append to `tests/test_flash_moe.py` inside `class TestFlashMoE`:

```python
    def test_output_matches_python_remap_reference(self, tmp_path):
        """mx.take-based remap must produce bit-identical output to the Python-list remap."""
        import mlx.core as mx

        hidden, inter, experts = 32, 16, 8
        flash_moe, store, _ = _setup_flash_moe(
            tmp_path, hidden, inter, experts, num_experts_per_tok=2
        )

        x = mx.random.normal((2, 4, hidden))
        # Mix of repeated and distinct experts to exercise the remap fully
        inds = mx.array(
            [
                [[0, 3], [1, 5], [0, 3], [2, 7]],
                [[4, 6], [1, 5], [2, 7], [0, 3]],
            ]
        )
        scores = mx.softmax(mx.random.normal(inds.shape).astype(mx.float32), axis=-1)

        out_new = flash_moe(x, inds, scores)
        mx.eval(out_new)

        # Reference: reconstruct output using a pure-Python remap on the same cached
        # weights.  Pull them back via store.load_experts (deterministic for the same set).
        B, L, K = inds.shape
        flat = inds.reshape(-1).tolist()
        unique = sorted(set(flat))
        loaded = store.load_experts(layer_idx=0, expert_indices=unique)
        idx_map = loaded.expert_index_map
        remap_py = mx.array(
            [idx_map[int(i)] for i in flat], dtype=mx.uint32
        ).reshape(B, L, K)

        x_expanded = mx.expand_dims(x, (-2, -3))
        if loaded.is_quantized:
            qkw = dict(
                transpose=True,
                group_size=loaded.group_size,
                bits=loaded.bits,
                mode=loaded.quant_mode,
            )
            g = mx.gather_qmm(
                x_expanded, loaded.gate_weight, loaded.gate_scales, loaded.gate_biases,
                rhs_indices=remap_py, **qkw,
            )
            u = mx.gather_qmm(
                x_expanded, loaded.up_weight, loaded.up_scales, loaded.up_biases,
                rhs_indices=remap_py, **qkw,
            )
            act = nn.silu(g) * u
            e = mx.gather_qmm(
                act, loaded.down_weight, loaded.down_scales, loaded.down_biases,
                rhs_indices=remap_py, **qkw,
            )
        else:
            g = mx.gather_mm(x_expanded, loaded.gate_weight.swapaxes(-1, -2), rhs_indices=remap_py)
            u = mx.gather_mm(x_expanded, loaded.up_weight.swapaxes(-1, -2), rhs_indices=remap_py)
            act = nn.silu(g) * u
            e = mx.gather_mm(act, loaded.down_weight.swapaxes(-1, -2), rhs_indices=remap_py)
        e = e.squeeze(-2)
        out_ref = (e * scores[..., None]).sum(axis=-2).astype(x.dtype)

        assert mx.allclose(out_new, out_ref, atol=0, rtol=0)
```

- [ ] **Step 2: Run test to verify it passes on current code**

Run: `uv run pytest tests/test_flash_moe.py::TestFlashMoE::test_output_matches_python_remap_reference -v`

Expected: PASS (the current code *is* the reference; this test pins output identity before we change the code). This is a regression guard — Task 2's actual verification is that this test still passes after the code change.

- [ ] **Step 3: Swap `FlashMoE.__call__` to use `mx.take` on the LUT**

In `olmlx/engine/flash/flash_moe.py`, function `FlashMoE.__call__`, replace lines 77-90 (the `# Collect unique expert indices...` block through the `.reshape(B, L, K)` end of the `remap` assignment):

```python
        # Collect unique expert indices for the SSD read list (Python-side, one eval per layer).
        mx.eval(inds)
        flat_inds = inds.reshape(-1).tolist()
        unique_experts = sorted(set(flat_inds))

        # Load experts from SSD (or RAM cache); LoadedExperts includes a device-side
        # remap LUT that maps global expert idx → local stack position.
        loaded = self.weight_store.load_experts(self.layer_idx, unique_experts)

        # Vectorized device-side remap: lut[inds] gives local positions.
        assert loaded.remap_lut is not None  # populated by FlashMoeWeightStore.load_experts
        remap = mx.take(loaded.remap_lut, inds.astype(mx.uint32))
```

Do NOT remove the `idx_map = loaded.expert_index_map` line below it — wait, it's inside the block being replaced. Confirm via `grep -n "idx_map" olmlx/engine/flash/flash_moe.py` — there should be zero remaining references after this edit.

- [ ] **Step 4: Run the bit-equal test and the full flash-MoE suite**

Run:
```bash
uv run pytest tests/test_flash_moe.py tests/test_flash_moe_weight_store.py -v
```

Expected: PASS (all tests, including the new `test_output_matches_python_remap_reference`).

- [ ] **Step 5: Run Nemotron + quantized variants to catch any regression**

Run:
```bash
uv run pytest tests/test_flash_moe.py::TestFlashMoENemotron -v
uv run pytest tests/test_flash_moe_weight_store.py::TestFlashMoeWeightStoreQuantized -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/flash/flash_moe.py tests/test_flash_moe.py
git commit -m "perf(flash-moe): vectorize expert remap with mx.take

Replace per-element Python dict lookup + mx.array allocation in
FlashMoE.__call__ with a single on-device mx.take against the
remap_lut built in load_experts. Eliminates ~flat_inds_len Python
function calls per MoE layer per token (48 layers × B·L·K entries
for Qwen3-Coder-Next). Bit-equal to the previous path."
```

---

### Task 3: Parallel I/O via `as_completed`

**Files:**
- Modify: `olmlx/engine/flash/moe_weight_store.py` — `load_experts()` future loop
- Test: `tests/test_flash_moe_weight_store.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_flash_moe_weight_store.py` inside `class TestFlashMoeWeightStore`:

```python
    def test_load_experts_tolerates_out_of_order_completion(self, tmp_path, monkeypatch):
        """load_experts must return correct stacked weights even if futures complete
        in a different order than submission."""
        import time
        from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore
        from olmlx.engine.flash.moe_bundler import bundle_moe_experts
        from tests.test_flash_moe_bundler import _make_synthetic_moe_weights
        import mlx.core as mx

        model_dir = _make_synthetic_moe_weights(
            hidden=32, inter=16, experts=8, num_layers=1, layer_offset=0, tmp_path=tmp_path
        )
        output_dir = tmp_path / "flash_moe"
        bundle_moe_experts(model_dir, output_dir)

        store = FlashMoeWeightStore(output_dir, num_io_threads=4, cache_budget_experts=0)
        try:
            # Baseline: serial read, cached order.
            baseline = store.load_experts(layer_idx=0, expert_indices=[5, 2, 7, 1])
            mx.eval(baseline.up_weight, baseline.down_weight)

            # Clear cache budget means each call re-reads; patch _read_expert to
            # delay the *first-requested* expert so it finishes last.
            original = store._read_expert
            delay_target = 5

            def delayed(layer_idx, expert_idx):
                if expert_idx == delay_target:
                    time.sleep(0.05)
                return original(layer_idx, expert_idx)

            monkeypatch.setattr(store, "_read_expert", delayed)

            reordered = store.load_experts(layer_idx=0, expert_indices=[5, 2, 7, 1])
            mx.eval(reordered.up_weight, reordered.down_weight)

            # Stacked tensors must be identical in input order regardless of completion order.
            assert mx.allclose(reordered.up_weight, baseline.up_weight, atol=0, rtol=0)
            assert mx.allclose(reordered.down_weight, baseline.down_weight, atol=0, rtol=0)
            assert baseline.expert_index_map == reordered.expert_index_map
        finally:
            store.close()
```

- [ ] **Step 2: Run test to verify it passes on current code**

Run: `uv run pytest tests/test_flash_moe_weight_store.py::TestFlashMoeWeightStore::test_load_experts_tolerates_out_of_order_completion -v`

Expected: PASS (current code already iterates `expert_indices` in order during stacking, so completion order doesn't matter). This pins the invariant before the refactor.

- [ ] **Step 3: Switch to `as_completed`**

In `olmlx/engine/flash/moe_weight_store.py`, at the top of the file, change the import:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

Then in `load_experts`, replace the current futures loop:

```python
        # Load missing experts via parallel I/O; consume in completion order so
        # slow readers don't block the fast ones.
        if missing:
            future_to_idx = {
                self._executor.submit(self._read_expert, layer_idx, idx): idx
                for idx in missing
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                data = future.result()
                cached[idx] = data
                self._cache.put(layer_idx, idx, data)
```

- [ ] **Step 4: Run the new test + full weight-store suite**

Run: `uv run pytest tests/test_flash_moe_weight_store.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/flash/moe_weight_store.py tests/test_flash_moe_weight_store.py
git commit -m "perf(flash-moe): consume expert-read futures via as_completed

Avoid stalling on the slowest future in submission order; cache
populates in completion order. Stacked output is unchanged because
the stacking loop iterates the caller-provided expert_indices, not
the cache insertion order."
```

---

### Task 4: Incremental KV dequantization side buffer

**Files:**
- Modify: `olmlx/engine/turboquant_cache.py` — `TurboQuantKVCache.__init__`, `update_and_fetch`, `trim`
- Test: `tests/test_turboquant.py` — extend `TestTurboQuantKVCache`

- [ ] **Step 1: Write the failing tests (equivalence over buffer-resize boundary)**

Append to `tests/test_turboquant.py` inside `class TestTurboQuantKVCache`:

```python
    def _oracle_cache_output(self, keys_per_step, values_per_step, bits, seed):
        """Build a fresh cache and feed updates one at a time; return final (k, v)."""
        import mlx.core as mx
        from olmlx.engine.turboquant_cache import TurboQuantKVCache
        from olmlx.engine.turboquant import TurboQuantRotation

        head_dim = keys_per_step[0].shape[-1]
        rk = TurboQuantRotation(head_dim=head_dim, seed=seed)
        rv = TurboQuantRotation(head_dim=head_dim, seed=seed + 1)
        cache = TurboQuantKVCache(bits=bits, rotation_key=rk, rotation_value=rv)
        k_out = v_out = None
        for k_step, v_step in zip(keys_per_step, values_per_step):
            k_out, v_out = cache.update_and_fetch(k_step, v_step)
        return k_out, v_out, cache

    @pytest.mark.parametrize("bits", [2, 4])
    @pytest.mark.parametrize("n_steps", [1, 8, 64, 256, 257, 513])
    def test_incremental_dequant_matches_full_dequant(self, bits, n_steps):
        """Incremental dequant in the side buffer must be bit-equal to full dequant."""
        import mlx.core as mx

        B, H, D = 1, 2, 32
        mx.random.seed(42)
        # Precompute per-step K/V, then feed to two caches and compare outputs.
        keys = [mx.random.normal((B, H, 1, D)).astype(mx.float16) for _ in range(n_steps)]
        values = [mx.random.normal((B, H, 1, D)).astype(mx.float16) for _ in range(n_steps)]

        # Run the current cache — which will, after Step 3, use the side buffer.
        k_new, v_new, _ = self._oracle_cache_output(keys, values, bits, seed=7)

        # Oracle: call turboquant_quantize + turboquant_dequantize directly on the
        # concatenated tensor (what the old full-cache dequant effectively did).
        from olmlx.engine.turboquant import (
            TurboQuantRotation, turboquant_quantize, turboquant_dequantize,
        )
        rk = TurboQuantRotation(head_dim=D, seed=7)
        rv = TurboQuantRotation(head_dim=D, seed=8)
        K = mx.concatenate(keys, axis=2)
        V = mx.concatenate(values, axis=2)
        k_idx, k_nrm = turboquant_quantize(K, rk, bits)
        v_idx, v_nrm = turboquant_quantize(V, rv, bits)
        k_ref = turboquant_dequantize(k_idx, k_nrm, rk, bits, dtype=mx.float16)
        v_ref = turboquant_dequantize(v_idx, v_nrm, rv, bits, dtype=mx.float16)

        assert mx.allclose(k_new, k_ref, atol=0, rtol=0)
        assert mx.allclose(v_new, v_ref, atol=0, rtol=0)

    def test_trim_preserves_incremental_side_buffer(self):
        """After trim, the side buffer must agree with a fresh cache fed the prefix."""
        import mlx.core as mx
        from olmlx.engine.turboquant_cache import TurboQuantKVCache
        from olmlx.engine.turboquant import TurboQuantRotation

        B, H, D, bits = 1, 2, 32, 4
        mx.random.seed(123)
        keys = [mx.random.normal((B, H, 1, D)).astype(mx.float16) for _ in range(20)]
        values = [mx.random.normal((B, H, 1, D)).astype(mx.float16) for _ in range(20)]

        rk = TurboQuantRotation(head_dim=D, seed=9)
        rv = TurboQuantRotation(head_dim=D, seed=10)
        cache = TurboQuantKVCache(bits=bits, rotation_key=rk, rotation_value=rv)
        for k, v in zip(keys, values):
            cache.update_and_fetch(k, v)
        # Trim last 5 tokens, then feed 3 more.
        cache.trim(5)
        for k, v in zip(keys[:3], values[:3]):
            cache.update_and_fetch(k, v)
        k_out, v_out = cache.update_and_fetch(keys[10], values[10])

        # Oracle: fresh cache with the equivalent sequence.
        rk2 = TurboQuantRotation(head_dim=D, seed=9)
        rv2 = TurboQuantRotation(head_dim=D, seed=10)
        oracle = TurboQuantKVCache(bits=bits, rotation_key=rk2, rotation_value=rv2)
        seq = list(zip(keys[:15], values[:15])) + list(zip(keys[:3], values[:3])) + [
            (keys[10], values[10])
        ]
        k_ref = v_ref = None
        for k, v in seq:
            k_ref, v_ref = oracle.update_and_fetch(k, v)
        assert mx.allclose(k_out, k_ref, atol=0, rtol=0)
        assert mx.allclose(v_out, v_ref, atol=0, rtol=0)

    def test_state_getter_excludes_side_buffer(self):
        """Cache state must still be [key_indices, key_norms, value_indices, value_norms]."""
        import mlx.core as mx
        from olmlx.engine.turboquant_cache import TurboQuantKVCache
        from olmlx.engine.turboquant import TurboQuantRotation

        B, H, D = 1, 2, 32
        rk = TurboQuantRotation(head_dim=D, seed=11)
        rv = TurboQuantRotation(head_dim=D, seed=12)
        cache = TurboQuantKVCache(bits=4, rotation_key=rk, rotation_value=rv)
        k = mx.random.normal((B, H, 4, D)).astype(mx.float16)
        v = mx.random.normal((B, H, 4, D)).astype(mx.float16)
        cache.update_and_fetch(k, v)
        state = cache.state
        assert len(state) == 4
        # Indices should be (B, H, offset, packed_dim); norms (B, H, offset, 1)
        assert state[0].shape[2] == 4
        assert state[1].shape[-1] == 1
```

Add `import pytest` at the top of the file if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_turboquant.py::TestTurboQuantKVCache -v -k "incremental or trim_preserves or state_getter"`

Expected: the `incremental` parametrized tests PASS on current code (full-cache dequant is equivalent), `trim_preserves` PASSES, `state_getter_excludes_side_buffer` PASSES. These pin behavior before refactor — they are regression guards. If any already fails, stop and investigate.

- [ ] **Step 3: Add side buffer fields to `__init__`**

In `olmlx/engine/turboquant_cache.py`, modify `TurboQuantKVCache.__init__` to add two fields after `self._value_norms: mx.array | None = None`:

```python
        self._key_dequant: mx.array | None = None
        self._value_dequant: mx.array | None = None
```

- [ ] **Step 4: Rewrite `update_and_fetch` to use the side buffer**

Replace the entire `update_and_fetch` method body in `olmlx/engine/turboquant_cache.py` with:

```python
    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Quantize new K/V, store bit-packed, and fetch dequantized history.

        Uses a persistent dequantized side buffer so each decode step dequantizes
        only the newly appended tokens — O(num_steps · head_dim²) per call rather
        than O(offset · head_dim²). Bit-equivalent to the previous full-cache path.
        """
        B, n_heads, num_steps, head_dim = keys.shape
        input_dtype = keys.dtype
        prev = self.offset

        k_idx, k_nrm = turboquant_quantize(keys, self.rotation_key, self._bits)
        v_idx, v_nrm = turboquant_quantize(values, self.rotation_value, self._bits)

        packed_dim = k_idx.shape[-1]

        # Allocate or grow index/norm/dequant buffers in lockstep.
        if self._key_indices is None or (prev + num_steps) > self._key_indices.shape[2]:
            new_steps = (num_steps + self.step - 1) // self.step * self.step
            idx_shape = (B, n_heads, new_steps, packed_dim)
            nrm_shape = (B, n_heads, new_steps, 1)
            deq_shape = (B, n_heads, new_steps, head_dim)

            if self._key_indices is not None:
                assert (
                    self._key_norms is not None
                    and self._value_indices is not None
                    and self._value_norms is not None
                    and self._key_dequant is not None
                    and self._value_dequant is not None
                )
                if prev % self.step != 0:
                    self._key_indices = self._key_indices[..., :prev, :]
                    self._key_norms = self._key_norms[..., :prev, :]
                    self._value_indices = self._value_indices[..., :prev, :]
                    self._value_norms = self._value_norms[..., :prev, :]
                    self._key_dequant = self._key_dequant[..., :prev, :]
                    self._value_dequant = self._value_dequant[..., :prev, :]
                self._key_indices = mx.concatenate(
                    [self._key_indices, mx.zeros(idx_shape, dtype=mx.uint8)], axis=2
                )
                self._key_norms = mx.concatenate(
                    [self._key_norms, mx.zeros(nrm_shape, dtype=mx.float32)], axis=2
                )
                self._value_indices = mx.concatenate(
                    [self._value_indices, mx.zeros(idx_shape, dtype=mx.uint8)], axis=2
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, mx.zeros(nrm_shape, dtype=mx.float32)], axis=2
                )
                self._key_dequant = mx.concatenate(
                    [self._key_dequant, mx.zeros(deq_shape, dtype=input_dtype)], axis=2
                )
                self._value_dequant = mx.concatenate(
                    [self._value_dequant, mx.zeros(deq_shape, dtype=input_dtype)], axis=2
                )
            else:
                self._key_indices = mx.zeros(idx_shape, dtype=mx.uint8)
                self._key_norms = mx.zeros(nrm_shape, dtype=mx.float32)
                self._value_indices = mx.zeros(idx_shape, dtype=mx.uint8)
                self._value_norms = mx.zeros(nrm_shape, dtype=mx.float32)
                self._key_dequant = mx.zeros(deq_shape, dtype=input_dtype)
                self._value_dequant = mx.zeros(deq_shape, dtype=input_dtype)

        assert (
            self._key_indices is not None
            and self._key_norms is not None
            and self._value_indices is not None
            and self._value_norms is not None
            and self._key_dequant is not None
            and self._value_dequant is not None
        )

        self.offset += num_steps
        self._key_indices[..., prev : self.offset, :] = k_idx
        self._key_norms[..., prev : self.offset, :] = k_nrm
        self._value_indices[..., prev : self.offset, :] = v_idx
        self._value_norms[..., prev : self.offset, :] = v_nrm

        # Dequantize only the newly appended slice and splice into the side buffer.
        k_new = turboquant_dequantize(
            self._key_indices[..., prev : self.offset, :],
            self._key_norms[..., prev : self.offset, :],
            self.rotation_key,
            self._bits,
            dtype=input_dtype,
        )
        v_new = turboquant_dequantize(
            self._value_indices[..., prev : self.offset, :],
            self._value_norms[..., prev : self.offset, :],
            self.rotation_value,
            self._bits,
            dtype=input_dtype,
        )
        self._key_dequant[..., prev : self.offset, :] = k_new
        self._value_dequant[..., prev : self.offset, :] = v_new

        return (
            self._key_dequant[..., : self.offset, :],
            self._value_dequant[..., : self.offset, :],
        )
```

- [ ] **Step 5: Update `trim` to keep buffers consistent**

Replace the existing `trim` method in the same file with:

```python
    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        if self.offset == 0:
            self._key_indices = None
            self._key_norms = None
            self._value_indices = None
            self._value_norms = None
            self._key_dequant = None
            self._value_dequant = None
        return n
```

- [ ] **Step 6: Run the full turboquant suite**

Run: `uv run pytest tests/test_turboquant.py -v`

Expected: PASS — all existing tests + the three new ones.

- [ ] **Step 7: Commit**

```bash
git add olmlx/engine/turboquant_cache.py tests/test_turboquant.py
git commit -m "perf(turboquant): cache dequantized KV in side buffer

Dequantize only newly appended tokens per update_and_fetch and splice
into a persistent float-dtype side buffer. Converts O(offset · head_dim²)
per step into O(num_steps · head_dim²), which matters most during prefill
and long-context decode. Bit-equal to the previous full-cache path;
validated across bits ∈ {2, 4} and lengths spanning the step=256 boundary
(including trim + resume). Memory cost: one extra input_dtype buffer per
quantized layer (~100-200 MB for Qwen3-Coder-Next at 4096 tokens)."
```

---

### Task 5: Validation — run full test suite + re-bench

**Files:** none modified. Validation only.

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest -q`

Expected: PASS. If anything unrelated fails, investigate before proceeding.

- [ ] **Step 2: Run ruff**

Run: `uv run ruff check . && uv run ruff format --check .`

Expected: clean. If format violations appear, run `uv run ruff format .` and re-commit.

- [ ] **Step 3: Start (or confirm) the olmlx server**

```bash
# If not already running:
curl -s --max-time 2 http://localhost:11434/api/version || (
  TS=$(date +%Y%m%d-%H%M%S)
  nohup uv run olmlx serve > "olmlx-serve-optimized-$TS.log" 2>&1 &
  sleep 20  # wait for startup; watch the log for "Application startup complete"
)
```

Expected: `curl` returns version JSON. Server log ends with `Application startup complete.`

- [ ] **Step 4: Re-run the identical baseline bench**

```bash
TS=$(date +%Y%m%d-%H%M%S)
uv run python -m olmlx.bench.api_bench \
  --models "mlx-community/Qwen3-Coder-Next-4bit:latest" \
  --apis "ollama-chat,openai-chat" \
  --modes "stream,nostream" \
  --prompts "factual,coding,reasoning" \
  --runs 2 --warmup 1 --max-tokens 256 --timeout 600 \
  > "bench-qwen3coder-optimized-$TS.log" 2>&1
```

Expected: all cells complete, final summary table printed.

- [ ] **Step 5: Compare against baseline**

```bash
ls -t ~/.olmlx/bench/api_bench_*.json | head -2
# Compare the newest two JSONs: baseline is the older one (2026-04-17T19:11:57Z
# per spec), optimized is the newer one.
uv run python -c "
import json, sys
from pathlib import Path
bench_dir = Path.home() / '.olmlx' / 'bench'
files = sorted(bench_dir.glob('api_bench_*.json'), key=lambda p: p.stat().st_mtime)
base, opt = json.loads(files[-2].read_text()), json.loads(files[-1].read_text())
def index(d): return {(r['api'], r['mode'], r['prompt']): r for r in d['results']}
b, o = index(base), index(opt)
print(f'{'cell':<38}{'base tok/s':>12}{'opt tok/s':>12}{'Δ%':>8}')
for key in b:
    if key not in o: continue
    bb = b[key].get('tokens_per_second_p50') or 0
    oo = o[key].get('tokens_per_second_p50') or 0
    d = (oo - bb) / bb * 100 if bb else 0
    print(f'{\"/\".join(key):<38}{bb:>12.2f}{oo:>12.2f}{d:>+7.1f}%')
"
```

Expected: no cell regresses by more than 2%; `coding` and `reasoning` decode p50 tok/s improved on both `ollama-chat` and `openai-chat`.

- [ ] **Step 6: If numbers look good, push**

```bash
git log --oneline origin/main..HEAD  # confirm commit list
git push
```

Expected: push succeeds.

---

## Self-Review

- **Spec coverage:**
  - A+E (vectorized remap) → Tasks 1, 2 ✓
  - B (`as_completed`) → Task 3 ✓
  - D (side buffer) → Task 4 ✓
  - Testing strategy (bit-equal, parametrized, trim, state) → Task 4 Steps 1-6 ✓
  - Throughput validation → Task 5 ✓
  - Regression safety → Tasks 2.5, 3.4, 4.6 ✓
- **Placeholder scan:** None.
- **Type consistency:** `remap_lut: mx.array | None = None` in Task 1, referenced as `loaded.remap_lut` in Task 2 with `assert loaded.remap_lut is not None`. `_key_dequant`/`_value_dequant` consistent between Task 4 Steps 3, 4, 5. Method names match the existing `TurboQuantKVCache` / `FlashMoeWeightStore` / `FlashMoE` class APIs.
