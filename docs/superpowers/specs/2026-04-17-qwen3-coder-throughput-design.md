# Qwen3-Coder-Next Throughput Optimization — Design

**Date:** 2026-04-17
**Target model:** `mlx-community/Qwen3-Coder-Next-4bit` (MoE, hybrid SSM + attention)
**Target mode:** `flash_moe: true`, `kv_cache_quant: turboquant:4`
**Scope:** Four identity-preserving optimizations to the Flash-MoE and TurboQuant KV cache hot paths. No behavioral or quality changes.

## Baseline

Measured on 2026-04-17 via `olmlx.bench.api_bench` against a running server (p50 across 2 runs per cell):

| api | mode | prompt | decode tok/s | TTFT (ms) | total (ms) |
|---|---|---|---|---|---|
| ollama-chat | stream | factual | 3.69 | 1487 | 2329 |
| ollama-chat | stream | coding | 9.15 | 1502 | 28198 |
| ollama-chat | stream | reasoning | 9.56 | 1845 | 15718 |
| ollama-chat | nostream | coding | 10.56 | — | 24236 |
| ollama-chat | nostream | reasoning | 10.95 | — | 12148 |
| openai-chat | stream | coding | 13.67 | 5610 | 26386 |
| openai-chat | stream | reasoning | 17.46 | 5935 | 15154 |

Steady-state decode is **~9–11 tok/s** (ollama) and **~13–17 tok/s** (openai stream). `factual` tok/s is dominated by TTFT because outputs are 8 tokens.

Full JSON baseline: `~/.olmlx/bench/api_bench_20260417T191157Z.json`.

## Problem Statement

Profiling (via code audit, not runtime profiler) identified four per-token overheads on the flash_moe + turboquant decode path that can be eliminated without changing sampling distribution, quantization precision, or token selection.

1. **Flash-MoE remap is Python-side.** `flash_moe.py:78–90` materializes routed indices to a Python list, runs a dict lookup per element, and allocates a new `mx.array` every token for every MoE layer (48× per token for Qwen3-Coder-Next).
2. **SSD I/O completes in submission order.** `moe_weight_store.py:343–346` calls `future.result()` in submission order, so parallel reads stall on the slowest future before later futures can be consumed.
3. **Full KV cache is re-dequantized every step.** `turboquant_cache.py:113–126` dequantizes the entire stored KV history on every `update_and_fetch` call — O(n·head_dim²) per decode step even though only one token was added.
4. **Remap tensor constructed on CPU.** The `mx.array([...])` in `flash_moe.py:87–90` triggers a host→device copy every layer; merging with (1) lets the remap stay on-device.

## Goals

- Preserve output bit-for-bit: unit tests compare new path against old path with `atol=0, rtol=0`.
- Measurable decode throughput lift on `coding` and `reasoning` prompts vs baseline.
- No regression on non-MoE, non-TurboQuant paths (they are untouched).
- No new config knobs, no new dependencies.

## Non-Goals

- Cross-layer MoE prefetch (candidate F from audit — deferred).
- Cached stacked `LoadedExperts` across tokens (candidate C — deferred; depends on steady-state hit rate data).
- Changes to sampling, logits processing, repetition penalty.
- Changes to prompt caching or hybrid-model trim behavior.

## Architecture

Four surgical edits in four files. No new modules, no API changes, no config changes. All changes sit behind the existing `flash_moe` and `kv_cache_quant` feature flags.

```
olmlx/engine/flash/flash_moe.py         ← A + E
olmlx/engine/flash/moe_weight_store.py  ← B + LUT build
olmlx/engine/turboquant_cache.py        ← D
olmlx/engine/turboquant.py              ← D (minor: helper for per-row dequant)
```

Each optimization is independent; can be landed as a stacked PR set or a single PR with one commit per optimization.

## Detailed Design

### A + E — Vectorized device-side expert remap

**Before** (`flash_moe.py:77–90`):
```python
mx.eval(inds)
flat_inds = inds.reshape(-1).tolist()
unique_experts = sorted(set(flat_inds))
loaded = self.weight_store.load_experts(self.layer_idx, unique_experts)
idx_map = loaded.expert_index_map
remap = mx.array(
    [idx_map[int(i)] for i in flat_inds],
    dtype=mx.uint32,
).reshape(B, L, K)
```

**After:**
```python
# Still need Python-side uniques for the SSD read list.
mx.eval(inds)
flat_inds = inds.reshape(-1).tolist()
unique_experts = sorted(set(flat_inds))
loaded = self.weight_store.load_experts(self.layer_idx, unique_experts)
remap = mx.take(loaded.remap_lut, inds.astype(mx.uint32))  # (B, L, K)
```

**Supporting change in `moe_weight_store.py`**: extend `LoadedExperts` with:
```python
remap_lut: mx.array  # shape (num_experts,), uint32
```

Built inside `load_experts` from the same `expert_index_map` that exists today:
```python
lut = np.full(layout.num_experts, 0xFFFFFFFF, dtype=np.uint32)
for eidx, pos in expert_index_map.items():
    lut[eidx] = pos
remap_lut = mx.array(lut)
```

This builds the LUT once per layer per forward pass (not once per token per layer), and the actual remap becomes a single on-device `mx.take`.

**Identity:** `mx.take(lut, inds)[b, l, k] == lut[inds[b,l,k]] == expert_index_map[inds[b,l,k].item()]`. Same uint32 value.

### B — Overlap SSD I/O with `as_completed`

**Before** (`moe_weight_store.py:339–346`):
```python
futures = {idx: self._executor.submit(self._read_expert, layer_idx, idx) for idx in missing}
for idx, future in futures.items():
    data = future.result()
    cached[idx] = data
    self._cache.put(layer_idx, idx, data)
```

**After:**
```python
from concurrent.futures import as_completed

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

**Identity:** `cached` is a dict — iteration order into the subsequent stacking loop is determined by `expert_indices` (the caller-provided list), not by insertion order into `cached`. So stacked tensors are identical regardless of which future resolved first.

### D — Incremental KV dequant with cached side buffer

**New state in `TurboQuantKVCache.__init__`:**
```python
self._key_dequant: mx.array | None = None    # (B, n_heads, capacity, head_dim) in input_dtype
self._value_dequant: mx.array | None = None
```

**Modified `update_and_fetch`:**

1. Quantize incoming `num_steps` tokens (unchanged — fills `_key_indices[prev:offset]` etc.).
2. Grow side buffers in lockstep with index buffers on resize (mirror the `mx.concatenate` block at lines 81-92).
3. Dequantize **only the new slice**:
   ```python
   k_new = turboquant_dequantize(
       self._key_indices[..., prev:self.offset, :],
       self._key_norms[..., prev:self.offset, :],
       self.rotation_key, self._bits, dtype=input_dtype,
   )
   v_new = turboquant_dequantize(...)
   self._key_dequant[..., prev:self.offset, :] = k_new
   self._value_dequant[..., prev:self.offset, :] = v_new
   ```
4. Return views into the side buffer:
   ```python
   return (
       self._key_dequant[..., :self.offset, :],
       self._value_dequant[..., :self.offset, :],
   )
   ```

**Resize path:** on buffer expansion, copy old dequant buffer forward alongside the index/norm buffers. When `prev % self.step != 0` (the truncation branch), also truncate the dequant buffer the same way.

**`trim(n)` path:** zero-out or shrink the dequant buffer tail alongside the index buffers; no re-dequantization needed because the preserved tail is still valid.

**`state` getter:** unchanged — side buffer is recoverable from indices+norms and is not part of the persisted state.

**Memory cost:** One additional float16 (or input_dtype) buffer of size `B × n_heads × capacity × head_dim` per quantized layer. For Qwen3-Coder-Next (12 quantized layers, head_dim=256, 4096 tokens, fp16, n_kv_heads ≈ 8): ~100-200 MB. Acceptable; can be capped in a follow-up if needed.

**Identity:** The per-row dequant for `[prev:offset]` uses the same `turboquant_dequantize` function with the same arguments that today's full-cache dequant uses for the same range. The concatenation of new + historical dequantized rows is mathematically the same tensor as calling dequant on the whole range (dequant is elementwise across the seq dim).

## Testing

**Correctness (written before implementation):**

1. `tests/test_flash_moe.py` — new or extended:
   - **A/E equivalence:** Construct tiny MoE (4 experts, 2-per-tok, quantized + fp16 paths). Run old code path (captured via a patched reference) and new path on identical inputs. Assert `mx.allclose(out_new, out_ref, atol=0, rtol=0)`.
   - **B ordering:** Mock `_read_expert` with variable per-index delay. Assert stacked `LoadedExperts` matches a deterministic baseline regardless of completion order.

2. `tests/test_turboquant_cache.py` — extend:
   - **D equivalence:** For N ∈ {1, 8, 64, 256, 513}, feed N tokens one at a time into the new cache. Compare final fetched K and V against an independent "oracle" cache that calls the old full-dequant path. Parametrized over bits ∈ {2, 4}. Covers the `step=256` boundary.
   - **D trim:** After `trim(k)`, fetched tensors must match a fresh cache fed tokens `[:offset-k]` then the same subsequent updates.
   - **D state:** `state` getter return value unchanged in shape and content.

**Integration:**
- Full `uv run pytest` must pass (no existing test expected to break).

**Throughput:**
- Re-run the identical `api_bench` invocation used for the baseline.
- Compare p50 tok/s and TTFT per cell against `~/.olmlx/bench/api_bench_20260417T191157Z.json`.
- Acceptance: no cell regresses more than 2%; `coding` and `reasoning` decode p50 improves.

**Quality spot-check:**
- Same coding prompt + fixed seed, pre- and post-optimization — outputs must match token-for-token.

## Risks

| Risk | Mitigation |
|---|---|
| D's side buffer erases TurboQuant's RAM savings | Flag in docs; follow-up PR can add a cap |
| A/E breaks non-quantized MoE path | Unit test covers both fp16 and quantized variants |
| B changes behavior when an expert read fails partway through | `.result()` still raises on failure; we cancel remaining futures in the exception path (already correct in submission code, preserved) |
| `mx.take` with `uint32` inds not supported on some MLX versions | Test on current pinned MLX version; fallback is the current code |
| D's buffer resize race with `trim()` | Both mutate `self._*` under the same caller (no threading inside cache); preserve single-threaded contract |

## Rollback

Each optimization is a local edit to one function. Revert = `git revert` of that commit. No migrations, no state changes.
