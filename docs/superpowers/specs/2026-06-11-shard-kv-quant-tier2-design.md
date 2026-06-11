# Shard KV quant — Tier 2: fused compressed-middle attention (#377)

Date: 2026-06-11
Issue: #377 (Tier 2 — "int4-compressed Q·K via a Metal kernel with RoPE folded onto Q")
Builds on: Tier 1 (#496) + follow-ups (#497)

## Problem

Tier 1's `ShardKVCache` honors the resident-memory contract (quantized middle +
FP16 sink/window only), but **every decode step materializes the full FP16
middle** via `_decompress_middle`: unpack → codebook gather → rank back-project
→ re-rope for K, gather → un-rotate for V — written to device memory, then read
again by `mx.fast.scaled_dot_product_attention`. At 32k context (H=8, D=128)
that is ~250 MB of traffic per layer per step on data whose packed form is
~20 MB. Decode is memory-bandwidth-bound on Apple Silicon, so this transient
round-trip is the dominant Tier-1 decode cost (the A/B bench's 52 tok/s).

Tier 2 removes the round-trip: compute attention **directly from the packed
form**, reading only int codes + norms from device memory.

## Approaches considered

**A. Single fused flash-decode kernel** (online softmax, K-score + V-accumulate
in one Metal kernel). Maximum performance, but a large, hard-to-validate kernel
(online softmax + two dequant schemes + RoPE + GQA in one body). Rejected for
v1 — can be a follow-up once the two-kernel layout is proven.

**B. Two slim Metal kernels + MLX softmax (chosen).**
- Kernel 1 (`shard_k_scores`): packed K codes + norms → per-q-head score row
  over the middle. Reconstructs each key *in registers* (gather → `c @ U_h` →
  `+ mean` → `× norm` → re-rope at its absolute position) and dots with the
  roped query. Writes only the `(n_q_heads, mid_len)` fp32 score tile.
- Kernel 2 (`shard_v_accumulate`): softmax weights + packed V codes + norms →
  weighted sum in *rotated* space; the orthonormal un-rotation is linear so it
  folds outside the sum (`Σ wⱼnⱼ(ṽⱼ @ R) = (Σ wⱼnⱼṽⱼ) @ R` — one D×D matmul
  after the kernel).
- Softmax (fp32) over `[sink | middle | window]` scores happens in plain MLX;
  the exact-region scores/output are two tiny matmuls (≤ 4 + 64 tokens).

Memory traffic per layer per step drops to roughly the packed size (~20 MB in
the example above) plus a score row — the Tier-2 contract — while each kernel
stays an independently parity-testable "gemv with inline dequant".

**C. Pure-MLX "RoPE-onto-Q" GEMM decomposition** (no custom kernel): the
relative-rotation identity `rope_i(q)·rope_j(k) = rope_{i−j}(q)·k` expands into
per-(position, freq) cos/sin weight matrices — but those intermediates are the
same size as the dequantized K, so there is no traffic win. Useful only as a
math cross-check; not the deliverable.

**Divergence from upstream, stated honestly:** Shard's "fold RoPE onto Q"
identity exists because a Triton int4 GEMM cannot rotate K per key. A custom
Metal kernel can — re-roping the reconstructed key in registers is
mathematically identical (same scores), bit-comparable with Tier 1's dequant
path, and simpler. What we keep from the Tier-2 framing is the part that
matters: **Q·K computed from the int4 form with no FP16 K ever materialized to
device memory.**

## Architecture

New module `olmlx/engine/shardquant_fused.py`:

1. **Reference ops** (plain MLX, run anywhere incl. CPU tests):
   - `shard_middle_scores_ref(q_roped, cache) -> (B, n_q, 1→, mid_len)` —
     identical math to `_decompress_middle`'s K path reordered, fp32.
   - `shard_middle_weighted_v_ref(weights, cache) -> (B, n_q, 1, D)` — gather +
     weighted-sum + folded un-rotate.
2. **Metal kernels** via `mx.fast.metal_kernel` (compiled once per
   (bits, layout) template; grid/shapes vary per call):
   - `shard_k_scores`: inputs = packed K codes (capacity buffer), K norms,
     `U_h` bases, per-head codebook, mean, rope freqs, roped q, scalars
     (mid_len, sink_len, rank, rope dims/traditional flag). One simdgroup per
     middle-token tile; `U_h` tiles staged in threadgroup memory; fp32
     accumulate; `simd_sum` reduction. Supports bits ∈ {2,4,8}, shared or
     per-head K codebooks (shared broadcast to (H, K) at the wrapper), partial
     rope dims, `rope_spec=None` (skip rotation), traditional + non-traditional
     pair layouts.
   - `shard_v_accumulate`: inputs = weights, packed V codes, V norms, VQ
     codebooks; per-(q-head, dim) accumulation. Un-rotate happens outside.
3. **`ShardFusedKV` handle** — small object carrying `k_exact`, `v_exact`
   (sink+window concat) and the cache reference. In fused mode
   `update_and_fetch` returns `(handle, handle)` instead of materialized K/V.
   Any *unpatched* consumer that feeds the handle to `mx.fast.sdpa` raises
   immediately — fail loud, never silently attend over a partial history.
4. **sdpa wrapper + patch installer.** mlx-lm attention modules bind
   `scaled_dot_product_attention` at import (`from .base import …`), so
   patching `mlx_lm.models.base` alone does nothing. `install_fused_sdpa(model)`
   walks `model.layers`, collects the defining module of each layer class, and
   swaps the module-global `scaled_dot_product_attention` for a wrapper
   (idempotent via marker attribute, original kept for delegation). The wrapper
   dispatches on `isinstance(keys, ShardFusedKV)`; everything else goes to the
   original untouched. Installed by `make_shard_cache` when fused mode is on;
   never uninstalled (isinstance dispatch + single-user serialized inference
   make it inert for non-shard models).

### Decode data flow (fused)

```
model ropes q at offset → cache.update_and_fetch(k,v)   [sink/window/spill: unchanged Tier-1 logic]
  └─ q_len==1, fused on, mid_len>0 → returns ShardFusedKV
wrapper sdpa(q, handle, handle, cache=…, scale, mask):
  scores_exact = q @ k_exactᵀ                      (≤ ~68+ tokens, MLX)
  scores_mid   = shard_k_scores(…)                 (Metal, or _ref off-GPU)
  w = softmax(scale · [s_sink | s_mid | s_win])    (fp32, MLX)
  out = w_exact @ v_exact + (shard_v_accumulate(w_mid,…) @ R)
```

### Fallback matrix (wrapper / update_and_fetch decide)

| Condition | Path |
|---|---|
| `q_len > 1` (prefill, chunked prefill) | Tier-1 full materialization (unchanged) |
| `mid_len == 0` | Tier-1 (exact-only concat) |
| `OLMLX_SHARD_FUSED=false` | Tier-1 everywhere |
| `sinks is not None` (gpt-oss-style) or unexpected mask | wrapper asks cache to materialize, delegates to original sdpa |
| Metal unavailable (CPU tests) | reference ops, same results |
| `B > 1` | Tier-1 (olmlx decode is B=1) |

Fallbacks reuse `_decompress_middle`, so correctness never depends on the
kernel being applicable.

## Config

`Settings.shard_fused: bool = True` (`OLMLX_SHARD_FUSED`) — kill switch.
Default ON: shard is already opt-in (env + calibration), Tier 2 is strictly a
faster evaluation of the same math, and parity is enforced by tests. Threaded
`model_manager → make_shard_cache(model, dir, bits, fused=…)`. Per-model
override at top level of `models.json` like other promoted shard knobs.

## Composition / safety

- **No new mutable state**: handle is created per `update_and_fetch` call,
  never stored. `state`, `trim`, deepcopy/snapshot (#284 eager-eval), radix
  takeover, and `_is_serializable_cache` gating are untouched.
- **No `mx.compile` around kernel calls** (custom kernels stay outside compiled
  graphs); kernels are JIT-compiled once per template and cached by mlx.
- Kernels read the **capacity-aligned** buffers and loop to `mid_len` (runtime
  scalar input) — same anti-retrace convention as Tier 1, no per-step rebuilds.
- Speculative decoders never see shard caches (own their caches); VLM path has
  no KV-quant; distributed out of scope — all unchanged.

## Testing (TDD, parity chain)

1. `tests/test_shardquant_fused.py`:
   - **ref == Tier 1**: decode-step attention output via reference fused path
     vs. `_decompress_middle` + sdpa, random caches over (bits ∈ {2,4,8},
     per-head/shared codebooks, rope traditional/partial/None, GQA groups,
     mean/None) — tight fp32 tolerance.
   - **kernel == ref**: same grid, GPU-marked (skip when Metal unavailable).
   - handle returned only when (q_len==1 ∧ fused ∧ mid_len>0); Tier-1 output
     otherwise; trim/snapshot behavior unchanged with fused on.
   - patch installer: idempotent, dispatches non-shard calls to original,
     unpatched-consumer failure is loud (TypeError on handle).
2. End-to-end greedy-parity: short generation fused on vs. off, same tokens
   (or same logits to tolerance) on a tiny real model — lives with the other
   real-model shard coverage in `tests/live/test_shard_quant_real.py`.
3. A/B bench: extend `scripts/shard_ab_bench.py` with `--fused {on,off}`;
   PR records decode tok/s + resident memory vs Tier 1 and `spectral:4`.

## Out of scope (v1)

Single-kernel online-softmax flash decode (follow-up), prefill fusion
(q_len>1), fused sink-attention (`sinks`) path, B>1, VLM/distributed/
speculative composition, disk spill (unchanged Tier-1 gate).
