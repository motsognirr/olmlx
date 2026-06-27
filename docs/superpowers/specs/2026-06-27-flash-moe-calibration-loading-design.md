# Flash-MoE-backed KV-quant calibration loading

**Date:** 2026-06-27
**Status:** Design approved

## Problem

Spectral and shard KV-cache quantization require a one-time calibration pass
(`olmlx spectral prepare <model>` / `olmlx shard prepare <model>`) that runs
forward passes through the model to collect post-RoPE K/V vectors. The shared
loader `_load_calibration_model` (in `olmlx/engine/spectralquant_calibrate.py`,
also used by `shardquant_calibrate.py`) loads the model with
`mlx_lm.load(lazy=False)`, eagerly materializing **all** weights.

For a large MoE model this is fatal: `mlx-community/Qwen3-235B-A22B-4bit` is
~125 GB of weights, far over the ~48 GB Metal wired limit on a 64 GB machine.
Calibration OOM-crashes (uncatchable Metal abort) at model load, before any
calibration work begins. Consequently spectral/shard quant — the calibrated
methods that hold quality better than `turboquant` at low bit-widths — are
unavailable for exactly the models that most need KV reduction.

## Goal

Let calibration run on models too large to fully load, by loading them through
the existing **Flash-MoE** path (router + attention resident, routed experts
streamed from SSD on demand) when a prepared `flash_moe/` bundle exists. This
is correctness-preserving for calibration: Flash-MoE replaces only the MoE FFN;
the **attention** layers — the sole source of the K/V vectors calibration
collects — are untouched, so a Flash-MoE-loaded model yields identical K/V.

Concrete target: produce a `spectral:2` calibration for
`Qwen3-235B-A22B-4bit` on a 64 GB machine.

## Non-Goals

- Refactoring `model_manager._load_flash_moe_model` to use the new shared
  helper. The hot inference loader stays untouched in this change; it can adopt
  the helper in a later PR.
- Adding Flash-MoE support to any other offline pipeline (dflash/eagle/mtp
  training, etc.).
- Sharding/distributed calibration. Flash-MoE is incompatible with distributed
  inference; that constraint is unchanged.

## Design

### Component 1 — shared loader helper

New free function in `olmlx/engine/flash/flash_moe_model.py`:

```
load_flash_moe_model(load_path, flash_moe_dir, *, cache_budget_experts, io_threads)
    -> (wrapped_model, tokenizer, store)
```

Encapsulates the recipe `model_manager._load_flash_moe_model` already performs:

1. `load_model_with_strict_fallback(load_path, lazy=True)` — lazy load so expert
   weights are never materialized (also handles the VL `strict=False` retry).
2. Read `flash_moe_dir / "flash_moe_config.json"` for architecture fields
   (`moe_layer_indices`, `hidden_size`, `intermediate_size`, `num_experts`,
   `num_experts_per_tok`).
3. Build `FlashMoeWeightStore(flash_moe_dir, num_io_threads=io_threads,
   cache_budget_experts=cache_budget_experts)`.
4. Wrap in `FlashMoeModelWrapper(model, store, ...)` (replaces SwitchGLU with
   the streaming FlashMoE layers).
5. `mx.eval(wrapped.parameters())` — materializes only the non-expert weights.
6. On any failure after the store is created, `store.close()` then re-raise.

Returns the `store` so the caller controls its lifetime.

### Component 2 — calibration loader branch

`_load_calibration_model(model_path)` gains a probe:

```
flash_moe_dir = Path(model_path) / "flash_moe"
if (flash_moe_dir / "flash_moe_layout.json").exists():
    model, tokenizer, store = load_flash_moe_model(
        model_path, flash_moe_dir,
        cache_budget_experts=_CALIBRATION_CACHE_BUDGET_EXPERTS,
        io_threads=_CALIBRATION_IO_THREADS,
    )
else:
    model, tokenizer = load_model_with_strict_fallback(model_path, lazy=False)
    store = None
```

The rest of `_load_calibration_model` is unchanged: `_get_backbone(model)`
resolves `.layers` through the wrapper's attribute proxying, and the returned
tuple is extended with `store` (None on the full-load path). `collect_kv_vectors`
runs unchanged — it hooks attention K/V, which Flash-MoE does not alter.

### Memory safety

- `_CALIBRATION_CACHE_BUDGET_EXPERTS = 8` (vs. the inference budget of 24).
  Calibration is latency-tolerant and the LRU budget only affects *which*
  experts are resident, never the K/V output, so we minimize resident footprint
  to leave room for the K/V-collection buffers that accumulate across samples.
- `_CALIBRATION_IO_THREADS = 32` (flash default).
- Both are module constants — bump if calibration is too slow.
- Footprint: ~4 GB non-expert resident + ~8 GB 8-expert LRU + a few GB of
  K/V-collection buffers (bounded by `samples × max_tokens_per_head`) sits well
  under the 48 GB Metal limit.

### Store lifecycle

`load_flash_moe_model` returns the store; `_load_calibration_model` returns it
up the chain; `calibrate_model` and `calibrate_model_shard` close it in a
`finally` after KV collection. The process exits right after calibration anyway,
but an explicit close tears down the SSD reader threads cleanly and is
test-observable.

### Error handling

- If the `flash_moe/` bundle dir is present, commit to the flash path. If it
  fails to load, **raise** (naming the bundle) rather than silently falling back
  to the full load that would OOM on a large model.
- No bundle → unchanged `mlx_lm.load(lazy=False)` path; small models behave
  exactly as today.

## Testing

TDD, reusing the `tests/test_flash_moe_model.py` harness (synthetic bundle via
`_make_synthetic_moe_weights` + `bundle_moe_experts`):

1. **Loader selection** — `_load_calibration_model` with a `flash_moe/` bundle
   present calls `load_flash_moe_model` and returns a `FlashMoeModelWrapper`;
   with no bundle it takes the `mlx_lm.load(lazy=False)` path (patched). Asserts
   the branch, not real weights.
2. **Helper build** — `load_flash_moe_model` returns a wrapped model whose MoE
   layers are `_FlashMoEBase` instances, a live store, and evals without
   materializing experts (reuse the synthetic Qwen3 mock).
3. **Store close** — `calibrate_model` closes the store in its `finally`
   (assert via a spy/mock store).
4. **K/V collection over flash wrapper** — `collect_kv_vectors` over a tiny
   flash-wrapped synthetic model returns the expected per-layer K/V shapes,
   proving attention K/V is unaffected by expert offloading.

CI cannot exercise the real 235 B model (no weights). **Acceptance test:** a live
`olmlx spectral prepare mlx-community/Qwen3-235B-A22B-4bit --avg-bits 2` run
completes without OOM, watching memory and the external volume, producing a
spectral calibration artifact the server then loads via `kv_cache_quant=spectral:2`.

## Rollout

1. Land helper + loader branch + unit tests (this spec → plan).
2. Run the live calibration on `Qwen3-235B-A22B-4bit`.
3. Switch the model's `kv_cache_quant` from `turboquant:4` to `spectral:2` in
   `~/.olmlx/models.json`; keep `flash_moe_cache_budget_experts: 24`.
