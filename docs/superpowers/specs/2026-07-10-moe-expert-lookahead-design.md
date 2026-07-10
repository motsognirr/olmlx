# Flash-MoE Trained Expert Lookahead: Prefetch + Scored Eviction

**Date:** 2026-07-10
**Status:** Approved

## Problem

The Flash-MoE path (`engine/flash/flash_moe.py`, `moe_weight_store.py`) is fully
synchronous: the router picks top-k experts, `mx.eval(inds)` syncs, then
`load_experts()` blocks the forward pass on SSD reads for every cache miss. The
RAM cache is a plain per-layer LRU (`LayerLruCache`, default 48 experts/layer).
There is no prefetching on the MoE path — every miss is paid on the critical
path, and the cache budget must be large enough that LRU recency alone keeps
the hit rate acceptable.

The dense flash path already solves the analogous problem for FFN neurons:
`LookaheadBank` (low-rank heads predicting layer L+1's active neurons from
layer L's hidden state) plus `Prefetcher` (background prediction + SSD I/O
overlapped with compute). This design ports that architecture to experts and
extends it with predictor-informed eviction.

## Goals

1. **Hide SSD I/O latency** — overlap expert loading for the next MoE layer
   with the current layer's load + compute.
2. **Reduce resident expert memory** — replace LRU victim selection with
   predicted-need scoring so `cache_budget_experts` can be lowered at equal or
   better hit rate.

Correctness is never at stake: prediction only warms the cache and picks
eviction victims. The synchronous `load_experts` path is unchanged and remains
the source of truth; a misprediction costs latency, never wrong output.

## Non-Goals

- Cross-token / multi-token-ahead prediction (next-token expert-set heads).
  Deferred; the prefetcher interface accommodates it later.
- Changes to the dense flash path or to `LayerLruCache` semantics for the
  dense store.
- Zero-training router-lookahead baseline (applying layer L+1's resident
  router to layer L's hidden state). Not built in v1; the offline eval in the
  training command makes a later comparison possible.

## Architecture

Four pieces, all under `engine/flash/`:

| Piece | File | Role |
|---|---|---|
| `MoeLookaheadBank` | `moe_predictor.py` (new) | Per-MoE-layer-pair low-rank heads: hidden state at MoE layer L → expert scores for the next MoE layer |
| Training command | `flash train-moe-lookahead` CLI subcommand | Record traces via a Flash-MoE-wrapped forward pass, train heads, save `<flash_dir>/moe_lookahead/` |
| `MoePrefetcher` | `moe_prefetch.py` (new) | Background predict + SSD prefetch, structural sibling of the dense `Prefetcher` |
| Scored eviction | `moe_weight_store.py` / `_ssd_base.py` | Cache victim = lowest predicted score instead of LRU-oldest, with LRU fallback |

### 1. Predictor — `MoeLookaheadBank`

One head per *consecutive MoE-layer pair* from `moe_layer_indices` — not
consecutive layer indices. Interleaved dense layers just add I/O lead time.

```
hidden (H) → Linear(H, rank≈128, no bias) → ReLU → Linear(rank, num_experts, no bias) → sigmoid
```

- **Input:** the hidden state entering MoE layer L's block (post-attention,
  pre-MLP — the `x` that `_FlashMoEBase.__call__` receives).
- **Target:** multi-hot top-k router selection at the next MoE layer.
- Same architecture family as `SparsityPredictor`; save/load/bank plumbing
  mirrors `LookaheadBank` with `num_experts` outputs instead of
  `intermediate_size`. Footprint ~130 KB/head at H=4096, rank=128, 256 experts.
- **Selection at inference:** top-`m` scored experts,
  `m = ceil(margin × num_experts_per_tok)`, margin configurable (default 1.5).
  Recall over precision: a false positive is one wasted SSD read; a false
  negative is a synchronous miss on the critical path.

### 2. Training — separate CLI step

`olmlx flash train-moe-lookahead` (precedent: `dflash` / `eagle` training
subcommands). Kept out of `moe-prepare` because bundling a 200 GB MoE is
expensive and predictors must be retrainable post-hoc; `moe-prepare` grows a
`--train-lookahead` convenience flag that chains both.

1. Load the model Flash-MoE-wrapped and lazy — the same kernel
   (`load_flash_moe_model`) used by Flash-MoE-backed KV-quant calibration.
2. Run calibration texts; record `(pre-MoE hidden state at L, router top-k at
   next MoE layer)` pairs. Targets come free from the resident routers.
3. Train each head with BCE + positive-class weighting (top-k/num_experts is
   ~3–6% positive; reuse the class-weighting machinery already in
   `prepare.py`'s dense lookahead training).
4. Save to `<flash_dir>/moe_lookahead/` next to the bundle, with a small JSON
   sidecar recording rank, margin default, layer-pair map, and training config.
5. **Offline eval baked in:** held-out recall@m per layer-pair printed at the
   end, so predictor quality is known before it touches serving.

### 3. Prefetch — `MoePrefetcher`

Structural sibling of the dense `Prefetcher`, reusing its concurrency pattern
verbatim: single dedicated prediction thread (`mx.eval` is materialized on the
calling thread first; prediction thread owns the only other eval), pending-map
with `wait()`, the issue-#242 in-flight guard, `PrefetchStats`-style counters.

Per decode step:

- `_FlashMoEBase.__call__` at MoE layer L calls `prefetcher.submit(L, x)`
  **before** its own synchronous `load_experts`. Prediction + SSD reads for
  the next MoE layer overlap with layer L's load, compute, and any intervening
  dense/attention work.
- At the next MoE layer: `wait()`, then the unchanged synchronous
  `load_experts`. Prefetched experts are cache hits; mispredictions fall
  through to the blocking path.
- `FlashMoeWeightStore.prefetch_experts(layer_idx, indices)` (new): read
  missing experts into the cache on the existing I/O pool — mirror of the
  dense store's `prefetch_neurons`.
- **Prefill guard:** skip prefetch when the position count exceeds a threshold
  (default 8). Long prefills activate most experts anyway; prediction work
  would be wasted.
- **Teardown:** prefetcher closes before the store (existing flash invariant,
  enforced in the model-manager unload path).

Gating: new `OLMLX_FLASH_MOE_PREFETCH` setting (default on); silently disabled
when `<flash_dir>/moe_lookahead/` is absent.

### 4. Predictor-informed eviction

`LayerLruCache` semantics stay untouched for the dense path. The MoE store
switches to a `ScoredLayerCache` subclass (in `_ssd_base.py`) that overrides
victim selection:

- After each prediction for the next MoE layer, the prefetcher pushes the
  score vector: `set_scores(layer_idx, scores)`.
- On insert overflow, the victim is the cached expert with the **lowest
  predicted score** among candidates *not part of the in-flight request*.
- Fallback to plain LRU order when scores are absent (first token, prefetch
  disabled) or stale.
- **Staleness:** scores are cleared once that layer's forward consumes them —
  a prediction from a previous token can never linger into the next.

This is the piece that lets `cache_budget_experts` shrink: the cache retains
what the predictor says the next pass needs, not what happened to be used
recently.

## Configuration

New settings (pydantic-settings, `OLMLX_` prefix):

- `flash_moe_prefetch: bool = True` — master switch (no-op without trained heads)
- `flash_moe_lookahead_margin: float = 1.5` — prefetch top-m multiplier
- `flash_moe_prefetch_max_positions: int = 8` — prefill guard threshold
- `flash_moe_scored_eviction: bool = True` — scored victim selection

Training hyperparameters (rank, epochs, calibration source) are CLI flags on
`train-moe-lookahead`, mirroring the dense predictor flags.

## Error Handling

- Prediction failure (exception on the predict thread): log warning, count in
  stats, `wait()` unblocks, forward pass proceeds synchronously. Same contract
  as the dense prefetcher.
- Missing/corrupt `moe_lookahead/` at load: log once, run with prefetch
  disabled. Never fail model load over an optional accelerator.
- Head/model shape mismatch (wrong num_experts or hidden size, e.g. stale
  predictors after re-bundling): detected at load via the JSON sidecar,
  prefetch disabled with a clear log line.
- Prefetch I/O failure: counted, logged, harmless — the synchronous path
  re-reads.

## Testing (TDD)

1. **Scored-eviction cache tests** — victim selection by score, LRU fallback
   when unscored, staleness clearing, in-flight-request protection.
2. **MoePrefetcher lifecycle/concurrency** — fake store, mirroring
   `test_flash_prefetch.py`: submit/wait ordering, pending-map cleanup,
   in-flight guard, close ordering.
3. **Head round-trip** — train on a synthetic router (known mapping), verify
   recall@m beats chance by a wide margin; save/load round-trip preserves
   predictions; non-contiguous `moe_layer_indices` pair mapping.
4. **Store `prefetch_experts`** — warms cache, dedups in-flight, counts stats.
5. **Metal-gated end-to-end smoke** — small real MoE: identical outputs with
   prefetch on vs off; hit-rate strictly higher with prefetch on.

## Success Criteria

- Expert cache hit rate (`ExpertCacheStats.hit_rate()`) rises at equal
  `cache_budget_experts`, and holds at a meaningfully reduced budget.
- Decode tok/s improves on an SSD-bound MoE model.
- Bit-identical outputs with prefetch on vs off.
- Offline recall@m (m = 1.5×k) materially above the trivial
  "same experts as layer L" baseline per layer-pair.
