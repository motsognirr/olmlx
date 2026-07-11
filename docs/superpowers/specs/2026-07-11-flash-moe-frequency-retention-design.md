# Flash-MoE Frequency-Aware Expert Retention — Design

**Date:** 2026-07-11
**Status:** Design (pre-implementation)
**Author:** Daniel Palmqvist (with Claude)

## Problem

Predictive expert prefetching (the `MoeLookaheadBank` / `MoePrefetcher` path in
`engine/flash/`) has never beaten a plain LRU expert cache on the canonical
flash-MoE target `mlx-community/Qwen3.5-35B-A3B-4bit` (35B-total / 3B-active MoE,
40 MoE layers × 256 experts, 8 routed/token, 17 GB SSD bundle, 32 GB machine).

Across ten benchmarked configurations (7-prompt `flash-moe` throughput set), every
prefetch variant lost **30–45%** decode throughput vs LRU, with high run-to-run
variance, while LRU reproduced exactly. The decisive observation: tripling real
decode predictor recall (0.11 → 0.28) changed throughput **not at all**. The
dominant knob was cache budget alone (48 → 128 experts/layer gave LRU **+34%**).

### Root-cause reframing

Two grounded facts from the current implementation reframe why prefetch loses:

1. **Prefetch and demand reads share one I/O pool.** `MoePrefetcher._do_prefetch`
   → `weight_store.prefetch_experts` dispatches onto the store's own
   `ThreadPoolExecutor` (`moe_weight_store.py:220`) — the same 32-thread pool the
   blocking demand path uses. Speculative reads therefore contend directly with
   demand reads for SSD bandwidth. This is a zero-sum game, and it explains the
   observed −45%.

2. **The OS page cache is not bypassed at serve time.** Expert files are opened
   with `bypass_cache=False` (`moe_weight_store.py:144`, `_ssd_base.py:61`). On a
   32 GB machine with a 17 GB bundle, a large fraction of experts live in the OS
   page cache, so a *RAM-cache miss* is frequently a page-cache copy plus a
   re-parse into `mx.array`s — **not** an SSD read. The real cost of an eviction is
   often `pread (page-cache) + parse/alloc`, not raw SSD bandwidth. This is
   consistent with the +34% from a larger parsed-array budget.

**Conclusion:** the lever is not better *prediction* (fetch bytes early) but better
*retention* (keep the parsed hot set resident). MoE routing is heavily skewed — a
minority of experts fire disproportionately often — yet plain LRU evicts a hot
expert after one cold window and then pays to re-read/re-parse it. A
frequency-aware retention policy keeps it, at **zero speculative I/O**. Its worst
case is a tie with LRU (cold-start / no-skew), versus prefetch's −45% worst case.

## Goal

Beat LRU **by any means** on the flash-MoE decode throughput bench, using a
retention policy that spends no speculative bandwidth. The neural predictor is out
of scope for this design — it stays where it is and stays off.

## Non-goals

- No changes to the demand read path's correctness or the gather-matmul dispatch.
- No revival or retraining of `MoeLookaheadBank`.
- No new speculative reads of any kind.
- No change to the Metal-stream / inference-lock handling.

## Approach: LFU + aging via the existing scored-eviction seam

### Hook points (all pre-existing)

- **Frequency state** lives in `FlashMoeWeightStore` (`moe_weight_store.py:123`): a
  new per-layer EMA map `freq[layer_idx][expert_idx] -> float`, updated on every
  *demand* `load_experts` call (`moe_weight_store.py:378`) from the `unique_experts`
  the router actually requested. Pure observation of demand traffic.
- **Eviction** reuses `ScoredLayerCache._pick_victim` (`_ssd_base.py:303`), which
  already evicts the minimum-score entry with an LRU tiebreak. We change only the
  *score source*: instead of the neural bank's consume-once predicted scores
  (pushed via `set_layer_scores`, `moe_weight_store.py:201`), we feed the persistent
  EMA frequencies before each demand load's `put()` cascade.
- **One config flag** — `flash_moe_freq_eviction: bool` (config.py, default
  `False`), distinct from the existing `flash_moe_scored_eviction` (which remains
  the predictor mode). The two are mutually exclusive; freq mode wins if both set,
  with a one-time warning.

### The policy

For layer `L` with per-layer budget `B` (`flash_moe_cache_budget_experts`):

- **Frequency signal (EMA over decode steps).** On each demand load of layer `L`
  with requested unique experts `E`:
  ```
  for e in all experts touched at L so far:
      hit = 1.0 if e in E else 0.0
      freq[L][e] = decay * freq[L][e] + (1 - decay) * hit
  ```
  In practice we update lazily — only experts currently resident plus those in `E`
  need scores — and apply the decay multiplicatively per step to resident entries.
  `decay` is a config knob (`flash_moe_freq_decay`, default ~0.98): higher = longer
  memory, adapts slower to conversation-level routing shifts.

- **Eviction.** When `put()` overflows `B`, `_pick_victim` evicts the resident
  expert with the **lowest** `freq[L][e]`, LRU-oldest as tiebreak (the existing
  tiebreak, unchanged).

- **Aging** is intrinsic to the EMA: an expert that stops firing decays toward 0 and
  becomes evictable, so stale-but-once-hot experts cannot ossify.

- **Insertion grace.** A just-inserted expert is protected from eviction for the
  load in which it was admitted — this already exists via `protect()`
  (`moe_weight_store.py:389`, `_ssd_base.py:284`) for in-flight experts of the
  current batch, which covers the LFU "never warms up" failure mode for one step.
  If the bench shows premature eviction of newly-hot experts, extend the grace to a
  small fixed window (`flash_moe_freq_grace_steps`, default 1); noted as a tuning
  lever, not built up front (YAGNI).

- **Cold start.** Before any skew emerges, frequencies are ~uniform, so the min-score
  victim reduces to the LRU-oldest tiebreak — i.e. **freq mode degrades gracefully to
  LRU**. This bounds the worst case to a tie.

## Instrumentation spike (Phase 0 — gates everything)

Runs and is analyzed *before* any policy work. Two decisive measurements:

1. **Skew probe — the go/no-go.** Over a decode bench, accumulate per-layer expert
   access counts and report top-k concentration and normalized entropy per MoE
   layer. **If decode routing is not meaningfully skewed, frequency retention cannot
   beat LRU — we stop and report that honestly** rather than running another
   inconclusive sweep.

2. **Miss-cost split.** Instrument `_read_expert` (`moe_weight_store.py:184`) to
   separate:
   - a **byte counter** (`layout.expert_byte_size` per read),
   - **pread wall time** vs **`mx.array` parse/alloc wall time**, and
   - an SSD-vs-page-cache signal via latency bucketing (page-cache preads are
     microseconds; SSD preads are ~100 µs–ms — a bimodal histogram separates them).

   These extend `ExpertCacheStats` (`moe_weight_store.py:77`) and surface as new
   Prometheus series alongside the existing `olmlx_flash_expert_cache_events_total`
   (`metrics.py:297`). They tell us *what* an avoided eviction actually saves
   (bandwidth vs parse), which sizes the achievable ceiling.

Instrumentation is behind the same low-overhead counter pattern already used by
`ExpertCacheStats`; latency timing is gated by a cheap flag so the default hot path
is unaffected.

## Success criteria

Benchmarked on the 7-prompt `flash-moe` throughput set with a minimal models config
(exclude dflash/turboquant entries), LRU as the reproducible baseline. Report a
**budget × policy** matrix:

- **Equal-budget win:** freq-retention beats LRU at the same budget. Largest room at
  budget 48 (highest eviction pressure); budget 128 already caches half of 256
  experts so there is little to save.
- **RAM win (likely the more valuable one):** freq-retention at budget 48–64
  *matches* LRU at 128 → equal throughput at ~half the resident expert RAM, which on
  a 32 GB machine buys bigger models / more KV cache.

A result is shippable if freq-retention is **≥ LRU at equal budget on every cell**
(never worse — the tie floor must hold) and wins on at least one of the two framings
above. If Phase 0 shows no skew, the shippable outcome is the negative report plus
the reusable instrumentation.

## Testing

- **Unit — EMA + victim selection.** Feed the tracker a synthetic skewed access
  stream; assert hot experts survive eviction where plain LRU would drop them.
  Assert cold-start (uniform frequencies) reproduces LRU eviction order exactly
  (the tie-floor guarantee).
- **Unit — decay/aging.** Assert an expert that stops firing decays below a
  competitor and becomes the victim within the expected number of steps.
- **Unit — mode exclusivity.** `flash_moe_freq_eviction` and
  `flash_moe_scored_eviction` both set → freq wins + one-time warning; predictor
  score path untouched when freq mode off.
- **Integration — bench matrix.** The budget × policy sweep above. LRU determinism
  (established in prior runs) makes the deltas trustworthy.

## Phasing

- **Phase 0 — Instrumentation + skew probe.** Byte/latency/parse counters + skew
  report. Run once. **Go/no-go gate.**
- **Phase 1 — EMA frequency tracker** in `FlashMoeWeightStore`, updated on demand
  loads. Unit-tested in isolation (no eviction wiring yet).
- **Phase 2 — Frequency-scored eviction** wired into `_pick_victim` behind
  `flash_moe_freq_eviction`; mode exclusivity + config plumbing
  (`config.py` → `FlashMoeConfig` → `registry.py` → `model_manager.py` →
  `wrap_flash_moe`).
- **Phase 3 — Bench matrix + analysis.** Decide equal-budget vs RAM-win framing
  from data; update `CLAUDE.md` flash-MoE section and the config docstring
  (`config.py:437`) with the outcome.

## Risks & open questions

- **No skew at decode** → approach dies at Phase 0. Cheap to discover; that is the
  point of gating on it.
- **Budget-128 headroom is small.** If the win only appears at low budgets, the
  RAM-win framing carries the result; make sure the bench covers 48/64/96/128.
- **EMA decay tuning could become another sweep.** Mitigation: fix a sensible
  default (0.98), treat decay/grace as single-shot tuning only if Phase 3 is
  borderline — do not open a large hyperparameter search.
- **Interaction with prefetch.** Out of scope; prefetch stays off. Freq mode and
  predictor-scored mode are mutually exclusive by construction.
