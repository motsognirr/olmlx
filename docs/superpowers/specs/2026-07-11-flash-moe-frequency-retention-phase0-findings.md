# Phase-0 Findings — Frequency-Aware Expert Retention

**Date:** 2026-07-11
**Verdict:** **NO-GO.** Frequency retention (LFU + aging) does not meaningfully
beat LRU on this model. Do not implement Phases 1–3 as designed.

## What was measured

Recorded the exact per-MoE-layer fired-expert sequence during **greedy decode**
of the 7 flash-moe bench prompts on `mlx-community/Qwen3.5-35B-A3B-4bit`
(40 layers × 256 experts, 8 routed/token), via the real Flash-MoE path
(`record_moe_router_traces_decode`, single-stream — no GDN corruption). 1088
decode steps total. Routing is cache-independent, so this fired-expert stream is
*exactly* the demand stream any per-layer expert cache sees.

Then simulated LRU vs LFU+aging (EMA decay ∈ {0.90, 0.98, 0.99}) vs Belady-optimal
(oracle) per-layer caches at budgets {48, 64, 96, 128}, counting misses. A miss =
one expert (re)loaded and parsed into `mx.array`s — the exact cost retention aims
to avoid. Sim is calibrated against reality: its LRU miss rates (32% @48 → 9.4%
@128) match the prior +34% throughput from budget 48→128.

## Result 1 — routing is near-uniform (modest skew)

Mean `H/Hmax = 0.913` (1.0 = perfectly uniform). ~217–256 of 256 experts fire at
each layer across the run. The top-48 experts capture only ~55% of accesses on
average (vs 18.75% under perfect uniformity, vs 100% if only 48 were ever used).
Middle layers are more specialized (H/Hmax ~0.86, layers 8/20) than edge layers
(~0.96) — a clean, structured MoE pattern confirming the trace is faithful.

## Result 2 — LFU+aging barely beats LRU, and only where it doesn't matter

Miss-count reduction vs LRU (positive = fewer misses = better):

| budget | LRU miss-rate | best LFU (@0.9) | LFU vs LRU | Belady (oracle) vs LRU |
|-------:|--------------:|----------------:|-----------:|-----------------------:|
| 48     | 32.06%        | 30.07%          | **+6.2%**  | +48.0% |
| 64     | 24.50%        | 23.36%          | +4.6%      | +49.3% |
| 96     | 14.90%        | 14.56%          | +2.3%      | +49.4% |
| 128    | 9.44%         | 9.33%           | **+1.2%**  | +47.8% |

- The win shrinks as budget grows and is **~1% at budget 128** — the
  throughput-optimal operating point from the prior sweep (LRU 20.4 tok/s). That
  is well inside the measured ±1.5 tok/s (~7%) run-to-run noise.
- Decay 0.98/0.99 make LFU **worse** than LRU (−5% to −22%). The policy is fragile
  and its best case is small.
- Throughput estimate (prior data: ~1.5% tok/s per 1pp miss-rate cut): LFU's best
  case is ~+3% at budget 48 and ~0% at budget 128. Not shippable.

## Root cause (and why it generalizes)

Modern MoE training adds an **auxiliary load-balancing loss** that explicitly
pushes expert utilization toward uniform. Near-uniform utilization is precisely
the property that defeats frequency-based caching: historical frequency stops
predicting future use, so LFU collapses onto LRU. This is not specific to
Qwen3.5 — any well-load-balanced MoE will show it.

## The one interesting signal — the Belady gap

Belady-optimal cuts misses ~48% at every budget, so large headroom *exists* — but
it requires knowing the future access sequence. Neither frequency (this work) nor
the learned per-token predictor (prior PR #650, recall ~0.35) captures it. The
gap is real but only an oracle reaches it; no cheap online history signal does.

## Implication for the remaining levers

For any **online** policy, the miss *rate* on this workload is already close to
LRU-optimal (at budget 128, LRU 9.4% vs oracle 4.9%). Policy is nearly irrelevant
at the good operating point. The remaining honest levers are:

1. **Miss *cost*, not miss *rate* (Path C):** the rate is irreducible by policy,
   but each miss still costs `pread + parse`. Cutting per-miss cost speeds decode
   regardless of policy. Needs the deferred pread-vs-parse split measurement to
   size it.
2. **Budget / RAM:** the only knob that reliably moved throughput (48→128 = +34%).
   A cheaper expert representation (more experts resident per GB) attacks this.

## Path C measurement (chosen continuation) — miss cost is parse-bound

Instrumented the real decode path at budget 128 (`scratchpad/measure_misscost.py`,
61k expert reads):

- Per expert: **2082 µs = 511 µs pread (24.6%) + 1571 µs parse (75.4%)**.
- pread p50 = 157 µs for 1.69 MiB ≈ **10 GiB/s = page-cache/memcpy speed** — reads
  are served from RAM, not SSD. **This definitively kills the I/O-bandwidth
  theories** and explains why prefetch never helped: there was no I/O latency to
  hide. The bottleneck is the host-side `mx.array` construction, not the disk.
- The parse (`_parse_expert_with_manifest`) does `raw[pos:pos+n]` (slice copy) →
  `np.frombuffer(...).copy()` (copy #2) → `mx.array(arr)` (copy #3) = **~3 host
  memcpys per component × 9 components/expert**. The in-situ 1571 µs (vs 81 µs
  isolated) is 32-way thread contention → memory-bandwidth bound.

### Proven fix (micro-benchmark `scratchpad/bench_parse.py`, mx.eval forced)

| parse variant | µs/expert |
|---|---:|
| current (`.copy()`) | 81.6 |
| **memoryview, no `.copy()`** | **34.2 (2.4×)** |
| single-buffer on-device views | 67.2 |

Dropping the redundant `.copy()` + bytes-slice (use a `memoryview`, let `mx.array`
do the one necessary host→MLX copy) is **2.4× faster in isolation** and should
help *more* in-situ (relieves the memory-bandwidth contention). **Safe:** the
current code's own correctness proves `mx.array` copies at construction (its local
numpy `.copy()` is GC'd before the deferred eval), so it never retains the view.

### Plan

1. Rewrite the three `_parse_*` methods (`moe_weight_store.py:238–363`) to parse
   from a `memoryview` without the intermediate `.copy()`. TDD: assert parsed
   arrays are bit-identical to current output; micro-bench confirms the speedup.
2. A/B end-to-end throughput at budgets {48, 64, 128} on the flash-moe bench.
   Expected: modest gain at 128 (read+parse is ~6% of decode wall there), larger
   at low budgets (misses 3.4× higher) → the **RAM win** (budget 48–64 matching
   higher-budget throughput).

## Reusable artifacts

- `scratchpad/record_experts.py` — faithful decode expert-trace recorder (any
  prepared flash-moe model).
- `scratchpad/analyze_skew.py` — skew stats + LRU/LFU/Belady miss simulator.
- `scratchpad/expert_traces.npz` — the recorded 1088-step trace.
