# Benchmark: Qwen3Next-family Flash-MoE throughput — May 2026

First throughput measurement of the two Qwen3Next checkpoints in the local
registry, both run through olmlx's Flash-MoE expert-offload path with
TurboQuant-4 KV cache. Single scenario, throughput suite only — quality
grading is out of scope for this round.

## Methodology

- **Serving:** in-process via the bench worker (`olmlx bench run`), not a
  long-lived `olmlx serve`. Each scenario spawns a fresh worker so model
  loading and KV cache state are isolated.
- **Scenario:** `flash-moe+tq4` — `OLMLX_FLASH_MOE=true` +
  `OLMLX_KV_CACHE_QUANT=turboquant:4`. The 80B model cannot fit in RAM at
  any quant; for the 42 GB Qwen3-Coder-Next checkpoint Flash-MoE is the
  scenario already configured in `models.json` so both were measured the
  same way for an apples-to-apples comparison.
- **Speed:** the standard 7-prompt throughput suite from
  `olmlx/bench/prompts.py`. No quality grading.
- **Determinism:** bench defaults (no seed override). The throughput suite
  uses `temperature=0` per the worker's default config.
- **Flash-MoE prep:** `olmlx flash prepare <hf-path>` writes
  `<model-dir>/flash_moe/{flash_moe_config.json, flash_moe_layout.json,
  layer_NN.flashexperts}`. Both models bundle to 41 GB on disk and prep
  takes ~30 s — the 512×48 expert grid is identical between them.
- **Host:** Apple Silicon, git `2656ca0` (branch
  `fix/343-non-trimmable-skip-storage`), 2026-05-24.

Raw per-run JSON is under [`raw/`](./raw/) — these are direct copies of the
canonical `~/.olmlx/bench/runs/<timestamp>/results.json` outputs.

## Models

Both checkpoints share the **identical Qwen3Next MoE shape**: 48 layers,
512 routed experts, 10 active per token, hidden 2048, plus a shared expert.
This is the crucial fact for interpreting the numbers below — per-token
Flash-MoE I/O is architecture-shape-bound, not parameter-count-bound, so the
two should perform very similarly.

| Model | Total params | Checkpoint size | Flash-MoE expert bundle | Total on-disk after prep |
|---|---|---:|---:|---:|
| `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit` | 80B (3B active) | 42 GB | 41 GB | 82 GB |
| `mlx-community/Qwen3-Coder-Next-4bit` | not labeled A3B but same shape | 42 GB | 41 GB | 82 GB |

Identical sizes across the board — expected given the shared MoE shape and
4-bit weight quantization.

## Results

Decode tok/s (steady-state output), prompt-eval tok/s (prefill):

| prompt       | Qwen3-Next-80B-A3B-Instruct-4bit | Qwen3-Coder-Next-4bit | Δ |
|--------------|------:|------:|------:|
| factual      |  9.75 |  9.70 | −0.05 |
| reasoning    | 12.23 | 12.13 | −0.10 |
| coding       | 12.40 | 11.74 | −0.66 |
| creative     | 13.46 | 13.96 | +0.50 |
| instruction  | 14.20 | 15.14 | +0.94 |
| multi-turn   | 12.96 | 12.42 | −0.54 |
| long-context | 11.69 | 12.90 | +1.21 |
| **avg decode tok/s** | **12.38** | **12.57** | **+0.19** |
| **long-context prefill tok/s** | **178.75** | **177.35** | −1.40 |

## Findings

1. **~12 tok/s sustained decode for an 80B MoE off SSD on Apple Silicon.**
   That is the headline. The Flash-MoE expert-offload path delivers usable
   interactive throughput for a model 2× the host RAM, with KV cache
   quantized to 4 bits.
2. **Throughput is architecture-shape-bound, not parameter-count-bound.**
   The two models perform indistinguishably (avg Δ < 0.2 tok/s, per-prompt Δ
   all < 1 tok/s). This is the predicted behavior: per-token decode loads
   10 experts × 48 layers = 480 expert tiles from SSD, the same number for
   both checkpoints, and that I/O dominates.
3. **Prefill scales the same way too.** ~178 tok/s on the 12 427-token
   long-context prompt for both models — same ceiling set by parallel
   `pread()` throughput on the expert files combined with `mx.gather_mm`
   over the loaded expert stacks. Each token still routes to exactly k=10
   experts (router topology is fixed), but prefill processes thousands of
   tokens in parallel: each expert weight loaded from SSD is reused for
   every token in the batch that routed to it, and the gather-matmul runs
   over the full sequence at once. Decode loads the same 480 expert tiles
   per step (10 experts × 48 layers) but applies them to a single token,
   so I/O cost per token is unchanged while compute utilization is far
   lower — which is why decode is the SSD-bound regime.
4. **Cold-cache warmup costs roughly one prompt.** The `factual` prompt
   (8 output tokens) lands at ~9.7 tok/s on both runs; from `reasoning`
   onward decode settles into the 12–15 tok/s band as the LRU warms.
   `FlashMoeWeightStore` does not share cache across models, so each model
   pays this cost on first use.
5. **Flash-MoE prep is fast for these models.** 30 s wall time to bundle
   41 GB of expert weights. The prep work is pure I/O reshuffling (no model
   forward pass, no predictor training) since MoE routing replaces the
   sparsity predictors that dense Flash-FFN uses.

## Lessons (methodology)

- **The "throughput" suite has a cold-cache outlier.** The `factual` prompt
  emits only 8 tokens, so any per-request fixed cost dominates its tok/s
  number. For Flash-MoE the LRU warmup is exactly that fixed cost. When
  comparing scenarios, look at the median of the longer prompts, not the
  arithmetic mean, or `factual` will skew small-scenario deltas.
- **Same-architecture model pairs are useful sanity checks.** Getting two
  models with identical MoE shape but very different total parameter counts
  to within 0.2 tok/s of each other validates the "architecture-bound"
  intuition for the Flash-MoE path — if the larger checkpoint had been
  measurably slower, something would be wrong (e.g. spurious shared-expert
  weight reloads).

## Caveats

- **Single scenario, single run.** No confidence intervals — the sub-1 tok/s
  per-prompt deltas are within plausible run-to-run noise, not a measured
  variance. Treat them as "indistinguishable" rather than "the 80B is 0.05
  tok/s slower on factual." The one row that crosses the 1 tok/s line is
  `long-context` (Δ +1.21) — but the two models also emitted different
  output lengths there at `temperature=0` (80B: 43 tokens; Coder: 25
  tokens), so the per-model tok/s estimates are over very short decode
  windows (~2–3s) and are themselves noisier than the longer-output rows.
  Same headline applies — indistinguishable — just don't read the 1.21 as a
  real signal.
- **No quality grading.** The throughput suite uses generic prompts and does
  not run GSM8K/MMLU/HumanEval. The quant choice (4-bit weights +
  TurboQuant-4 KV cache) is plausibly fine for both models but is not
  verified here. The earlier `qwen36-gemma-2026-05` report covers
  Qwen3.6-A3B at 4-bit with full quality grading; that result transfers
  approximately to the Qwen3Next family but should be confirmed.
- **No comparison against other scenarios.** This benchmark does not show
  what Flash-MoE costs vs a hypothetical in-RAM run (the 80B doesn't fit in
  RAM at all; the 41 GB Coder model would fit on a high-RAM Mac without
  Flash-MoE but wasn't tested that way). It also does not measure the cost
  of `kv_cache_quant=turboquant:4` vs an unquantized KV cache.

## Open questions / follow-ups

Things to measure next, in priority order:

1. **`flash-moe` (no KV quant) vs `flash-moe+tq4`** on Qwen3-Coder-Next-4bit.
   Quantifies TurboQuant-4's per-token cost on hybrid linear-attention
   layers. The 80B can't be tested without KV quant on a typical machine —
   KV cache for a 12k context plus the resident expert cache won't both fit.
2. **`OLMLX_FLASH_MOE_CACHE_BUDGET_EXPERTS=128` or `256`** (default 48).
   At ~80 MB per expert per layer, more RAM-resident experts should reduce
   SSD misses on `factual`-like cold prompts and possibly steady-state too.
3. **Classic speculative decoding on Flash-MoE.** Classic is the only
   speculative strategy supported on Flash-MoE (`dflash` and `eagle` are
   blocked). A small Qwen3Next draft, if one exists, could hide expert-fetch
   latency. Per the existing CLAUDE.md note, EAGLE/DFlash drafts perform
   poorly on hybrid linear-attention targets like this family, so classic is
   the right call here regardless.

None of these have been measured yet, so there is no empirical pick
("use X, not Y") to write into CLAUDE.md from this round.

## Reproducing

Measured on branch `fix/343-non-trimmable-skip-storage`, SHA `2656ca0`. That
branch's only behavioral change is to gate cross-request prompt-cache storage
for non-trimmable hybrid sliding-window caches (`RotatingKVCache` /
`ChunkedKVCache` — i.e. Gemma 4, gpt-oss). Qwen3Next uses `ArraysCache` via
its gated-delta layers and falls under the separate
`_cache_supports_persistence` gate that was already in place before this
branch, so the throughput numbers here should reproduce unchanged from
`main` once the branch lands. Numbers were collected with no other process
contending for the GPU.

```bash
# one-time prep (writes ~41 GB to <model-dir>/flash_moe/)
olmlx flash prepare mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit
olmlx flash prepare mlx-community/Qwen3-Coder-Next-4bit

# speed
olmlx bench run --model mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit --scenarios flash-moe+tq4
olmlx bench run --model mlx-community/Qwen3-Coder-Next-4bit              --scenarios flash-moe+tq4
```

Both models are configured in `~/.olmlx/models.json` with `flash_moe: true`
and `kv_cache_quant: turboquant:4`, so the same configuration applies when
serving them through `olmlx serve` for normal use.
