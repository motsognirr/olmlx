# #503 Phase 0 — Prefill measurement findings

**Status:** measured 2026-07-19. Prefill confirmed as the dominant cost on a cold
big agentic prompt; **recommend deferring** draft-assisted prefill (no lossless
headroom) and treating prompt-cache reuse as the already-shipped win. One bug in
the Phase-0 instrumentation surfaced (fresh/covered mis-attribution on the
checkpoint path) — see Caveats.

## Setup
- Model: `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` — a `qwen3_5` hybrid
  linear-attention model, 1M context, plain KV (non-speculative,
  non-quantized cache). Chosen because it reliably holds 69k tokens and
  prefills in a bounded time; the configured 32B agentic coder model
  (Qwen3-32B-4bit) has a ~40k context ceiling and would truncate the prompt,
  and Qwen3.5-27B-4bit + turboquant prefilled at only ~115 tok/s (>10 min,
  timed out).
- Prompt: `agentic-69k` (tokenized to **69,445 tokens**, tool-defs system
  segment + 4-message transcript), `num_predict=32`.
- Driver: local server on the merged Phase-0 code (`olmlx serve`), `POST
  /api/chat` with `stream:true`. Numbers read from the server log's measured
  `prefill Xs (fresh F/N tok, cache-covered C)` line and the `/api/chat`
  response stats. `olmlx bench run` was **not** used: its CLI exposes only
  `--prompt-set {throughput,quality,all}` (no per-prompt `--prompts` filter,
  contrary to the plan's assumed command), and the measured value lives in the
  log/span, not the bench results table.

## Numbers

| metric | cold (cache-miss) | warm (identical prefix, cache-hit) |
| --- | --- | --- |
| **measured prefill wall-clock** (`ttft`, Task 1) | **190.17 s** | **0.56 s** |
| decode wall-clock (32 tokens) | ~2.5 s (12.9 tok/s) | ~2.5 s |
| total (`total_duration`) | 192.68 s | 3.06 s |
| **prefill / (prefill+decode)** | **98.7 %** | ~18 % |
| real prefill throughput | 69,445 tok / 190.17 s = **365 tok/s** | — |
| prompt-cache prefill speedup | — | **340×** (190.17 s → 0.56 s) |
| target_lane_ns / draft_lane_ns (spec-only, Task 2) | not exercised (model is non-speculative) — see Follow-ups |

**Heuristic vs measured:** for the same cold run the *reconstructed*
`prompt_eval_duration` reported **0.15 s** (and a bogus "6.7 tok/s / 1 prompt
token" in the completion log) versus the **~190 s** the request actually spent
in prefill — the heuristic is wrong by **>1000×** on this path. This is the
concrete justification for the measured `ttft` Phase 0 added: the pre-existing
split was not just imprecise, it was unusable for deciding #503.

## Caveats (bugs the measurement exposed)

- **`fresh`/`cache-covered` mis-attribution on the checkpoint/deferred-prefill
  path.** Both runs logged `fresh 1/69445 tok, cache-covered 69444` — including
  the **cold** run, where the cache was empty (server had just restarted; the
  190 s wall-clock proves all 69,445 tokens were freshly prefilled). On hybrid
  GDN / `qwen3_5` models the prefill is driven by `_drive_segmented_prefill`
  (deferred), so mlx-lm's stream sees only the 1-token seeding forward and
  reports `token.prompt_tokens == 1`. Task 1 computes `_fresh = token.prompt_tokens`,
  so it attributes the deferred-prefilled tokens to "cache-covered" and would
  set the `cache_hit` span attr True on a cold miss. **The measured wall-clock
  itself is correct and is the reliable signal** (190 s cold vs 0.56 s warm
  distinguishes miss from hit unambiguously); only the fresh/covered *split* is
  unreliable on this path. Fix (follow-up): derive fresh/covered from the
  cache-reuse decision (`already_covered` / the prompt-cache offset) rather than
  from mlx-lm's post-hoc `token.prompt_tokens`. Low risk, log/span-only.

## Decision

- **Is prefill the measured bottleneck on the agentic case?** **Yes, decisively**
  — 98.7 % of cold compute wall-clock, and the absolute cost is large (190 s for
  69k tokens even on a 9B; it scales with model size and was >10 min on a 27B).
- **Build #503 proper / scope down / defer? → Defer the lossless build; the
  already-shipped prompt cache is the real win; only an *approximate* strategy
  could cut the residual cold cost.** Rationale:
  1. Prefill is compute-saturated (365 tok/s = the GPU doing dense matmuls over
     every token), so — as Phase 0's premise held — there is **no lossless
     "speculative prefill" headroom**: every prompt token's KV must be computed
     to decode correctly, and there is no free batch dimension or second GPU.
  2. The dominant *repeated-prefix* cost is **already** eliminated by prompt-cache
     reuse: 190 s → 0.56 s (340×). In real agentic multi-turn the stable
     system+history prefix is cache-covered after turn 1; only the growing
     suffix is freshly prefilled. So the 190 s cold number is the **first-turn /
     cache-evicted** case, not the steady state.
  3. What remains — the cold first-turn 69k prefill — can only be reduced by an
     **approximate** token-pruning strategy (SpecPrefill-style: a cheap pass
     scores prompt-token importance, target prefills only the important subset).
     That alters the output distribution (a sibling to `proxy_tuning`, not an
     exactness-preserving speculative strategy) and is a much larger commitment.
  4. **Recommendation:** do not build draft-assisted prefill now. Revisit only
     if cold first-turn prefill latency on big agentic prompts becomes a
     measured user pain that prompt-cache reuse cannot cover (e.g. frequent
     cache eviction, or first-turn latency SLOs). If revisited, scope it as the
     approximate-pruning direction behind an explicit opt-in, with its own spec.

## Follow-ups
1. Fix the fresh/cache-covered mis-attribution on the checkpoint path (Caveats)
   — derive from the reuse offset, not `token.prompt_tokens`. Small, log/span-only.
2. If a per-lane (target vs draft) prefill number is wanted, run a
   **classic-draft or PLD** speculative config (Task 2's breakdown only
   populates on `_drive_spec_prefill`, i.e. classic/PLD on the reuse path) with
   `OLMLX_LOG_LEVEL=DEBUG` to capture the `spec prefill breakdown [...]` line.
   Not needed to answer the Phase-0 question.
3. Optional: adopt the measured `ttft` as the client-facing
   `stats.prompt_eval_duration` (currently the >1000×-wrong heuristic reaches
   the Ollama API / bench table) — its own PR, per the design's deferred note.
