# Draft-assisted prefill (#503) — Phase 0: measure before building

## Context

Issue #503 proposes **draft-assisted / speculative prefill**: use a draft model (or
a cheap heuristic) to skip, batch, or overlap portions of the target prefill on long
agentic prompts (the ~69k-token case from the speculative prefill-cancellation notes).
The issue itself gates the work: *"Worth it once prefill latency on big agentic prompts
is the measured bottleneck."*

Two facts make measurement the correct first step rather than jumping to an
implementation:

1. **No lossless headroom on a single Apple GPU.** Prefill is compute-bound and already
   saturated — one big batched matmul over all N prompt tokens. Unlike decode
   (memory-bandwidth-bound, where speculation verifies K draft tokens for ~the cost of
   one forward because the batch dimension is free), prefill has no free batch dimension
   and no second GPU to overlap onto. Every prompt token's KV must be computed to decode
   correctly, so the decode-style *exact* speculation trick has essentially zero
   theoretical win here. Real wins would require an *approximate* (output-altering)
   token-pruning approach — a much larger commitment we should not make on an unmeasured
   hunch.

2. **Today's prefill/decode split is an estimate, not a measurement.** `prompt_eval_duration`
   is *reconstructed* in `_derive_timing_stats` (`inference.py`) from mlx-lm's reported
   `prompt_tps`/`gen_tps` rates split out of a single combined `eval_timer_ns`, with
   50/50 fallbacks when a rate is unknown. The `ttft_ns` span attribute
   (`inference.py` ~3785) is then set from that *derived* value — its own comment
   concedes "the prefill forward runs lazily inside the stream's first iteration
   (worker thread), so the prefill span can't time it." No real first-token timestamp is
   ever captured.

**Phase 0 goal:** produce a *real, measured* prefill wall-clock and a within-prefill
breakdown for the ~69k agentic case, reproducibly, so the build-or-defer decision on
#503 proper rests on numbers. Phase 0 changes **no** prefill algorithm.

## Current prefill paths (what we are measuring)

There are three worker-thread prefill paths; the measured-TTFT primary is path-agnostic,
and the breakdown instrumentation targets the two explicit drive sites:

- **Flat mlx-lm** (`async_mlx_stream`): dense / non-checkpoint models. Prefill happens
  inside mlx-lm's `generate_step`; we do not touch its internals.
- **Segmented checkpoint** (`_drive_segmented_prefill`, `inference.py`): hybrid-GDN /
  message-boundary reuse path, driven via the `deferred_prefill` closure on the
  generation worker.
- **Speculative** (`_drive_spec_prefill`, `speculative.py`): fills each lane
  (target + draft) **serially** over the prompt — the one place the draft-lane cost the
  issue worries about is directly measurable.

All three run start-to-finish on one generation worker thread and materialise cache
state per sub-chunk via `_eval_cache` (an `mx.eval`), so a wall-clock `Timer` at these
sites is accurate without any added synchronisation.

## Component 1 — Measured TTFT (primary, path-agnostic)

In the streaming generate loop (`inference.py`, the `async for token in stream` block
around line 3688):

- Capture a `time.perf_counter_ns()` timestamp on the **first** loop iteration.
  `ttft_measured_ns = first_token_ts − prefill_start_ts`. Because the lazy prefill
  forward is forced inside the stream's first iteration on the worker thread,
  time-to-first-token **is** the prefill wall-clock, and it is identical across all
  three paths.
- Set the existing `_decode_span` `ttft_ns` attribute to the **measured** value
  (replacing the heuristic-sourced value at ~line 3785).
- Emit one concise log line, e.g.
  `prefill 4.2s (fresh 61k/69k tok, cache-covered 8k)`. Fresh-vs-covered is derived from
  `token.prompt_tokens` (what mlx-lm re-prefilled) vs `len(full_prompt_tokens)` (full
  request) — already computed for the `cache_hit` attribute, so it costs nothing extra.

**Decision (a):** the client-facing `stats.prompt_eval_duration` (Ollama API,
`olmlx bench` table) **stays on the existing heuristic in Phase 0**, to keep the blast
radius minimal. Adopting the measured value there is an obvious, isolated follow-up
noted below — deliberately deferred so the measurement change cannot perturb existing
API/bench output.

## Component 2 — Within-prefill breakdown (secondary)

Lightweight `Timer`s + trace-span attributes at the two explicit worker-thread drive
sites (no extra sync — each sub-chunk already `_eval_cache`s):

- **`_drive_spec_prefill`** (`speculative.py`): per-lane ns (`draft_lane_ns`,
  `target_lane_ns`), `n_chunks`, `covered_tokens`, `fresh_tokens` — set on the existing
  `spec.prefill` span. The serial lane fill means these sum to the drive wall-clock.
- **`_drive_segmented_prefill`** (`inference.py`): `drive_ns`, `n_chunks`,
  `covered_tokens`, `fresh_tokens`, `boundary_depth` — for hybrid-GDN / segmented
  targets.

These attributes are additive; nothing consumes them programmatically yet — they are for
reading during the profiling exercise.

## Component 3 — Reproducible driver: `agentic-69k` bench prompt

Add a deterministic ~69k-token agentic `BenchPrompt` to `olmlx/bench/prompts.py`:

- New category `agentic`.
- A realistic **system prompt containing tool definitions** + a multi-turn tool-call
  conversation + embedded code context. The tool-defs system segment is exactly what
  creates the message boundary the issue flags (and what the pure-RotatingKVCache /
  checkpoint paths key on).
- **Decision (b):** the body is **synthetic and deterministically padded** to ~69k
  tokens (repeated realistic code/log blocks truncated to a fixed length, mirroring the
  existing `_LONG_CONTEXT_BODY` approach) — reproducible across runs, no PII, safe to
  commit. Not built from a real captured transcript.
- `max_tokens` small (e.g. 32) so prefill dominates the sample.

Driven via `olmlx bench run --prompts agentic-69k`.

## Testing (TDD, all Metal-free)

Write failing tests first:

- **Measured-TTFT capture:** a fake async stream that `await asyncio.sleep(δ)` before
  yielding its first token → assert the recorded `ttft` ≥ δ and that the span attr / log
  reflects the *measured* value, not the heuristic `prompt_eval_duration`.
- **`_drive_spec_prefill` lane timing:** stubbed target + draft models (each a callable
  recording invocation) → assert `draft_lane_ns` and `target_lane_ns` are both recorded
  and consistent with the serial fill (target-only when no draft lane).
- **`agentic-69k` prompt:** deterministic across constructions; tokenizes to ~69k tokens
  (± tolerance) with a reference tokenizer; `to_dict`/`from_dict` round-trips; category
  is `agentic`.

## Out of scope (Phase 0)

- No change to any prefill algorithm; no draft-assisted skipping, batching, pruning, or
  overlap.
- No rewire of the client-facing `stats.prompt_eval_duration` (see Decision (a)).
- The decision to build, scope down, or defer #503 proper is made **after** reading the
  Phase-0 numbers, and recorded as a short findings note (mirroring
  `2026-07-11-flash-moe-frequency-retention-phase0-findings.md`).

## Follow-ups (explicitly deferred)

- Adopt the measured TTFT as `stats.prompt_eval_duration` across the non-batched text
  path (improves the Ollama-API and bench-table accuracy) — isolated change, its own PR.
- If the numbers justify it: the approximate token-pruning direction, as a separate
  non-exactness-preserving strategy (sibling to `proxy_tuning`), with its own spec.

## Deliverable

`olmlx bench run --prompts agentic-69k` against the agentic model → read the measured
prefill wall-clock + breakdown → write the findings note → decide.
