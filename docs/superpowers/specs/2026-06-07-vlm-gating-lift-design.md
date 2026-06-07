# Lift VLM-path gating: prompt cache, grammar, speculative (#429)

**Date:** 2026-06-07
**Issue:** #429 — Lift prompt-cache / grammar / speculative gating on the VLM (image) path
**Status:** Design approved; pending implementation plan

## Problem

Three engine features are gated off whenever a request carries images:

1. **Prompt caching** — multi-turn vision chats re-prefill the (often large) image
   prefix every turn.
2. **Structured outputs / grammar** — `response_format` / `format` is silently
   skipped with images.
3. **Speculative decoding** — skipped with images.

The gating rationale in CLAUDE.md (and the issue) is **stale**. The installed
`mlx_vlm 0.4.4` `generate_step` / `stream_generate` already accept `prompt_cache`,
`logits_processors`, and `sampler`, and `stream_generate` ships a native
`PromptCacheState` cross-turn reuse mechanism (with correct image-in-prefix
handling) plus an optional `vision_cache`. Most of the work is **removing stale
olmlx-side gates and wiring mlx_vlm's own machinery**, not building new inference
plumbing.

A latent bug compounds this: the streaming VLM path is *already allowed* to use the
prompt cache (`(stream or not lm.is_vlm)` gate), and `_setup_prompt_cache` has VLM
`input_ids` branches — but those pass *text-only* `input_ids` to mlx_vlm, which then
treats `input_ids` as authoritative and pops `pixel_values=None`, **silently dropping
the image** on fresh streaming vision requests. This path appears untested.

## Decisions (from brainstorming)

| Question | Decision |
| --- | --- |
| Scope | All three sub-tasks (full #429) in one spec. |
| VLM prompt-cache approach | **mlx_vlm-native `prompt_cache_state`**, wrapped in a tiny per-`cache_id` LRU on the olmlx side. Delete the `input_ids` hack. |
| Speculative on VLM | **Out of scope** for v1; document the decision in CLAUDE.md. |
| Non-streaming VLM caching | **Yes** — switch the non-streaming VLM path from `mlx_vlm.generate` to draining `mlx_vlm.stream_generate` (also fixes `eval_count==0`). |

## Architecture

Four coordinated changes in `engine/inference.py`, one new small store, one config
knob, plus CLAUDE.md updates. Unifying insight: mlx_vlm already owns the hard parts
(prefix matching, KV trim, image-in-prefix detection, state update); olmlx supplies
a bounded LRU of `PromptCacheState` objects keyed by `cache_id` and lifts its own
gates.

Data flow for a vision request:

```
generate_chat → VLM branch
  → build prompt (_apply_chat_template_vlm, unchanged)
  → compute prompt token ids
  → VlmPromptCacheStore.get(cache_id) → PromptCacheState (or fresh)
  → grammar: _install_grammar_processor (VLM gate lifted) → logits_processors
  → stream:     async_mlx_stream(..., prompt_cache_state=state, **gen_kwargs)
    non-stream: drain mlx_vlm.stream_generate(..., prompt_cache_state=state, ...)
  → mlx_vlm trims KV to common prefix, encodes only NEW images, auto-updates state
  → store keeps state (LRU); /api/ps reports reuse
```

## Components

### 2a. Grammar on VLM
**Files:** `engine/inference.py` (`_install_grammar_processor`, `_resolve_model_vocab_size`)

- Delete the `if lm.is_vlm: return False` early-return (currently ~line 138). Keep
  the distributed and tools gates.
- Extend `_resolve_model_vocab_size` to descend into `model.language_model` (a VLM's
  `lm_head` lives there, not under `model` / `model.model`), otherwise the bitmask
  can't be sized and grammar warn-skips on every VLM.
- `logits_processors` already forwards through `async_mlx_stream` →
  `mlx_vlm.stream_generate` → `generate_step`. Update the stale docstrings
  (currently ~lines 117–135) and the `not (lm.is_vlm and images)` clause
  (currently ~line 4087).

### 2b. `VlmPromptCacheStore` (new)
**File:** `engine/prompt_cache/vlm_state.py`

- Tiny LRU keyed by `cache_id` → `mlx_vlm.generate.PromptCacheState`, modeled on the
  speculative `_SpecCacheStore`: `enabled()`, `get(cache_id)`, `insert(cache_id, state)`,
  `clear()`; `capacity == 0` is the kill switch (`enabled()` False, `insert` no-op).
- Held per `LoadedModel`, analogous to `prompt_cache_store`. All inference is
  serialized under the inference lock, so no internal locking is needed.
- Because olmlx holds the `PromptCacheState`, compute `cache_read_tokens` ourselves
  via `state.find_prefix_length(full_ids)` *before* the generate call — mlx_vlm does
  not expose `reused_prefix_len` in `GenerationResult`. `cache_creation_tokens =
  len(full_ids) - cache_read_tokens`.

### 2c. Remove the image-dropping hack
**File:** `engine/inference.py` (`_setup_prompt_cache`)

- Delete the VLM `input_ids` branches (currently ~lines 3065–3066, 3091–3092). VLM no
  longer routes through `_setup_prompt_cache`; it uses `VlmPromptCacheStore` +
  `prompt_cache_state`. This removes the source of the silent image drop.

### 2d. Non-streaming VLM → drain `stream_generate`
**File:** `engine/inference.py` (`_full_completion_inner`, currently ~lines 4105 / 4151)

- Replace both `mlx_vlm.generate(...)` calls with a loop draining
  `mlx_vlm.stream_generate(...)`, accumulating text and capturing the final
  `GenerationResult` for `prompt_tokens` / `generation_tokens` (mirrors how the
  non-streaming *text* path already drains `mlx_lm.stream_generate`). Thread
  `prompt_cache_state` and `logits_processors` through. Fixes the documented
  `prompt_eval_count` / `eval_count == 0` gap.

### 2e. Speculative on VLM
No code change. Keep the existing skip + debug log. Document the out-of-scope
decision and rationale in CLAUDE.md.

## Config & gating

**File:** `config.py`
- New `vlm_prompt_cache_slots: int = 2` (env `OLMLX_VLM_PROMPT_CACHE_SLOTS`; `0` = off).

**File:** `engine/inference.py` (`use_prompt_cache` decision, currently ~line 4506)
- Add a dedicated VLM branch so VLM caching engages for **both** stream and
  non-stream (drop the `(stream or not lm.is_vlm)` restriction for the VLM case).
- Memory-pressure handling: since VLM bypasses `_setup_prompt_cache`, the VLM
  branch performs its own pressure check (reusing `memory_utils.is_memory_pressure_high`)
  and `clear()`s the `VlmPromptCacheStore` before falling back to a fresh
  `PromptCacheState`; the model-manager pressure flush also `clear()`s it.

## v1 limits (to document in CLAUDE.md)

- **VLM prompt cache:** in-memory only, keyed by `cache_id` — no radix-takeover, no
  disk spill, no KV-quant. Mirrors the speculative KV-reuse scoping. 2 slots bound
  memory; not counted against `OLMLX_MEMORY_LIMIT_FRACTION`.
- **Speculative on VLM:** out of scope — drafts are text-only and the
  classic/eagle/mtp decoders don't thread image features through the target forward.
- **Distributed VLM:** out of scope (unchanged).

## Error handling

- Grammar vocab still unresolvable for a VLM → keep the current warn-and-skip
  (return `False`); the request proceeds unconstrained.
- VLM + KV-quant requested → VLM cache path bypasses `_make_prompt_cache_for_lm`
  quant (uses mlx_vlm's plain cache); documented as a v1 limit, no error.
- Memory pressure high → clear the VLM store, fall back to fresh prefill.

## Testing (TDD — failing tests first)

**Unit (mocked, no model):** `tests/` — `VlmPromptCacheStore` LRU semantics: insert /
evict-past-capacity / get-hit / get-miss / disabled (capacity 0) kill switch /
`find_prefix_length`-derived read-token accounting.

**Live (`real_model`, placed outside `tests/integration/` to dodge that dir's autouse
MLX mock)** — new `tests/live/test_vlm_cache_grammar.py`:

1. *Image-drop regression:* a fresh streaming VLM request with an image produces
   image-grounded output (guards the 2c fix).
2. *Grammar:* `response_format={"type":"json_schema",...}` on an image request →
   schema-valid JSON (acceptance criterion 2).
3. *Prompt cache:* a 3-turn vision conversation reuses the image-prefix KV
   (`cache_read_tokens > 0`, observed via `/api/ps`), with output **identical** to the
   `OLMLX_VLM_PROMPT_CACHE_SLOTS=0` uncached path (acceptance criterion 1).
4. *Token counts:* a non-streaming VLM request returns `prompt_eval_count` /
   `eval_count > 0`.

## Risks

- **Cross-thread lazy-graph (#284 family):** the `PromptCacheState` cache persists in
  the store across requests and is mutated on the generation worker thread. mlx_vlm
  `mx.eval`s the cache during and after generation, so it should be materialized
  before the next cross-thread reuse — but verify explicitly; if a lazy graph
  survives, eager-eval the cache on store insert (`mx.eval([c.state for c in cache])`),
  the same remedy used for `snapshot_cache_for_persistence`.
- **Image-token-id trim:** mlx_vlm's prefix-trim reuse keys off
  `model.config.image_token_index` / `image_token_id`. Present for Gemma-family; if
  absent, mlx_vlm safely re-encodes the image (slower, not incorrect).
- **Acceptance-criterion determinism:** the "identical output" assertion requires
  greedy decoding (temperature 0) so the cached and uncached paths match
  bit-for-bit.

## Out of scope

- Distributed VLM.
- Video / audio modalities (#427 / #426) — though the prompt-cache fix should
  generalize.
- Speculative decoding on the image path (documented decision, not implemented).
- Radix index / disk spill / KV-quant for the VLM cache (v1 limits).

## Acceptance criteria (from #429)

- [ ] A 3-turn vision conversation reuses the image-prefix KV across turns (verified
  via `/api/ps` cache metrics), output identical to the uncached path.
- [ ] `response_format={"type":"json_schema",...}` produces schema-valid JSON on an
  image request.
- [ ] The speculative-with-images decision is documented in CLAUDE.md.
- [ ] (Added) Non-streaming VLM requests report non-zero token counts.

## Touched files

- `engine/inference.py` — grammar gate, vocab resolution, remove `input_ids` hack,
  non-streaming drain, VLM cache branch in `use_prompt_cache`.
- `engine/prompt_cache/vlm_state.py` — new `VlmPromptCacheStore`.
- `engine/model_manager.py` — own a `VlmPromptCacheStore` per VLM `LoadedModel`;
  clear it on memory-pressure flush.
- `config.py` — `vlm_prompt_cache_slots`.
- `CLAUDE.md` — rewrite the VLM-gating design note (grammar supported, prompt cache
  via `prompt_cache_state` with v1 limits, speculative out of scope).
- `tests/` — unit tests for the store; `tests/live/test_vlm_cache_grammar.py`.
