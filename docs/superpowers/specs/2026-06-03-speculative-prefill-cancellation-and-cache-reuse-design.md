# Speculative prefill: cancellation + cross-request KV cache reuse

**Date:** 2026-06-03
**Status:** Design — pending implementation
**Author:** Daniel Palmqvist (with Claude)

## Motivation

A long agentic coding session (Claude-Code-style) against
`unsloth/Qwen3.6-27B-MLX-8bit` with classic speculative decoding appeared to
"freeze" while pinning the GPU at ~90%, with flat memory. Investigation of
`/tmp/olmlx.log` found two independent root causes.

### Observed timeline (request `db164fe6`)

```
17:01:41  request starts (17 msgs, 11 tools, speculative, prompt cache DISABLED)
          → begins prefilling the full ~69k-token conversation
  ...      client disconnects mid-prefill (cancel_event set; 10s fallback join expires)
17:08:12  WARNING: "Inference thread still alive after cleanup attempts — deferring..."
17:08:15  next request a5971dbe arrives → "Waiting for deferred GPU cleanup"
17:08:45  a5971dbe → 503: "deferred GPU cleanup did not complete within 30.0s"
17:09:51  db164fe6 thread finally exits (~8 min of non-cancellable prefill)
```

### Root cause 1 — non-cancellable speculative prefill

`speculative_stream_generate` (`speculative_stream.py:75`) calls
`decoder.prefill(prompt_arr)` as a single blocking call; `cancel_event` is only
checked *after* prefill (`:105-107`, `:129`, `:135`). Inside,
`SpeculativeDecoder.prefill` → `_prefill_last_logit` / `_chunked_prefill`
(`speculative.py:105,127`) loop over `_PREFILL_CHUNK` chunks but take **no
cancel callback**. A client disconnect during a multi-minute prefill cannot
interrupt it; the GPU stays pinned and the inference lock stays held until
prefill completes, 503-ing every subsequent request.

The non-speculative path already solves this: a `prompt_progress_callback`
returns `False` when `cancel_event.is_set()` (`streaming.py:617`), checked
between prefill chunks.

### Root cause 2 — prompt cache disabled under speculative

`inference.py:4328` gates `use_prompt_cache` off whenever `lm.is_speculative`.
So every agent turn re-prefills the entire growing conversation (~69k tokens by
mid-session). Across the logged run, turns cost 9–33 min each, almost entirely
prefill; outputs were tiny (95–155 tokens typical). Speculative's ~1.3–1.9×
*decode* speedup is irrelevant when prefill is ~99% of wall-clock and is repaid
in full every turn.

`SpeculativeDecoder.prefill` (`speculative.py:378`) creates fresh
`_target_cache` and `_draft_cache` via `make_prompt_cache` on every call and
never persists them; the decoder lives entirely outside the prompt-cache store
(whose `CachedPromptState` holds one cache per entry).

## Goals

1. A client disconnect during speculative prefill interrupts it within one
   sub-chunk (≤ ~1s), releasing the GPU and lock promptly — no deferred-cleanup
   503 for the next request.
2. Speculative requests reuse KV cache across turns so each turn prefills only
   the new suffix, restoring prompt-cache-class latency for long sessions while
   keeping the decode-time speculative speedup.

## Non-goals

- Cache reuse for `dflash` / `eagle` strategies (experimental; 2–6% acceptance;
  their GDN rollback + capture state make persisted caches risky for little
  gain). They keep fresh-prefill.
- Disk spill of speculative snapshots (v1 is in-memory only).
- Counting speculative snapshots against `OLMLX_MEMORY_LIMIT_FRACTION` (v1).
- Distributed / VLM speculative (already unsupported / out of scope).

## Part 1 — Cancellable speculative prefill

### Design

- New exception `PrefillCancelled(Exception)` in `speculative.py`.
- Add `cancel_event: threading.Event | None = None` to the `prefill` method of
  the `SpeculativeDecoderProtocol` (`speculative_stream.py:48`) and every
  implementer (`SpeculativeDecoder`, `SpeculativeFlashDecoder`, `DFlashDecoder`,
  `EagleDecoder`, `PromptLookupDecoder`).
- Thread `cancel_event` into `_chunked_prefill(model, tokens, cache,
  cancel_event=None)` and `_prefill_last_logit(model, prompt, cache,
  cancel_event=None)`. At the top of each `_PREFILL_CHUNK` iteration, if
  `cancel_event is not None and cancel_event.is_set()`, raise `PrefillCancelled`.
- `speculative_stream_generate` (`speculative_stream.py:53`) passes the
  `cancel_event` it already receives into `decoder.prefill(...)`, and wraps that
  call:

  ```python
  try:
      first_token = decoder.prefill(prompt_arr, cancel_event=cancel_event)
  except PrefillCancelled:
      return  # clean generator exit; no token yielded
  ```

- dflash/eagle: accept the kwarg for protocol conformance; honor it in any
  chunk loop they own, accept-and-ignore otherwise. Not on the hot path.

### Behavior

Cancel granularity drops from "whole prefill" to one 2048-token chunk
(sub-second to ~1s for a 27B target). The inference thread reaches the
`speculative_stream.py:129` checkpoint and exits, so `drain_and_join` completes
normally and `_schedule_deferred_inference_cleanup` is never reached — no 503.

## Part 2 — Decoder-owned persistent caches (Approach B)

### Component: `_SpecCacheStore`

A small LRU held on the `SpeculativeDecoder` instance. The decoder is a single
instance per model (`LoadedModel.speculative_decoder`) and all inference is
serialized under the inference lock, so the store needs no internal locking.

Entry shape:

```python
@dataclass
class _SpecCacheEntry:
    tokens: list[int]      # token sequence the snapshots represent (ends at a message boundary)
    target_snap: list      # snapshot_cache_for_persistence(target_cache)
    draft_snap: list       # snapshot_cache_for_persistence(draft_cache)
```

- Backed by `OrderedDict`, capacity `OLMLX_SPECULATIVE_CACHE_SLOTS` (default
  **2** — each entry is a full 27B + draft KV snapshot, multi-GB).
- **Lookup = longest token-prefix match** by linear scan over the (tiny) entry
  set. No `cache_id` dependency: matching purely on token prefix is what makes
  reuse robust to unstable client-supplied ids (the same problem the non-spec
  path's radix index solves; at 2–4 slots a trie is unnecessary).

### Rewritten `prefill(prompt, *, segmented=None, cancel_event=None)`

`segmented` is a `SegmentedPrompt` (from `tokenize_segmented_chat`) carrying
message boundaries; threaded in from `inference.py` (see Integration).

1. **Lookup.** Find the LRU entry whose `tokens` is the longest prefix of the
   new prompt tokens. Compute `common = _find_common_prefix(prompt_tokens,
   entry.tokens)`.
2. **Reuse decision.**
   - **Trimmable target** (working cache layers all in
     `_TRIMMABLE_CACHE_CLASSES` — detected from the cache instances, not from
     `lm`, which the decoder has no handle to): deepcopy the entry's snapshots
     into the working caches, `trim_prompt_cache` both to `common`;
     `already_covered = common`.
   - **Non-trimmable target** (Qwen3.6 / `ArraysCache`): reuse only if
     `common == len(entry.tokens)` (the stored tokens are a *full* prefix —
     strict extension, exactly the chat-append workload). Otherwise discard the
     hit. On reuse, deepcopy snapshots into working caches; `already_covered =
     len(entry.tokens)`.
   - **Miss / discard:** `self._target_cache = make_prompt_cache(self._target)`,
     `self._draft_cache = make_prompt_cache(self._draft)`; `already_covered = 0`.

   Deepcopy-on-reuse is mandatory: `step()` mutates the working caches in place
   (KV growth + GDN rollback trims), which must never touch the stored snapshot.
3. **Drive prefill** over the uncovered span `[already_covered ..
   len(prompt)-1]` on the **default stream** (see invariant 1), replicating the
   two-chunk + boundary-snapshot logic of `_drive_segmented_prefill`
   (`inference.py:2462`):
   - Split at the deepest interior message boundary > `already_covered` and
     < `len(prompt)`.
   - Sub-chunk each span at `_PREFILL_CHUNK`, `mx.eval(flatten_cache_state)` +
     `mx.clear_cache()` per sub-chunk, checking `cancel_event` (raises
     `PrefillCancelled`).
   - Target's final token produces `_last_target_logit` (`_prefill_last_logit`
     semantics) → `first_token`. Draft span just populates KV (no logit).
   - At the boundary, snapshot **both** caches via
     `snapshot_cache_for_persistence(..., eager_eval=False)` and insert/refresh
     the LRU entry with `tokens = prompt_tokens[:deepest_boundary]`; evict LRU
     past capacity. (Mirrors the non-spec store storing at the deepest interior
     boundary — the point the *next* turn reuses.)
   - Pure-rotating speculative targets (if any) follow the single-chunk
     end-snapshot variant, same as `_drive_segmented_prefill`.
4. `self._cache_seq_len = len(prompt_tokens)`; `self._pending_token =
   first_token`; return `first_token`.

### Integration (`inference.py`)

- The `not lm.is_speculative` clause at `inference.py:4328` is relaxed: prompt
  cache reuse for speculative is internal to the decoder, but the speculative
  branch in `_stream_completion` now builds a `SegmentedPrompt` (reusing
  `tokenize_segmented_chat`) and passes `segmented` into
  `async_speculative_stream` → `speculative_stream_generate` →
  `decoder.prefill`. The existing "Prompt cache disabled … speculative" log line
  is updated to reflect that speculative now self-manages reuse.
- Gate Part 2 to `classic` and `pld` strategies. dflash/eagle decoders ignore
  `segmented` and always fresh-prefill.

## Correctness invariants

1. **Single stream.** Speculative prefill and `step()` decode both run on the
   **default stream** (`inference.py:3990`: "Speculative decoding does not use
   mlx_lm's generation_stream"). Part 2 replicates `_drive_segmented_prefill`
   rather than calling it precisely because that driver forces
   `generation_stream`; prefilling there and decoding on the default stream
   would leave the GatedDeltaNet recurrent state a cross-stream lazy graph whose
   materialization corrupts MoE expert routing on Qwen3-Next-family targets
   (#284 / #396 hazard family).
2. **Snapshot hygiene.** Snapshots are deepcopied + eager-eval'd before storage
   (`snapshot_cache_for_persistence`), and working caches are deepcopied *from*
   snapshots on reuse — no Metal-stream-bound graph survives across the
   request/thread boundary (#284 / #343 / #396).
3. **Hybrid = strict-extension only.** Non-trimmable caches reuse only on a full
   prefix match; divergent branches fall back to a correct (un-accelerated)
   fresh prefill.
4. **GDN capture suppressed during prefill** (existing behavior at
   `speculative.py:401`) is preserved; snapshots are taken with capture off.

## Known limitations (v1)

- In-memory only; snapshots not counted against the memory limit. Default 2
  slots bounds this; documented because the target is a 27B.
- ~30 lines of chunking/snapshot logic are duplicated from
  `_drive_segmented_prefill` (cross-referenced in comments). Extracting a
  stream+model-parameterized helper was considered but rejected: the two call
  sites diverge (two models, last-logit capture, default vs generation stream)
  enough that duplication is lower-risk than over-abstraction. Revisit if a
  third caller appears.

## Configuration

- `OLMLX_SPECULATIVE_CACHE_SLOTS` (int, default `2`): max persisted speculative
  cache lineages. `0` disables reuse (fresh prefill every turn — the current
  behavior, useful as a kill switch).

## Testing (TDD — failing test first)

### Part 1
- Unit: a fake decoder whose `_chunked_prefill` blocks on a spin; set
  `cancel_event` and assert `prefill` raises `PrefillCancelled` within one
  chunk.
- Integration: a cancelled speculative stream releases the inference lock
  without entering deferred cleanup (assert no "deferring Metal sync" warning,
  next request not 503).

### Part 2
- `_SpecCacheStore`: longest-prefix lookup; LRU eviction at capacity; `0` slots
  disables.
- Trimmable path: partial-prefix hit trims working caches to `common`.
- Hybrid path: partial-prefix hit is *discarded* (fresh prefill); full-prefix
  hit reuses with `already_covered = len(entry.tokens)`.
- Deepcopy isolation: mutating a working cache after reuse does not change the
  stored snapshot.
- End-to-end: turn 2 with a strict-extension prompt prefills only the suffix
  (assert covered-token count) and yields token-identical output to a fresh
  prefill (regression guard against cross-stream / chunking corruption).

## Files touched

- `olmlx/engine/speculative.py` — `PrefillCancelled`, cancel-aware
  `_chunked_prefill` / `_prefill_last_logit`, `_SpecCacheStore`, rewritten
  `SpeculativeDecoder.prefill`.
- `olmlx/engine/speculative_stream.py` — protocol signature, pass `cancel_event`
  + `segmented`, catch `PrefillCancelled`.
- `olmlx/engine/inference.py` — build/pass `SegmentedPrompt` on the speculative
  branch; update the gating log line.
- `olmlx/config.py` — `OLMLX_SPECULATIVE_CACHE_SLOTS`.
- `olmlx/engine/flash/speculative.py`, `dflash/decoder.py`, `eagle/decoder.py` —
  `prefill` signature conformance (kwarg accept).
- `tests/` — new tests per above.
- `CLAUDE.md` — update the speculative + prompt-cache design notes.
