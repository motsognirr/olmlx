# mlx 0.32.0 / mlx-lm 0.31.3 / mlx-vlm 0.5.0 bump — thread-local stream migration (#499)

**Date:** 2026-07-10
**Issue:** #499 (Adapt to mlx >= 0.31.2 thread-local streams)
**Approach:** Sync-in-worker discipline ("Approach B") — chosen over trusting
cross-thread proxy synchronization semantics empirically.

## Background

mlx 0.31.2 made default streams thread-local ("MLX can be used by multiple
threads for independent computations") and broke the previous model where a
module-level stream created on the main thread could be adopted by worker
threads. olmlx pinned `mlx<0.31.2` and tracked the migration in #499, gated
on mlx-lm shipping a thread-local `generation_stream`.

That gate is satisfied: mlx-lm 0.31.3 defines
`generation_stream = mx.new_thread_local_stream(mx.default_device())` — a
**ThreadLocalStream proxy** that resolves to a per-thread stream instance at
use time. mlx-vlm 0.5.0 ported the same change (upstream #1050). mlx 0.32.0
additionally fixes the `mx.fast.rope` B>1/L==1 decode-shape corruption
(upstream #3498) that `engine/ropefix.py` works around.

Verified semantics (from #499 empirics + upstream release notes):

- Ops must be created and evaluated on the same thread; only materialized
  arrays may cross threads.
- A worker thread cannot adopt a foreign thread's stream via
  `with mx.stream(s)`.
- A ThreadLocalStream proxy can be *captured* anywhere, but resolves to the
  **calling thread's** instance — including inside `mx.synchronize(proxy)`.

## Scope

One PR:

- `pyproject.toml`: `mlx>=0.32.0,<0.33`, `mlx-lm>=0.31.3,<0.32`,
  `mlx-vlm>=0.5.0,<0.6`; regenerate `uv.lock`. `transformers>=5.5.0`
  unchanged (mlx-vlm 0.5.0 imposes no cap; the `<5.13` cap only arrives with
  mlx-vlm 0.6.4, which is a follow-up project).
- Rewrite the mlx pin rationale comment in pyproject (floor is 0.32.0 for the
  rope fix + thread-local streams; the old "<0.31.2" rationale is obsolete).
- Thread-local stream migration (below).
- Ropefix retirement (below).

Out of scope: mlx-vlm 0.5.0 → 0.6.4 (separate follow-up: video input #427,
hybrid-VLM reroute lift, new-model matrix), the remaining items of #614, any
use of new 0.32.0 features (nvfp4, `qmv_wide`, `new_thread_unsafe_stream`).

mlx-vlm 0.4.4 → 0.5.0 is included because 0.4.4's
`generation_stream = mx.new_stream(...)` is created at import on the main
thread and would crash VLM inference from worker threads under mlx ≥ 0.31.2.
0.5.0 predates the 0.6.0 generation-engine refactor, so the internals olmlx
touches (`mlx_vlm.generate.PromptCacheState`, `mlx_vlm.utils.MODEL_REMAPPING`,
the gemma4 module quirks, `mlx_vlm.load`/`stream_generate`/
`apply_chat_template`) are expected to survive — but 0.5.0 added APC prompt
caching, so `PromptCacheState` compatibility must be verified during
implementation.

## Design

### Stream ownership principle

**Every stream is owned by the thread that created it; all syncs happen on
the owning thread; cross-thread completion is guaranteed by thread-join,
never by cross-thread sync.**

Most of the codebase already conforms:

- `CancellableStream._run`'s finally block (`utils/streaming.py`) imports
  `generation_stream` and syncs **inside the worker thread**. Under 0.31.3
  the import yields the proxy, which resolves in-thread — correct as-is.
  Comment update only.
- `_generate_sync` (non-streaming) syncs at the end of its own worker-thread
  call. No change.
- Speculative decoders and the continuous-batching worker each run all their
  GPU work on a single thread (that thread's default/generation stream) —
  self-consistent under thread-local semantics. No change.

Changes:

- **Delete** `_resolve_generation_streams()` / `_generation_streams` /
  `_sync_generation_streams()` (`engine/inference.py`). A loop-thread sync of
  a thread-local proxy syncs the loop thread's (idle) instance — dead weight
  at best, false confidence at worst.
- `_safe_sync()` becomes `mx.synchronize()` only — fencing the loop thread's
  own eager ops (cache deepcopy/eval, eviction). New docstring rationale:
  worker GPU work is already fenced by the worker's own final sync plus
  `drain_and_join`.

### Checkpoint prefill moves into the generation thread (the load-bearing change)

`_drive_segmented_prefill` currently runs on the **event-loop thread** under
`with mx.stream(_generation_streams[0])`; decode then runs on a worker
thread. Today both name the same module-level stream object. Under
thread-local streams they resolve to **different instances**, silently
reintroducing the #284 GDN corruption (coherent-but-off-prompt output on
Qwen3-Next-family hybrids at ~16k+ tokens).

Fix: the drive becomes a **deferred closure executed at the start of the
generation thread**, before decode:

- Streaming path: inside `gen_factory`, before `mlx_lm.stream_generate`.
- Non-streaming path: at the top of the `_generate_sync` worker-thread call.

Same thread ⇒ same thread-local `generation_stream` ⇒ the #284 "prefill and
decode on the same stream" contract maps onto "same thread", exactly as #499
predicted. `prefill_stream` resolution changes from `_generation_streams[0]`
to importing the proxy in-thread.

Consequences handled:

- `store.insert_checkpoint` and the drive's snapshots now run on the worker
  thread. Snapshots must be eager-evaled **on that thread** before the loop
  thread touches them — already the case (`mx.eval` per chunk;
  `snapshot_cache_for_persistence` evals eagerly). Verify
  `insert_checkpoint`'s thread-safety from a non-loop thread.
- Cache-setup results (suffix tokens, cache_read/creation counts) are
  currently returned synchronously to build `gen_kwargs` before the stream
  starts. The deferred drive reports these via a small result holder the
  worker fills before yielding its first token (mechanism detailed in the
  implementation plan).
- Side bonus: removes the event-loop-blocking checkpoint prefill called out
  in #614 (which stays open — it has other items).

### Ropefix retirement

- Delete `engine/ropefix.py`; call sites in `engine/batching.py`,
  `engine/dflash/selfgen.py` (including its re-export),
  `scripts/dflash_gen_data.py`.
- Delete the removal-gate test and import-compat tests in
  `tests/test_batching.py` / `tests/test_dflash_selfgen.py`.
- The live batched-parity canary stays — now running unpatched, it **is** the
  regression test for upstream's rope fix.

### Secondary audits (expected no-ops, verified in the plan)

From #499's checklist:

- Prompt-cache disk-spill `asyncio.to_thread` sites, `_SpecCacheStore`
  deepcopy-on-reuse, radix takeover: each must eval lazy state on the thread
  that created it (existing eager-eval discipline should already satisfy
  this).
- Flash prefetcher: `flash/predictor.py` pool-thread `mx.eval(scores)`
  evaluates a graph built on the generation thread — materialize before
  submitting to the I/O pool.
- Test infrastructure referencing `generation_stream` /
  `_generation_streams`: `tests/__init__.py`,
  `tests/integration/conftest.py`, `tests/test_inference.py`,
  `tests/test_inference_checkpoint_drive.py` — updated to the new structure.

## Testing

TDD order:

1. **Metal-gated semantics probe test** pinning the behaviors this design
   relies on: worker-thread proxy resolution; foreign-stream adoption fails;
   materialized arrays cross threads. Fails loudly if mlx changes the model
   again.
2. **Metal-gated checkpoint-drive parity test on a hybrid GDN model**:
   multi-message prompt through the drive + decode; output must match a
   single-thread reference. Written to fail under the old loop-thread drive
   on 0.32.0, proving it catches the hazard.
3. Bump + migration to green.
4. Full `uv run pytest` (Metal-gated included) + live server smoke on a small
   text model, a Qwen3.x hybrid (GDN), and a VLM — the three architectures
   the stream invariants protect.

## Rollback

Single PR; reverting it (including `uv.lock`) restores the old world exactly.

## Follow-ups (not this PR)

- mlx-vlm 0.5.0 → 0.6.4: video input (#427), hybrid-VLM reroute lift,
  new-model compatibility matrix, `transformers<5.13` cap.
- Evaluate mlx 0.32.0 features: nvfp4 / `qmv_wide` for the KV-quant stack
  (#629, #506), `new_thread_unsafe_stream` for panel parallelism.
- Update CLAUDE.md invariants after landing: the "Metal stream hazard" entry
  gains the thread-local framing; the "mlx 0.31.x rope bug" entry is removed.
