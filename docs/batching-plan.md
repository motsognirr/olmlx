# Implementation Plan: Continuous Batching via mlx-lm `BatchGenerator`

Status: **Phases 0–3 implemented** (Phase 3 = consumer backpressure,
aggregate KV admission, per-model batching config, and fairness-guard
tuning — see §10/§11). (engine/ropefix.py,
engine/batching.py,
inference.py `_batch_eligible` / `_get_batch_scheduler` /
`_stream_completion_batched` / `_full_completion_batched` /
`_setup_batched_prompt_cache`, settings, LoadedModel fields + teardown;
tests/test_batching.py + tests/live/test_batching_real.py). Measured on this
machine (M-series, Qwen3-4B-4bit, ~100-token chat generations, prefill
included): 80 tok/s single stream → 120 aggregate at 2 concurrent (1.49×),
135 at 4 (1.68×), 142 at 8 (1.76×); exact greedy parity with the exclusive
path at 3-way concurrency. Phase 2 adds the prompt-cache round trip
(move semantics via `PromptCacheStore.take`/`async_take`, re-store from the
worker's done event), the non-streaming path, per-sequence grammar, batch
occupancy on `/api/ps` + `olmlx_batch_*` Prometheus metrics, and the two
PR #507 refactors (shared `_enter_inference_lock`/`_exit_inference_lock`,
one `StopScanner`). Phases 3+ remain.

Two integration landmines surfaced by the full suite, both fixed with
regression tests: (1) Mock vacuity — a `MagicMock` model's `make_cache()` is
truthy but iterates empty, so the plain-KV probe passed vacuously and tests
that patch `inference.settings` (truthy `.batching`) routed mocked requests
onto the batched path; probes now materialize the cache list and the
predicate gates on `settings.batching is True`. (2) Loop rebinding — the
scheduler lives on `LoadedModel` and can outlive the event loop that created
its manager task (pytest-asyncio runs one loop per test); a submit from a
new loop after the bound loop closed now rebinds (fresh wakeup event +
manager, stale inbox cancelled) instead of queueing into a manager that can
never run again, which hung the suite indefinitely. Target: concurrent text-chat requests (Claude-Code-style parallel
subagents) batched on one model, replacing FIFO head-of-line blocking behind the
global inference lock with continuous batching. Decode on Apple Silicon is
memory-bandwidth-bound, so B concurrent sequences cost ~one sequence's weight
traffic per step: aggregate throughput scales near-linearly until KV reads or
compute dominate, and a queued request's prefill interleaves with running decodes
instead of waiting behind them.

## 1. Verified groundwork

What the installed mlx-lm (0.31.x) already provides — all line refs into
`.venv/lib/python3.11/site-packages/mlx_lm/generate.py`:

- `BatchGenerator` (1486) implements true continuous batching: `insert()` /
  `insert_segments()` add sequences while others are mid-decode; each `_next()`
  tick (1762) decodes the running batch, then advances pending prefills by
  `prefill_step_size` chunks and migrates finished prefills into the generation
  batch. Everything runs under `mx.stream(generation_stream)` (1846) — one
  thread, one stream, consistent with olmlx's #284 discipline.
- Per-sequence state is first-class: samplers, `logits_processors` lists, KV
  caches, `max_tokens`, and Aho-Corasick `SequenceStateMachine` stop matchers
  (943) are all per-uid.
- `insert(caches=...)` accepts pre-computed per-sequence caches (merged via
  `merge()`, 874–883); `extract_cache(uids)` / `remove(uids,
  return_prompt_caches=True)` (1716, 1692) hand per-sequence caches back —
  the hooks the prompt-cache store needs. `prompt_cache_nbytes` (1843) gives
  live KV memory for admission control.
- Responses are **token-level** (`GenerationBatch.Response`: uid, token,
  logprobs, finish_reason — 1238): olmlx must detokenize per sequence.
  `TokenizerWrapper.detokenizer` returns a fresh streaming-detokenizer instance
  per access (tokenizer_utils.py:447–451), so per-sequence detokenizers are
  cheap.
- It sets the Metal wired limit on construction and restores it in `close()`
  (1540–1552).

Hard constraints discovered:

- **Cache-type restriction** (`_make_cache`, 838–867): only `KVCache` →
  `BatchKVCache`, `ArraysCache` (native left_padding), `RotatingKVCache` with
  `keep == 0`, and `CacheList` are batch-convertible. Anything else raises.
  Consequences: olmlx's TurboQuant/Spectral/Shard KV-quant caches **cannot
  batch** in v1; `RotatingKVCache` with keep tokens (sink-style SWA configs)
  cannot either.
- **`mx.fast.rope` bug** (mlx 0.31.x, pinned `<0.31.2`): corrupts batch rows
  ≥ 1 at B>1, L==1 — exactly the batched-decode shape. The exact workaround
  exists as `safe_rope_patch` (engine/dflash/selfgen.py:57, folds B into the
  heads dim) but is currently scoped to selfgen. This is the single hard
  correctness blocker; see §6.
- **Seed**: `_apply_seed` (engine/inference.py:1278) sets the *global*
  `mx.random` state. Per-request seeds are not reproducible under concurrent
  batching.

olmlx-side seams that stay unchanged:

- `generate_chat`'s per-request async-generator contract (`{"text", "done",
  "stats"}` chunks). Routers, `thinking_split`, `streaming_common`, tool
  parsing all consume text deltas downstream and need no changes.
- Stop-sequence matching is already olmlx-side text matching in the consumer
  loop (inference.py:2850–2878) — it moves into the per-request consumer
  unchanged. EOS goes through `BatchGenerator`'s `stop_tokens` state machine.
- Sampler/processors are already built per request by `_convert_options`
  (inference.py:1212–1245) via `make_sampler`/`make_logits_processors` —
  exactly the shape `insert(samplers=..., logits_processors=...)` wants.

## 2. Architecture

New module `engine/batching.py` (single module; split later if it grows):

```
┌ HTTP request (eligible) ─────────────────────────────────────┐
│ generate_chat → _stream_completion_batched                   │
│   tokenize, build sampler/processors, fetch prompt-cache      │
│   await scheduler.submit(SeqRequest) → AsyncIterator[Event]  │
└──────────────────────────────────────────────────────────────┘
            │ inbox queue (thread-safe)        ▲ per-uid output queues
            ▼                                  │ (loop.call_soon_threadsafe)
┌ BatchScheduler (one per LoadedModel, lazy) ──────────────────┐
│ owns _inference_lock while sequences in flight               │
│ one dedicated worker thread:                                 │
│   with safe_rope_patch():                                    │
│     loop: drain inbox → gen.insert_segments(...)             │
│           prompt_resp, gen_resp = gen.next()                 │
│           dispatch tokens/progress per uid                   │
│           finished/cancelled → gen.remove(..., caches=True)  │
│     idle (no seqs, empty inbox) → sync, release lock, park   │
└──────────────────────────────────────────────────────────────┘
```

Key decisions:

- **One worker thread for the whole batch loop.** Replaces the
  thread-per-request `CancellableStream` model on this path. All MLX work for
  batched requests happens on this thread on `generation_stream` — the same
  single-thread/single-stream discipline as today, so the #284 hazard family
  is not re-opened. The async bridge reuses the queue + `call_soon_threadsafe`
  pattern from `utils/streaming.py`.
- **Lock interplay (v1): the scheduler is just another lock holder.** When the
  first sequence arrives, the scheduler acquires `_inference_lock` (same
  `_acquire_inference_lock` path, same deferred-cleanup handshake); it holds it
  while any sequence is in flight and releases after drain +
  `mx.synchronize(generation_stream)`. Non-batchable requests acquire the lock
  normally and therefore wait for drain. This keeps the Metal-exclusivity
  invariant in exactly one mechanism and avoids redesigning the lock.
  - **Fairness guard**: if exclusive waiters are queued (`_queue_depth > 0`),
    the scheduler stops accepting *new* inserts (inbox stays queued) and
    releases the lock once running sequences finish. Otherwise a steady stream
    of batched requests starves embeddings/rerank/other-model requests forever.
- **Submission API**:

  ```python
  @dataclass
  class SeqRequest:
      tokens: list[int]                 # full prompt token ids
      max_tokens: int
      sampler: Callable | None
      logits_processors: list | None
      cache: list | None                # from prompt-cache store (moved, not copied)
      cache_id: str
      # output: asyncio queue of TokenEvent(token:int) | Progress | Done(reason, cache)

  class BatchScheduler:
      async def submit(self, req: SeqRequest) -> AsyncIterator[Event]: ...
      def cancel(self, uid: int) -> None       # consumer-driven (disconnect/timeout)
      def close(self) -> None                  # drain, join thread, restore wired limit
  ```

- **Per-request consumer** (in `_stream_completion_batched`): owns a fresh
  `tokenizer.detokenizer`, feeds tokens through it, applies olmlx stop-sequence
  text matching, gpt-oss channel filter (moot in v1 — see eligibility),
  `inference_timeout`, and emits today's chunk dicts. On `GeneratorExit` /
  timeout it calls `scheduler.cancel(uid)`; the scheduler removes the sequence
  at the next tick (prefill removal latency is bounded by one
  `prefill_step_size` chunk — this *generalizes* prefill cancellation, which
  currently only speculative decoders implement).

## 3. Eligibility predicate (v1)

Route to the batch engine only when **all** hold; otherwise the existing
exclusive path runs untouched:

| Condition | Why |
|---|---|
| `settings.batching` enabled (default off) | opt-in rollout |
| text LLM (not VLM / whisper / TTS / reranker / embeddings) | only chat decode batches |
| no images/audio | as above |
| model not speculative-configured | spec decoders own their caches; different goal (single-stream latency) |
| not distributed, not flash / flash-MoE | per-rank sideband and per-token SSD I/O assume one sequence |
| KV quant off for this model | `_make_cache` rejects custom cache classes |
| cache probe passes | one-time per model: every `make_cache()` element ∈ {KVCache, ArraysCache, RotatingKVCache(keep==0), CacheList} |
| no `seed` in options | global PRNG; reproducibility promise can't be kept in a batch |
| ~~no grammar~~ (lifted in Phase 2) | per-sequence processor calls verified — see §11 |
| not gpt-oss channel-format (v1) | usually rotating-cache w/ keep anyway; channel filter untested against batched detok |
| same model as currently scheduled | one BatchGenerator per model; cross-model → exclusive path waits for drain |

Conservative v1 allowlist in practice: dense full-attention models on plain
`KVCache` (Qwen3 dense, Llama-family, etc.). Hybrid GDN models (`ArraysCache`)
are *mechanically* supported by mlx-lm's batching but carry the #284-family
risk surface — gate them behind a follow-up flag after parity testing.

## 4. Request-flow changes (`engine/inference.py`)

`_stream_completion` grows an early branch: eligible → delegate to
`_stream_completion_batched(...)`; the function signature and yielded chunk
shapes are identical, so `generate_chat` callers don't change.

1. **Tokenize**: full token list (reuse `prompt_tokens` /
   `tokenize_for_cache`); `insert_segments` can take message-boundary segments
   later, plain `insert` in v1.
2. **Prompt cache**: `store.take(cache_id)` (move semantics — the entry leaves
   the store while in the batch, mirroring how `gen_kwargs["prompt_cache"]`
   works today) → trim to common prefix exactly as `_setup_prompt_cache` does →
   pass via `caches=[...]`. On finish, `remove(uid, return_prompt_caches=True)`
   → store under `cache_id` with prompt+generated tokens (radix reuse
   preserved). **Verify in tests** that `BatchKVCache.extract(idx)`
   (models/cache.py:841) strips left padding so a re-stored cache is a clean
   B=1 prefix.
3. **Sampler/processors**: already per-request from `_convert_options`; pass
   through.
4. **Memory admission**: extend `_kv_cache_preflight_check` to add
   `scheduler.gen.prompt_cache_nbytes` (live batch KV) to the projected usage.
   Over budget → behave like today (flush/queue/503), i.e. the request falls
   back to waiting rather than joining the batch.
5. **Stats**: `prompt_eval_count` from `PromptProcessingBatch.Response.progress`,
   `eval_count` from token count, TTFT at first generated token.
   `TimingStats`-based `/metrics` emission unchanged (same seams). Document
   that per-request decode tok/s drops while aggregate rises.
6. **Non-streaming** (`_full_completion`): Phase 2 — internally consume the
   batched stream and aggregate (the existing `collect_stream` pattern).
7. **Keep-alive / eviction**: wrap each batched sequence in the existing
   `_inference_ref(lm, keep_alive)` so `active_refs` keeps the model pinned
   while it has batch members.

## 5. ModelManager / lifecycle

- `LoadedModel.batch_scheduler: BatchScheduler | None`, created lazily on
  first eligible request (like the speculative decoder).
- `_close_loaded_model`: `scheduler.close()` runs **first** (before grammar
  drop, mirroring that ordering rule) — drains/cancels sequences, joins the
  thread, restores the wired limit.
- Server shutdown (`lifespan`): close all schedulers before model teardown.

## 6. The rope bug (correctness gate)

**Phase 0 finding (verified empirically on mlx 0.31.1):** the corruption only
affects the **scalar-offset** kernel path at B>1/L==1. The per-row
vector-offset path — which is exactly what `BatchKVCache` passes during
batched decode with left padding — is correct, as is a uniform offset
*vector*. So `BatchGenerator` decode does not actually hit the bug; only
scalar-offset batched callers (dflash selfgen's uniform-length batches over
plain `KVCache`) do.

- `safe_rope_patch` now lives in `engine/ropefix.py` (selfgen re-imports) and
  folds **only** the scalar-offset B>1/L==1 shape; vector-offset calls pass
  through untouched (a per-row offset has no single fold — the original
  always-fold version broke `batch_generate` with a shape error).
- The scheduler thread still holds the patch around its loop as defense in
  depth: it is exact, free for non-buggy shapes, and covers any model whose
  cache exposes a scalar offset at decode.
- Removal gates: `tests/test_batching.py::
  test_rope_bug_still_present_remove_patch_when_this_fails` (Metal-gated,
  asserts the raw bug still reproduces — fails on the mlx release that fixes
  it, #499) and the live parity canary
  (`tests/live/test_batching_real.py`: B=4 identical greedy prompts ≡ each
  other ≡ unbatched `generate()`; mixed-length batch coherent — both pass
  today on Qwen3-4B-4bit).

## 7. Config

New settings (pydantic, `OLMLX_` prefix), per-model overridable at top level of
`models.json` (consistent with flash/speculative):

```
batching: bool = False                  # OLMLX_BATCHING
batch_completion_size: int = 8          # BatchGenerator completion_batch_size
batch_prefill_size: int = 4             # prefill_batch_size
batch_prefill_step: int = 2048          # prefill_step_size (matches _PREFILL_CHUNK)
```

Queue timeout reuses `inference_queue_timeout` (submission honors it while the
fairness guard has the inbox paused).

## 8. Observability

- Metrics: `olmlx_batch_active_sequences` gauge, `olmlx_batch_inserts_total`,
  `olmlx_batch_aggregate_tokens_total`; batch occupancy on `/api/ps`.
- Tracing: per-request `inference` span stays (consumer side); scheduler emits
  a low-cardinality `batch.step` span only under sampling (same stance as
  per-step speculative spans).

## 9. Testing (TDD)

Unit (`tests/`, fake BatchGenerator — note the autouse MLX mock in
`tests/integration/`):

1. Scheduler: insert/dispatch/finish; cancel mid-prefill and mid-decode frees
   the slot; drain releases the lock; fairness guard pauses inserts when an
   exclusive waiter appears; idle park/wake; close() joins.
2. Eligibility predicate: every row of the §3 table.
3. Consumer: detokenizer correctness across token boundaries, stop-sequence
   truncation parity with the exclusive path, timeout → cancel.

Live (`tests/live/test_batching_real.py`, `real_model`):

4. **Parity**: same prompt, greedy — batch-of-1 output ≡ exclusive-path output.
5. **Rope canary** (§6).
6. **Concurrency win**: N=4 parallel HTTP requests complete with aggregate
   tok/s ≥ ~2× the serial baseline; TTFT of request 4 ≪ serial queueing.
7. **Prompt-cache round trip**: cache_id turn 1 (batched) → turn 2 reuses the
   prefix (assert `cache_read_tokens > 0`); radix sibling takeover still works
   on entries that passed through the batch.
8. **Interleave**: long-prompt batched request + concurrent embeddings request
   — embeddings completes after drain without deadlock; with a second batched
   request it interleaves (TTFT bounded).
9. Churn/stress: staggered inserts/removes for several minutes; memory bound
   respected (admission control kicks in rather than Metal OOM).

## 10. Phases

- **Phase 0 — prep** (small): promote `safe_rope_patch` + canary test;
  cache-probe helper; settings plumbing. No behavior change.
- **Phase 1 — core** (the bulk): `BatchScheduler` + async bridge +
  `_stream_completion_batched`; streaming chat only; no prompt-cache reuse
  (fresh prefill per request), no grammar; dense-`KVCache` models only;
  `OLMLX_BATCHING=true` opt-in. Even `completion_batch_size=8` with fresh
  prefills delivers the two headline wins (aggregate throughput + prefill
  interleaving).
- **Phase 2 — integration** (done): prompt-cache store move/extract round
  trip (`_setup_batched_prompt_cache` takes the entry out of the store —
  `take`/`async_take`, no shared mutable cache across lockless consumers —
  seeds the sequence via `insert(caches=…, all_tokens=…)`, and the worker
  hands the final KV back in the done event, eager-evaluated on the worker
  thread per #284; stop-hit cancellations still store, timeout/disconnect
  invalidate); non-streaming path (`_full_completion_batched` drains the
  batched stream — and gains inference_timeout enforcement, which the
  exclusive non-streaming path can't do); grammar per-sequence (verified:
  `GenerationBatch._step` calls per-sequence processors with `[1, vocab]`
  rows + per-sequence `TokenBuffer` history, prompt included on first call
  — exactly the `GrammarLogitsProcessor` contract; live JSON-mode batch
  test); `/api/ps` `batch_metrics` + `olmlx_batch_*` metrics. Also from the
  PR #507 review: shared `_enter_inference_lock`/`_exit_inference_lock`
  helpers (used by `_inference_locked`, `_stream_completion`, and the batch
  scheduler's `acquire_gpu`/`release_gpu`), and one `StopScanner`
  (`engine/stop_sequences.py`) replacing the three inline copies.
- **Phase 3 — hardening** (done): consumer backpressure (done —
  `OLMLX_BATCH_CONSUMER_LAG_LIMIT`, drop-the-laggard, see §11) and user
  docs (done — USER_MANUAL "Continuous Batching"). Per-request KV admission
  vs `OLMLX_MEMORY_LIMIT_FRACTION` landed in Phase 2 (`_batched_kv_preflight`);
  aggregate (cross-sequence) admission **done** (`OLMLX_BATCH_KV_ADMISSION`,
  default on — see §11). Per-model config **done**: a `batching: bool` at
  the top of a `models.json` entry overrides the global `OLMLX_BATCHING`
  opt-in for that model ("default-on per-model" — enable batching for one
  known-good dense model without flipping the global, or opt a model out;
  mechanical eligibility still applies on top), and `batch_completion_size`
  / `batch_prefill_size` / `batch_prefill_step` are likewise per-model
  overridable (resolved in `_get_batch_scheduler`, `ModelConfig` →
  `LoadedModel` → `_batch_eligible` mirroring the `prompt_cache`/
  `enable_thinking` override plumbing). Fairness-guard tuning **done**:
  `OLMLX_BATCH_FAIRNESS_QUANTUM` (seconds, default `0.0` = today's
  immediate yield; per-model overridable) gives the batch a minimum
  admission-open service window per busy period before it latches the
  exclusive-pending pause — a throughput floor under interleaved mixed
  traffic, bounding the *extra* wait imposed on the exclusive waiter to one
  quantum (`held` only grows, so the latch always fires within a quantum —
  exclusive is never starved). Observability: a `batch_fairness_pauses`
  counter (`/api/ps` + `olmlx_batch_fairness_pauses_total`) plus an info log
  on each yield, so the formerly-misleading `_queue_depth` "queued"
  semantics (a depth of 1 behind a batch means "waiting for K sequences to
  drain") are now visible from the batch side. The deep "starvation in
  reverse" fix (drain-and-switch, mid-generation preemption) stays Phase 4.
- **Phase 4 — extensions** (separate decisions): hybrid `ArraysCache` (GDN)
  models behind a flag; batch-capable KV-quant caches (upstream-shaped work);
  drain-and-switch policy for multi-model juggling; `insert_segments` +
  message-boundary segmentation for checkpoint-style reuse.

## 11. Risks / open questions

- **xgrammar in batch** (resolved, Phase 2): `GenerationBatch._step` calls
  per-sequence processors with `logits[e:e+1]` (`[1, vocab]`) and a
  per-sequence `TokenBuffer` context seeded with the sequence's full token
  history (prompt included — `PromptProcessingBatch.prompt()` appends the
  prompt to `self.tokens` before the generation transition). That matches
  `GrammarLogitsProcessor`'s first-call-is-prompt contract exactly. The
  context is an `mx.array` (not a list) — the processor's `len()`/slice/
  iterate usage handles both. Gated by the live JSON-mode batch test.
- **`BatchKVCache.extract` padding semantics** (resolved, Phase 2): extract
  slices `[idx, :, left_padding[idx]:_idx]` through `mx.contiguous` and
  resets `offset = keys.shape[2]` — a clean B=1 prefix. The live round-trip
  test stores from a mixed-length batch (real left-padding) and reuses at
  exact greedy parity with the exclusive path.
- **Hybrid models**: mlx-lm batches `ArraysCache` mechanically, but olmlx has
  a documented history of GDN/Metal-stream corruption (#284/#396). Keep out of
  v1; admit only with dedicated long-prompt parity tests.
- **Detokenizer fidelity**: streaming detokenizers differ subtly from
  `stream_generate`'s text deltas (e.g. byte-level BPE boundary handling);
  parity test #4 covers it.
- **Starvation in reverse** (partially mitigated, Phase 3): the fairness
  guard means heavy exclusive traffic (e.g. KV-quant model in rotation) can
  keep collapsing the batch. `OLMLX_BATCH_FAIRNESS_QUANTUM` (Phase 3) softens
  this — a non-zero quantum guarantees each busy period a minimum admission
  window, so under interleaved traffic the batch holds a duty cycle instead
  of collapsing to one sequence, at the cost of bounded extra exclusive
  latency. It does not preempt a running sequence, so it cannot bound the
  *drain* tail; the full fix (drain-and-switch / mid-generation preemption)
  stays Phase 4. Default `0.0` keeps the immediate-yield behavior.
- **Consumer backpressure** (PR #507 review; **resolved, Phase 3**): the
  per-sequence event bridge is still an unbounded `asyncio.Queue`, but the
  worker now drops a sequence whose unconsumed backlog exceeds
  `OLMLX_BATCH_CONSUMER_LAG_LIMIT` (default 2048; `0` disables). Lag is
  `BatchSequence._emitted` (worker writer) minus `_consumed` (consumer
  writer, bumped via `note_consumed()` after every `out.get()`) — each is
  single-writer so plain-int reads are torn-free under the GIL, the same
  discipline as the scheduler counters. `_sweep_lagging` flags over-limit
  sequences (sets `lagged`, clears `want_cache`) and `_sweep_cancelled`
  removes them with a `{"truncated": "lag"}` done event; the consumer
  treats that as a benign truncation (ends cleanly, stores nothing —
  parity with the timeout invalidate path) rather than the model-unload
  cancel (which still raises). Unlike CancellableStream's bounded
  `Queue(32)` *blocking* backpressure, a batch can't block one slow reader
  without stalling its co-tenants, so the v1 policy is drop-the-laggard.
- **Aggregate KV admission** (Phase 3, **resolved**): per-request
  `_batched_kv_preflight` rejects a single request whose own KV would blow
  `OLMLX_MEMORY_LIMIT_FRACTION`, but N consumers each pass it concurrently
  against the *same* pre-admission Metal reading and can collectively
  oversubscribe (none sees the others' not-yet-allocated KV). The worker
  closes the gap in `_admit` (`OLMLX_BATCH_KV_ADMISSION`, default on): it
  measures headroom once per tick (`limit − get_metal_memory()`, which nets
  out resident sequences' KV) and accumulates a `promised` total for
  sequences admitted earlier in the *same* tick whose prefill hasn't run yet
  (so their bytes aren't in the Metal reading). A candidate that would push
  `promised + estimate` over headroom is **deferred** — re-queued to the
  inbox tail, admission stops for the tick — and retried once a co-tenant
  finishes and headroom recovers (backpressure, not rejection: the whole
  point of batching is to admit when memory frees). An **empty batch always
  admits its first sequence** regardless of estimate — that lone request's
  fit is the per-request preflight's job, and gating it here would deadlock
  (nothing would ever free KV). The estimate is plain-fp16
  `estimate_kv_cache_bytes(suffix + max_tokens)` (batched eligibility
  excludes KV-quant; a reused prefix is already-resident, not re-counted).
  Because deferred sequences live in the real inbox (not worker-local
  state), a fairness pause that ends the busy period leaves them for the
  manager's `inbox-not-empty → re-arm` to pick up next period — no
  stranding.
- **Batched timing semantics differ**: `prompt_eval_duration` (TTFT) on the
  batched path includes batch-queue wait and co-tenant prefill interleave,
  and `eval_duration` is wall time shared with co-batched sequences — they
  measure user-perceived latency, not isolated model speed. The exclusive
  path's mlx-derived numbers remain the per-model benchmark reference.
- **Lock-holder restructuring** (resolved, Phase 3): the scheduler holding
  `_inference_lock` long-term changes `_queue_depth` log semantics ("queued"
  now often means "waiting for a whole batch to drain"). The batch manager
  already logs its own acquisition via a batch-specific `queued_log`; Phase 3
  adds an info log on each fairness yield (`"yielding lock to exclusive
  waiter after %.2fs service, %d running"`) and the `batch_fairness_pauses`
  counter, so an operator seeing a depth-1 "queued" line on an exclusive
  request can correlate it with the batch that's draining ahead of it.
