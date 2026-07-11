# olmlx

Ollama-compatible API server using Apple MLX for local inference on Apple Silicon.

## Usage Context

Single-user, localhost-only inference server:
- No auth, rate limiting, or TLS — localhost-bound by default
- Error messages may include internal details (paths, model names)
- The inference lock provides natural backpressure

## Project Structure

```
olmlx/
├── app.py              # FastAPI app factory, middleware, router registration
├── config.py           # Settings (pydantic-settings, OLMLX_ env prefix)
├── cli.py              # CLI subcommands (serve, chat, service, models, config, dflash, eagle, flash, spectral, shard, bench)
├── chat/               # In-process terminal chat + MCP agent loop
│   └── voice/          # Push-to-talk STT (Whisper) + Kokoro TTS
├── engine/
│   ├── inference.py    # generate_chat/completion/embeddings/transcription/rerank
│   ├── panel.py        # Multi-model panel + judge coordinator (per-turn reconciler)
│   ├── model_manager.py
│   ├── speculative_loaders.py
│   ├── spec_decoder_base.py  # SpecDecoderBase: shared mechanics for all speculative strategies
│   ├── speculative.py  # classic + PLD decoders
│   ├── proxy_tuning.py # Decode-time logit arithmetic (base + α·(expert−antiexpert))
│   ├── grammar.py      # xgrammar JSON-mode / JSON-Schema logits processor
│   ├── chat_templating.py
│   ├── tool_parser.py  # Qwen/GLM/Mistral/Llama/DeepSeek/MiniMax/bare-JSON
│   ├── template_caps.py
│   ├── prompt_cache/   # cache_id LRU + radix prefix index + disk spill
│   ├── turboquant*.py / spectralquant*.py / shardquant*.py
│   ├── rerank/         # MLX XLM-RoBERTa cross-encoder
│   ├── distributed.py
│   ├── flash/          # Sparse FFN + Flash-MoE (SSD-backed)
│   ├── dflash/         # Block-diffusion speculative decoder + training
│   ├── eagle/          # EAGLE speculative decoder + training
│   ├── mtp/            # Qwen3.6 native MTP head decoder
│   └── agent/          # Autonomous agent: orchestrator/store/memory/skills/delegate (OLMLX_AGENT_*)
├── routers/
│   ├── anthropic.py / openai.py / responses.py / audio.py / rerank.py / metrics.py
│   ├── streaming_common.py  # shared tools-mode buffering + keepalive
│   ├── thinking_split.py    # single <think>/Gemma-4-channel state machine
│   └── *.py            # Ollama /api/* routes
├── schemas/
└── utils/
    ├── streaming.py / metrics.py / timing.py / tracing.py
```

## Non-Obvious Invariants

**Metal stream hazard** — All non-speculative inference (prefill + decode) must run on `generation_stream`. Mixing streams leaves GatedDeltaNet/MoE recurrent state as a cross-stream lazy graph that corrupts expert routing on Qwen3.x hybrids — output looks coherent but is pretraining-data dumps. Speculative decoders are the exception: both prefill and decode run on `default_stream` (a split would reintroduce the same hazard). `snapshot_cache_for_persistence` does deepcopy + eager `mx.eval` before snapshots cross worker threads — lazy graphs bound to one stream crash on reuse from another.

Since mlx 0.32.0 (#499) streams are **thread-local**: `generation_stream` is a per-thread proxy, so "same stream" is enforced as "same thread" — the checkpoint-prefill drive is deferred into the generation worker thread (`deferred_prefill` closure), never run on the event loop. Worker GPU work is fenced by the worker's own end-of-run sync + thread join; the event-loop thread never syncs generation streams. `PromptCacheStore` stays loop-affine — worker-side checkpoint inserts marshal through `loop.call_soon_threadsafe`.

**Pure-RotatingKVCache prefill is single-chunk** — Models with only SWA (`RotatingKVCache`, no `ArraysCache`) — gpt-oss, Step-3.5, Gemma 3 — must prefill in ONE `model()` call. Splitting at a message boundary corrupts windowed attention: output is coherent but unrelated, and tool calls are silently skipped (tool defs add the system segment that creates the boundary). Detected by `_is_pure_rotating_cache`.

**Grammar tokenizer identity** — CPython recycles `id()` addresses; each grammar cache entry carries a `weakref` to its tokenizer and validates referent identity before serving. A stale entry with a recycled id would serve a wrong-vocabulary grammar. `drop_for_tokenizer` in `_close_loaded_model` runs **first** so no other close-step failure can skip it.

**TurboQuant deepcopy** — `TurboQuantKVCache.__deepcopy__` shares `mx.Dtype` by reference (pickle rejects it) and eager-evals private dequant side buffers — they're not exposed by `flatten_cache_state` and would leave a Metal-stream-bound lazy graph that crashes on cross-thread reuse.

**TurboQuant dequant-buffer shed-on-store** — `TurboQuantKVCache` holds a full-precision dequant side buffer (`_key_dequant`/`_value_dequant`, ~4–8x the packed footprint at 4-bit). `_estimate_state_bytes` walks `__dict__` and counts it, so an unshed cache trips the `prompt_cache_ram_budget_gb` soft-eviction at only ~30k tokens — and with disk spill off (default) the entry is silently dropped, forcing a full re-prefill every turn after. `_shed_transient_buffers` (store.py) calls `release_dequant_buffers()` from `_set_in_memory` — the single insertion chokepoint for **every** store path (`async_set`/`insert_checkpoint`/disk-restore/`takeover`/preflight), so the shed can't be missed at an individual call site. `update_and_fetch` rebuilds the buffers lazily from packed indices+norms on resume (so the rebuild must run *before* the resize/splice paths), and `trim`'s buffer-shrink branch guards against the shed (`None`) buffers. Only TurboQuant is affected — Spectral dequantizes to locals, Shard dequantizes on read.

**Rotation matrices must be eager-eval'd at construction** — `TurboQuantRotation.__init__` and `SpectralRotation.__init__` precompute `matrix_T`/`V_T` as `matrix.T` — a *lazy* transpose op. The prompt cache (and its rotations) is built on the **event-loop thread** (`_make_prompt_cache_for_lm`), while prefill/decode run on a separate `asyncio.to_thread` worker. Under mlx ≥0.31.2 thread-local streams (#499) the lazy op stays bound to the constructing thread's stream, so evaluating any graph that references it from the worker thread raises `There is no Stream(gpu, N) in current thread` — surfacing at flash-MoE's `mx.eval(inds)` on the first token (bricked every `kv_cache_quant`-configured flash-MoE model). Both `__init__`s call `mx.eval(matrix, matrix_T)` so the rotations are materialized leaves, safe to read from any thread (materialized arrays carry no stream binding — same rationale as `_pin_state_to_offset`). NOTE: a *second*, separate `no Stream(gpu, N)` crash still hits some prompts on small-`flash_moe_cache_budget_experts` models (GLM-Air, 235B) under heavy expert eviction — an unfixed suspected race in the flash-store io-thread path, distinct from this rotation fix.

**Hybrid VLM routing** — VLMs whose `text_config.layer_types` contains `"linear_attention"` (Qwen3.5, Qwen3_5_moe) route through mlx-lm, not mlx-vlm. The mlx-vlm path crashes with a Metal stream error on text inference.

**Distributed: eval before first forward** — After `model.shard()`, must `mx.eval(model.parameters())` on every rank before any forward pass. The combined lazy materialization + `all_sum` Metal command buffer times out (~10s GPU timeout) on 32B+ models.

**Distributed: signal handlers before spawn** — Worker cleanup signal handlers must be installed before spawning SSH workers. The default SIGTERM disposition skips atexit, which orphans SSH workers if the coordinator dies during the pre-uvicorn startup window.

**MTP concat order** — In `MTPDecoder`, the fc input is `concat([embed, hidden])` — embed first, which is the **opposite** of EAGLE's order. Getting it wrong drops acceptance to ~0.006 while remaining exactness-preserving (correct but useless). Determined empirically via `scripts/mtp_decoder_probe.py`.

**MTP prefill chunking** — `MTPDecoder.prefill` pass-1 must chunk through `_chunked_prefill`. A single forward over a long prompt forces `lm_head` over every position, producing a `[1, seq-1, vocab]` float32 tensor that OOMs Metal (~44 GB at 73k tokens vs ~41.7 GB limit). GDN capture is suppressed (`use_buffer(None)`) during the chunked prefix — the capture call appends per forward, so leaving it active rebloats memory across sub-chunks on hybrid GDN targets.

**DFlash training** — Always use `--self-generate`. Ground-truth dataset text → ~0.4% acceptance. Target-generated responses → real acceptance (~45% p1 on Qwen3-0.6B). This is the upstream recipe.

**TTS priming** — The router primes the PCM source generator's first chunk before starting the response. Priming the ffmpeg-encoded stream instead wouldn't work: ffmpeg emits its container header before reading stdin, so an upstream `ValueError` from a non-TTS model would leak after the 200 has been sent.

**Flash prefetcher teardown order** — Prefetcher must be closed before the weight store on model unload. Prefetch tasks submit work into the store's thread pool; reversing the order leaves in-flight tasks referencing a closed store.

**Flash-MoE router dispatch order** — `_replace_moe_layers` picks a replacement class by gate shape, and the order of the `elif` chain is load-bearing. A plain `nn.Linear`/`QuantizedLinear` gate returns logits; a custom gate (DeepSeek/Kimi) returns `(inds, scores)` directly. The two linear-gate special cases (Qwen3-Next = `shared_expert_gate`, MiniMax = `e_score_correction_bias`) and the plain-Qwen3Moe case (`_FlashMoEQwen3`, e.g. Qwen3-235B-A22B) must all be matched **before** the `gate is not None` DeepSeek branch — otherwise a linear gate falls into `_FlashMoEDeepSeek._route`, which unpacks `inds, scores = self.gate(x)` and dies with `not enough values to unpack (expected 2, got 1)` on the first forward.

**Flash-MoE expert parse skips a defensive copy** — an expert-cache miss's cost is ~75% parse (host-side `mx.array` construction), not I/O: experts are page-cache-resident, so `pread` runs at memcpy speed (~157 µs for a 1.69 MiB expert). The three `_parse_*` methods (`moe_weight_store.py`) therefore parse from a **zero-copy `memoryview`** and let `mx.array` do the *one* necessary host→device copy — no bytes-slice, no `np.frombuffer(...).copy()`. This is safe because `mx.array` copies host data at construction (so parsed arrays never alias `raw`, even though `raw` is freed before the deferred eval); `test_parse_does_not_alias_source_buffer` locks that invariant. Removing the redundant copy was worth **+7–23% decode throughput** (bigger at low cache budgets, where misses dominate). Corollary — the reason predictive expert prefetch/lookahead (#650) never beat LRU and was **removed** (2026-07-11): I/O was never the bottleneck (experts are page-cache-resident), so speculative reads only added CPU/copy contention, and the router is load-balanced (near-uniform) so no causal predictor can anticipate reuse — the Flash-MoE store is now plain `LayerLruCache`, no `ScoredLayerCache`, no `MoePrefetcher`/`MoeLookaheadBank`. The layout offset-table `.copy()` at load time is unrelated and correct (the backing `f.read()` bytes go out of scope).

**Autonomous agent rides ChatSession** — `engine/agent/` (gated on `OLMLX_AGENT_ENABLED`) drives goal-pursuit by wrapping the existing `ChatSession` ReAct loop, never calling MLX directly — so the inference-lock / Metal-stream handling is reused unchanged (same constraint as the panel coordinator). The one behavioral extension is *continuation semantics*: a stop without the `finish` tool injects a nudge and the run continues. `finish` is detected from the `tool_call` **event stream**, not a return value; `AgentToolManager` subclasses `BuiltinToolManager` so `ChatSession` dispatches the control tools (`finish`/`remember`/`recall`/`create_skill`/`delegate`) transparently. `AgentStore` shares one SQLite connection across `asyncio.to_thread` worker threads (`check_same_thread=False` + a `threading.Lock`); all runtime I/O is offloaded so the event loop never blocks on disk. Subagents share the global inference lock — `DelegateRunner` serializes child generation behind a semaphore (a concurrency-limited queue, not wall-clock parallelism).

**Panel coordinator routes through generate_chat** — `engine/panel.py` is a per-turn reconciler that rides the client's tool loop: it returns a `generate_chat`-shaped result whose text is either the judge's prose or canonical Qwen `<tool_call>` blocks, so both routers' existing `parse_model_output` paths handle it unchanged. Every classifier/panelist/judge call MUST go through `generate_chat` — never call MLX directly — or the Metal-stream/inference-lock handling is bypassed. Panels need `max_loaded_models ≥ members + judge + classifier` to avoid reload thrash.

**Proxy-tuning is a non-exactness-preserving speculative strategy** — `engine/proxy_tuning.py`'s `ProxyTuningDecoder` registers as `speculative_strategy="proxy_tuning"` to reuse the speculative lifecycle/stream/dispatch plumbing, but it *alters* the output distribution (`base + α·(expert − antiexpert)`), so it deliberately has no draft→verify→accept and no cache trimming — every model advances over the same committed tokens. Expert/anti-expert are loaded **inline by the loader** (held by the decoder, not in `ModelManager._loaded`), so no `max_loaded_models` bump is needed — unlike the panel coordinator. All three models must share one exact tokenizer/vocabulary (`check_vocab_identity` enforces token-mapping equality, stricter than `_check_vocab_match`'s size-only test). Supports **dense and hybrid GDN/MoE** bases: it installs no GDN capture because that only undoes rejected draft tokens and proxy-tuning never speculates (no draft→verify→accept, no trimming), so per-token forward + `make_prompt_cache` keeps GDN recurrent state correct on the default stream; verified on a hybrid Qwen3.5 base (`tests/test_proxy_tuning_hybrid.py`). Grammar disables it per-request like every other speculative strategy.

**LoRA-adapter hot-swap shares base weights via a structural copy** — `base:adapter` entries (`engine/registry.py` `AdapterConfig`, `"adapters"` section of models.json) load through `ModelManager._load_adapter_model`, which `structural_copy`s the base model and applies `mlx_lm.tuner.utils.load_adapters`. The copy is deliberately neither `copy.copy` (shallow — shares the `layers` list/submodules, so `load_adapters` would corrupt the base) nor `copy.deepcopy` (duplicates `mx.array` weights, and crashes on quantized models because `mx.Dtype` is unpicklable): it rebuilds the `Module`/list/dict tree with `mx.array` leaves and scalar attrs (`bits`, dtypes) shared by reference, and copies the per-module `_no_grad` set so freezing one instance can't affect another. The adapter `LoadedModel` reuses the **same tokenizer object** as the base, so `_close_loaded_model` skips `drop_for_tokenizer` for adapter entries (`lm.adapter_base` set) — the base drops it once its last adapter is gone. The base is pinned against LRU/expiry eviction while `_adapter_child_refs > 0` (incremented under the manager lock at load, decremented via `_detach_adapter_locked` at every `_loaded` removal); unloading a base with live adapters raises `ActiveRequestsError`. Needs `max_loaded_models ≥ base + adapters` (each occupies a slot). VLM/flash/distributed/speculative/KV-quant/whisper/tts/reranker bases are rejected (`_reject_adapter_base`). Verified end-to-end on a real model in `tests/test_adapter_loading.py` (Metal-gated).

## Development

```bash
uv sync --no-editable
uv run olmlx          # http://localhost:11434
uv run pytest
```

Models config at `~/.olmlx/models.json` (auto-created). Override for dev: `OLMLX_MODELS_CONFIG=models.json`.

### TDD

Write failing tests first, then implement to make them pass. For bug fixes, write a test that reproduces the bug before writing the fix.

## Git

- Remote: `git@github.com:motsognirr/olmlx.git`
- Author: Daniel Palmqvist <daniel.u.palmqvist@gmail.com>

## Type Annotations

- **TypedDict**: `total=True` for required keys, `total=False` for optional. Never empty — use `dict[str, Any]` for freeform JSON.
- **Pydantic field types**: Prefer `dict[str, Any]` over TypedDict for free-shape fields (e.g. `Tool.function`) — Pydantic v2 enforces required keys and silently strips extras.
- **Protocol return types**: `__aexit__` returns `bool | None` (not `None`) to allow suppression. `__call__` must match actual return type (mlx-lm models return `mx.array`, not `dict`).
- **AsyncIterator**: If yielding both `str` and `dict`, annotate `AsyncIterator[str | dict[str, Any]]`.
- **TYPE_CHECKING**: Guard annotation-only imports when module has `from __future__ import annotations`.
- **Protocol return values**: Match the real type (e.g. `ClientSession.initialize()` returns `InitializeResult`).
