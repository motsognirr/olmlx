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
├── cli.py              # CLI subcommands (serve, chat, service, models, config, dflash, eagle, flash, spectral, shard)
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
│   ├── tool_parser.py  # Qwen/Mistral/Llama/DeepSeek/MiniMax/bare-JSON
│   ├── template_caps.py
│   ├── prompt_cache/   # cache_id LRU + radix prefix index + disk spill
│   ├── turboquant*.py / spectralquant*.py / shardquant*.py
│   ├── rerank/         # MLX XLM-RoBERTa cross-encoder
│   ├── distributed.py
│   ├── flash/          # Sparse FFN + Flash-MoE (SSD-backed)
│   ├── dflash/         # Block-diffusion speculative decoder + training
│   ├── eagle/          # EAGLE speculative decoder + training
│   └── mtp/            # Qwen3.6 native MTP head decoder
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

**Pure-RotatingKVCache prefill is single-chunk** — Models with only SWA (`RotatingKVCache`, no `ArraysCache`) — gpt-oss, Step-3.5, Gemma 3 — must prefill in ONE `model()` call. Splitting at a message boundary corrupts windowed attention: output is coherent but unrelated, and tool calls are silently skipped (tool defs add the system segment that creates the boundary). Detected by `_is_pure_rotating_cache`.

**Grammar tokenizer identity** — CPython recycles `id()` addresses; each grammar cache entry carries a `weakref` to its tokenizer and validates referent identity before serving. A stale entry with a recycled id would serve a wrong-vocabulary grammar. `drop_for_tokenizer` in `_close_loaded_model` runs **first** so no other close-step failure can skip it.

**TurboQuant deepcopy** — `TurboQuantKVCache.__deepcopy__` shares `mx.Dtype` by reference (pickle rejects it) and eager-evals private dequant side buffers — they're not exposed by `flatten_cache_state` and would leave a Metal-stream-bound lazy graph that crashes on cross-thread reuse.

**Hybrid VLM routing** — VLMs whose `text_config.layer_types` contains `"linear_attention"` (Qwen3.5, Qwen3_5_moe) route through mlx-lm, not mlx-vlm. The mlx-vlm path crashes with a Metal stream error on text inference.

**Distributed: eval before first forward** — After `model.shard()`, must `mx.eval(model.parameters())` on every rank before any forward pass. The combined lazy materialization + `all_sum` Metal command buffer times out (~10s GPU timeout) on 32B+ models.

**Distributed: signal handlers before spawn** — Worker cleanup signal handlers must be installed before spawning SSH workers. The default SIGTERM disposition skips atexit, which orphans SSH workers if the coordinator dies during the pre-uvicorn startup window.

**MTP concat order** — In `MTPDecoder`, the fc input is `concat([embed, hidden])` — embed first, which is the **opposite** of EAGLE's order. Getting it wrong drops acceptance to ~0.006 while remaining exactness-preserving (correct but useless). Determined empirically via `scripts/mtp_decoder_probe.py`.

**MTP prefill chunking** — `MTPDecoder.prefill` pass-1 must chunk through `_chunked_prefill`. A single forward over a long prompt forces `lm_head` over every position, producing a `[1, seq-1, vocab]` float32 tensor that OOMs Metal (~44 GB at 73k tokens vs ~41.7 GB limit). GDN capture is suppressed (`use_buffer(None)`) during the chunked prefix — the capture call appends per forward, so leaving it active rebloats memory across sub-chunks on hybrid GDN targets.

**DFlash training** — Always use `--self-generate`. Ground-truth dataset text → ~0.4% acceptance. Target-generated responses → real acceptance (~45% p1 on Qwen3-0.6B). This is the upstream recipe.

**mlx 0.31.x rope bug** — `mx.fast.rope` corrupts batch rows ≥1 at B>1, L==1 (decode shape). `safe_rope_patch` in `engine/ropefix.py` folds B into the heads dim as the workaround. Applied in the continuous batching worker loop and dflash selfgen. Remove once mlx fixes it (guarded by a Metal-gated unit test).

**TTS priming** — The router primes the PCM source generator's first chunk before starting the response. Priming the ffmpeg-encoded stream instead wouldn't work: ffmpeg emits its container header before reading stdin, so an upstream `ValueError` from a non-TTS model would leak after the 200 has been sent.

**Flash prefetcher teardown order** — Prefetcher must be closed before the weight store on model unload. Prefetch tasks submit work into the store's thread pool; reversing the order leaves in-flight tasks referencing a closed store.

**Panel coordinator routes through generate_chat** — `engine/panel.py` is a per-turn reconciler that rides the client's tool loop: it returns a `generate_chat`-shaped result whose text is either the judge's prose or canonical Qwen `<tool_call>` blocks, so both routers' existing `parse_model_output` paths handle it unchanged. Every classifier/panelist/judge call MUST go through `generate_chat` — never call MLX directly — or the Metal-stream/inference-lock handling is bypassed. Panels need `max_loaded_models ≥ members + judge + classifier` to avoid reload thrash.

**Proxy-tuning is a non-exactness-preserving speculative strategy** — `engine/proxy_tuning.py`'s `ProxyTuningDecoder` registers as `speculative_strategy="proxy_tuning"` to reuse the speculative lifecycle/stream/dispatch plumbing, but it *alters* the output distribution (`base + α·(expert − antiexpert)`), so it deliberately has no draft→verify→accept and no cache trimming — every model advances over the same committed tokens. Expert/anti-expert are loaded **inline by the loader** (held by the decoder, not in `ModelManager._loaded`), so no `max_loaded_models` bump is needed — unlike the panel coordinator. All three models must share one exact tokenizer/vocabulary (`check_vocab_identity` enforces token-mapping equality, stricter than `_check_vocab_match`'s size-only test). v1 is **dense-only** (no GDN capture installed); grammar disables it per-request like every other speculative strategy.

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
