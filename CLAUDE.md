# olmlx

Ollama-compatible API server using Apple MLX for local inference on Apple Silicon.

## Usage Context

This is a **single-user, localhost-only** inference server. Implications for code review:

- **No authentication/authorization** — localhost-bound by default; no user accounts or API keys.
- **No rate limiting** — one user, one machine. The inference lock provides natural backpressure.
- **No TLS** — traffic stays on the loopback interface.
- **Error messages may include internal details** (paths, model names) — acceptable for a local tool.
- **No Cache-Control headers** — no shared proxy or browser caching concerns.

## Project Structure

```
olmlx/
├── app.py              # FastAPI app factory, middleware, router registration
├── config.py           # Settings (pydantic-settings, OLMLX_ env prefix)
├── cli.py              # CLI subcommands (serve, chat, service, models, config, dflash, eagle, flash, spectral)
├── chat/               # In-process terminal chat (ChatSession + MCP tool client + Rich TUI)
├── engine/
│   ├── inference.py    # generate_chat, generate_completion, generate_embeddings, generate_transcription
│   ├── model_manager.py # Model loading/unloading, keep-alive, LRU eviction
│   ├── registry.py     # Ollama name → HF repo via models.json
│   ├── distributed.py  # Multi-machine ring tensor parallelism (TCP sideband)
│   ├── grammar.py      # xgrammar JSON-mode / JSON-Schema logits processor
│   ├── speculative.py  # Classic + PLD speculative decoders
│   ├── tool_parser.py  # Qwen/Mistral/Llama/DeepSeek/MiniMax/bare-JSON tool call parsing
│   ├── template_caps.py # Chat template capability detection (tools, thinking)
│   ├── prompt_cache/   # KV cache reuse: cache_id LRU + radix prefix index, disk spill, checkpoint path
│   ├── turboquant*.py  # Random-rotation + Lloyd-Max KV quantization
│   ├── spectralquant*.py # Eigenvector-rotation + non-uniform KV quantization (requires calibration)
│   ├── flash/          # Sparse FFN + SSD-backed neuron loading (dense + MoE)
│   ├── dflash/         # Block-diffusion speculative decoder + training pipeline
│   └── eagle/          # EAGLE autoregressive speculative decoder + training pipeline
├── models/             # ModelStore: local-first model storage and manifests
├── routers/
│   ├── anthropic.py    # /v1/messages — Anthropic Messages API
│   ├── openai.py       # /v1/chat/completions, /v1/completions, /v1/models, /v1/embeddings
│   ├── audio.py        # /v1/audio/transcriptions — Whisper STT
│   └── *.py            # /api/{chat,generate,tags,show,ps,pull,copy,create,delete,embed,blobs,version} (Ollama)
├── schemas/            # Pydantic request/response models per API surface
└── utils/
    ├── streaming.py    # async_mlx_stream, safe_ndjson_stream
    └── timing.py       # Timer, TimingStats
```

## Key Design Decisions

- **Anthropic router**: Buffers full model output before emitting SSE so `<think>` and `<tool_call>` raw text from Qwen-family models can be split into proper content blocks. Also exposes `/v1/messages/count_tokens`.
- **Thinking toggle**: Resolved per-request to engine's `enable_thinking: bool | None`. Ollama uses native top-level `think` field; OpenAI uses `reasoning_effort` (presence → on) or `chat_template_kwargs.enable_thinking` (authoritative, only clean OFF). Mapping helpers in `routers/common.py`. Omission on `/api/chat` and `/v1/chat/completions` falls back to per-model default (`ModelConfig.enable_thinking`), then engine default. `/api/generate` stays off-by-default and ignores the per-model default.
- **Structured outputs** (`engine/grammar.py`): JSON-mode / JSON-Schema via xgrammar as an mlx-lm logits processor. Surfaces on OpenAI `response_format` and Ollama `format` (both `/api/chat` and `/api/generate`). Compiled grammars cached per (tokenizer-id, spec-hash). VLM, distributed, and speculative paths skip with a warning — `logits_processors` isn't threaded through any of them.
- **Tool call parsing**: Qwen `<tool_call>`, Mistral `[TOOL_CALLS]`, Llama 3.x `<function=>`, DeepSeek, MiniMax `<minimax:tool_call>`, bare JSON. `tool_result` blocks → `role: "tool"` messages; `tool_use` blocks → `tool_calls`; thinking blocks in history are skipped.
- **Tool format conversion**: Anthropic tool defs converted to OpenAI-style `{"type": "function", ...}` for `apply_chat_template()`.
- **Tool-result message rendering**: Before templating, `role: "tool"` turns are rewritten to whatever the chat template can render, keyed off `TemplateCaps`. Gemma-style templates (`uses_tool_responses`) take a `tool_responses` array merged into the preceding assistant turn; minimal templates that only allow user/system/assistant and raise otherwise (`handles_tool_role` False — e.g. Devstral/Mistral) get tool calls + results folded into user-message text via `_convert_tool_messages_to_user_text`. Templates that natively handle the `tool` role pass through unchanged. Applied once before the VLM/text split in `generate_chat`.
- **Model storage**: HF repo path (e.g. `Qwen--Qwen3-8B`). Direct HF paths auto-register in `models.json` on first load or pull.
- **Active inference protection**: `LoadedModel.active_refs` prevents LRU eviction and expiry of models currently serving requests.
- **Memory safety**: After load, checks Metal active+cache against `OLMLX_MEMORY_LIMIT_FRACTION`; oversized rejected with HTTP 503 to prevent uncatchable Metal OOM. `OLMLX_INFERENCE_HEADROOM_FRACTION` reserves space *below* the limit for KV/activations (effective weight budget = `limit - headroom`). Under pressure, `ensure_loaded` first flushes resident models' prompt caches, then evicts idle (`active_refs == 0`) LRU models. Active models are never evicted.
- **Stream cleanup**: NDJSON routers use `safe_ndjson_stream()` (shared async gen with error handling + guaranteed `aclose()`). OpenAI/Anthropic have SSE-specific cleanup.
- **Prompt caching**: Cross-request KV reuse keyed by `cache_id` (LRU). Radix prefix index (`PrefixCacheIndex`) catches sibling-prefix takeover (Claude-Code-style branching): on miss, walks the trie and re-keys the matched entry to the new `cache_id` — no KV copy, but the old `cache_id` loses its entry. Disk spill for cold entries; async wrappers offload disk I/O via `asyncio.to_thread`. Tunables under `OLMLX_PROMPT_CACHE_*`; metrics on `/api/ps`. Speculative decoders gated off (own their target/draft caches); non-streaming VLM gated off (`mlx_vlm.generate` doesn't accept `prompt_cache`).
- **Message-boundary checkpoint path** (prompt cache): For non-trimmable cache layouts — `ArraysCache` SSM hybrids (Qwen3.5, Nemotron-H, Jamba) and mixed Rotating+Arrays (Qwen3-Next) — prefill runs in **at most two chunks** with one snapshot at the deepest interior message boundary. Each chunk's `model()` forward runs on mlx-lm's `generation_stream` (the stream `stream_generate` decodes on) and is sub-chunked at `_PREFILL_CHUNK` (2048). **The stream is load-bearing**: prefilling on `mx.default_stream` and then decoding on `generation_stream` left the `gated_delta_kernel` recurrent state a cross-stream lazy graph whose materialization corrupted MoE expert routing on Qwen3-Next-family quantized targets (Qwen3-Coder-Next, Qwen3.6-35B-A3B) — coherent-but-off-prompt pretraining-data dumps once the prompt exceeded ~16k tokens (same Metal-stream hazard family as #284). Running prefill on the decode stream fixes it; the message-boundary chunking was never the cause. Sub-chunking bounds activation memory (a ~70k-token single forward OOM'd Metal at 123 GB). `snapshot_cache_for_persistence` does deepcopy + eager `mx.eval` to materialize lazy graphs before the cross-thread crossing (#284, #343, #396).
- **Pure-`RotatingKVCache` (SWA) prefill is single-chunk**: sliding-window models with no `ArraysCache` layers (gpt-oss, Step-3.5, Gemma 3 — detected by `_is_pure_rotating_cache`) are prefilled in **one** `model()` call (not sub-chunked), with one snapshot at the end (KV depth `len-1`). Splitting prefill at an interior message boundary corrupts their windowed attention — coherent-but-unrelated text, skipped tool calls (only with `tools`, since tool defs add the `system`/`developer` segments that create the boundary). So they skip the two-chunk message-boundary split *and* the `_PREFILL_CHUNK` sub-chunking that the GatedDeltaNet family uses; both branches still run on `generation_stream`.
- **KV-quant + checkpoint composition**: `TurboQuantKVCache.__deepcopy__` shares the `mx.Dtype` singleton by reference (default reductor falls back to pickle, which rejects `mx.Dtype`) and eager-evals private `_key_dequant`/`_value_dequant` side buffers that `flatten_cache_state` doesn't expose (otherwise the snapshot leaves a Metal-stream-bound graph and crashes on cross-thread reuse — generalized #284 hazard). `SpectralQuantKVCache` uses the default walk. Disk-save remains separately gated by `_is_serializable_cache` (packed-index layout has no safetensors form).
- **Hybrid VLM detection**: VLMs combining vision + linear-attention hybrid text towers (Qwen3.5, Qwen3_5_moe) ship a text-only mlx-lm module. `_detect_model_kind` routes them through mlx-lm because the mlx-vlm path crashes with a Metal stream error on text inference (#284). Discriminator: `text_config.layer_types` contains `"linear_attention"`.
- **KV quantization** (`OLMLX_KV_CACHE_QUANT`):
  - `turboquant:{2,4}` — random rotation → Lloyd-Max → bit-packed (~7.5× / ~3.9× compression). Drop-in `KVCache` replacement. `update_and_fetch` dequantizes only new tokens into a reused per-cache side buffer (avoids re-dequantizing history each step at ~100-200 MB extra buffer).
  - `spectral:{2,4}` — data-driven eigenvector rotation, non-uniform bit allocation (more bits for high-variance semantic dims, fewer for tail). Requires `olmlx spectral prepare <model>` first; missing calibration raises HTTP 400 with the exact command. Auto-calibrate via `OLMLX_KV_CACHE_AUTO_CALIBRATE=true`.
- **Flash inference** (`engine/flash/`): SSD-backed sparse FFN for models > GPU memory. One-time prep: `olmlx flash prepare <model>` bundles FFN to `.flashweights`, trains sparsity predictors. Runtime: `FlashModelWrapper` consults predictor per token, loads active neurons via `FlashWeightStore` (`LayerLruCache` + optional preallocated buffer + optional `O_DIRECT`/`F_NOCACHE`). Promoted knobs: `OLMLX_FLASH`, `*_SPARSITY_THRESHOLD`, `*_MIN/MAX_ACTIVE_NEURONS`, `*_MEMORY_BUDGET_FRACTION`, `*_PREFETCH`, `*_SPECULATIVE*`. Advanced knobs remain under `OLMLX_EXPERIMENTAL_FLASH_*`. Per-model overrides for promoted fields go at top level of `models.json`; legacy `experimental.flash_*` for promoted keys raises a migration error. **Per-model overrides silently ignored on distributed workers** — `worker_main` reads globals, not the registry.
- **Flash-MoE** (`engine/flash/flash_moe*.py`): SSD-backed routed-expert offloading. Router, shared experts, attention, embeddings stay in RAM; routed expert weight matrices live on SSD, loaded per-token via per-layer LRU + parallel `pread()`. Supported: DeepSeek-V3, Kimi-K2.5, Qwen3-Next MoE, MiniMax, gpt-oss, Gemma4 VLM, Step-3.5. Knobs: `OLMLX_FLASH_MOE`, `_CACHE_BUDGET_EXPERTS`, `_IO_THREADS`. **Incompatible with distributed** (expert weights can't be sharded) and with feature-conditioned speculative (`dflash`, `eagle`); `classic` and `pld` speculative are supported.
- **Distributed inference**: Tensor-parallel across Apple Silicon via MLX ring backend. `distributed_strategy` is fixed to `"tensor"` (pipeline literal removed; dormant code retained for #273). Sideband server starts in CLI before uvicorn because `import transformers` can take minutes. Worker sideband connect retries with exponential backoff (up to 120s). **After `model.shard()`, must `mx.eval(model.parameters())` on every rank before any forward pass** — otherwise the combined lazy materialization + `all_sum` Metal command buffer exceeds the ~10s GPU timeout for 32B+ models. Hostfile: `~/.olmlx/hostfile.json`. Flash + distributed shards only attention; FlashMLP layers per-rank from local SSD (correct because sharded `o_proj` replicates output via `all_sum`).
- **Speculative decoding** (`OLMLX_SPECULATIVE_STRATEGY`): Four strategies share `prefill`/`step`/`reset` + `verify_draft_greedy`.
  - `classic` (default): standalone draft LM, λ candidates per verify. `SpeculativeFlashDecoder` extends with prefetcher + adaptive window. **Empirically best for hybrid linear-attention targets (Qwen3.5/3.6)**: ~82% acceptance, 1.33-1.92× speedup with a 0.8B draft.
  - `dflash`: block-diffusion draft, cross-attention over target hidden states (`DFlashAttention`). Universal target via `_LayerHook` + `_patch_model`. `_GDNStateCapture` monkey-patches `GatedDeltaNet.__call__` for rejection rollback on linear-attention targets. Loads drafts with nested `dflash_config.{target_layer_ids, mask_token_id}`. Acceptance ~2% on Qwen3.5 (feature-space drift + MASK convention mismatch).
  - `eagle` (arxiv 2401.15077): autoregressive draft over `concat([h_target, embed(token_prev)])`, shares `lm_head` via `bind()`. Same `_patch_model` + GDN rollback. Loads drafts with `eagle_config.{block_size, target_layer_id}`. Acceptance ~6% on Qwen3.5 (compounding feature-chain error + per-token `.item()` Metal sync + draft cache growth `num_accepted - 1` per step).
  - `pld`: prompt-lookup decoding — no draft model, n-gram match on prompt+history. **Only strategy compatible with Flash-MoE**. Reuses `_GDNStateCapture` for hybrid targets. `step()` returns `(accepted, num_drafted)`; EMA acceptance skips no-match steps so streaks don't drag the rate to 0. Tunables: `*_PLD_MAX_NGRAM` (3), `*_PLD_MIN_NGRAM` (1), `*_PLD_LOOKUP_WINDOW` (8192).
  - All compose with prompt caching, tools, streaming. Per-model overrides at top level of `models.json` (not `experimental`); legacy `OLMLX_EXPERIMENTAL_DFLASH*` raises migration error.
- **DFlash / EAGLE draft training** (`olmlx {dflash,eagle} prepare`): Load target via mlx-lm, install `_patch_model` hooks, build `DraftConfig` from target's config (vocab/head dims/GQA shape inherited), `bind()` borrows `embed_tokens`/`lm_head`, stream batches from HuggingFace (default `HuggingFaceH4/ultrachat_200k`). Frozen target via `mx.stop_gradient`; `nn.value_and_grad(draft, ...)`. AdamW + cosine LR. Output: `<model-dir>/{dflash,eagle}/` in upstream-compatible schema. Pad contamination handled in `_select_pivot` (per-batch unpadded prefix) and `_draft_loss` (zero-weights pad targets). Multimodal targets supported via `_text_config()` descent.
  - DFlash: `--distill` (KL against target logits, `(1-α)·CE + α·T²·KL`), `--train-windows-per-step N` (amortise target forward across K non-overlapping masked windows).
  - DFlash precompute (`olmlx dflash precompute`): dump `(input_ids, target_hidden)` shards to skip per-step target forward in subsequent training. Same shards work for both DFlash and EAGLE.
  - EAGLE: `--use-precomputed` only. Pairing is `(h_{t-1}, token_t) → token_{t+1}` (off-by-one collapses bench acceptance to ~1%). `--sample-positions N` (default 256) subsamples vocab projection (~10× speedup, unbiased CE estimator). `iter_precomputed_shards` retries transient `mx.load()` failures (macOS mmap/fd pressure).
- **Speculative prefetch** (`OLMLX_FLASH_PREFETCH`): Pre-loads neurons from SSD ahead of need. Cross-layer (layer L predicts L+1 from L's pre-MLP hidden, optional dedicated `LookaheadBank`), draft-informed (bulk prefetch for all target layers from draft hiddens). **Prediction synchronous** (MLX `mx.eval` deadlocks under concurrent multi-thread use); only I/O is async via dedicated `ThreadPoolExecutor`. **Prefetcher must be closed before weight store** on model unload (prefetch tasks submit into the store's pool).
- **Audio transcription**: OpenAI-compatible `/v1/audio/transcriptions` via mlx-whisper. Whisper is a first-class `ModelManager` kind: `_detect_model_kind` matches `model_type == "whisper"` or mlx-whisper dims (`n_mels`/`n_audio_state`); `LoadedModel.is_whisper` guards LLM-only prompt-cache/KV-quant paths. `generate_transcription` injects the managed model into `mlx_whisper.transcribe.ModelHolder` (the module-level singleton — race-free under the serialized lock) and runs transcribe in a worker thread. Formats: `json`, `verbose_json`, `text`, `srt`, `vtt`. `ForceJSONMiddleware` skips `multipart/form-data` so upload boundary survives. **Requires ffmpeg on PATH**. Streaming and diarization out of scope.
- **Terminal chat**: `olmlx chat` runs in-process via `ModelManager`/`generate_chat()` — no HTTP. Connects to external MCP servers (stdio/SSE) with a full agent loop. MCP config in Claude Desktop format at `~/.olmlx/mcp.json`.
- **Model load timeout**: `OLMLX_MODEL_LOAD_TIMEOUT` → `ModelLoadTimeoutError` (HTTP 504).

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
