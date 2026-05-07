# olmlx

Ollama-compatible API server using Apple MLX for local inference on Apple Silicon.

## Usage Context

This is a **single-user, localhost-only** inference server. It is not designed for multi-tenant or public-facing deployment. Implications for code review:

- **No authentication/authorization** — the server binds to localhost by default; there are no user accounts or API keys.
- **No rate limiting** — one user, one machine. The inference lock provides natural backpressure.
- **No TLS** — traffic stays on the loopback interface.
- **Error messages may include internal details** (paths, model names) — acceptable for a local tool.
- **No Cache-Control headers** — no shared proxy or browser caching concerns.

## Project Structure

```
olmlx/
├── app.py              # FastAPI app factory, middleware, router registration
├── config.py           # Settings (pydantic-settings, OLMLX_ env prefix)
├── cli.py              # CLI with subcommands (serve, chat, service, models incl. search, config)
├── __main__.py         # Entry point (delegates to cli.py)
├── chat/
│   ├── __init__.py     # Package exports
│   ├── config.py       # ChatConfig dataclass, load_mcp_config()
│   ├── mcp_client.py   # MCPClientManager: connect/discover/call MCP servers
│   ├── session.py      # ChatSession: agent loop, message history, tool execution
│   └── tui.py          # ChatTUI: Rich-based terminal UI with streaming markdown
├── engine/
│   ├── distributed.py  # DistributedCoordinator/Worker sideband protocol (TCP)
│   ├── distributed_worker.py # Worker entry point for non-rank-0 nodes (launched via SSH)
│   ├── inference.py    # generate_chat, generate_completion, generate_embeddings
│   ├── model_manager.py # Model loading/unloading, keep-alive, LRU eviction
│   ├── registry.py    # Ollama name → HuggingFace repo mapping via models.json, fuzzy search
│   ├── speculative.py  # SpeculativeDecoder: model-agnostic draft→target verification
│   ├── speculative_stream.py # Streaming adapter for speculative decoding
│   ├── template_caps.py # Chat template capability detection (tools, thinking)
│   ├── tool_parser.py  # Multi-format tool call parsing (Qwen, Mistral, Llama, DeepSeek, bare JSON)
│   ├── spectralquant.py # SpectralQuant: eigenvector rotation, non-uniform quantization
│   ├── spectralquant_cache.py # SpectralQuantKVCache: drop-in KVCache replacement with SpectralQuant compression
│   ├── spectralquant_calibrate.py # Eigenspectral calibration: collect KV vectors, eigendecompose, fit codebooks
│   ├── turboquant.py   # TurboQuant_mse: rotation, Lloyd-Max codebooks, bit-packed quantize/dequantize
│   ├── turboquant_cache.py # TurboQuantKVCache: drop-in KVCache replacement with TurboQuant compression
│   ├── dflash/
│   │   ├── adapters.py      # TargetAdapter ABC + Qwen3Adapter for hidden state extraction
│   │   ├── decoder.py       # DFlashDecoder: block-diffusion speculative decoding
│   │   └── draft_model.py   # DFlashDraftModel: cross-attention draft architecture
│   └── flash/
│       ├── bundler.py       # Bundle FFN weights into per-layer .flashweights files
│       ├── flash_mlp.py     # FlashMLP: sparse FFN that loads active neurons from SSD; WindowManager
│       ├── flash_model.py   # FlashModelWrapper: wraps mlx-lm model with FlashMLP layers + Prefetcher
│       ├── predictor.py     # SparsityPredictor, PredictorBank, LookaheadBank (cross-layer)
│       ├── prefetch.py      # Prefetcher: background neuron prefetch with thread pool + stats
│       ├── prepare.py       # prepare_model_for_flash pipeline, predictor training
│       ├── speculative.py   # SpeculativeFlashDecoder: extends SpeculativeDecoder with prefetch
│       ├── _ssd_base.py     # LayerLruCache, HeaderSpec codec, full_pread — shared by dense + MoE stores
│       ├── weight_store.py  # FlashWeightStore: SSD I/O + LayerLruCache + preallocated buffers
│       ├── flash_moe.py     # Flash-MoE sparse expert loading
│       ├── flash_moe_model.py # Flash-MoE model wrapper
│       ├── moe_bundler.py   # Bundle MoE expert weights
│       ├── moe_prepare.py   # MoE preparation pipeline
│       ├── moe_weight_store.py # MoE weight store
│       └── speculative_stream.py # Re-exports from engine/speculative_stream.py
├── models/
│   ├── manifest.py     # Model manifest/metadata
│   └── store.py        # Local model storage
├── routers/
│   ├── anthropic.py    # /v1/messages — Anthropic Messages API (Claude Code)
│   ├── openai.py       # /v1/chat/completions, /v1/completions, /v1/models, /v1/embeddings
│   ├── chat.py         # /api/chat (Ollama)
│   ├── generate.py     # /api/generate (Ollama)
│   ├── models.py       # /api/tags, /api/show, /api/ps (Ollama)
│   ├── manage.py       # /api/pull, /api/copy, /api/create, /api/delete (Ollama)
│   ├── embed.py        # /api/embed, /api/embeddings (Ollama)
│   ├── blobs.py        # /api/blobs (Ollama)
│   └── status.py       # /, /api/version (Ollama)
├── schemas/
│   ├── anthropic.py    # Anthropic API request/response models
│   ├── openai.py       # OpenAI API request/response models
│   ├── common.py       # Shared schema types
│   ├── chat.py         # Ollama chat schemas
│   ├── generate.py     # Ollama generate schemas
│   ├── embed.py        # Ollama embedding schemas
│   ├── models.py       # Ollama model schemas
│   ├── manage.py       # Ollama management schemas
│   ├── pull.py         # Ollama pull schemas
│   └── status.py       # Ollama status schemas
└── utils/
    ├── streaming.py    # async_mlx_stream, safe_ndjson_stream — streaming bridge + error wrapper
    └── timing.py       # Timer, TimingStats
```

## Key Design Decisions

- **Anthropic router** (`routers/anthropic.py`): Buffers full model output before emitting SSE events to properly parse `<think>` blocks (→ thinking content blocks) and `<tool_call>` blocks (→ tool_use content blocks). This is necessary because Qwen 3.5 outputs these as raw text. Also exposes `/v1/messages/count_tokens`.
- **Tool format conversion**: Anthropic tool definitions are converted to OpenAI-style `{"type": "function", "function": {...}}` format for `tokenizer.apply_chat_template()`.
- **Tool call parsing**: Supports Qwen (`<tool_call>`), Mistral (`[TOOL_CALLS]`), Llama 3.x (`<function=Name>`), DeepSeek, MiniMax (`<minimax:tool_call>`), and bare JSON formats.
- **Message conversion**: `tool_result` blocks → `role: "tool"` messages; `tool_use` blocks → `tool_calls` array; `thinking` blocks in history are skipped.
- **Model storage**: Models stored by HF repo path (e.g. `Qwen--Qwen3-8B`). `ModelManager` takes a `ModelStore` dependency for local-first config loading and auto-download.
- **Active inference protection**: `LoadedModel.active_refs` prevents LRU eviction and expiry of models currently serving requests.
- **Memory safety**: After loading, checks Metal memory (active + cache) against `OLMLX_MEMORY_LIMIT_FRACTION` of system RAM. Rejects oversized models with `MemoryError` (HTTP 503) to prevent uncatchable Metal OOM crashes.
- **Stream cleanup**: NDJSON routers (generate, chat, manage) use `safe_ndjson_stream()` from `utils/streaming.py` — a shared async generator that wraps a source with error handling and guaranteed `aclose()`. OpenAI and Anthropic routers have their own cleanup patterns due to SSE framing differences.
- **Auto-registration**: Direct HF paths (e.g. `Qwen/Qwen3-8B`) are auto-registered in `models.json` on first load or pull.
- **Prompt caching**: KV cache reuse across requests when prompts share a common prefix. Works with both mlx-lm (text) and mlx-vlm (vision) models. Controlled via `OLMLX_PROMPT_CACHE` setting. `PromptCacheStore` async wrappers (`async_get`, `async_set`, `async_evict_all_to_disk`) offload disk I/O to `asyncio.to_thread()` to avoid blocking the event loop. All `_entries` mutations happen on the event loop; an eviction-generation counter prevents re-insertion of stale disk data after bulk eviction. Hybrid SSM-style models (Qwen3.5, Qwen3-Next gated-delta layers using `ArraysCache`) skip cross-request storage — `_cache_supports_persistence` returns False because reusing `ArraysCache` state across worker threads crashes mlx-lm with a Metal stream error during the next prefill (issue #284). Within-request reuse during a single generation still works.
- **Hybrid VLM detection**: VLMs that combine vision with hybrid linear+full attention (Qwen3.5, Qwen3_5_moe) ship a dedicated text-only module in mlx-lm (`qwen3_5.py`, `qwen3_5_moe.py`). `_detect_model_kind` routes these through the mlx-lm text path because the mlx-vlm path crashes with a Metal stream error on text inference (issue #284). Discriminator: `text_config.layer_types` contains `"linear_attention"`. Standard VLMs (Gemma 4, Qwen2-VL) lack this field and continue to load through mlx-vlm.
- **TurboQuant KV cache quantization**: Compresses KV cache using TurboQuant_mse (arxiv 2504.19874). Algorithm: random rotation → Lloyd-Max scalar quantization per coordinate → bit-packed storage → inverse rotation on fetch. Supports 2-bit (~7.5x compression) and 4-bit (~3.9x compression). `TurboQuantKVCache` is a drop-in replacement for mlx-lm's `KVCache`. Per-layer rotation matrices via QR decomposition. Codebooks are standard Gaussian Lloyd-Max centroids scaled by 1/√(head_dim). Indices bit-packed: 2 per byte (4-bit) or 4 per byte (2-bit). Head dim detected from `model.args.head_dim` or K projection weight shape. Incompatible with disk cache offload (guarded in `_save_to_disk`). Each `update_and_fetch` dequantizes only the newly appended tokens (O(num_steps·head_dim²)) and splices them into a per-cache dequantized side buffer that is reused across calls; this costs one extra `input_dtype` buffer per quantized layer (~100-200 MB for Qwen3-Coder-Next at 4096 tokens) but avoids re-dequantizing the full history each step. Config: `OLMLX_KV_CACHE_QUANT=turboquant:4` (or `:2`), validated at startup.
- **SpectralQuant KV cache quantization**: Improves on TurboQuant by using data-driven eigenvector rotations instead of random rotations, enabling non-uniform bit allocation (more bits for high-variance semantic dimensions, fewer for low-variance tail dimensions). Requires one-time calibration via `olmlx spectral prepare <model>` which: collects KV vectors from calibration data, computes per-head covariance + eigendecomposition (`mx.linalg.eigh` on CPU stream), derives effective dimensionality via participation ratio, and fits separate Lloyd-Max codebooks for semantic and tail regimes. Calibration data saved to `<model_dir>/spectral/` (safetensors + JSON config). Config: `OLMLX_KV_CACHE_QUANT=spectral:4` (or `:2`).
- **Model load timeout**: Configurable via `OLMLX_MODEL_LOAD_TIMEOUT` with dedicated `ModelLoadTimeoutError` (HTTP 504).
- **Terminal chat** (`chat/`): `olmlx chat` runs inference directly in-process via `ModelManager`/`generate_chat()` — no HTTP server needed. Connects to external MCP servers (stdio/SSE) for tool use with a full agent loop (model → tool calls → MCP execution → results fed back → continue). Uses `parse_model_output()` from `engine/tool_parser.py` for thinking/tool extraction. MCP config uses Claude Desktop format (`~/.olmlx/mcp.json`).
- **Distributed inference** (experimental): Splits models across multiple Apple Silicon machines using MLX's ring distributed backend. Startup sequence: CLI generates ring hostfile → launches workers via SSH → ring init (`mx.distributed.init`) → starts sideband server (TCP, port 32400) → starts uvicorn. Key constraints:
  - Sideband server starts in CLI before uvicorn (not in app lifespan) because `import transformers` can take minutes.
  - Worker sideband connection retries with exponential backoff (up to 120s) to handle startup race conditions.
  - After `model.shard()`, must materialize weights with `mx.eval(model.parameters())` on both coordinator and worker before any forward pass — otherwise the combined lazy weight materialization + all_sum Metal command buffer exceeds the ~10s GPU timeout for 32B+ models.
  - Coordinator broadcasts inference params via sideband before `stream_generate`; a lightweight `all_sum` barrier synchronizes ranks before heavy compute.
  - Worker and coordinator must load the same model (all_sum requires matching tensor shapes).
  - **Flash + distributed**: When both Flash and distributed are enabled (tensor strategy), `FlashModelWrapper.shard()` shards only attention projections, leaving FlashMLP layers unsharded. Each rank independently loads active neurons from its local SSD. This is correct because `o_proj` (sharded-to-all) replicates its output via `all_sum`, so every rank feeds identical input to FlashMLP. Pre-sharding is skipped (MLP weights live on SSD). Flash env vars are forwarded to workers. Flash-MoE + distributed remains blocked (expert weights cannot be sharded).
  - Config: `OLMLX_EXPERIMENTAL_DISTRIBUTED=true`, hostfile at `~/.olmlx/hostfile.json` with `{"hosts": ["ip1", "ip2"], "model": "hf-path"}`.
- **Speculative decoding** (`engine/speculative.py`): Model-agnostic two-tier architecture. Base `SpeculativeDecoder` runs a draft model to produce λ candidate tokens autoregressively; the target verifies all candidates in one forward pass using greedy argmax verification, persistent KV caches, and cache trimming on rejection. `SpeculativeFlashDecoder` (`engine/flash/speculative.py`) extends the base with Flash-specific prefetcher integration (Path B: draft hidden state capture for bulk neuron prediction) and adaptive neuron window sizing. Both share the same streaming adapter (`engine/speculative_stream.py`). Composes with prompt caching, tool calls, and SSE/NDJSON streaming. Config: `OLMLX_SPECULATIVE=true`, `OLMLX_SPECULATIVE_DRAFT_MODEL=<hf-path>`, `OLMLX_SPECULATIVE_TOKENS=4`, or `--speculative` / `--speculative-draft-model` / `--speculative-tokens` on `olmlx serve`. Per-model overrides go at the top level of a `models.json` entry (`speculative`, `speculative_draft_model`, `speculative_tokens`) — not under `experimental` (loading a config that still puts them under `experimental` raises a clear migration error).
- **DFlash block-diffusion decoding** (experimental): Alternative speculative strategy where the draft model is conditioned on hidden states from specific target layers, using cross-attention (DFlashAttention) over concatenated target + current states. Model-specific `TargetAdapter` handles hidden state extraction and KV cache rollback. Implements the same `prefill`/`step`/`reset` interface as `SpeculativeDecoder` for seamless streaming integration. Config: `OLMLX_EXPERIMENTAL_DFLASH=true`, `OLMLX_EXPERIMENTAL_DFLASH_DRAFT_MODEL=<hf-path>`, `OLMLX_EXPERIMENTAL_DFLASH_BLOCK_SIZE=4`. Requires a pre-trained dflash draft model with `config.json` containing `target_layer_ids`.
- **Speculative prefetch** (experimental): Predicts and pre-loads neuron weights from SSD before they're needed, hiding I/O latency during Flash inference. Three paths:
  - **Path A — Cross-layer**: While layer L computes, predicts layer L+1's active neurons using L's pre-MLP hidden state and starts background SSD reads. Optional `LookaheadBank` provides dedicated cross-layer predictors (trained with `OLMLX_EXPERIMENTAL_FLASH_PREFETCH=true` during preparation); falls back to reusing layer L+1's sparsity predictor.
  - **Path B — Draft-informed**: During speculative decoding, captures the draft model's hidden states (pre-lm_head, via `draft.model()` + `draft.lm_head()`) and submits bulk prefetch for all target layers before verification. Maps draft positions to target layers by depth ratio so early/deep layers get appropriate signals. Deduplicates predictor calls when multiple layers share the same draft position.
  - **Prefetcher** (`prefetch.py`): Owns a dedicated `ThreadPoolExecutor` (default 16 threads) separate from the weight store's I/O pool. Prediction runs synchronously on the calling thread (MLX `mx.eval` deadlocks under concurrent multi-thread use); only I/O is async. `PrefetchStats` tracks submitted/hits/misses/failures. Lifecycle: prefetcher must be closed before weight store on model unload (prefetcher tasks submit into the weight store's pool).
  - Config: `OLMLX_EXPERIMENTAL_FLASH_PREFETCH=true`. Also controls whether `olmlx flash prepare` trains lookahead predictors. Tuning: `OLMLX_EXPERIMENTAL_FLASH_PREFETCH_CONFIDENCE_THRESHOLD` (default 0.3), `_MIN_NEURONS` (64), `_MAX_NEURONS`, `_IO_THREADS` (16).

## Development

```bash
uv sync --no-editable
uv run olmlx          # starts on http://localhost:11434
uv run pytest              # run tests
```

Models are configured in `~/.olmlx/models.json` (auto-created on first run). For dev, override with `OLMLX_MODELS_CONFIG=models.json`.

### TDD

Use test-driven development: write failing tests first, then implement the code to make them pass. For bug fixes, write a test that reproduces the bug before writing the fix.

## Git

- Remote: `git@github.com:motsognirr/olmlx.git`
- Author: Daniel Palmqvist <daniel.u.palmqvist@gmail.com>

## Type Annotations

Maintain consistent type safety throughout the codebase:

- **TypedDict for required fields only**: Use `TypedDict` with `total=True` (default) for required keys. Use `TypedDict, total=False` for optional keys. Never use an empty TypedDict — use `dict[str, Any]` for freeform JSON.

- **Pydantic model field types**: Be careful when using TypedDict as a Pydantic field type — Pydantic v2 enforces required keys. Prefer `dict[str, Any]` for fields that should accept any shape (e.g., `Tool.function`) to avoid silently stripping extra keys.

- **Protocol return types**: `__aexit__` must return `bool | None` (not `None`) so implementations can suppress exceptions. `__call__` must match actual return types (e.g., mlx-lm models return `mx.array`, not `dict`).

- **AsyncIterator return types**: If a function yields both `str` and `dict`, annotate as `AsyncIterator[str | dict[str, Any]]`, not just `AsyncIterator[str]`.

- **TYPE_CHECKING for annotation-only imports**: If a type is only used in annotations and the module has `from __future__ import annotations`, guard the import under `TYPE_CHECKING` or remove it entirely. Avoid runtime imports of types that are never evaluated.

- **Protocol return values**: Match the real type's return value (e.g., `ClientSession.initialize()` returns `InitializeResult`, not `None`).
