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
│   ├── template_caps.py # Chat template capability detection (tools, thinking)
│   ├── tool_parser.py  # Multi-format tool call parsing (Qwen, Mistral, Llama, DeepSeek, bare JSON)
│   ├── turboquant.py   # TurboQuant_mse: rotation, Lloyd-Max codebooks, bit-packed quantize/dequantize
│   └── turboquant_cache.py # TurboQuantKVCache: drop-in KVCache replacement with TurboQuant compression
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
- **Tool call parsing**: Supports Qwen (`<tool_call>`), Mistral (`[TOOL_CALLS]`), Llama 3.x (`<function=Name>`), DeepSeek, and bare JSON formats.
- **Message conversion**: `tool_result` blocks → `role: "tool"` messages; `tool_use` blocks → `tool_calls` array; `thinking` blocks in history are skipped.
- **Model storage**: Models stored by HF repo path (e.g. `Qwen--Qwen3-8B`). `ModelManager` takes a `ModelStore` dependency for local-first config loading and auto-download.
- **Active inference protection**: `LoadedModel.active_refs` prevents LRU eviction and expiry of models currently serving requests.
- **Memory safety**: After loading, checks Metal memory (active + cache) against `OLMLX_MEMORY_LIMIT_FRACTION` of system RAM. Rejects oversized models with `MemoryError` (HTTP 503) to prevent uncatchable Metal OOM crashes.
- **Stream cleanup**: NDJSON routers (generate, chat, manage) use `safe_ndjson_stream()` from `utils/streaming.py` — a shared async generator that wraps a source with error handling and guaranteed `aclose()`. OpenAI and Anthropic routers have their own cleanup patterns due to SSE framing differences.
- **Auto-registration**: Direct HF paths (e.g. `Qwen/Qwen3-8B`) are auto-registered in `models.json` on first load or pull.
- **Prompt caching**: KV cache reuse across requests when prompts share a common prefix. Works with both mlx-lm (text) and mlx-vlm (vision) models. Controlled via `OLMLX_PROMPT_CACHE` setting. `PromptCacheStore` async wrappers (`async_get`, `async_set`, `async_evict_all_to_disk`) offload disk I/O to `asyncio.to_thread()` to avoid blocking the event loop. All `_entries` mutations happen on the event loop; an eviction-generation counter prevents re-insertion of stale disk data after bulk eviction.
- **TurboQuant KV cache quantization** (experimental): Compresses KV cache using TurboQuant_mse (arxiv 2504.19874). Algorithm: random rotation → Lloyd-Max scalar quantization per coordinate → bit-packed storage → inverse rotation on fetch. Supports 2-bit (~7.5x compression) and 4-bit (~3.9x compression). `TurboQuantKVCache` is a drop-in replacement for mlx-lm's `KVCache` — dequantizes on each `update_and_fetch` call (O(n) per step, same as attention). Per-layer rotation matrices via QR decomposition. Codebooks are standard Gaussian Lloyd-Max centroids scaled by 1/√(head_dim). Indices bit-packed: 2 per byte (4-bit) or 4 per byte (2-bit). Head dim detected from `model.args.head_dim` or K projection weight shape. Incompatible with disk cache offload (guarded in `_save_to_disk`). Config: `OLMLX_EXPERIMENTAL_KV_CACHE_QUANT=turboquant:4` (or `:2`), validated at startup.
- **Model load timeout**: Configurable via `OLMLX_MODEL_LOAD_TIMEOUT` with dedicated `ModelLoadTimeoutError` (HTTP 504).
- **Terminal chat** (`chat/`): `olmlx chat` runs inference directly in-process via `ModelManager`/`generate_chat()` — no HTTP server needed. Connects to external MCP servers (stdio/SSE) for tool use with a full agent loop (model → tool calls → MCP execution → results fed back → continue). Uses `parse_model_output()` from `engine/tool_parser.py` for thinking/tool extraction. MCP config uses Claude Desktop format (`~/.olmlx/mcp.json`).
- **Distributed inference** (experimental): Splits models across multiple Apple Silicon machines using MLX's ring distributed backend. Startup sequence: CLI generates ring hostfile → launches workers via SSH → ring init (`mx.distributed.init`) → starts sideband server (TCP, port 32400) → starts uvicorn. Key constraints:
  - Sideband server starts in CLI before uvicorn (not in app lifespan) because `import transformers` can take minutes.
  - Worker sideband connection retries with exponential backoff (up to 120s) to handle startup race conditions.
  - After `model.shard()`, must materialize weights with `mx.eval(model.parameters())` on both coordinator and worker before any forward pass — otherwise the combined lazy weight materialization + all_sum Metal command buffer exceeds the ~10s GPU timeout for 32B+ models.
  - Coordinator broadcasts inference params via sideband before `stream_generate`; a lightweight `all_sum` barrier synchronizes ranks before heavy compute.
  - Worker and coordinator must load the same model (all_sum requires matching tensor shapes).
  - Config: `OLMLX_EXPERIMENTAL_DISTRIBUTED=true`, hostfile at `~/.olmlx/hostfile.json` with `{"hosts": ["ip1", "ip2"], "model": "hf-path"}`.

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
