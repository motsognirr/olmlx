# MLX Ollama

Ollama-compatible API server using Apple MLX for local inference on Apple Silicon.

## Project Structure

```
mlx_ollama/
├── app.py              # FastAPI app factory, middleware, router registration
├── config.py           # Settings (pydantic-settings, MLX_OLLAMA_ env prefix)
├── cli.py              # CLI with subcommands (serve, service install/uninstall/status)
├── __main__.py         # Entry point (delegates to cli.py)
├── engine/
│   ├── inference.py    # generate_chat, generate_completion, generate_embeddings
│   ├── model_manager.py # Model loading/unloading, keep-alive, LRU eviction
│   ├── registry.py    # Ollama name → HuggingFace repo mapping via models.json
│   ├── template_caps.py # Chat template capability detection (tools, thinking)
│   └── tool_parser.py  # Multi-format tool call parsing (Qwen, Mistral, Llama, DeepSeek, bare JSON)
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
│   └── ...             # Ollama API schemas
└── utils/
    ├── streaming.py    # async_mlx_stream — sync-to-async streaming bridge
    └── timing.py       # Timer, TimingStats
```

## Key Design Decisions

- **Anthropic router** (`routers/anthropic.py`): Buffers full model output before emitting SSE events to properly parse `<think>` blocks (→ thinking content blocks) and `<tool_call>` blocks (→ tool_use content blocks). This is necessary because Qwen 3.5 outputs these as raw text.
- **Tool format conversion**: Anthropic tool definitions are converted to OpenAI-style `{"type": "function", "function": {...}}` format for `tokenizer.apply_chat_template()`.
- **Tool call parsing**: Supports Qwen (`<tool_call>`), Mistral (`[TOOL_CALLS]`), Llama 3.x, DeepSeek, and bare JSON formats.
- **Message conversion**: `tool_result` blocks → `role: "tool"` messages; `tool_use` blocks → `tool_calls` array; `thinking` blocks in history are skipped.
- **Model storage**: Models stored by HF repo path (e.g. `Qwen--Qwen3-8B`). `ModelManager` takes a `ModelStore` dependency for local-first config loading and auto-download.
- **Active inference protection**: `LoadedModel.active_refs` prevents LRU eviction and expiry of models currently serving requests.
- **Stream cleanup**: All streaming routers use `try/finally` with `await result.aclose()` to ensure GPU resources are released on client disconnect.
- **Auto-registration**: Direct HF paths (e.g. `Qwen/Qwen3-8B`) are auto-registered in `models.json` on first load or pull.

## Development

```bash
uv sync --no-editable
uv run mlx-ollama          # starts on http://localhost:11434
uv run pytest              # run tests
```

Models are configured in `~/.mlx_ollama/models.json` (auto-created on first run). For dev, override with `MLX_OLLAMA_MODELS_CONFIG=models.json`.

### TDD

Use test-driven development: write failing tests first, then implement the code to make them pass. For bug fixes, write a test that reproduces the bug before writing the fix.

## Git

- Remote: `git@github.com:motsognirr/olmlx.git`
- Author: Daniel Palmqvist <daniel.u.palmqvist@gmail.com>
