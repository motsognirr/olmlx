# MLX Ollama

Ollama-compatible API server using Apple MLX for local inference on Apple Silicon.

## Project Structure

```
mlx_ollama/
├── app.py              # FastAPI app factory, middleware, router registration
├── config.py           # Settings (pydantic-settings, MLX_OLLAMA_ env prefix)
├── __main__.py         # Entry point (uv run mlx-ollama)
├── engine/
│   ├── inference.py    # generate_chat, generate_completion, generate_embeddings
│   ├── model_manager.py # Model loading/unloading, keep-alive, LRU eviction
│   └── registry.py    # Ollama name → HuggingFace repo mapping via models.json
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
- **Tool call parsing**: Supports Qwen (`<tool_call>`), Mistral (`[TOOL_CALLS]`), and bare JSON formats.
- **Message conversion**: `tool_result` blocks → `role: "tool"` messages; `tool_use` blocks → `tool_calls` array; `thinking` blocks in history are skipped.

## Development

```bash
uv sync --no-editable
uv run mlx-ollama          # starts on http://localhost:11434
```

Models are configured in `models.json` (gitignored). Copy `models.json.example` to start.

## Git

- Remote: `git@github.com:dpalmqvist/mlx_ollama.git`
- Author: Daniel Palmqvist <daniel.u.palmqvist@gmail.com>
