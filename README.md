# olmlx

Drop-in Ollama API replacement powered by Apple's [MLX](https://github.com/ml-explore/mlx) framework. Get faster inference on Mac M-series hardware while using any tool that speaks the Ollama REST API.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Install

### Option 1: Global install (recommended)

```bash
# Install globally — no clone needed
uv tool install git+ssh://git@github.com/motsognirr/olmlx.git

# Start the server
olmlx
```

On first run, `~/.olmlx/models.json` is created with example model mappings.

### Option 2: From source

```bash
git clone <repo-url> && cd mlx-for-claude
uv sync --no-editable
uv run olmlx
```

The server starts on `http://localhost:11434` — the same default port as Ollama.

## Auto-start on Login (macOS)

```bash
# Install as a launchd service — starts on login, restarts on crash
olmlx service install

# Check status
olmlx service status

# Remove the service
olmlx service uninstall
```

The service writes logs to `~/.olmlx/olmlx.log`.

## Model Configuration

Edit `~/.olmlx/models.json` to map Ollama-style model names to HuggingFace repos. MLX-format models from [mlx-community](https://huggingface.co/mlx-community) work best:

```json
{
  "llama3.2:latest": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "mistral:7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
  "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
  "gemma2:2b": "mlx-community/gemma-2-2b-it-4bit"
}
```

You can also pass HuggingFace paths directly in API calls — any model name containing `/` is treated as an HF repo ID.

## Usage

### Pull and chat

```bash
# Pull a model (downloads from HuggingFace)
curl http://localhost:11434/api/pull -d '{"model": "llama3.2:latest"}'

# Generate a completion (non-streaming)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:latest",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# Chat (streaming by default)
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:latest",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

### OpenAI-compatible API

The server also exposes OpenAI-format endpoints at `/v1/`, so you can point any OpenAI SDK client at it:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:latest",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")
response = client.chat.completions.create(
    model="llama3.2:latest",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Use with the Ollama Python library

```python
import ollama

client = ollama.Client(host="http://localhost:11434")
response = client.chat(
    model="llama3.2:latest",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response["message"]["content"])
```

## API Endpoints

### Ollama API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — returns `"Ollama is running"` |
| `/api/version` | GET | Server version |
| `/api/tags` | GET | List available models |
| `/api/ps` | GET | List currently loaded models |
| `/api/generate` | POST | Text completion (streaming/non-streaming) |
| `/api/chat` | POST | Chat completion (streaming/non-streaming) |
| `/api/show` | POST | Model details and metadata |
| `/api/pull` | POST | Download a model from HuggingFace |
| `/api/copy` | POST | Create a model alias |
| `/api/create` | POST | Create model from Modelfile |
| `/api/delete` | DELETE | Delete a local model |
| `/api/embed` | POST | Generate embeddings |
| `/api/embeddings` | POST | Generate embeddings (legacy format) |
| `/api/blobs/:digest` | HEAD/POST | Check or upload blobs |

### OpenAI-compatible API

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completion (SSE streaming supported) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List models |
| `/v1/embeddings` | POST | Generate embeddings |

### Anthropic Messages API

| Endpoint | Method | Description |
|---|---|---|
| `/v1/messages` | POST | Anthropic Messages API (SSE streaming supported) |

This endpoint allows using the server as a backend for tools that speak the Anthropic API, such as Claude Code. It supports thinking blocks, tool use, and streaming.

## Configuration

All settings can be overridden with `OLMLX_`-prefixed environment variables or a `.env` file in the project root:

| Variable | Default | Description |
|---|---|---|
| `OLMLX_HOST` | `0.0.0.0` | Bind address |
| `OLMLX_PORT` | `11434` | Port |
| `OLMLX_MODELS_DIR` | `~/.olmlx/models` | Where downloaded models are stored |
| `OLMLX_MODELS_CONFIG` | `~/.olmlx/models.json` | Path to model mapping file |
| `OLMLX_DEFAULT_KEEP_ALIVE` | `5m` | How long idle models stay loaded (`0` = unload immediately, `-1` = never unload) |
| `OLMLX_MAX_LOADED_MODELS` | `1` | Max models loaded concurrently (LRU eviction when exceeded) |

## How It Works

Instead of GGUF models and llama.cpp, this server uses [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) to run inference directly on Apple Silicon's GPU via the Metal framework. Models are downloaded from HuggingFace Hub in MLX safetensor format.

The server translates Ollama API requests into `mlx_lm.generate()` / `mlx_lm.stream_generate()` calls, with a sync-to-async streaming bridge so the FastAPI event loop stays responsive during generation.

Key internals:
- **Model Manager** — loads/unloads models with configurable keep-alive TTLs and LRU eviction
- **Registry** — resolves Ollama model names to HuggingFace paths via `models.json`
- **Streaming Bridge** — runs `mlx_lm.stream_generate()` in a thread, feeds tokens through an `asyncio.Queue`
- **Chat Templates** — uses each model's built-in `tokenizer.apply_chat_template()` for correct prompt formatting

## Troubleshooting

### "Model not found" errors

When you see `Model 'X' not found`, the model name isn't recognized. Fix it by:

1. **Add a mapping** to `~/.olmlx/models.json`:
   ```json
   {
     "my-model:latest": "mlx-community/Model-Repo-Name"
   }
   ```

2. **Use a HuggingFace path directly** in API calls:
   ```bash
   curl http://localhost:11434/api/generate -d '{
     "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
     "prompt": "Hello"
   }'
   ```

### Metal GPU crashes

If the server crashes during inference:

1. **Check available memory** — unload other GPU-heavy apps (video editors, 3D apps)
2. **Reduce model size** — use 4-bit quantized models instead of 8-bit or 16-bit
3. **Limit concurrent requests** — set `OLMLX_MAX_LOADED_MODELS=1`
4. **Check logs** — if running as a service, view `~/.olmlx/olmlx.log`

### Model won't unload / memory pressure

Models stay loaded based on `OLMLX_DEFAULT_KEEP_ALIVE`:

- `5m` (default) — unload after 5 minutes idle
- `0` — unload immediately after use
- `-1` — never unload

To force unload a model:
```bash
curl -X POST http://localhost:11434/api/unload -d '{"model": "llama3.2:latest"}'
```

To see loaded models:
```bash
curl http://localhost:11434/api/ps
```

The `active_refs` field shows how many requests are currently using each model. Models with `active_refs > 0` cannot be unloaded until requests complete.

### Context window limits

If responses get cut off or you see context-related errors:

1. **Shorten your prompt** — remove old messages from the conversation
2. **Use a model with larger context** — some models support 32K+ tokens
3. **Check model documentation** — verify the model's actual context limit

### Tool calling not working

Tool calling requires:
1. A model with tool calling capability (Qwen 2.5, Llama 3.1+, Mistral Nemo)
2. Tools passed in the request
3. A chat template that supports tools

If the template doesn't support tools, olmlx falls back to injecting tool descriptions into the system message.

## Model Compatibility

| Model Family | Chat | Tools | Thinking | Vision |
|---|---|---|---|---|
| Qwen 2.5/3 | ✓ | ✓ | ✓ (Qwen 3) | ✗ |
| Llama 3.1/3.2 | ✓ | ✓ | ✗ | ✗ |
| Mistral/Nemo | ✓ | ✓ | ✗ | ✗ |
| Gemma 2 | ✓ | ✗ | ✗ | ✗ |
| Phi 3 | ✓ | ✗ | ✗ | ✗ |
| LLava-based | ✓ | ✗ | ✗ | ✓ |

Check a model's chat template on HuggingFace to verify feature support.
