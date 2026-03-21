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
git clone <repo-url> && cd olmlx
uv sync --no-editable
uv run olmlx
```

The server starts on `http://localhost:11434` — the same default port as Ollama.

## CLI

```bash
olmlx                      # Start the server (default)
olmlx serve                # Start the server (explicit)
olmlx chat <model>         # Interactive terminal chat with MCP tool support
olmlx models list          # List locally downloaded models
olmlx models pull <name>   # Download a model
olmlx models show <name>   # Show model details
olmlx models delete <name> # Delete a local model
olmlx config show          # Show current configuration
olmlx service install      # Install as launchd service
olmlx service status       # Check service status
olmlx service uninstall    # Remove the service
```

## Terminal Chat

`olmlx chat` provides an interactive terminal chat that runs inference directly in-process — no server needed. It supports MCP tool servers for agent-style workflows where the model can call external tools.

```bash
# Basic chat
olmlx chat qwen3:8b

# With a system prompt
olmlx chat qwen3:8b --system "You are a helpful coding assistant"

# With MCP tools (reads ~/.olmlx/mcp.json by default)
olmlx chat qwen3:8b --mcp-config path/to/mcp.json

# Disable thinking or MCP
olmlx chat qwen3:8b --no-thinking --no-mcp
```

### Slash commands

| Command | Description |
|---|---|
| `/exit` | Quit the chat |
| `/clear` | Clear conversation history |
| `/tools` | Show available MCP tools |
| `/skills` | Show loaded skills |
| `/system <prompt>` | Set or show the system prompt |
| `/model <name>` | Switch to a different model |

Multiline input is supported with a trailing `\`.

### MCP tool servers

Configure MCP servers in `~/.olmlx/mcp.json` using the same format as Claude Desktop:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "remote": {
      "url": "http://localhost:8080/sse"
    }
  }
}
```

Entries with `command` use stdio transport; entries with `url` use SSE transport. When tools are available, the model can call them and the results are automatically fed back for the model to continue — a full agent loop.

### Skills

Skills are markdown files that provide specialized instructions the model can load on demand. Instead of stuffing everything into the system prompt, skill descriptions are listed briefly and the model uses a `use_skill` tool to load the full content only when relevant.

```bash
# Create the skills directory and copy the examples
mkdir -p ~/.olmlx/skills
cp examples/skills/*.md ~/.olmlx/skills/

# Chat with skills enabled (default)
olmlx chat qwen3:8b

# List loaded skills in chat
/skills

# Disable skills
olmlx chat qwen3:8b --no-skills

# Use a custom skills directory
olmlx chat qwen3:8b --skills-dir /path/to/skills
```

Skill files use a simple frontmatter format:

```markdown
---
name: code-review
description: Structured code review focusing on correctness, clarity, and maintainability
---

When reviewing code, follow this structured approach...
```

The `name` field is required; `description` is optional but recommended — it's shown in the system prompt so the model knows when to use each skill.

**Included example skills** (in `examples/skills/`):

| Skill | Description |
|---|---|
| `code-review` | Structured review: correctness, clarity, maintainability, security |
| `explain` | Explain code or concepts, adapting depth to the question |
| `commit-message` | Write clear conventional commit messages from diffs |
| `debug` | Systematic debugging: reproduce, isolate, fix, verify |
| `refactor` | Safe refactoring — improve structure without changing behavior |

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

## Vision-Language Models

olmlx supports vision-language models (VLMs) that can process images alongside text. VLMs are automatically detected and loaded using mlx-vlm.

### Using a VLM

```bash
# Chat with an image (base64-encoded in message)
curl http://localhost:11434/api/chat -d '{
  "model": "llava:1.5-7b",
  "messages": [{
    "role": "user",
    "content": "What is in this image?",
    "images": ["iVBOR..."]
  }]
}'
```

Note: images should be raw base64 without a `data:image/...;base64,` prefix.

### VLM Model Configuration

Add VLM mappings to `~/.olmlx/models.json`:

```json
{
  "llava:1.5-7b": "mlx-community/llava-1.5-7b-4bit"
}
```

VLMs are automatically detected by inspecting `config.json` for vision-related keys (`vision_config`, `vision_tower`, etc.) and loaded via mlx-vlm instead of mlx-lm.

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
| `/v1/messages/count_tokens` | POST | Count tokens for a messages request |

This endpoint allows using the server as a backend for tools that speak the Anthropic API, such as Claude Code. It supports thinking blocks, tool use, streaming, and prompt caching (KV cache reuse across requests).

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
| `OLMLX_MEMORY_LIMIT_FRACTION` | `0.75` | Max fraction of system RAM for Metal GPU memory (0–1). Models exceeding this are rejected to prevent OOM crashes |
| `OLMLX_MODEL_LOAD_TIMEOUT` | `None` | Timeout in seconds for model loading (no timeout by default) |
| `OLMLX_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `OLMLX_PROMPT_CACHE` | `true` | Enable KV cache reuse across requests for faster inference |
| `OLMLX_PROMPT_CACHE_MAX_TOKENS` | `32768` | Invalidate the KV cache after a conversation exceeds this many tokens. Use a very large value to effectively disable |
| `OLMLX_CORS_ORIGINS` | `http://localhost:*`, `http://127.0.0.1:*` | Allowed CORS origins |

### Distributed inference settings (experimental)

| Variable | Default | Description |
|---|---|---|
| `OLMLX_EXPERIMENTAL_DISTRIBUTED` | `false` | Enable distributed inference |
| `OLMLX_EXPERIMENTAL_DISTRIBUTED_HOSTFILE` | `~/.olmlx/hostfile.json` | Path to hostfile with hosts and model |
| `OLMLX_EXPERIMENTAL_DISTRIBUTED_BACKEND` | `ring` | MLX distributed backend |
| `OLMLX_EXPERIMENTAL_DISTRIBUTED_PORT` | `32323` | Base port for ring backend (increments per rank) |
| `OLMLX_EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT` | `32400` | TCP port for coordinator↔worker sideband |
| `OLMLX_EXPERIMENTAL_DISTRIBUTED_SECRET` | *(empty)* | Shared secret for worker authentication |
| `OLMLX_EXPERIMENTAL_DISTRIBUTED_REMOTE_WORKING_DIR` | *(empty)* | Working directory on remote workers |
| `OLMLX_EXPERIMENTAL_DISTRIBUTED_REMOTE_PYTHON` | `python` | Python command on remote workers |

## Distributed Inference (Experimental)

Run models across multiple Apple Silicon machines connected via network (Thunderbolt recommended for best performance). This lets you run models that don't fit on a single machine — e.g. a 72B model split across two 64GB Mac Minis.

### Setup

1. **Both machines** need olmlx installed and the same model downloaded:
   ```bash
   cd ~/Documents/olmlx_distributed
   git clone <repo-url> . && uv sync --no-editable
   ```

2. **Passwordless SSH** from the coordinator to all workers:
   ```bash
   ssh-copy-id user@worker-ip
   ssh-keyscan -H worker-ip >> ~/.ssh/known_hosts
   ```

3. **Create a hostfile** on the coordinator at `~/.olmlx/hostfile.json`:
   ```json
   {
     "hosts": ["10.0.1.1", "10.0.1.2"],
     "model": "mlx-community/Qwen2.5-32B-Instruct-4bit"
   }
   ```
   The first host is the coordinator (rank 0). All hosts must be reachable via SSH.

4. **Configure** the coordinator with a `.env` file or environment variables:
   ```bash
   OLMLX_EXPERIMENTAL_DISTRIBUTED=true
   OLMLX_EXPERIMENTAL_DISTRIBUTED_HOSTFILE=~/.olmlx/hostfile.json
   OLMLX_EXPERIMENTAL_DISTRIBUTED_BACKEND=ring
   OLMLX_EXPERIMENTAL_DISTRIBUTED_PORT=32323
   OLMLX_EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT=32400
   OLMLX_EXPERIMENTAL_DISTRIBUTED_REMOTE_WORKING_DIR=~/Documents/olmlx_distributed
   OLMLX_EXPERIMENTAL_DISTRIBUTED_REMOTE_PYTHON=.venv/bin/python
   OLMLX_HOST=0.0.0.0
   ```

5. **Start the server** on the coordinator only — workers are launched automatically via SSH:
   ```bash
   .venv/bin/python -m olmlx serve
   ```

6. **Send requests** using the model from the hostfile:
   ```bash
   curl http://coordinator-ip:11434/api/chat -d '{
     "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
     "messages": [{"role": "user", "content": "Hello!"}],
     "stream": false
   }'
   ```

### How it works

- The coordinator generates an MLX ring hostfile from the hosts list and launches workers on remote machines via SSH
- MLX's ring distributed backend (`mx.distributed.init`) connects all ranks
- A sideband TCP channel (separate from the ring) broadcasts inference parameters to workers
- On each request, the coordinator broadcasts prompt/params, both ranks run `stream_generate` in lockstep, and `all_sum` operations synchronize partial results
- Only the coordinator returns results; worker output is discarded

### Limitations

- The requested model must match the model in the hostfile (workers pre-load it at startup)
- VLM (vision) models are not supported in distributed mode
- If the coordinator crashes mid-inference, workers hang indefinitely (MLX has no timeout on collective operations) — the atexit handler kills worker processes
- Distributed adds per-token latency from network synchronization — it's slower than single-machine for models that fit in memory, but enables models that otherwise wouldn't run at all

## How It Works

Instead of GGUF models and llama.cpp, this server uses [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) to run inference directly on Apple Silicon's GPU via the Metal framework. Models are downloaded from HuggingFace Hub in MLX safetensor format.

The server translates Ollama API requests into `mlx_lm.generate()` / `mlx_lm.stream_generate()` calls, with a sync-to-async streaming bridge so the FastAPI event loop stays responsive during generation.

Key internals:
- **Model Manager** — loads/unloads models with configurable keep-alive TTLs and LRU eviction
- **Registry** — resolves Ollama model names to HuggingFace paths via `models.json`
- **Streaming Bridge** — runs `mlx_lm.stream_generate()` in a thread, feeds tokens through an `asyncio.Queue`
- **Chat Templates** — uses each model's built-in `tokenizer.apply_chat_template()` for correct prompt formatting
- **Prompt Caching** — reuses KV cache across requests when the prompt shares a common prefix, reducing time-to-first-token. Works with both text models (mlx-lm) and vision models (mlx-vlm)

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

### Metal GPU crashes / out of memory

If a model is too large for your system, olmlx will reject it at load time with a clear error message (HTTP 507) instead of crashing. The server checks Metal GPU memory usage after loading and compares it against `OLMLX_MEMORY_LIMIT_FRACTION` (default: 75% of system RAM).

If you still experience crashes or want to adjust the threshold:

1. **Use a smaller model** — try 4-bit quantized models instead of 8-bit or 16-bit
2. **Increase the limit** — set `OLMLX_MEMORY_LIMIT_FRACTION=0.85` if you have headroom
3. **Free memory** — close other GPU-heavy apps (video editors, 3D apps)
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
| Qwen 2.5/3/3.5 | ✓ | ✓ | ✓ (Qwen 3+) | ✗ |
| Llama 3.1/3.2 | ✓ | ✓ | ✗ | ✗ |
| Mistral/Nemo | ✓ | ✓ | ✗ | ✗ |
| Gemma 2 | ✓ | ✗ | ✗ | ✗ |
| Phi 3 | ✓ | ✗ | ✗ | ✗ |
| LLava-based | ✓ | ✗ | ✗ | ✓ |

Check a model's chat template on HuggingFace to verify feature support.
