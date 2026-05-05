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
olmlx                        # Start the server (default)
olmlx serve                  # Start the server (explicit)
olmlx chat <model>           # Interactive terminal chat with MCP tool support
olmlx models list            # List locally downloaded models
olmlx models pull <name>     # Download a model
olmlx models show <name>     # Show model details
olmlx models delete <name>   # Delete a local model
olmlx models search <query>  # Search for models by name (fuzzy matching)
olmlx flash prepare <model>  # Prepare a model for flash inference
olmlx flash info <model>     # Show flash preparation status
olmlx bench run [--model M]  # Run benchmark scenarios
olmlx bench compare <a> <b>  # Compare two benchmark runs
olmlx bench list             # List past benchmark runs
olmlx config show            # Show current configuration
olmlx service install        # Install as launchd service
olmlx service status         # Check service status
olmlx service uninstall      # Remove the service
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
| `OLMLX_MAX_TOKENS_LIMIT` | `131072` | Maximum tokens allowed per request |
| `OLMLX_CORS_ORIGINS` | `http://localhost:*`, `http://127.0.0.1:*` | Allowed CORS origins |
| `OLMLX_SPECULATIVE` | `false` | Enable speculative decoding with a draft model (also `--speculative` on `olmlx serve`) |
| `OLMLX_SPECULATIVE_DRAFT_MODEL` | `None` | HuggingFace path of the draft model (also `--speculative-draft-model`) |
| `OLMLX_SPECULATIVE_TOKENS` | `4` | Candidate tokens generated per verification step (also `--speculative-tokens`) |
| `OLMLX_KV_CACHE_QUANT` | `None` | KV cache quantization: `turboquant:4` (~3.9x), `turboquant:2` (~7.5x), `spectral:4` (~5.9x), or `spectral:2` (also `--kv-cache-quant`) |

### Flash inference settings (experimental)

| Variable | Default | Description |
|---|---|---|
| `OLMLX_EXPERIMENTAL_FLASH` | `false` | Enable LLM in a Flash inference |
| `OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD` | `0.5` | Activation sparsity threshold (0-1] |
| `OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS` | `128` | Minimum active neurons per layer |
| `OLMLX_EXPERIMENTAL_FLASH_IO_THREADS` | `32` | I/O threads for SSD weight loading |
| `OLMLX_EXPERIMENTAL_FLASH_CACHE_BUDGET_NEURONS` | `1024` | Budget for cached neurons in memory |
| `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE` | `false` | Enable speculative decoding with draft model |
| `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL` | `None` | Draft model name or HuggingFace path |
| `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS` | `4` | Candidate tokens per speculative step |
| `OLMLX_EXPERIMENTAL_FLASH_PREFETCH` | `false` | Enable speculative neuron prefetching |

### Flash-MoE settings (experimental)

| Variable | Default | Description |
|---|---|---|
| `OLMLX_EXPERIMENTAL_FLASH_MOE` | `false` | Enable Flash-MoE expert offloading for MoE models |
| `OLMLX_EXPERIMENTAL_FLASH_MOE_CACHE_BUDGET_EXPERTS` | `48` | Experts cached per layer (LRU eviction) |
| `OLMLX_EXPERIMENTAL_FLASH_MOE_IO_THREADS` | `32` | I/O threads for expert loading |

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

## Speculative Decoding

Speculative decoding pairs a small *draft* model with a larger *target* model: the draft proposes several tokens autoregressively, and the target verifies them in a single forward pass. When most drafts are accepted you get multiple tokens per target step, which lowers latency. Verification is performed with greedy argmax against the target's logits — so for `temperature=0` requests the output is bit-identical to plain greedy decoding. With `temperature > 0`, output quality is preserved but the token stream is no longer drawn from the target's sampled distribution, so individual completions can differ from a non-speculative run with the same seed.

```bash
OLMLX_SPECULATIVE=true \
OLMLX_SPECULATIVE_DRAFT_MODEL=mlx-community/Qwen3-0.6B-4bit \
olmlx serve
```

Or, equivalently, with CLI flags:

```bash
olmlx serve \
  --speculative \
  --speculative-draft-model mlx-community/Qwen3-0.6B-4bit \
  --speculative-tokens 4
```

You can also pin per-model defaults in `~/.olmlx/models.json`:

```json
{
  "mlx-community/Qwen3.5-27B-4bit:latest": {
    "hf_path": "mlx-community/Qwen3.5-27B-4bit",
    "speculative": true,
    "speculative_draft_model": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    "speculative_tokens": 4
  }
}
```

### Picking a draft model

- Use the same model family — vocabulary mismatches are rejected at load.
- Smaller is better for latency *if* the draft tracks the target's distribution. A 0.5–1B draft for a 7–32B target is a good starting point.
- Quantization matters less for the draft than for the target — pick whatever fits comfortably alongside the target in memory.

### Expected speedup

Real-world speedup typically lands between **1.4x and 2x** on Apple Silicon for code/chat workloads, dropping toward **1x** on highly creative or out-of-distribution prompts where the draft is frequently rejected. The token budget (`--speculative-tokens`, default 4) trades draft work for verification savings — try 4–8 for routine workloads, lower if your prompts are noisy.

### When not to use it

- The target is already small (under ~3B). The draft+target overhead can dominate.
- You're memory-constrained: both models stay resident, including their KV caches.
- Prompts are highly stochastic / high-temperature — acceptance drops and overhead dominates.
- You're using vision-language models that don't expose `.language_model` (most VLMs work, but a few do not).

### Migration from `OLMLX_EXPERIMENTAL_SPECULATIVE_*`

The settings have been promoted out of `experimental`. The new env vars are `OLMLX_SPECULATIVE`, `OLMLX_SPECULATIVE_DRAFT_MODEL`, and `OLMLX_SPECULATIVE_TOKENS`. The legacy `OLMLX_EXPERIMENTAL_SPECULATIVE*` names are still honoured for one release: their values are forwarded to the new settings (the new names win when both are set) and a deprecation warning is logged at startup. Per-model `models.json` entries that previously placed these keys under `"experimental": {...}` now go at the top level — loading an old config raises a clear migration error pointing at the new location.

**Important — also rename the keys in your `.env`.** pydantic-settings reads the project's `.env` file to populate `Settings`; if your `.env` still has the old `OLMLX_EXPERIMENTAL_SPECULATIVE*` names, the deprecation banner and per-field forwarder honour them too. Rename the keys in `.env` to the new names; the legacy names are scanned and forwarded for one release only.

**Important — `.env` opt-outs during the deprecation window.** The legacy env-var forwarder cannot distinguish "field was never set" from "field was explicitly written to its schema default in `.env`" (e.g. `OLMLX_SPECULATIVE=false`). If you have an explicit-default `.env` opt-out **and** the old `OLMLX_EXPERIMENTAL_SPECULATIVE*` still set in your shell, the legacy value will silently overwrite the `.env` value — and speculative will be re-enabled despite your `.env` saying otherwise. Two ways to avoid this surprise:

- Remove the legacy `OLMLX_EXPERIMENTAL_SPECULATIVE*` exports from your shell profile before upgrading.
- Watch the startup logs for `Forwarding legacy …` warnings — they fire per-field whenever the forwarder applies a value, so you can spot an unwanted override immediately.

**Behaviour change — Settings now validate on assignment.** As part of this change, `Settings` runs Pydantic validators on every programmatic field assignment (not just the speculative ones). Code that previously did `settings.port = 0` to test error handling will now raise `ValidationError` instead of silently accepting the bad value. This is intentional — invalid settings should never be reachable — but it is a behaviour change for anyone who was monkey-patching settings in tests or tools.

## LLM in a Flash (Experimental)

Run models larger than available GPU memory by keeping only active neurons in RAM and loading the rest from SSD on demand.

### Setup

```bash
# 1. Prepare the model (one-time)
olmlx flash prepare mlx-community/Qwen2.5-32B-Instruct-4bit

# 2. Check preparation status
olmlx flash info mlx-community/Qwen2.5-32B-Instruct-4bit

# 3. Start with flash enabled
OLMLX_EXPERIMENTAL_FLASH=true olmlx serve
```

### Speculative decoding

Combine flash with a small draft model for faster token generation:

```bash
OLMLX_EXPERIMENTAL_FLASH=true \
OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE=true \
OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit \
olmlx serve
```

The draft model generates candidate tokens in-memory, then the flash model verifies them in one pass — producing multiple tokens per SSD read.

### Flash-MoE

For Mixture-of-Experts models (DeepSeek-V3, Kimi-K2.5, Qwen3-Next MoE, MiniMax, gpt-oss), Flash-MoE keeps only the router in RAM and loads routed experts from SSD on demand:

```bash
OLMLX_EXPERIMENTAL_FLASH_MOE=true olmlx serve
```

### TurboQuant KV cache

Compress the KV cache ~4-8x using TurboQuant quantization, enabling longer context windows. Add to your `.env` file:

```bash
# 4-bit (~3.9x compression)
OLMLX_KV_CACHE_QUANT=turboquant:4

# 2-bit (~7.5x compression, lower quality)
OLMLX_KV_CACHE_QUANT=turboquant:2
```

**SpectralQuant** improves on TurboQuant with data-driven eigenvector rotations and non-uniform bit allocation (~19% better compression, +2.6pp cosine similarity). Requires one-time calibration:

```bash
# Calibrate the model (one-time, ~15 seconds)
olmlx spectral prepare <model>

# Use spectral quant
OLMLX_KV_CACHE_QUANT=spectral:4
```

Note: TurboQuant and SpectralQuant are incompatible with disk cache offload.

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

### Combining with Flash inference

Flash and distributed can be used together for dense (non-MoE) models. Attention is distributed across ranks while each rank loads active MLP neurons from its local SSD. Each machine must independently run `olmlx flash prepare <model>`. Enable with `OLMLX_EXPERIMENTAL_FLASH=true` alongside distributed settings. See [DISTRIBUTED.md](DISTRIBUTED.md) for details.

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
| DeepSeek | ✓ | ✓ | ✗ | ✗ |
| Gemma 2 | ✓ | ✗ | ✗ | ✗ |
| Phi 3 | ✓ | ✗ | ✗ | ✗ |
| LLava-based | ✓ | ✗ | ✗ | ✓ |

Check a model's chat template on HuggingFace to verify feature support.
