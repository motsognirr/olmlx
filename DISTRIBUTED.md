# Distributed Inference

Run models across multiple Apple Silicon machines using MLX's ring distributed backend. This enables models too large for a single machine — e.g. a 72B model split across two Mac Minis.

## Architecture

```
┌─────────────────────┐         SSH          ┌─────────────────────┐
│   Coordinator (R0)  │ ───────────────────> │    Worker (R1)      │
│                     │                      │                     │
│  olmlx serve        │   Ring Backend       │  distributed_worker │
│  ├─ ring init ◄─────┼──── all_sum ────────►│  ├─ ring init       │
│  ├─ sideband server │   (Thunderbolt)      │  ├─ sideband client │
│  ├─ uvicorn/FastAPI  │                      │  ├─ model.shard()   │
│  └─ model.shard()   │   Sideband (TCP)     │  └─ stream_generate │
│                     │ ◄──── ready ─────── │                     │
│  On request:        │ ────── params ─────►│                     │
│  broadcast → generate│ ◄── all_sum ───────►│  generate (lockstep)│
└─────────────────────┘                      └─────────────────────┘
        ▲
        │ HTTP :11434
   ┌────┴────┐
   │  Client │
   └─────────┘
```

**Ring backend**: MLX's `mx.distributed.init(backend="ring")` connects ranks via TCP. Each rank listens on its own port and connects to the next rank. Collective operations (`all_sum`) synchronize partial results during the forward pass.

**Sideband channel**: A separate TCP connection (port 32400) from coordinator to workers broadcasts inference parameters (prompt, max_tokens, gen_kwargs) before each request. The ring backend only handles tensor synchronization.

## Startup Sequence

1. **CLI** generates ring hostfile from `~/.olmlx/hostfile.json`
2. **CLI** launches workers on remote hosts via SSH
3. **CLI** sleeps 3s, then calls `mx.distributed.init()` (ring handshake)
4. **CLI** starts sideband server (before uvicorn — avoids slow `import transformers` blocking workers)
5. **Worker** completes ring init, connects to sideband (retries up to 120s)
6. **Worker** loads model, shards, materializes weights, sends ready signal
7. **Uvicorn** starts, app lifespan retrieves pre-created group + coordinator
8. **Lifespan** waits for all workers to report ready (up to 60s)
9. **Server** accepts requests

## Performance Results

Tested with two M4 Mac Minis (64GB + 24GB) connected via Thunderbolt.

### Qwen2.5-14B-Instruct-4bit (~8GB)

| Setup | Time | Notes |
|-------|------|-------|
| Standalone (64GB machine) | — | Fits easily, no need for distributed |
| Distributed (WiFi) | 66.2s | ~300 tokens |
| Distributed (Thunderbolt) | 46.9s | ~30% faster than WiFi |

### Qwen2.5-32B-Instruct-4bit (~18GB)

| Setup | Time | Notes |
|-------|------|-------|
| Standalone (64GB machine) | 28.4s | Fits on one machine |
| Distributed (Thunderbolt) | 74.9s | ~9GB per shard |

### Key Observations

- **Distributed adds per-token latency** from `all_sum` network synchronization. For models that fit on a single machine, standalone is faster.
- **Thunderbolt is ~30% faster than WiFi** for distributed inference due to lower `all_sum` latency.
- **The value of distributed is enabling models that don't fit on one machine** — e.g. a 72B model (~40GB) across two 64GB machines, where standalone would OOM.

## Setup Guide

### Prerequisites

- Two or more Apple Silicon Macs on the same network (Thunderbolt recommended)
- Passwordless SSH from coordinator to all workers
- Same olmlx version and model downloaded on all machines

### 1. Install on all machines

```bash
git clone <repo-url> ~/Documents/olmlx_distributed
cd ~/Documents/olmlx_distributed
uv sync --no-editable
```

### 2. Set up passwordless SSH (from coordinator)

```bash
ssh-copy-id user@worker-ip
ssh-keyscan -H worker-ip >> ~/.ssh/known_hosts
# Verify: ssh -o BatchMode=yes user@worker-ip hostname
```

### 3. Create hostfile on coordinator

```bash
cat > ~/.olmlx/hostfile.json << 'EOF'
{
  "hosts": ["10.0.1.1", "10.0.1.2"],
  "model": "mlx-community/Qwen2.5-32B-Instruct-4bit"
}
EOF
```

The first host is the coordinator. Use Thunderbolt IPs for best performance.

### 4. Configure coordinator

Create `~/Documents/olmlx_distributed/.env`:

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

### 5. Pre-download the model on all machines

```bash
.venv/bin/python -c "import mlx_lm; mlx_lm.load('mlx-community/Qwen2.5-32B-Instruct-4bit')"
```

### 6. Start (coordinator only)

```bash
cd ~/Documents/olmlx_distributed
.venv/bin/python -m olmlx serve
```

Workers are launched automatically via SSH. Monitor worker progress:

```bash
tail -f ~/.olmlx/worker-1.log
```

### 7. Test

```bash
curl http://coordinator-ip:11434/api/chat -d '{
  "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

The requested model **must match** the model in the hostfile.

## Troubleshooting

### Worker fails to connect to sideband

The worker retries the sideband connection for up to 120s. If it still fails, the coordinator's uvicorn startup is too slow (heavy `import transformers`). Check that the sideband server starts before uvicorn in the coordinator log:

```
Distributed coordinator listening on 0.0.0.0:32400
```

### Metal GPU timeout on large models

After `model.shard()`, weights must be materialized before inference. Without this, the first forward pass combines lazy weight evaluation with `all_sum` in a single Metal command buffer that exceeds the ~10s GPU timeout. This is handled automatically in the code.

### Ring init hangs

Both coordinator and worker must call `mx.distributed.init()` within each other's ~31s retry window. If the coordinator starts too late (heavy imports), increase the delay in `cmd_serve()`. Check that `MLX_RANK` and `MLX_HOSTFILE` are set correctly on both sides.

### Worker shows `errno 32` (EPIPE)

The coordinator crashed first, breaking the ring socket. Check the coordinator log for the root cause (usually Metal GPU timeout or OOM).

### Model mismatch crash

The worker loads the model from the hostfile at startup. If you request a different model via the API, the `all_sum` tensor shapes won't match, causing a Metal timeout. Always request the same model specified in the hostfile.

## Combining with Flash Inference

Flash and distributed can be used together for dense (non-MoE) models. This combines two independent speedups:

- **Attention**: distributed across ranks (sharded projections, `all_sum` synchronization)
- **MLP**: each rank independently loads only active neurons from its local SSD (Flash sparsity)

### How it works

When both `OLMLX_EXPERIMENTAL_FLASH=true` and `OLMLX_EXPERIMENTAL_DISTRIBUTED=true` are set:

1. `FlashModelWrapper.shard()` shards only attention projections (`q/k/v/o_proj`), leaving FlashMLP layers unsharded
2. `o_proj` uses `ShardedToAllLinear` which calls `all_sum` — its output is replicated on all ranks
3. Every rank feeds identical input to FlashMLP → same neuron predictions → same output
4. Pre-sharding is automatically skipped (MLP weights live on SSD, not in safetensors)

### Setup

Each machine must independently prepare the model for Flash:

```bash
# On EVERY machine (coordinator AND workers):
olmlx flash prepare mlx-community/Qwen2.5-32B-Instruct-4bit
```

Then configure the coordinator:

```bash
OLMLX_EXPERIMENTAL_DISTRIBUTED=true
OLMLX_EXPERIMENTAL_FLASH=true
# Flash tuning params are forwarded to workers automatically
OLMLX_EXPERIMENTAL_FLASH_IO_THREADS=32
```

### Memory advantage

Flash + distributed uses less RAM per rank than either feature alone:

- MLP weights (~2/3 of model) live on SSD — no RAM cost
- Attention weights (~1/3 of model) are split across ranks
- Each rank holds: `attention_size/N` + embeddings + lm_head + predictors + neuron cache

### Constraints

- **Dense models only**: Flash-MoE + distributed is explicitly blocked (expert weights cannot be sharded via `all_sum`)
- **Head count divisibility**: `n_heads` and `n_kv_heads` must be evenly divisible by world size
- **Local SSD required**: Each worker needs flash-prepared data on its own SSD — there is no network weight loading

## Limitations

- Only one model can be used (specified in hostfile, loaded at worker startup)
- VLM (vision-language) models are not supported in distributed mode
- If the coordinator crashes mid-inference, workers hang indefinitely (MLX has no timeout on collective operations)
- Prompt caching is disabled in distributed mode
- All machines must have the model downloaded locally (each loads full weights, then shards)
