# Looped Qwen (experiment)

Converts a Qwen2/Qwen3 checkpoint into a *looped* transformer — prelude →
shared core run N times → coda, in the style of Inkling 2 Mini — and recovers
quality by distilling from the original model. Pure MLX, standalone: nothing in
`olmlx/` imports it, and olmlx cannot serve the result yet (that needs a
registered model type).

```
embed → prelude (frozen, shared with teacher)
      → core (K shared layers) × N passes, message gate between passes
      → coda (copy of last layers, trainable) → norm → lm_head
```

`prelude + core × loops + coda` must equal the base model's layer count, so the
student costs the same FLOPs per token with fewer layer parameters.

* `--mode teacher` (default): core = mean of the replaced layer groups, plus a
  per-loop LoRA initialised by SVD of each original layer's difference from the
  mean, exact per-loop biases/norms, and a gate that is exactly open at init. The student starts
  close to the teacher, and exactly equal to it at full rank.
* `--mode inkling`: the gate starts closed (every pass sees only the prelude
  output) and opens during training.

## Usage

```bash
pip install mlx mlx-lm datasets

# quick recipe check on 0.5B (24 layers = 3 + 6×3 + 3)
python scripts/looped/looped_qwen.py train --model Qwen/Qwen2.5-0.5B --out q05-looped \
  --prelude 3 --core 6 --coda 3 --loops 3 \
  --hf-dataset HuggingFaceFW/fineweb-edu --hf-subset sample-10BT \
  --batch 8 --seq 1024 --accum 4 --steps 2000

# Qwen2.5-3B (defaults fit 36 layers: 6 + 8×3 + 6)
python scripts/looped/looped_qwen.py train --model Qwen/Qwen2.5-3B --out q3b-looped \
  --hf-dataset HuggingFaceFW/fineweb-edu --hf-subset sample-10BT \
  --batch 4 --seq 1024 --accum 8 --steps 3000

python scripts/looped/looped_qwen.py eval --out q3b-looped --data heldout.jsonl
python scripts/looped/looped_qwen.py generate --out q3b-looped --prompt "Hello" --chat
```

* `--data` takes `.jsonl` (`{"text": ...}`) or `.txt`. Evaluate on a held-out
  file, because `eval` on the same HF stream reads the data that training
  started on.
* `--resume` continues the LR schedule from the saved step. The optimizer
  moments restart.
* The trainer logs tok/s and peak memory every `--log-every` steps.

Rough estimate for an M4 Pro with 64 GB on the 3B (computed, not measured):
~40–45 GB peak at batch 4 × 1024 tokens (≈1.2B fp32 trainable params with
AdamW, the bf16 teacher, and full-vocab logits), and on the order of
150–300 tok/s.

## Self-check

```bash
python scripts/looped/check_looped.py
```

Runs on tiny random Qwen2/Qwen3 models (CPU is fine). It checks that the
full-rank teacher init matches the teacher, that KV-cache decoding matches a
full forward, and that gradients are identical with and without checkpointing.
It is deliberately not a pytest module (no `test_` prefix), so the repo's CI
never collects it.
