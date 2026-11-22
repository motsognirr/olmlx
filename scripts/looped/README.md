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
  mean, exact per-loop biases/norms, and a gate that is exactly open at init. At
  full rank the student equals the teacher. At a practical rank it does **not**
  start close: real layers are nearly unrelated (`|W - mean|` ≈ 0.8 `|W|`, with a
  flat spectrum), so rank 64 recovers ~15 % of each difference and rank 256 ~35 %,
  and the student collapses. Measured on Qwen3-0.6B (5 + 6×3 + 5): teacher CE
  3.06; init KL 0.0006 at rank 1024, 11.7 at rank 64, 11.9 at rank 256, 11.2 at
  rank 0, and 10.5 in inkling mode. Distillation does all the recovery work.
* `--mode inkling`: the gate starts closed (every pass sees only the prelude
  output) and opens during training.
* `--teacher <model>`: distill from a separate, e.g. larger, model instead of
  the base (Qwen3-4B for a looped Qwen3-0.6B). The student is still built from
  `--model`. The teacher must share the tokenizer exactly, since the KL is taken
  token by token over the vocabulary. It gets its own full forward pass, so
  training is slower. `eval` then reports base, student and teacher perplexity.

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

* `--data` takes `.jsonl` (`{"text": ...}`) or `.txt`. A `.txt` is one
  document: it is read and tokenized whole, every epoch, so use `.jsonl` or
  `--hf-dataset` for a large corpus. Evaluate on a held-out file, because
  `eval` on the same HF stream reads the data that training started on.
* The base must be unquantized (bf16/fp16), e.g. `Qwen/Qwen3-0.6B`, not an
  `mlx-community/...-4bit` build: the core is a mean of the layers' weights.
* `--resume` continues the LR schedule from the saved step. The optimizer
  moments restart. The schedule is rebuilt from the current `--lr`/`--warmup`/
  `--steps`, so a larger `--steps` extends the run with a re-spanned cosine (a
  warm restart); the trainer prints a note when it differs from the saved one.
  `--steps` must exceed the checkpoint's step. To continue the data stream
  exactly, resume re-reads and re-tokenizes the data already trained on and
  skips it, which adds minutes to a resume after a long run.
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
