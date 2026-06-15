# Proxy-Tuning Stage 2 — Train M⁺ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the proxy-tuning expert/anti-expert pair: M⁻ = `Qwen3-1.7B-Base` (used as-is) and M⁺ = M⁻ LoRA-fine-tuned on the Stage-1 dataset, fused and 4-bit quantized — both sharing the Qwen3 tokenizer (vocab 151936) required to steer a dense Qwen3 base.

**Architecture:** Stage 2 is **mostly orchestration of `mlx-lm` CLIs** (`convert`, `lora`, `fuse`) run on Apple Silicon, plus **one** unit-testable Python helper that gates the result (tokenizer-identity verifier). Validation is by **artifact checks** (does it load? generate coherently? share the tokenizer?), not unit tests — except the verifier, which is TDD. A tiny smoke-train precedes the full run to catch wiring errors cheaply.

**Tech Stack:** `mlx-lm` 0.31.2 (`convert`/`lora`/`fuse`/`generate`), MLX, the Stage-1 dataset (`data/proxy_tuning/{train,valid}.jsonl`), pytest (verifier only).

**Scope:** Stage 2 only of `docs/superpowers/specs/2026-06-15-olmlx-proxy-tuning-pair-design.md` §6. Stage 3 (α-sweep eval + serve) is a separate plan, blocked on this stage's artifacts.

**Grounded facts (verified during planning):**
- `Qwen/Qwen3-1.7B-Base` exists on HF and **ships a `chat_template`** (4,116 chars) → LoRA on our chat-format data needs no template workaround.
- No prebuilt MLX base (`mlx-community/Qwen3-1.7B-Base-bf16` does **not** exist) → we `mlx_lm convert` from the HF base. (`mlx-community/Qwen3-1.7B-bf16` exists but is the **instruct** variant — wrong for M⁻, which must be untuned.)
- `mlx_lm convert -q --q-bits --q-group-size`, `mlx_lm lora --train --model --data --fine-tune-type lora --num-layers --batch-size --iters --learning-rate --max-seq-length --mask-prompt --steps-per-eval --val-batches --adapter-path --grad-checkpoint`, and `mlx_lm fuse --model --adapter-path --save-path` are all confirmed present in 0.31.2.

**Execution note (important):** only **Task 1** (the verifier) is TDD code suitable for subagent/inline execution. **Tasks 2–8 are an operator runbook** — they run real `mlx-lm` on the user's Mac with their models (downloads + ~30–90 min training), like Stage-1's real-run tasks. The agent implements Task 1; the operator runs Tasks 2–8 and pastes results back.

**All artifacts live under** `~/.olmlx/proxy_tuning/` (checkout-independent, near the user's other models; the Stage-3 serving config will reference these by absolute path).

---

## Execution status (updated 2026-06-15)

- ✅ **Task 1** done — verifier committed. **Deviation:** the committed verifier checks each model's `config.json` `vocab_size` (151936 — the logits width proxy-tuning's arithmetic needs), **not** `len(tokenizer.get_vocab())` (151669). The Task-1 code block below is the original; the committed code reflects this fix. Task 8's `--base-vocab 151936` is therefore correct as written.
- ✅ **Task 2** done — M⁻ converted to bf16. **Deviation:** the `chat_template` survived as a sidecar `chat_template.jinja` (4116 chars), so the Step-3 `tokenizer_config.json` check prints `False` but the loaded tokenizer *does* expose the template (`apply_chat_template` works). **The Task-2 Step-3 "copy from instruct sibling" remedy is unnecessary — skip it.**
- ✅ **Task 3** done — clean `sft/` dir (7954 train + 692 valid).
- ✅ **Task 4** done — 20-iter smoke-train passed (val loss 2.804→2.220, peak mem 6.5 GB); smoke adapters removed.
- ✅ **Task 5** done, **with a deviation from the constant-LR recipe.** A first run at constant `lr 1e-4` plateaued (val min ~2.017 @ iter 800, then rose). Re-ran with a **linear-warmup → cosine-decay** schedule (`~/.olmlx/proxy_tuning/lora_cosine.yaml`: warmup 100 @ 1e-7→1e-4, cosine 1e-4→1e-5 over 3900 post-warmup steps). Val bottomed at **iter 1800 (1.814)** then overfit, so the run was stopped early. **Best checkpoint = iter 1800**, staged into `~/.olmlx/proxy_tuning/adapters_best/` (md5-verified copy of `0001800_adapters.safetensors`).
- ✅ **Task 6** done — fused M⁻ + the **iter-1800** adapter (from `adapters_best/`, not the final weights) → `qwen3-1.7b-olmlx-expert-bf16/`; generates coherent, code/olmlx-flavored text.
- ✅ **Task 7** done — both quantized to 4-bit (4.501 bpw, 934 MB each): `qwen3-1.7b-olmlx-expert-4bit/`, `qwen3-1.7b-base-4bit/`; both load + generate.
- ✅ **Task 8** done — gate PASSED: `OK: M-/M+ share one tokenizer and match the base vocabulary.`
- **Hazard hit:** a parallel session checked this shared checkout out to `main` mid-train (per the parallel-session interference noted in project memory). All commits survived on `feat/proxy-tuning-stage2`; recovered with `git checkout`. Watch for branch thrash on long runs.

**Stage 2 complete.** Stage-3 serving artifacts:
- M⁺ (expert): `~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-4bit`
- M⁻ (anti-expert): `~/.olmlx/proxy_tuning/qwen3-1.7b-base-4bit`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `olmlx/proxy_tuning_pipeline/verify.py` | `assert_serveable_pair()` — gate that M⁻/M⁺ share one tokenizer and match the base vocab; thin CLI | Create |
| `tests/test_proxy_tuning_verify.py` | Unit tests for the verifier (injected fake tokenizers) | Create |
| `~/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16/` | M⁻ in bf16 (training base) | Artifact (Task 2) |
| `~/.olmlx/proxy_tuning/sft/{train,valid}.jsonl` | Clean SFT data dir for `mlx_lm lora` | Artifact (Task 3) |
| `~/.olmlx/proxy_tuning/adapters/` | LoRA adapter weights | Artifact (Task 5) |
| `~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-bf16/` | M⁺ fused, bf16 | Artifact (Task 6) |
| `~/.olmlx/proxy_tuning/qwen3-1.7b-{base,olmlx-expert}-4bit/` | M⁻ and M⁺ at 4-bit (serving) | Artifacts (Task 7) |

---

## Task 1: Tokenizer-identity / serveability verifier (TDD)

**Files:**
- Create: `olmlx/proxy_tuning_pipeline/verify.py`
- Test: `tests/test_proxy_tuning_verify.py`

**Context:** Proxy-tuning adds logits across M, M⁻, M⁺ token-by-token, so all three must share one exact tokenizer. M⁺ inherits M⁻'s tokenizer by construction (it *is* M⁻ fine-tuned), but quantization/fuse round-trips can in principle perturb tokenizer files — and the pair must also match the steered base's vocab (Qwen3 = 151936). This verifier is the **gate** before serving. It reuses the already-tested `check_vocab_identity` from the engine (`olmlx/engine/proxy_tuning.py`) for the M⁻↔M⁺ mapping check, and adds the base-vocab-size check.

- [ ] **Step 1: Write the failing test**

Create `tests/test_proxy_tuning_verify.py`:

```python
"""Tests for the Stage-2 M-/M+ serveability verifier."""

from __future__ import annotations

import pytest

from olmlx.proxy_tuning_pipeline.verify import assert_serveable_pair


class _FakeTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def _loader_for(mapping: dict[str, dict[str, int]]):
    def _load(path: str):
        return _FakeTokenizer(mapping[path])

    return _load


def test_assert_serveable_pair_passes_for_matching_pair():
    vocab = {tok: i for i, tok in enumerate(["a", "b", "c"])}
    loader = _loader_for({"m_minus": dict(vocab), "m_plus": dict(vocab)})
    # No raise == pass.
    assert_serveable_pair("m_minus", "m_plus", base_vocab_size=3, loader=loader)


def test_assert_serveable_pair_raises_on_token_mapping_diff():
    loader = _loader_for(
        {"m_minus": {"a": 0, "b": 1, "c": 2}, "m_plus": {"a": 0, "b": 2, "c": 1}}
    )
    with pytest.raises(ValueError, match="vocab"):
        assert_serveable_pair("m_minus", "m_plus", base_vocab_size=3, loader=loader)


def test_assert_serveable_pair_raises_on_base_vocab_mismatch():
    vocab = {"a": 0, "b": 1, "c": 2}
    loader = _loader_for({"m_minus": dict(vocab), "m_plus": dict(vocab)})
    with pytest.raises(ValueError, match="base"):
        assert_serveable_pair("m_minus", "m_plus", base_vocab_size=151936, loader=loader)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_verify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.proxy_tuning_pipeline.verify'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/proxy_tuning_pipeline/verify.py`:

```python
"""Gate a trained M-/M+ pair: do they share one tokenizer + match the base vocab?

Proxy-tuning combines logits across M (base), M- (anti-expert), M+ (expert)
token-by-token, so all three must share one exact tokenizer. Run this before
registering the pair for serving.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

from olmlx.engine.proxy_tuning import check_vocab_identity


def _load_tokenizer(path: str) -> Any:
    from mlx_lm import load

    _model, tokenizer = load(path)
    return tokenizer


def assert_serveable_pair(
    anti_expert_dir: str,
    expert_dir: str,
    base_vocab_size: int,
    *,
    loader: Callable[[str], Any] = _load_tokenizer,
) -> None:
    """Raise ValueError unless M-/M+ share a token->id mapping and match the base.

    ``base_vocab_size`` is the steered model's vocabulary size (Qwen3 = 151936).
    """
    tok_anti = loader(anti_expert_dir)
    tok_expert = loader(expert_dir)
    # M- <-> M+ token-mapping identity (reuses the engine's tested guard).
    check_vocab_identity(
        tok_anti,
        tok_expert,
        reference_label="anti-expert (M-)",
        other_label="expert (M+)",
    )
    vocab = tok_anti.get_vocab()
    if len(vocab) != base_vocab_size:
        raise ValueError(
            f"M-/M+ vocab size ({len(vocab)}) does not match the steered base "
            f"vocabulary ({base_vocab_size}). All three models must share one "
            f"tokenizer — confirm the base, M-, and M+ are the same family."
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Verify an M-/M+ proxy-tuning pair is serveable."
    )
    ap.add_argument("anti_expert", help="M- (untuned base) model directory")
    ap.add_argument("expert", help="M+ (fine-tuned) model directory")
    ap.add_argument(
        "--base-vocab",
        type=int,
        default=151936,
        help="steered base vocab size (Qwen3 dense = 151936)",
    )
    args = ap.parse_args(argv)
    assert_serveable_pair(args.anti_expert, args.expert, args.base_vocab)
    print("OK: M-/M+ share one tokenizer and match the base vocabulary.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_verify.py -v`
Expected: 3 passed

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/proxy_tuning_pipeline/verify.py tests/test_proxy_tuning_verify.py && uv run ruff format olmlx/proxy_tuning_pipeline/verify.py tests/test_proxy_tuning_verify.py`
Expected: no errors

```bash
git add olmlx/proxy_tuning_pipeline/verify.py tests/test_proxy_tuning_verify.py
git commit -m "feat(proxy-tuning): M-/M+ serveability verifier (tokenizer identity + base vocab)"
```

---

## Task 2: Source + convert M⁻ (Qwen3-1.7B-Base → bf16 MLX)

**Files:** Artifact only — `~/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16/` (operator-run; needs network for the HF download)

> M⁻ is the **untuned base** — used as-is as the anti-expert, and the starting point M⁺ is fine-tuned from. bf16 (not quantized) because LoRA trains on it; a separate 4-bit copy for serving comes in Task 7.

- [ ] **Step 1: Convert the HF base to bf16 MLX**

Run:
```bash
mkdir -p ~/.olmlx/proxy_tuning
uv run python -m mlx_lm convert \
  --hf-path Qwen/Qwen3-1.7B-Base \
  --mlx-path ~/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16 \
  --dtype bfloat16
```
Expected: downloads `Qwen/Qwen3-1.7B-Base`, writes `model.safetensors`, `config.json`, `tokenizer*.json`, `tokenizer_config.json` into the target dir; exits 0.

- [ ] **Step 2: Confirm it loads and generates coherent (untuned) text**

Run:
```bash
uv run python -m mlx_lm generate \
  --model ~/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16 \
  --prompt "A KV cache in transformer inference is" --max-tokens 40
```
Expected: coherent continuation (base-model completion, not chat). Confirms weights + tokenizer round-tripped.

- [ ] **Step 3: Confirm the chat_template survived the conversion**

Run:
```bash
uv run python -c "import json; c=json.load(open('$HOME/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16/tokenizer_config.json')); print('chat_template present:', bool(c.get('chat_template')))"
```
Expected: `chat_template present: True` (required for Task 5's chat-format LoRA). If `False`, copy it from the instruct sibling:
```bash
uv run python - <<'PY'
import json
from huggingface_hub import hf_hub_download
src = json.load(open(hf_hub_download("Qwen/Qwen3-1.7B", "tokenizer_config.json")))
dst_path = f"{__import__('os').path.expanduser('~')}/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16/tokenizer_config.json"
dst = json.load(open(dst_path))
dst["chat_template"] = src["chat_template"]
json.dump(dst, open(dst_path, "w"), indent=2)
print("chat_template copied")
PY
```

- [ ] **Step 4: No commit** — artifacts are gitignored (outside the repo). Record the path.

---

## Task 3: Prepare a clean SFT data directory

**Files:** Artifact only — `~/.olmlx/proxy_tuning/sft/{train,valid}.jsonl`

> `mlx_lm lora --data DIR` reads `train.jsonl`/`valid.jsonl`/`test.jsonl` by name. Copy just those two out of `data/proxy_tuning/` (which also holds `raw.jsonl`/`raw.meta.json`) into a clean dir so nothing else is in scope. The Stage-1 files are already mlx-lm chat format (`{"messages": [...]}`), which `mlx_lm lora` auto-detects.

- [ ] **Step 1: Copy the rebalanced train/valid into a clean dir**

Run (from the repo root, where `data/proxy_tuning/` lives):
```bash
mkdir -p ~/.olmlx/proxy_tuning/sft
cp data/proxy_tuning/train.jsonl data/proxy_tuning/valid.jsonl ~/.olmlx/proxy_tuning/sft/
wc -l ~/.olmlx/proxy_tuning/sft/train.jsonl ~/.olmlx/proxy_tuning/sft/valid.jsonl
```
Expected: `train.jsonl` ~7954 lines, `valid.jsonl` ~692 lines (the rebalanced 8,646-example set).

- [ ] **Step 2: Confirm the format is the chat shape mlx-lm expects**

Run:
```bash
head -1 ~/.olmlx/proxy_tuning/sft/train.jsonl | uv run python -c "import sys,json; r=json.loads(sys.stdin.read()); print('keys:', list(r)); print('roles:', [m['role'] for m in r['messages']])"
```
Expected: `keys: ['messages']` and `roles: ['user', 'assistant']`.

- [ ] **Step 3: No commit** — artifacts only.

---

## Task 4: Smoke-train (20 iters) — validate the wiring cheaply

**Files:** Artifact only — a throwaway `~/.olmlx/proxy_tuning/adapters_smoke/`

> Catch data-format / template / memory errors in ~1 minute before committing to the full run.

- [ ] **Step 1: Run a 20-iteration LoRA smoke train**

Run:
```bash
uv run python -m mlx_lm lora \
  --model ~/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16 \
  --train --data ~/.olmlx/proxy_tuning/sft \
  --fine-tune-type lora --num-layers 16 --batch-size 4 \
  --iters 20 --steps-per-eval 10 --val-batches 5 \
  --learning-rate 1e-4 --max-seq-length 2048 --mask-prompt --grad-checkpoint \
  --adapter-path ~/.olmlx/proxy_tuning/adapters_smoke
```
Expected: prints `Loading pretrained model`, tokenizes the data without a chat-template error, runs 20 iters printing `Iter N: Train loss ...`, a couple of `Val loss` lines, and writes `adapters.safetensors` under `adapters_smoke/`. Train loss should be finite and trending down.

- [ ] **Step 2: If it errors**, diagnose before the full run:
  - `Tokenizer does not have a chat template` → re-do Task 2 Step 3 (copy the template).
  - `No such file ... train.jsonl` → re-check Task 3 (data dir).
  - OOM → add/keep `--grad-checkpoint`, lower `--batch-size` to 2.

- [ ] **Step 3: Clean up the smoke adapters**

Run: `rm -rf ~/.olmlx/proxy_tuning/adapters_smoke`

- [ ] **Step 4: No commit.**

---

## Task 5: Full LoRA fine-tune → M⁺ adapters

**Files:** Artifact only — `~/.olmlx/proxy_tuning/adapters/`

> ~7,954 train examples / batch 4 ≈ ~1,989 iters/epoch. `--iters 4000` ≈ 2 epochs — a sound starting point for a LoRA style/convention delta on a 1.7B. Expect **~30–90 min** on Apple Silicon. Watch the validation loss: if it bottoms then rises, that's overfitting — mlx-lm saves adapter checkpoints (`--steps-per-eval` cadence), so an earlier checkpoint can be used.

- [ ] **Step 1: Run the full LoRA train**

Run:
```bash
uv run python -m mlx_lm lora \
  --model ~/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16 \
  --train --data ~/.olmlx/proxy_tuning/sft \
  --fine-tune-type lora --num-layers 16 --batch-size 4 \
  --iters 4000 --steps-per-eval 200 --val-batches 25 \
  --learning-rate 1e-4 --max-seq-length 2048 --mask-prompt --grad-checkpoint \
  --adapter-path ~/.olmlx/proxy_tuning/adapters 2>&1 | tee ~/.olmlx/proxy_tuning/train.log
```
Expected: periodic `Iter N: Train loss X, ...` and `Iter N: Val loss Y` lines; final `Saved final weights to .../adapters/adapters.safetensors`. Train loss should fall well below the iter-1 value; val loss should fall then plateau.

- [ ] **Step 2: Sanity-check the adapter applied (M⁻ + adapter generates)**

Run:
```bash
uv run python -m mlx_lm generate \
  --model ~/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16 \
  --adapter-path ~/.olmlx/proxy_tuning/adapters \
  --prompt "Explain the olmlx Metal stream invariant." --max-tokens 80
```
Expected: coherent, and noticeably more olmlx-flavored / on-topic than the base (Task 2 Step 2) — a qualitative sign the delta took. (Deep quality is Stage-3's eval gate, not here.)

- [ ] **Step 3: Note the final val loss** from `~/.olmlx/proxy_tuning/train.log` (the `Val loss` of the last eval). If val loss was still falling at iter 4000, consider re-running with more iters; if it rose after a minimum, note the iter of the minimum for a possible earlier checkpoint.

- [ ] **Step 4: No commit** — artifacts only.

---

## Task 6: Fuse → M⁺ (full bf16 weights)

**Files:** Artifact only — `~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-bf16/`

> Proxy-tuning needs M⁺ as a standalone full model that produces logits (not a base+adapter pair). `fuse` merges the LoRA adapters into the base weights.

- [ ] **Step 1: Fuse adapters into M⁻ to produce M⁺**

Run:
```bash
uv run python -m mlx_lm fuse \
  --model ~/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16 \
  --adapter-path ~/.olmlx/proxy_tuning/adapters \
  --save-path ~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-bf16
```
Expected: writes a full model dir (`model.safetensors`, `config.json`, `tokenizer*`) and exits 0.

- [ ] **Step 2: Confirm M⁺ loads and generates standalone**

Run:
```bash
uv run python -m mlx_lm generate \
  --model ~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-bf16 \
  --prompt "Explain the olmlx Metal stream invariant." --max-tokens 80
```
Expected: coherent output similar to Task 5 Step 2 (fusing is exact — same behavior as base+adapter). Confirms the fused weights are valid.

- [ ] **Step 3: No commit** — artifacts only.

---

## Task 7: Quantize M⁻ and M⁺ to 4-bit (serving copies)

**Files:** Artifacts only — `~/.olmlx/proxy_tuning/qwen3-1.7b-base-4bit/`, `~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-4bit/`

> Serve both at 4-bit to match the steered base's precision and minimize the decode-time memory/compute of running M⁻+M⁺ every token (q-group-size 64 matches mlx-community's Qwen3 4-bit convention).

- [ ] **Step 1: Quantize M⁺ (expert)**

Run:
```bash
uv run python -m mlx_lm convert \
  --hf-path ~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-bf16 \
  --mlx-path ~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-4bit \
  -q --q-bits 4 --q-group-size 64
```
Expected: writes the 4-bit dir; exits 0.

- [ ] **Step 2: Quantize M⁻ (anti-expert)**

Run:
```bash
uv run python -m mlx_lm convert \
  --hf-path ~/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16 \
  --mlx-path ~/.olmlx/proxy_tuning/qwen3-1.7b-base-4bit \
  -q --q-bits 4 --q-group-size 64
```
Expected: writes the 4-bit dir; exits 0.

- [ ] **Step 3: Confirm both 4-bit models load + generate**

Run:
```bash
for d in qwen3-1.7b-base-4bit qwen3-1.7b-olmlx-expert-4bit; do
  echo "=== $d ==="
  uv run python -m mlx_lm generate --model ~/.olmlx/proxy_tuning/$d \
    --prompt "olmlx is" --max-tokens 15
done
```
Expected: both produce coherent output (no shape/load errors).

- [ ] **Step 4: No commit** — artifacts only.

---

## Task 8: Gate — verify the serveable pair (uses Task 1)

**Files:** none (runs the Task-1 verifier on the real 4-bit artifacts)

> This is the **go/no-go gate** for Stage 3: confirm the 4-bit M⁻ and M⁺ share one tokenizer and match the steered base's vocab (Qwen3 = 151936). If this fails, the loader's `check_vocab_identity` would reject the pair at serve time.

- [ ] **Step 1: Run the verifier on the 4-bit pair**

Run:
```bash
uv run python -m olmlx.proxy_tuning_pipeline.verify \
  ~/.olmlx/proxy_tuning/qwen3-1.7b-base-4bit \
  ~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-4bit \
  --base-vocab 151936
```
Expected: `OK: M-/M+ share one tokenizer and match the base vocabulary.`

- [ ] **Step 2: If it raises** a vocab mismatch, the quantize/fuse perturbed the tokenizer (rare) — re-copy `tokenizer*.json` + `tokenizer_config.json` from `~/.olmlx/proxy_tuning/qwen3-1.7b-base-bf16/` into both 4-bit dirs and re-run Step 1.

- [ ] **Step 3: Record the final artifact paths** for Stage 3 serving:
  - M⁺ (expert): `~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-4bit`
  - M⁻ (anti-expert): `~/.olmlx/proxy_tuning/qwen3-1.7b-base-4bit`

- [ ] **Step 4: No commit** — this is a validation gate.

---

## Self-Review

**Spec coverage (against §6 of the design spec):**
- ✅ Source M⁻ — prebuilt MLX base doesn't exist, so `mlx_lm convert` from `Qwen/Qwen3-1.7B-Base` (verified) → bf16 (Task 2). True **base** (non-instruct) checkpoint.
- ✅ LoRA fine-tune (`mlx_lm.lora`, `--mask-prompt`, chat data) → adapters (Task 5), with a smoke-train first (Task 4).
- ✅ Fuse → full bf16 M⁺ (Task 6).
- ✅ 4-bit quantize M⁺ **and** M⁻ for serving (Task 7).
- ✅ Assert tokenizer identity (`get_vocab()` equality via the engine's `check_vocab_identity`) **and** vocab size == base (151936) — the verifier (Task 1) run on the real artifacts (Task 8).
- ✅ Validate by artifact checks (loads, generates coherently) at every stage; smoke-train before the full run.

**Placeholder scan:** No TBD/TODO. Every command is concrete and runnable; the one conditional (Task 2 Step 3 / Task 8 Step 2) gives the exact remedy command, not a vague instruction. Hyperparameters are concrete starting values with stated rationale (iters≈2 epochs, lr 1e-4 LoRA-standard, q-group 64 matches mlx-community).

**Type/name consistency:** `assert_serveable_pair(anti_expert_dir, expert_dir, base_vocab_size, *, loader=...)` is defined in Task 1 and invoked via the CLI in Task 8 with the same argument order (M⁻ then M⁺). Artifact paths (`qwen3-1.7b-base-bf16` → `adapters` → `qwen3-1.7b-olmlx-expert-bf16` → `*-4bit`) are consistent across Tasks 2→5→6→7→8.

**Execution-mode note:** Task 1 is TDD code (agent-runnable). Tasks 2–8 are an **operator runbook** — real `mlx-lm` on the user's Apple Silicon with their models and ~30–90 min of training; the operator runs them and pastes results back, as in Stage 1.
