# Benchmark: Qwen3.6-27B (8bit vs 4bit) vs Gemma 4 family — May 2026

Speed + quality comparison across nine locally-served models, run through the
olmlx server (`/api/chat`) on Apple Silicon. The original question was an
8bit-vs-4bit quant comparison of `Qwen3.6-27B`; it grew into a small cross-model
survey. Two later additions extend the survey to the extremes of the size range:
`gpt-oss-120b` (120B MoE, run via flash-MoE since it is ~2× the machine's RAM)
and `Nemotron-Cascade-2-30B-A3B` (a `nemotron_h` hybrid Mamba/attention MoE).

## Methodology

- **Serving:** each model loaded via the normal olmlx stack (`olmlx serve` →
  `ModelManager`), `kv_cache_quant=turboquant:4`, **plain decode** (no
  speculative, no flash unless required to load — see caveats).
- **Determinism:** `temperature=0`, `seed=42` for every request.
- **Speed:** `olmlx bench run --scenarios baseline` (7-prompt throughput suite),
  reported as average decode tok/s.
- **Quality:** the bundled mini task-sets in `olmlx/bench/task_prompts.py`
  graded with `olmlx/bench/quality.py`:
  - **GSM8K** (20 problems) — `numeric` grader on the `#### N` final line.
  - **MMLU** (20 four-choice) — `regex_match` on the trailing `Answer: X`.
  - **HumanEval** (10 problems) — sandboxed `code_exec` (opt-in enabled).
- **Token caps:** GSM8K/HumanEval @ **1024**, MMLU @ **1024**. The original
  defaults (MMLU 128, GSM8K/HumanEval 512) truncated these verbose reasoning
  models before they emitted the graded answer — see "Lessons" below.

Driver script: [`scripts/grade_quant_compare.py`](../../../scripts/grade_quant_compare.py).
Raw per-prompt grader output is under [`raw/`](./raw).

## Results

| Model | Class | tok/s | GSM8K | MMLU | HumanEval | Total |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B 8bit | dense 27B | 7.1 | 20/20 | 20/20 | 6/10 | **46/50** |
| Qwen3.6-27B 4bit | dense 27B | 15.3 | 17/20 | 19/20 | 8/10 | **44/50** |
| gemma-4-31B | dense 31B | 12.0 | 17/20 | 20/20 | 10/10 | **47/50** |
| gemma-4-26B-A4B | MoE (flash-MoE) | 21.1¹ | 18/20 | 20/20 | 9/10 | **47/50** |
| gemma-4-e2b | ~2B | 100.1 | 19/20 | 20/20 | 8/10 | **47/50** |
| Devstral-Small 2505 6bit | dense (code) | 12.7 | 20/20 | 20/20 | 10/10 | **50/50** |
| Ternary-Bonsai-8B 2bit | 2-bit 8B | 82.1 | 20/20 | 20/20 | 10/10 | **50/50** |
| Nemotron-Cascade-2-30B-A3B 4bit | hybrid MoE (nemotron_h) | 95.1 | 20/20 | 20/20 | 10/10 | **50/50** |
| gpt-oss-120b MXFP4-Q4 | 120B MoE (flash-MoE) | 7.1¹ | 20/20 | 20/20 | 10/10 | **50/50** |

¹ gemma-4-26B-A4B and gpt-oss-120b speeds are **flash-MoE (SSD expert offload)**,
not directly comparable to the dense plain-decode rows. gpt-oss-120b (115 GB on a
64 GB machine) is SSD-I/O-bound on nearly every expert load.

### Qwen3.6-27B: 8bit → 4bit quant

| | 8bit | 4bit |
|---|---:|---:|
| Decode throughput | ~7.1 tok/s | ~15.3 tok/s (**2.1×**) |
| On-disk size | 32 GB | 15 GB |
| Quality (graded) | 46/50 | 44/50 |

≈2× faster and half the footprint for a 2-point quality delta (−3 GSM8K,
−1 MMLU, +2 HumanEval) that is within noise at n=10–20 per set.

## Findings

- **The mini-suites are saturated.** Eight of nine models score 46–50/50, and
  the four perfect scores span a 2-bit 8B, a 6bit code model, a hybrid MoE, and
  a 120B MoE. A ~2B model (e2b) ties the 27B-8bit. These 10–20-problem sets
  discriminate "broken vs working," not fine quality — any ranking within this
  band is noise-limited. Real discrimination needs the full splits (GSM8K 1319,
  MMLU 14042, HumanEval 164).
- **Speed/size are the clean differentiators.** Nemotron-Cascade A3B (95 tok/s)
  and e2b (100 tok/s) dominate; Qwen-8bit and gpt-oss-120b are slowest at 7
  (the latter SSD-bound under flash-MoE); the 26B MoE's 21 tok/s shows flash-MoE
  working well.
- **Quant cost on Qwen3.6-27B is small** (8→4bit: 46→44, ≈2× speedup).
- **A3B activation is the sweet spot here.** Nemotron-Cascade-2-30B-A3B keeps
  ~3B params hot while the 30B capacity fits in RAM at 4-bit, landing near-e2b
  throughput (95 tok/s) at a perfect 50/50 — the best speed-at-50/50 in the set.
- **flash-MoE scales to 120B.** gpt-oss-120b (115 GB) runs to a perfect 50/50 on
  a 64 GB machine via SSD expert offload, at the cost of 7 tok/s — proof the
  flash-MoE path holds quality on a model nearly 2× the machine's RAM.

## Lessons (methodology)

- **Token-cap truncation was the dominant artifact** at every step. A reasoning
  model that opens with a long "thinking process" gets cut off before the
  graded answer line, and the grader then extracts a stray intermediate number
  (GSM8K) or no answer at all (MMLU). Symptoms looked like catastrophic quality
  drops (MMLU 3/20) but were pure truncation — at 1024 tokens the same model
  scored 20/20. Any future grading harness should default to a generous cap.
- **Verbosity differs by model**, so a fixed cap silently penalizes the more
  verbose ones — a fairness problem, not just a measurement one.

## Caveats

- **Not the same publisher/recipe everywhere** (e.g. unsloth 8bit vs
  mlx-community 4bit for Qwen3.6-27B). Same base weights, slightly different
  quant recipes.
- **gemma-4-26B-A4B is a vision MoE** and would not load in plain decode
  (`Missing 211 parameters: vision_tower...`); it was run via its flash-MoE
  path. Quality is unaffected by flash-MoE; throughput is not comparable to the
  dense rows.
- **gpt-oss-120b (115 GB) only loads via flash-MoE** on this 64 GB machine —
  router/attention/embeddings stay in RAM, the 128 routed experts live on SSD.
  Quality is unaffected; the 7.1 tok/s is SSD-I/O-bound and not comparable to
  the dense rows. Run with `flash-moe+tq4`.
- **Nemotron-Cascade is a `nemotron_h` hybrid** (Mamba/attention MoE, top-6 of
  experts, ~3B active). It loads in plain decode under mlx-lm 0.31.2 and fits in
  RAM at 4-bit. As a hybrid SSM model its KV cache is not reused across requests
  (same `ArraysCache` exclusion as Qwen3.5/Qwen3-Next, issue #284); within-request
  reuse is unaffected and grading runs one prompt at a time regardless.
- **code_exec** runs model-generated Python in a subprocess with rlimits
  (opt-in). Acceptable for a single-user local tool.

## Reproducing

```bash
# speed (dense / in-RAM models)
olmlx bench run --model <hf-path> --scenarios baseline
# speed (models too large for RAM — gpt-oss-120b)
olmlx bench run --model <hf-path> --scenarios flash-moe+tq4

# quality (per model)
python scripts/grade_quant_compare.py <hf-path> out.json --sets gsm8k,humaneval --max-tokens 1024
python scripts/grade_quant_compare.py <hf-path> out_mmlu.json --sets mmlu --max-tokens 1024
# for gpt-oss-120b, prefix with OLMLX_FLASH_MOE=true OLMLX_KV_CACHE_QUANT=turboquant:4
```
