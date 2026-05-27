# Extended speed + quality benchmark — 23 configured models

**Date:** 2026-05-24
**Status:** Design — ready for plan
**Predecessor:** `docs/benchmarks/qwen36-gemma-2026-05/README.md`

## Goal

Produce a complete report comparing the 23 models in `~/.olmlx/models.json` on both speed and quality, with deep enough quality grading that ranking signals do not bottom out on saturated mini-suites. Output is a markdown report plus matplotlib charts under `docs/benchmarks/extended-2026-05/`, including a "future research directions" section derived from gaps the run exposes.

## Constraints

- **Wall-clock budget: ~48 h.** Drives every sizing choice. Realistic total is 50–52 h with model-load overhead; the run is started on a Friday evening.
- **Single Apple-Silicon machine.** No farm; serial per-model execution with a single `olmlx serve` process per model load.
- **Heterogeneous fleet.** Throughput spans 7 tok/s (`gpt-oss-120b` under flash-MoE) to ~150 tok/s (small drafts). Naive uniform suite sizing blows the budget on the slow rows.
- **Mini-suites are saturated.** The May 2026 report scored 9/10 models at 46–50/50 on GSM8K-20 + MMLU-20 + HumanEval-10; this run must use larger or harder sets to discriminate.

## Approach

A **tiered task ladder** runs a discriminating Core suite on all 23 models, an Extended suite on a 13-model user-facing subset, and a small Ablation on 2 starred models. One report covers all rows with the asymmetry visible (extended cells show `—` for core-only models).

## Architecture

### Reuses (no new code)

- `olmlx/bench/quality.py` graders (`numeric`, `regex_match`, `code_exec`, `contains`, `regression_snapshot`)
- `olmlx/bench/prompts.py` (`BenchPrompt` dataclass)
- `olmlx/bench/worker.py` (per-model subprocess lifecycle)
- `olmlx/bench/api_bench.py` for the speed suite
- `olmlx bench run` CLI

### New files

| File | Purpose |
|---|---|
| `olmlx/bench/extended_suites.py` | Builds the seven new prompt sets (HumanEval+, MBPP+, GSM8K, MATH-500, MMLU-Pro, GPQA-Diamond, IFEval) and a small RULER-style needle-in-haystack generator. Caches under `~/.olmlx/bench-cache/`. |
| `olmlx/bench/ifeval_grader.py` | Vendored IFEval verifiable-constraint checks, registered into `quality.GRADERS` as `"ifeval"`. |
| `scripts/run_extended_bench.py` | Orchestrator: reads the model list, applies the tier table, drives one olmlx server per model, writes per-row JSON. |
| `scripts/build_extended_report.py` | Pure post-processor: reads result JSON, renders charts (matplotlib) and the README markdown. Idempotent; can re-render without re-benching. |

### Data layout

```
docs/benchmarks/extended-2026-05/
├── README.md
├── charts/
│   ├── frontier.png
│   ├── suite_heatmap.png
│   ├── quant_pairs.png
│   ├── ablation_delta.png
│   └── ruler_position.png
└── raw/
    ├── <safe-model-name>.json    # per-row: tier, per-prompt grades, speed
    └── ablation/
        └── <model>-<knob>.json
```

`~/.olmlx/bench-cache/` holds downloaded source datasets, content-addressed by upstream commit/checksum so repeat runs don't re-download.

## Quality suites

### Core (~180 prompts, runs on all 23 models)

| Suite | Count | Grader | Max tokens | Source |
|---|---:|---|---:|---|
| HumanEval+ subset (stratified) | 50 | `code_exec` | 4096 | evalplus (MIT) |
| GSM8K subset (test split, length-stratified) | 70 | `numeric` | 4096 | OpenAI GSM8K (MIT) |
| GPQA-Diamond | 60 | `regex_match` `Answer: X` | 1024 | Allen AI GPQA (CC-BY) |

### Extended (~250 prompts, runs on 13 models)

| Suite | Count | Grader | Max tokens | Notes |
|---|---:|---|---:|---|
| HumanEval+ full | 164 | `code_exec` | 4096 | augmented test cases |
| MBPP+ subset (sanitized) | 50 | `code_exec` | 4096 | |
| MATH-500 subset | 50 | `numeric` (boxed) | 4096 | competition math |
| MMLU-Pro subset (stratified) | 50 | `regex_match` | 1024 | |
| IFEval subset (verifiable-constraint only) | 50 | `ifeval` (new) | 1024 | rubric subset excluded |
| RULER S-NIAH-1 (4k + 8k) | 20 | `contains` | 1024 | tests prompt-cache + KV-quant + sliding-window |

**Token-cap policy:** 1024 for short-answer (MMLU-Pro, GPQA, IFEval, RULER); **4096** for tasks where verbose `<think>` reasoning can otherwise truncate the answer line (HumanEval+, MBPP+, GSM8K, MATH-500). This is informed by the May 2026 report's observation that Qwen3.6-35B-A3B scored 42/50 at a 1024 cap and 50/50 at 2048 purely from `<think>`-block truncation.

## Model tiering

23 models grouped by what coverage each gets.

### Extended tier (13 models — Core + Extended + speed)

| Model | Class | Notes |
|---|---|---|
| `mlx-community/Qwen3-Coder-Next-4bit` ⭐ | A3B coder MoE, flash-MoE + TQ-4 | TurboQuant ablation anchor |
| `mlx-community/Qwen3.6-35B-A3B-4bit` ⭐ | qwen3_5_moe | speculative ablation anchor |
| `mlx-community/Qwen3.6-35B-A3B-6bit` | qwen3_5_moe | quant comparison vs 4bit |
| `mlx-community/Qwen3.6-27B-4bit` | dense | flagship dense |
| `mlx-community/Nemotron-Cascade-2-30B-A3B-4bit` | nemotron_h hybrid | |
| `mlx-community/gemma-4-31B-it-OptiQ-4bit` | dense, speculative on | |
| `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` | gemma-4 VLM, flash-MoE | |
| `mlx-community/Qwen3-8B-4bit` | dense, speculative on | |
| `lmstudio-community/Devstral-Small-2505-MLX-6bit` | dense code-specialist | |
| `mlx-community/Qwen3-4B-4bit` | small dense | |
| `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` | tiny coder | |
| `prism-ml/Ternary-Bonsai-8B-mlx-2bit` | 2-bit weight quant | quant-frontier datapoint |
| `mlx-community/gemma-4-e2b-it-OptiQ-4bit` | ~2B | tiny baseline |

### Core-only tier (10 models — Core + speed, no Extended)

- **Heavy flash-MoE (4):** `gpt-oss-120b-MXFP4-Q4`, `MiniMax-M2.7-5bit`, `Qwen3-Next-80B-A3B-Instruct-4bit`, `Step-3.5-Flash-6bit`
- **Redundant quant (1):** `unsloth/Qwen3.6-27B-MLX-8bit` (4bit covers the architecture)
- **Superseded base (1):** `mlx-community/Qwen3.5-27B-4bit`
- **Unknown composition (1):** `clowncar/generalist`
- **Pure draft models (3):** `Qwen3-0.6B-4bit`, `Qwen3.5-0.8B-MLX-4bit`, `Qwen2.5-Coder-0.5B-Instruct-4bit` — graded so the report can document what each draft can do solo

### Runtime triage rule (automatic, not configured)

After loading a model, the orchestrator runs a 100-token warmup prompt and measures decode tok/s. If observed throughput × remaining-budget cannot complete Core, the row drops GPQA-Diamond (the most tokens-per-prompt) and runs HumanEval+ 50 + GSM8K 70 only. Triage decisions are recorded in the per-row JSON so the report can show exactly which suite each row actually ran. Expected to fire only on `gpt-oss-120b` and `MiniMax-M2.7-5bit`.

## Ablation (2 ⭐ models, ~150 prompts each)

Re-run HumanEval+ 80 + GSM8K 70 (the Core code + math) under alternative configurations. Same `seed=42`, `temperature=0`.

1. **TurboQuant-4 cost on `Qwen3-Coder-Next-4bit`:** KV cache = off vs `turboquant:4` vs `spectral:4` (last skipped if no spectral calibration on disk for this model).
2. **Speculative-decoding cost on `Qwen3.6-35B-A3B-4bit`:** speculative off (production) vs speculative on with `Qwen3.5-0.8B-MLX-4bit` as draft. Expected delta is 0 at temp=0; nonzero is a correctness bug.

## Speed measurements

Per-model `olmlx bench run` with the scenario matching each model's `models.json` config: `baseline` for plain dense, `flash-moe+tq4` for flash-MoE rows, `speculative` for the two models with speculative on. Report **decode tok/s p50** and **TTFT p50** at three prompt-length buckets (short factual, medium coding, long reasoning) using the existing 7-prompt suite.

## Determinism & fairness

- All quality runs use `temperature=0`, `seed=42`, `top_p=1.0`, overriding per-model `options` in `models.json` so two models aren't compared at different sampling distributions.
- **Subset selection is deterministic.** Every "subset" or "stratified" suite picks its members by sorting source IDs and selecting evenly through the list (e.g. for HumanEval+ 50 from 164: indices `round(i * 164/50)` for `i` in `0..49`). MMLU-Pro / GPQA stratify across category labels first, then sample evenly within each. The selected ID list is recorded in `raw/<model>.json` so a re-run on a different machine grades the same prompts.
- **Composite score** = unweighted mean of per-suite pass rates (each suite contributes equally regardless of prompt count). Reported as a percentage in the headline table. Per-suite pass rates are reported alongside so weighting is recoverable.
- KV-quant adds length-dependent floating-point noise that varies with sequence length (carried forward from the May 2026 caveat). Not load-bearing at this set size; would be at full-split sizes. Recorded in the report's Caveats section.
- `<think>` blocks left at the model's default; max-tokens generous enough to let the verbose reasoners finish.
- HumanEval+ / MBPP+ run with `--enable-code-exec` (sandboxed subprocess with rlimits — acceptable for a single-user local tool per `CLAUDE.md`).

## Report structure

`docs/benchmarks/extended-2026-05/README.md`, mirroring the May 2026 precedent's style so this reads as a continuation:

1. Methodology
2. Headline table (23 rows × Core + Extended columns + composite + tok/s + TTFT, extended cells `—` where not run)
3. Discrimination plot — which suites separate which tiers; explicitly names any newly saturated suite for the next report to retire
4. Coding deep-dive (HumanEval+ + MBPP+: pass@1, error-type distribution, problems all coders failed)
5. Math deep-dive (GSM8K + MATH-500, separating arithmetic vs multi-step errors)
6. Knowledge (GPQA + MMLU-Pro per subject, plus extraction-failure rate)
7. Steerability + long-context (IFEval by constraint type; RULER pass@k by needle position)
8. Quant / speculative ablation — ⭐ results, plotted as quality delta vs production
9. Quant pair comparisons (35B-A3B 4↔6bit; 27B 4↔8bit cross-referenced to prior report)
10. Findings (4–6 bullets, prior report's style)
11. Future research directions — derived after the fact from data, not pre-written
12. Caveats
13. Reproducing — exact commands

### Charts (matplotlib, saved as PNG)

- `frontier.png` — scatter: tok/s vs composite quality, color = class, size = params
- `suite_heatmap.png` — 23 × 9 cell shading by pass rate; saturated columns visually obvious
- `quant_pairs.png` — grouped bars for matched quant comparisons
- `ablation_delta.png` — per-suite quality delta from the ablation toggles
- `ruler_position.png` — long-context pass rate vs needle-position bin per context length

## Budget estimate

Core (~180 prompts × ~500 tok avg = 90k tok/model):

| Speed bucket | Models | Time/model | Subtotal |
|---|---:|---:|---:|
| Drafts (~150 tok/s) | 3 | 0.2 h | 0.5 h |
| Fast (80–100 tok/s) | 7 | 0.3 h | 2.0 h |
| Medium (15–30 tok/s) | 5 | 1.1 h | 5.7 h |
| Medium-slow (10–15 tok/s) | 3 | 1.7 h | 6.3 h |
| Slow flash-MoE (~10 tok/s) | 3 | 2.5 h | 7.5 h |
| Very-slow flash-MoE (~7–12 tok/s) | 2 | 4.3 h | 8.6 h |
| **Core subtotal** | **23** | | **~30.6 h** |

Extended (~250 prompts × ~500 tok avg = 125k tok/model) on 13 models, summed: **~16.7 h**.

Ablation (~150 prompts × 3 configs on 2 models, both fast/medium): **~3 h**.

Model-load + warmup overhead: **~4–6 h** (cold loads at 30 s – 15 min × 23 models, plus speed-suite reruns).

**Grand total: ~54–56 h.** Realistic Friday-evening start, finishing late Monday morning. Slightly over weekend; acceptable given the runtime-triage rule for the slowest rows.

## What's out of scope

- Vision-language quality (no VLM benchmark suite wired up — gemma-4 VLM is graded on its text behavior only)
- Multi-turn / agentic evaluation (no SWE-bench, no τ-bench)
- Full-split MMLU (14k prompts), full-split GSM8K (1319 prompts) — left for a longer follow-up run informed by what this report saturates
- Quality-vs-feature-flag matrix across all 23 models (replaced by the focused 2-model ablation; see Approach decision)
- Distributed-inference scenarios (single machine only)

## Migration / cleanup

None — purely additive. Existing `olmlx/bench/task_prompts.py` and `goldens.py` continue to work for fast smoke tests; the new extended suites live alongside.

## Reproducing (post-run, for the report's "Reproducing" section)

```bash
# Run the full extended benchmark (writes raw/ and overwrites README.md)
python scripts/run_extended_bench.py \
    --models-config ~/.olmlx/models.json \
    --output docs/benchmarks/extended-2026-05/ \
    --enable-code-exec

# Re-render report + charts only (no re-benching)
python scripts/build_extended_report.py docs/benchmarks/extended-2026-05/
```
