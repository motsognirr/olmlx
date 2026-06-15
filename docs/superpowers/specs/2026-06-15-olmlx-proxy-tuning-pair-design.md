# olmlx-Domain Proxy-Tuning Pair — Design Spec

**Status:** Draft for review
**Date:** 2026-06-15
**Author:** Daniel Palmqvist (with Claude)
**Depends on:** Proxy-tuning decode mode (PR #525, merged `8204a5e`) — `engine/proxy_tuning.py`, `speculative_strategy="proxy_tuning"`.

---

## 1. Background & Goal

olmlx now ships a **proxy-tuning** decode mode (Liu et al. 2024): a large base model `M` is steered at decode time, without touching its weights, by a small tuned **expert M⁺** and small untuned **anti-expert M⁻**, combining per token as `logits = M + α·(M⁺ − M⁻)`. The mechanism is built and merged; what it lacks is a *real* M⁺/M⁻ pair (the smoke test used a contrived, meaningless pair).

**Goal:** produce a **production** M⁺/M⁻ pair whose tuning delta encodes **olmlx's own domain** — its code, conventions, the rich "why" documentation (CLAUDE.md invariants, code comments, `docs/` specs), and adjacent **MLX / inference-optimization** knowledge — so that steering a dense Qwen3 base nudges its output toward olmlx idioms and conventions.

**What "production" means here:** the pair is trained on a real, curated dataset, validated against a held-out eval that *measures the steering lift* (not just "it runs"), and registered for serving on the user's live olmlx instance.

### Realistic expectations (load-bearing)
Proxy-tuning transfers **behavior, style, and conventions** well; it is **weak at injecting precise new facts** (logit arithmetic from a 1.7B cannot reliably make a 32B recall exact API names). The design therefore optimizes for **convention/idiom transfer** and treats fact transfer as a bonus, and it makes the eval an explicit **ship gate** rather than an assumption.

---

## 2. Goals / Non-Goals

**Goals**
- A curated ~10k-example domain SFT dataset derived from the olmlx repo.
- A trained, fused, quantized **M⁺** plus its untouched **M⁻**, sharing the exact Qwen3 tokenizer required by the loader's `check_vocab_identity` guard.
- An eval harness that sweeps α and A/B-tests base-vs-steered on held-out olmlx-style prompts, gating "ship."
- Registration of the pair via the existing proxy-tuning config (env + `models.json`).

**Non-Goals**
- No new training infrastructure in olmlx — fine-tuning uses **`mlx-lm`'s** mature LoRA tooling (`mlx_lm.lora` / `mlx_lm.fuse` / `mlx_lm.convert`).
- No changes to the proxy-tuning decode mode itself (already merged).
- No hybrid/GDN base support — proxy-tuning is **dense-only** in v1, so the steered base must be a dense Qwen3 (8B/14B/32B), **not** Qwen3-Next / Qwen3-Coder-Next.
- Not a general-purpose instruction-tuned model — M⁺ exists only to produce a useful **delta**, not to be served directly.

---

## 3. Key Decisions (with rationale)

| Decision | Choice | Rationale |
|---|---|---|
| **Domain** | olmlx repo + conventions + adjacent MLX/inference | User's target; strong fit for proxy-tuning's convention-transfer strength. |
| **M± size** | **Qwen3-1.7B-Base** | Best capacity/cost balance across the 8B→32B target range. Decode runs *both* M± every token (cost ≈ `M + 2·M±`); a 4B pair ≈ doubles per-token cost on an 8B base and collapses the worth-it ratio. 1.7B is cheap on 8B (+~43%), near-free on 32B (+~11%). |
| **Steered base M** | Dense Qwen3 8B→32B | Dense-only constraint; worth-it ratio `M ≫ M±` favorable. |
| **Data strategy** | Mechanical-extraction **backbone** + LLM **expansion** (Option C) | Extraction guarantees grounding/coverage; LLM expansion gives the instruction diversity that actually teaches conventions as behavior. |
| **Expansion generator** | **OpenAI GPT-5.4-mini** | User's choice; cheap/fast for grounded synthetic instruction data. Uses the OpenAI SDK + `OPENAI_API_KEY`; independent of the (provider-agnostic) training/serving path. |
| **Dataset size** | **~10k** examples, ~5–10% held out | Sweet spot for a LoRA *style/convention delta* on a 1.7B; diversity matters more than raw count past this. |
| **Training** | LoRA (bf16) → fuse → 4-bit quantize | LoRA is light on a 1.7B; fuse produces full M⁺ weights; quantize for serving parity with the 4-bit base. |
| **Base sourcing** | Prebuilt MLX Qwen3-1.7B base if available, else `mlx_lm.convert` from `Qwen/Qwen3-1.7B-Base` | Must be a true **base** (non-instruct) checkpoint, bf16 for training. |
| **Ship gate** | α-sweep + A/B vs base on held-out olmlx prompts, LLM-judged + manual spot-check | Proxy-tuning's transfer is uncertain for fact-heavy content; measure the lift, don't assume it. |

---

## 4. Architecture Overview

Three stages, built and validated in order. Each stage's output is a concrete artifact consumed by the next.

```
┌─ Stage 1: Data pipeline ──────────────────────────────────────────────┐
│  olmlx repo                                                            │
│   ├─ source (1.4k funcs, comments)   ┐                                 │
│   ├─ CLAUDE.md invariants            │  1a. mechanical extraction      │
│   ├─ docs/ specs (54 files)          ├─────────────► (source, seed)    │
│   ├─ tests (3.8k test fns)           │      units (grounded, $0)       │
│   └─ git log (419 commits)           ┘            │                    │
│                                                   ▼                    │
│                              1b. GPT-5.4-mini expansion (OpenAI SDK)   │
│                                  grounded chunk → diverse instructions │
│                                                   │                    │
│                                                   ▼                    │
│                              1c. curate (dedup, filter, split)         │
│                                                   │                    │
└───────────────────────────────────────────────────┼───────────────────┘
                                                     ▼
                                    train.jsonl  +  valid.jsonl
                                                     │
┌─ Stage 2: Train M⁺ ────────────────────────────────┼───────────────────┐
│  Qwen3-1.7B-Base (bf16)  ── M⁻ (used as-is) ───────┐│                    │
│        │                                           ││                    │
│        └─ mlx_lm.lora (LoRA, bf16) ◄───────────────┘▼                    │
│                 │                                                        │
│                 └─ mlx_lm.fuse ─► M⁺ (full bf16) ─► quantize ─► M⁺ 4-bit │
└────────────────────────────────────────────────────┬───────────────────┘
                                                      ▼
                              M⁺ (4-bit MLX)   +   M⁻ (Qwen3-1.7B-Base, 4-bit MLX)
                                                      │
┌─ Stage 3: Evaluate & serve ─────────────────────────┼──────────────────┐
│  held-out olmlx prompts × α∈{0,0.5,1.0,1.5}                             │
│        │                                                                │
│        ├─ generate via proxy-tuning (olmlx, dense Qwen3 base)           │
│        ├─ LLM-judge convention-adherence + coherence                    │
│        └─ ship-gate decision ──► register M⁺/M⁻ in env + models.json    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Tokenizer invariant (spans all stages):** M⁻, M⁺, and the steered base M must share one exact tokenizer (Qwen3 vocab 151936). M⁺ inherits M⁻'s tokenizer by construction (it *is* M⁻ fine-tuned). The base↔pair identity is enforced at load by `check_vocab_identity`; we additionally assert it at the end of Stage 2.

---

## 5. Stage 1 — Data Pipeline

**Output artifacts:** `data/train.jsonl`, `data/valid.jsonl` in mlx-lm chat format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 5a. Mechanical extraction backbone (`$0`, pure Python, no LLM)
Walk the repo into grounded `(source_context, seed)` units. Sources and the seed shape each yields:

| Source | Extractor | Seed kind |
|---|---|---|
| Functions + docstrings (`ast`) | signature + docstring + body | "explain / implement this" + completion |
| CLAUDE.md "Non-Obvious Invariants" | each invariant block | "explain this invariant / why it holds" |
| `docs/` specs & plans (54 files) | section chunks | Q&A grounded in the doc |
| Tests (3.8k `test_*`) | test fn + target under test | "what behavior does this enforce" / spec→test |
| `git log` (419 commits) | message + diff | "implement this change" (message → diff) |

Requirements: chunk to fit the generator context; **strip secrets / tokens**; record provenance (file:line) per unit for grounding and debugging; dedup near-identical units before expansion.

### 5b. GPT-5.4-mini expansion (OpenAI SDK)
For each unit, prompt GPT-5.4-mini with the **source context as grounding** to produce several diverse instruction→response pairs (vary instruction phrasing, task type — explain / implement / review / convert). Generate ~N pairs per unit to reach ~10k after curation.

- **Provider:** OpenAI SDK, `OPENAI_API_KEY`. (Not the Claude SDK — this step is deliberately a different provider per the user's choice.)
- **Cost control:** batch requests; reuse the grounding chunk across the several generations per unit; cap output length.
- **Grounding discipline:** the generator must answer *from the provided source*, not invent olmlx APIs — system prompt instructs grounding + "say what's not in the source." This bounds hallucination (the main synthetic-data risk).

### 5c. Curate
- **Dedup:** near-duplicate filter (e.g. embedding/MinHash) — diversity matters more than count for a style delta; synthetic data repeats.
- **Quality/length filters:** drop truncated, empty, or degenerate pairs; basic schema validation.
- **Split:** hold out ~5–10% as `valid.jsonl` (enough to measure training loss meaningfully; the *real* eval is Stage 3).
- **Target:** ~10k `train` examples after curation. Log how many raw pairs were generated vs kept (no silent truncation).

### Testing (Stage 1)
- Unit-test each extractor on a small fixture repo subset (deterministic, no LLM): correct seed shape, provenance, secret-stripping.
- Mock the OpenAI client to test the expansion driver's batching, grounding-prompt assembly, and JSONL formatting without network.
- Curation: unit-test dedup, filters, and the split ratio on synthetic inputs.

---

## 6. Stage 2 — Train M⁺

**Output artifacts:** `models/qwen3-1.7b-olmlx-expert` (M⁺, 4-bit MLX) and the resolved path to **M⁻** (`Qwen3-1.7B-Base`, 4-bit MLX).

### Steps
1. **Source M⁻:** locate a prebuilt MLX `Qwen3-1.7B-Base`; if absent, `mlx_lm.convert` from `Qwen/Qwen3-1.7B-Base` to bf16 MLX. Must be the **base** (non-instruct) checkpoint. Keep a bf16 copy for training and a 4-bit copy for serving as M⁻.
2. **LoRA fine-tune:** `mlx_lm.lora --train --model <Qwen3-1.7B-Base-bf16> --data data/ --fine-tune-type lora` with tuned rank / epochs (default ~2–4 epochs; tune against `valid.jsonl` loss; watch for overfitting on the small set).
3. **Fuse:** `mlx_lm.fuse` → full bf16 M⁺ weights.
4. **Quantize:** 4-bit MLX M⁺ for serving parity with the 4-bit base.
5. **Assert tokenizer identity:** confirm M⁺ and M⁻ produce identical `get_vocab()` (they must — M⁺ is M⁻ fine-tuned) and that vocab size matches the intended steered base (151936). This pre-flights the loader's `check_vocab_identity`.

### Testing (Stage 2)
- This stage is mostly orchestration of `mlx-lm` CLIs; validate by **artifact checks**, not unit tests: M⁺ loads, generates coherently standalone, `get_vocab()` == M⁻'s, quantized sizes sane.
- A tiny "smoke train" (few steps on a handful of examples) to validate the pipeline wiring before the full 10k run.

---

## 7. Stage 3 — Evaluate & Serve

**Output:** a ship/no-ship decision + (on ship) registered config.

### Eval harness
- **Held-out prompt set:** ~50–100 olmlx-style prompts — explain-this-invariant, "implement X following olmlx conventions", convention Q&A. Authored fresh (not drawn from training data) to test generalization.
- **α sweep:** generate base (α=0) vs steered at α∈{0.5, 1.0, 1.5} for each prompt, via the real olmlx proxy-tuning path on a dense Qwen3 base (8B for cost; spot-check 32B).
- **Scoring:** LLM judge (GPT-5.4-mini or Claude) on a rubric — **convention-adherence** (uses olmlx idioms, respects documented invariants, matches code style) + **coherence** (no degradation). Plus a manual spot-check by the user.
- **Ship gate:** ship if steered at best α **beats base by a clear margin on convention-adherence without losing coherence**. If no lift: the delta isn't transferring — revisit (more diversity, higher α, larger M±, or accept proxy-tuning's limits for this content) rather than shipping a null result.

### Serve (on ship)
- Register via the existing proxy-tuning config (built in PR #525): global env `OLMLX_SPECULATIVE_PROXY_EXPERT_MODEL` / `_ANTIEXPERT_MODEL` / `_ALPHA` (best α from the sweep) in the server's `.env`, and per-model `speculative: true, speculative_strategy: "proxy_tuning"` on the chosen dense Qwen3 base in `models.json`.
- Mind the live-server constraints already learned: the running server is the `olmlx-1` checkout on :11436; deploy by updating that checkout + config (see project memory).

### Testing (Stage 3)
- Unit-test the eval harness scaffolding (prompt loading, α iteration, result aggregation, judge-call assembly) with a mocked judge + a mocked olmlx endpoint.
- The eval *run* itself is a real-model activity (Metal + the trained pair) — its output is the gate, executed manually.

---

## 8. Inter-Stage Interfaces (artifacts)

| Producer | Artifact | Consumer |
|---|---|---|
| Stage 1 | `data/train.jsonl`, `data/valid.jsonl` (mlx-lm chat format) | Stage 2 `mlx_lm.lora --data` |
| Stage 2 | M⁺ (4-bit MLX dir), M⁻ path (Qwen3-1.7B-Base 4-bit) | Stage 3 + serving config |
| Stage 3 | best α + ship decision | `.env` / `models.json` registration |

Clean artifact boundaries mean each stage is independently runnable and testable, and a re-run of a later stage doesn't force re-running an earlier one.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **Delta doesn't transfer** (fact-heavy content) | Eval is an explicit ship gate; α sweep; design optimizes for convention transfer, treats facts as bonus. |
| **Synthetic-data homogeneity / hallucinated olmlx APIs** | Grounding discipline in 5b; dedup/diversity curation in 5c; provenance tracking. |
| **Delta contamination** (M⁺ drifts to generic GPT-style instruct) | 10k (not 50k) examples; few epochs; validate via the *delta's* effect (Stage 3), not M⁺'s standalone quality. |
| **Tokenizer mismatch** base↔pair | Stage 2 asserts `get_vocab()` identity; loader `check_vocab_identity` is the runtime floor. |
| **No prebuilt MLX 1.7B base** | `mlx_lm.convert` fallback from the HF base checkpoint. |
| **Decode cost on 8B** | 1.7B pair chosen specifically to keep overhead ~+43% on 8B; α tunable. |
| **Cost overrun on generation** | Batch + cache grounding chunks; cap output; 10k cap; log kept-vs-generated. |

---

## 10. Decomposition into Implementation Plans

This spec covers three subsystems with clean artifact boundaries. Each becomes its own implementation plan (spec → plan → build), in order:

1. **Plan 1 — Data pipeline** (Stage 1): extractors + GPT-5.4-mini expansion + curation → `train/valid.jsonl`. TDD-able (mock OpenAI; fixture repo).
2. **Plan 2 — Training** (Stage 2): M⁻ sourcing + LoRA + fuse + quantize + tokenizer assert. Orchestration + artifact checks; smoke-train first.
3. **Plan 3 — Eval & serve** (Stage 3): eval harness + α sweep + judge + ship gate + registration.

Build order is strict — each is blocked on the prior stage's artifact. We start with Plan 1.

---

## 11. Open Questions / Future

- Whether to later train a **second pair at 4B** for a 32B-exclusive deployment (better delta, ~25% overhead on 32B) — deferred; revisit if 32B becomes the dominant target.
- Whether to add per-model `models.json` overrides for the proxy fields (currently global-only in olmlx v1) — independent olmlx feature, out of scope here.
- Whether "grammar-after-combination" (vs the current grammar-disables-proxy-tuning) is worth building for constrained olmlx-convention output — out of scope here.
