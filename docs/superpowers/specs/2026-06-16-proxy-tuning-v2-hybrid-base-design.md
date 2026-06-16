# Proxy-Tuning v2 — Hybrid Base Support — Design Spec

**Status:** Draft for review
**Date:** 2026-06-16
**Author:** Daniel Palmqvist (with Claude)
**Depends on:** Proxy-tuning v1 (`engine/proxy_tuning.py`, `speculative_strategy="proxy_tuning"`) and the Stage-1/2/3 proxy-tuning pair + eval harness (merged PRs #525, #528, #530).

---

## 1. Background & Goal

Proxy-tuning v1 is documented as **dense-only**: it steers a dense Qwen3 base `M` with a small expert/anti-expert pair via `logits = M + α·(M⁺ − M⁻)`. The Stage-3 eval established the v1 pair gives only a modest, sub-threshold lift on dense bases (8B +0.29, 32B +0.11 convention; NO-SHIP). The natural next target is olmlx's *own* domain model family — the **Qwen3-Next / Qwen3.5** generation — which is **hybrid GatedDeltaNet (linear attention) + (sometimes) MoE**, and explicitly listed as a v1 non-goal.

**Goal:** enable proxy-tuning to steer hybrid GDN(+MoE) bases, verified empirically and by α-sweep eval.

### Load-bearing insight (verified during planning)
The "dense-only" label was **conservative, not enforced**. Investigation found:
- Proxy-tuning has **no draft→verify→accept and no cache rollback** — every model advances exactly one token per `step()` over the same committed sequence. The GDN-capture machinery (`engine/gdn_rollback.py`, `spec_decoder_base._install_gdn_capture`) that makes EAGLE/DFlash/MTP "dense-only" exists **solely to undo rejected draft tokens**. Proxy-tuning never rejects, so it needs none of it.
- There is **no code guard**: `_load_proxy_tuning_decoder` (`engine/speculative_loaders.py:780`) checks only vocabulary identity, nothing architectural.
- Per-token forward + `make_prompt_cache` maintains GDN recurrent state correctly (chunking-invariant; the cache materializes recurrent state between calls). MoE routing is internal to the forward; the decoder takes only final-position logits, so MoE is transparent.
- Proxy-tuning already runs everything on `default_stream`, so the GDN Metal-stream hazard (a cross-stream lazy graph) does not apply.

So the hypothesis is: **the decoder already works on hybrid bases**, and v2 is *verification-first* — prove it, fix only what surfaces, lift the constraint — not "build GDN rollback."

### Hard environment constraint (verified)
This machine has **64 GB RAM** (MLX recommended working set **55.7 GB**). Model footprints:
- `Qwen3-Coder-Next-4bit` (qwen3_next, 512-expert MoE): **82 GB** → does **not** fit; needs olmlx Flash-MoE (SSD-streamed experts).
- `Qwen3.5-27B-4bit` (qwen3_5, hybrid, non-MoE): **~16 GB** → fits.
- `Qwen3.5-0.8B-4bit` (qwen3_5, hybrid): tiny → fits; ideal cheap GDN-path test vehicle.

This footprint reality drives the decomposition below.

---

## 2. Decomposition (three sequential sub-projects)

Build order is forced by dependencies and gets cheaper-risk results first.

| Sub-project | What | Depends on | Eval feasible here? |
|---|---|---|---|
| **v2a — Hybrid decode-path support** | Prove `ProxyTuningDecoder` runs correctly on a GDN(+MoE) base; lift the dense-only constraint; regression test. Verified on the tiny `Qwen3.5-0.8B`. | — | correctness only (no steering pair at this vocab) |
| **v2b — Qwen3.5 pair + eval** | Train a *new* M⁺/M⁻ at vocab 248320 from `Qwen3.5-4B-Base` on the **existing Stage-1 dataset** (reusable chat text); fuse, 4-bit, verify; α-sweep eval on `Qwen3.5-27B-4bit`. | v2a | **yes** (27B ≈ 16 GB) |
| **v2c — Coder-Next via Flash-MoE** | Bundle `Qwen3-Coder-Next` for Flash-MoE; integrate proxy-tuning × Flash-MoE base; functional smoke (the existing 151936 pair is tokenizer-compatible). | v2a | smoke only (SSD streaming too slow for a full sweep) |

Each sub-project gets its own implementation plan. **This spec details v2a; v2b and v2c are scoped here and will be expanded into their own specs when their turn comes.**

---

## 3. v2a — Hybrid Decode-Path Support (detailed)

**Output:** a verified, documented capability — proxy-tuning steers a hybrid GDN(+MoE) base — plus a regression test. No new training, no eval (no 248320 pair exists yet).

### 3.1 Verification strategy (the core deliverable)
Because there is no steering pair at the Qwen3.5 vocab yet, v2a verifies **decode-path correctness**, not steering quality:

- **Correctness smoke (Metal):** load the tiny hybrid `Qwen3.5-0.8B-4bit` and pass it as **all three** of base/expert/anti-expert to `ProxyTuningDecoder` (one model object → three independent caches via `make_prompt_cache`). With an identical trio, `M⁺ − M⁻ = 0`, so for any α the combined logits equal the base's — therefore the decoder's output **must match a plain `mlx_lm` greedy generation** of the same model on the same prompt. Any GDN-state corruption in the chunked prefill or per-token step would diverge (degeneration / repetition / pretraining-dump signature).
- Run on both a **short** prompt (single-chunk prefill) and a **long** prompt (multi-chunk prefill) — the long prompt is what exercises GDN recurrent-state threading across chunks.

### 3.2 Single contingency fix (conditional)
The one plausible real defect is **GDN chunked-prefill corruption**, analogous to the pure-RotatingKVCache "must prefill in one `model()` call" invariant. If §3.1's long-prompt smoke diverges from the reference while the short-prompt smoke matches, add:
- A hybrid-base detector (sniff cache types from `make_prompt_cache`, modeled on the existing `_is_pure_rotating_cache` in `engine/speculative.py:227` / `engine/inference.py:1838`), and
- a branch in `ProxyTuningDecoder._prefill_impl` that forces **single-chunk prefill** for hybrid bases (skip `_chunked_prefill`, do one `model(prompt)` call) — accepting the memory cost on long prompts as the trade for correctness.

This is scoped as a contingency; if the smoke passes as-is, this code is **not** written (YAGNI).

### 3.3 Lift the constraint + document
- Remove "v1 targets dense families only / dense-only" wording from `engine/proxy_tuning.py` (docstrings at lines ~20-24, ~114-126) and replace with what's actually supported: **dense and hybrid GDN/MoE bases, given a vocabulary-identical pair**; note the no-rollback reason GDN capture is unneeded.
- Update the CLAUDE.md proxy-tuning invariant paragraph accordingly (it currently says "v1 is dense-only (no GDN capture installed)").

### 3.4 Regression test
A **Metal-gated integration test** (guarded like other engine Metal tests, skipped in CI) that runs the §3.1 identical-trio correctness check on `Qwen3.5-0.8B-4bit` and asserts the proxy output equals the reference greedy generation (or, if exact-match is brittle, asserts non-degeneracy + high token-overlap). Unit-level fakes cannot catch GDN state issues, so this must use a real hybrid model.

### 3.5 Files (v2a)
| File | Change |
|---|---|
| `olmlx/engine/proxy_tuning.py` | docstring: drop dense-only; (contingency) `_is_hybrid` detect + single-chunk prefill branch in `_prefill_impl` |
| `CLAUDE.md` | update proxy-tuning invariant (dense+hybrid supported) |
| `tests/test_proxy_tuning_hybrid.py` | new Metal-gated integration test (identical-trio correctness on Qwen3.5-0.8B) |

### 3.6 v2a testing
- Metal-gated integration test (§3.4) — the real gate; run locally.
- Existing `tests/ -k proxy_tuning` must stay green (no regressions to v1 dense behavior).

---

## 4. v2b — Qwen3.5 Pair + Eval (scoped; own spec later)

- **Data:** reuse the existing Stage-1 dataset (`data/proxy_tuning/{train,valid}.jsonl`) — it's tokenizer-agnostic chat text, no regeneration needed.
- **Train:** `mlx_lm convert` `Qwen/Qwen3.5-4B-Base` → bf16; `mlx_lm lora` on it; fuse; 4-bit quantize. Anti-expert = the untuned 4B base. Verify the pair at **vocab 248320** with the Stage-2 `assert_serveable_pair` (`--base-vocab 248320`).
- **Eval:** Stage-3 harness (`--base mlx-community_Qwen3.5-27B-4bit`, no-think, α-sweep) — the real test of whether steering a hybrid base transfers the olmlx delta. **Requires v2a.**
- **Risks to design around:** does `mlx_lm.lora` train the `qwen3_5` (GDN) architecture cleanly; 4B pair ≈ doubles per-token decode overhead vs the 1.7B; bf16 4B base footprint for training (~8 GB) fits.

## 5. v2c — Coder-Next via Flash-MoE (scoped; own spec later)

- **Bundle:** `moe_prepare`/`moe_bundler` over `Qwen3-Coder-Next` → SSD Flash-MoE bundle (one-time; mind the external-drive caution from the Step-3.5 incident).
- **Integrate:** load Coder-Next as a Flash-MoE base under `speculative_strategy=proxy_tuning`; the flash wrapper exposes `__call__(inputs, cache=cache)` — the interface the decoder already calls — and `pld`/`self_speculative` already compose with Flash-MoE (precedent). Verify the prefetcher lifecycle + GDN layers behave under the proxy decode loop. **Requires v2a.**
- **Verify:** functional smoke only — the existing 151936 pair is tokenizer-compatible; coherent, steered, non-degenerate output. **No full α-sweep eval** (SSD expert streaming × 3-model proxy forward × 768 tok × N prompts is impractically slow).
- **Risks:** flash-MoE bundling of qwen3_next's specific MoE (512 experts + shared expert) — bundler detects `num_experts` and handles `shared_experts`, but untested on this arch; proxy × flash-MoE composition untested; perf.

---

## 6. Risks & Mitigations (v2)

| Risk | Mitigation |
|---|---|
| Decoder corrupts GDN state on hybrid base | §3.1 correctness smoke catches it; §3.2 single-chunk-prefill contingency is the known fix |
| "It already works" is wrong in a way the small-model smoke misses (scale/MoE) | v2b runs a real eval on 27B; v2c smokes the actual MoE Coder-Next — both exercise larger/MoE paths |
| Coder-Next 82 GB unloadable | v2c uses Flash-MoE SSD streaming; smoke-only on quality |
| `mlx_lm.lora` can't train qwen3_5 | v2b spec verifies with a smoke-train first (as Stage 2 did) |
| Regression test not CI-portable (needs real model) | Metal-gated + skipped in CI, like existing engine tests; run locally |

## 7. Non-Goals (v2)
- No GDN draft-rollback machinery (proxy-tuning never rejects — explicitly unneeded).
- No change to the proxy-tuning *algorithm* (`combine_proxy_logits` unchanged).
- v2a does **not** train any model or run a steering eval (no 248320 pair exists yet — that's v2b).
- No full α-sweep eval on Coder-Next (v2c is smoke-only by design).
