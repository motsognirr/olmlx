# DFlash Draft Training & Benchmark for Gemma4 31B

**Date:** 2026-06-11  
**Goal:** Train a DFlash block-diffusion speculative draft for `mlx-community/gemma-4-31b-it-4bit` and benchmark it against a greedy baseline.

---

## Background

DFlash training currently has two bugs that affect Gemma4 targets. Both must be fixed before training.

### Bug 1 — `rope_theta` not extracted for Gemma4 (`engine/dflash/prepare.py`)

Gemma4's `text_config.rope_parameters` is keyed by attention type:

```json
{
  "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25},
  "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
}
```

`_build_draft_config` tries `rope_params_inner.get("rope_theta")` which returns `None` (the dict has no flat `rope_theta` key), then falls through to `rope_theta = 10000.0` with a `logger.error`. Fix: after the flat-key attempt, walk the attention-type sub-dicts and extract `rope_theta` from `"full_attention"` (preferred, since the draft defaults to all-full-attention) with `"sliding_attention"` as fallback. The `logger.error` only fires if the walk also finds nothing.

### Bug 2 — `final_logit_softcapping` not propagated (`engine/dflash/prepare.py`)

Gemma4 uses logit softcapping (`final_logit_softcapping: 30.0`). `_build_draft_config` never reads this from the target config, so `DraftConfig.final_logit_softcapping` stays `None`. The draft applies the shared `lm_head` but without softcapping — a train/infer mismatch that silently degrades acceptance. Fix: add `final_logit_softcapping=text_cfg.get("final_logit_softcapping")` to the `DraftConfig(...)` call in `_build_draft_config`. No schema changes needed — `_draft_config_to_disk` already serialises the field when non-None.

---

## New Bench Scenario (`bench/scenarios.py`)

Add a dedicated `dflash` scenario immediately after `speculative` in `SCENARIOS`:

```python
Scenario(
    name="dflash",
    description="DFlash block-diffusion speculative decoding",
    env_overrides={
        "OLMLX_SPECULATIVE": "true",
        "OLMLX_SPECULATIVE_STRATEGY": "dflash",
    },
    should_skip=_requires_speculative_draft,
),
```

The existing `speculative` scenario sets `OLMLX_SPECULATIVE=true` but inherits `OLMLX_SPECULATIVE_STRATEGY` from the environment (defaulting to `classic`). The new scenario pins the strategy, making bench results unambiguous and enabling targeted `--scenarios dflash` runs for any future DFlash target.

---

## Training Recipe

After the bug fixes, train with `--self-generate` (the upstream recipe; empirically decisive for acceptance vs ground-truth training):

```bash
uv run olmlx dflash prepare mlx-community/gemma-4-31b-it-4bit \
  --self-generate \
  --selfgen-seqs 1500 \
  --selfgen-max-new 640 \
  --steps 2000 \
  --batch-size 4 \
  --block-size 16
```

Default values kept for `--num-hidden-layers 5`, `--num-target-layers 5`, `--lr 5e-4` (paper values). `--distill` is incompatible with `--self-generate` and is omitted.

**Output:** `~/.olmlx/models/mlx-community_gemma-4-31b-it-4bit/dflash/`  
**Estimated time:** ~1–2 hrs on M4 Pro (1500-seq self-generate phase + 2000 training steps)

---

## Benchmark Workflow

After training, wire the draft via `models.json`:

```json
"mlx-community/gemma-4-31b-it-4bit": {
    "speculative": true,
    "speculative_strategy": "dflash",
    "speculative_draft_model": "<absolute-path-to-dflash-dir>"
}
```

Run both scenarios in sequence:

```bash
uv run olmlx bench run mlx-community/gemma-4-31b-it-4bit --scenarios baseline,dflash
```

Then compare:

```bash
uv run olmlx bench compare <baseline-timestamp> <dflash-timestamp>
```

The bench worker inherits the models.json config, so the draft wires up automatically. Success metrics: tok/s speedup (target ≥ 1.5×), TTFT overhead acceptable (< 20% increase).

---

## Files Changed

| File | Change |
|------|--------|
| `olmlx/engine/dflash/prepare.py` | Fix `rope_theta` extraction for per-attention-type `rope_parameters`; propagate `final_logit_softcapping` to `DraftConfig` |
| `olmlx/bench/scenarios.py` | Add `dflash` scenario after `speculative` |

No new tests needed: the existing `test_build_draft_config_*` suite covers `_build_draft_config`; adding a Gemma4-shaped fixture case pins the fix. The `dflash` scenario is covered by the existing `test_bench_scenarios.py` pattern.
