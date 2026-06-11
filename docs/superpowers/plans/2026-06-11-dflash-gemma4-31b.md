# DFlash Gemma4 31B Training & Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two bugs in `_build_draft_config` that break Gemma4 DFlash training, add a dedicated `dflash` bench scenario, train a DFlash draft for `mlx-community/gemma-4-31b-it-4bit`, and benchmark it against the greedy baseline.

**Architecture:** Two targeted bug fixes in `engine/dflash/prepare.py` (rope_theta extraction + final_logit_softcapping propagation), one new scenario in `bench/scenarios.py`, then an operational training run followed by a bench run. No new files; all changes extend existing patterns.

**Tech Stack:** MLX, mlx-lm, pytest, `olmlx dflash prepare`, `olmlx bench run`

---

## File Map

| File | Change |
|------|--------|
| `olmlx/engine/dflash/prepare.py` | Extend `_build_draft_config`: handle per-attention-type `rope_parameters`; add `final_logit_softcapping` to `DraftConfig(...)` call |
| `olmlx/bench/scenarios.py` | Add `dflash` scenario after `speculative` in `SCENARIOS` |
| `tests/test_dflash_prepare.py` | Add two tests in `TestDraftConfigDerivation` |
| `tests/test_bench_scenarios.py` | Add assertion that `dflash` scenario is present with correct env keys |

---

### Task 1: Fix `rope_theta` extraction for per-attention-type `rope_parameters`

Gemma4's `text_config.rope_parameters` is `{"full_attention": {"rope_theta": 1000000.0, ...}, "sliding_attention": {"rope_theta": 10000.0, ...}}`. The current cascade tries `rope_params_inner.get("rope_theta")` which returns `None` for this shape, firing a `logger.error` and falling back to 10000.0.

**Files:**
- Modify: `tests/test_dflash_prepare.py` (add test in `TestDraftConfigDerivation`)
- Modify: `olmlx/engine/dflash/prepare.py:273-299` (`_build_draft_config`)

- [ ] **Step 1: Write the failing test**

Add inside `class TestDraftConfigDerivation` in `tests/test_dflash_prepare.py`:

```python
def test_rope_theta_from_per_attention_type_rope_parameters(self):
    """Gemma4 uses rope_parameters keyed by attention type.

    The existing cascade checks rope_params_inner.get("rope_theta"), which
    returns None when the dict is {"full_attention": {...}, "sliding_attention":
    {...}}. The fix must walk the sub-dicts and prefer "full_attention".
    """
    from olmlx.engine.dflash.prepare import _build_draft_config

    target_cfg = {
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "vision_config": {"hidden_size": 1152},
        "text_config": {
            "vocab_size": 262144,
            "hidden_size": 5376,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "intermediate_size": 21504,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 131072,
            "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
            },
        },
    }
    cfg = _build_draft_config(
        target_cfg,
        target_layer_ids=[12, 24, 36, 48, 59],
        num_hidden_layers=5,
        block_size=16,
        mask_token_id=1,
    )
    # full_attention is preferred when present
    assert cfg.rope_theta == 1000000.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_dflash_prepare.py::TestDraftConfigDerivation::test_rope_theta_from_per_attention_type_rope_parameters -v
```

Expected: `FAILED` — `AssertionError: assert 10000.0 == 1000000.0`

- [ ] **Step 3: Implement the fix**

In `olmlx/engine/dflash/prepare.py`, replace the `else:` branch of the rope_theta cascade (lines 285–299) with a new branch that walks per-attention-type sub-dicts before falling through to the error. The full cascade block becomes:

```python
    if text_cfg.get("rope_theta") is not None:
        rope_theta = float(text_cfg["rope_theta"])
    elif (
        isinstance(rope_params_inner, dict)
        and rope_params_inner.get("rope_theta") is not None
    ):
        rope_theta = float(rope_params_inner["rope_theta"])
    elif (
        isinstance(rope_params_outer, dict)
        and rope_params_outer.get("rope_theta") is not None
    ):
        rope_theta = float(rope_params_outer["rope_theta"])
    elif isinstance(rope_params_inner, dict):
        # Per-attention-type rope_parameters (Gemma4): the dict's values are
        # per-layer-kind sub-configs, each carrying their own rope_theta.
        # Prefer "full_attention" (the draft is all-full-attention by default);
        # fall back to any sub-dict that has rope_theta.
        _found: float | None = None
        for _key in ("full_attention", "sliding_attention"):
            _sub = rope_params_inner.get(_key)
            if isinstance(_sub, dict) and _sub.get("rope_theta") is not None:
                _found = float(_sub["rope_theta"])
                break
        if _found is None:
            for _sub in rope_params_inner.values():
                if isinstance(_sub, dict) and _sub.get("rope_theta") is not None:
                    _found = float(_sub["rope_theta"])
                    break
        if _found is not None:
            rope_theta = _found
        else:
            rope_theta = 10000.0
            logger.error(
                "No 'rope_theta' found at the top level or under "
                "'rope_parameters' in the target config — falling back to "
                "10000.0. Long-context targets (Qwen3.5+, Qwen3.6) typically "
                "use ~10_000_000; verify the target's config.json."
            )
    else:
        rope_theta = 10000.0
        logger.error(
            "No 'rope_theta' found at the top level or under "
            "'rope_parameters' in the target config — falling back to "
            "10000.0. Long-context targets (Qwen3.5+, Qwen3.6) typically "
            "use ~10_000_000; verify the target's config.json."
        )
```

- [ ] **Step 4: Run the new test and the full rope_theta suite to confirm no regressions**

```bash
uv run pytest tests/test_dflash_prepare.py::TestDraftConfigDerivation -v
```

Expected: all `TestDraftConfigDerivation` tests pass, including the pre-existing `test_picks_up_rope_theta_from_rope_parameters_block` and `test_rope_parameters_falls_back_to_top_level_for_vlm`.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/dflash/prepare.py tests/test_dflash_prepare.py
git commit -m "fix(dflash): handle per-attention-type rope_parameters for Gemma4"
```

---

### Task 2: Propagate `final_logit_softcapping` from target config to `DraftConfig`

Gemma4 uses `final_logit_softcapping: 30.0`. The draft applies the shared `lm_head` but `_build_draft_config` never reads this field, so `DraftConfig.final_logit_softcapping` stays `None` — a train/infer mismatch that silently degrades acceptance.

**Files:**
- Modify: `tests/test_dflash_prepare.py` (add test in `TestDraftConfigDerivation`)
- Modify: `olmlx/engine/dflash/prepare.py:307-327` (the `return DraftConfig(...)` call)

- [ ] **Step 1: Write the failing test**

Add inside `class TestDraftConfigDerivation` in `tests/test_dflash_prepare.py`:

```python
def test_final_logit_softcapping_propagated_from_target(self):
    """_build_draft_config must propagate final_logit_softcapping (Gemma4 = 30.0).

    The field is in text_config; the fix adds it to the DraftConfig(...) call.
    DraftConfig.final_logit_softcapping defaults to None so without the fix
    the draft skips softcapping at both train and infer time.
    """
    from olmlx.engine.dflash.prepare import _build_draft_config

    target_cfg = {
        "vocab_size": 262144,
        "hidden_size": 5376,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "head_dim": 256,
        "intermediate_size": 21504,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 131072,
        "final_logit_softcapping": 30.0,
    }
    cfg = _build_draft_config(
        target_cfg,
        target_layer_ids=[1, 8, 15, 22, 29],
        num_hidden_layers=5,
        block_size=16,
        mask_token_id=1,
    )
    assert cfg.final_logit_softcapping == 30.0

def test_final_logit_softcapping_absent_stays_none(self):
    """Models without softcapping must produce DraftConfig.final_logit_softcapping=None."""
    from olmlx.engine.dflash.prepare import _build_draft_config

    target_cfg = {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 11008,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
    }
    cfg = _build_draft_config(
        target_cfg,
        target_layer_ids=[5, 11, 17, 23],
        num_hidden_layers=4,
        block_size=4,
        mask_token_id=0,
    )
    assert cfg.final_logit_softcapping is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_dflash_prepare.py::TestDraftConfigDerivation::test_final_logit_softcapping_propagated_from_target -v
```

Expected: `FAILED` — `AssertionError: assert None == 30.0`

- [ ] **Step 3: Implement the fix**

In `olmlx/engine/dflash/prepare.py`, add `final_logit_softcapping` to the `return DraftConfig(...)` call. The full call becomes:

```python
    return DraftConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=int(text_cfg["vocab_size"]),
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        block_size=block_size,
        num_target_layers=len(target_layer_ids),
        target_layer_ids=list(target_layer_ids),
        mask_token_id=mask_token_id,
        rope_scaling=text_cfg.get("rope_scaling") or target_cfg.get("rope_scaling"),
        final_logit_softcapping=text_cfg.get("final_logit_softcapping"),
    )
```

- [ ] **Step 4: Run both new tests and the full prepare suite**

```bash
uv run pytest tests/test_dflash_prepare.py -v
```

Expected: all tests pass. Confirm `test_final_logit_softcapping_propagated_from_target` and `test_final_logit_softcapping_absent_stays_none` are both `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/dflash/prepare.py tests/test_dflash_prepare.py
git commit -m "fix(dflash): propagate final_logit_softcapping from target config"
```

---

### Task 3: Add `dflash` bench scenario

The `speculative` scenario sets `OLMLX_SPECULATIVE=true` but inherits `OLMLX_SPECULATIVE_STRATEGY` from the caller's environment (defaulting to `classic`). A dedicated `dflash` scenario locks the strategy in so bench results are unambiguous.

**Files:**
- Modify: `olmlx/bench/scenarios.py` (add scenario after `speculative`)
- Modify: `tests/test_bench_scenarios.py` (assert `dflash` is present)

- [ ] **Step 1: Write the failing test**

In `tests/test_bench_scenarios.py`, add to class `TestScenariosList`:

```python
def test_dflash_scenario_present_with_correct_env(self):
    names = {s.name: s for s in SCENARIOS}
    assert "dflash" in names, "dflash scenario must be present"
    dflash = names["dflash"]
    assert dflash.env_overrides.get("OLMLX_SPECULATIVE") == "true"
    assert dflash.env_overrides.get("OLMLX_SPECULATIVE_STRATEGY") == "dflash"
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_bench_scenarios.py::TestScenariosList::test_dflash_scenario_present_with_correct_env -v
```

Expected: `FAILED` — `AssertionError: dflash scenario must be present`

- [ ] **Step 3: Add the scenario**

In `olmlx/bench/scenarios.py`, insert after the `speculative` `Scenario(...)` block (after line 234):

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

- [ ] **Step 4: Run the bench scenario test suite**

```bash
uv run pytest tests/test_bench_scenarios.py -v
```

Expected: all tests pass. The `test_has_scenarios` count check uses `>= 16` so the new entry doesn't break it.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add olmlx/bench/scenarios.py tests/test_bench_scenarios.py
git commit -m "feat(bench): add dflash scenario"
```

---

### Task 4: Train the DFlash draft for Gemma4 31B

This is an operational step — no code changes. Run it in a terminal and let it finish before Task 5.

**Files:** None (output: `~/.olmlx/models/mlx-community_gemma-4-31b-it-4bit/dflash/`)

- [ ] **Step 1: Start training**

```bash
uv run olmlx dflash prepare mlx-community/gemma-4-31b-it-4bit \
  --self-generate \
  --selfgen-seqs 1500 \
  --selfgen-max-new 640 \
  --steps 2000 \
  --batch-size 4 \
  --block-size 16
```

Expected output progression:
1. `Training DFlash draft for mlx-community/gemma-4-31b-it-4bit...` header
2. Self-generate phase: `Self-generating 1500 training sequences...` (takes ~40–60 min on M4 Pro)
3. Training phase: `step 1/2000 loss=...` logs every 50 steps (takes ~60–90 min)
4. `Saved DFlash draft to ~/.olmlx/models/mlx-community_gemma-4-31b-it-4bit/dflash/`

- [ ] **Step 2: Verify the output exists**

```bash
ls ~/.olmlx/models/mlx-community_gemma-4-31b-it-4bit/dflash/
```

Expected:
```
config.json
model-00001-of-00001.safetensors
```

- [ ] **Step 3: Spot-check the saved config**

```bash
python3 -c "
import json
cfg = json.load(open('$HOME/.olmlx/models/mlx-community_gemma-4-31b-it-4bit/dflash/config.json'))
print('block_size:', cfg['block_size'])
print('rope_theta:', cfg['rope_theta'])
print('final_logit_softcapping:', cfg.get('final_logit_softcapping'))
print('target_layer_ids:', cfg['dflash_config']['target_layer_ids'])
print('num_hidden_layers:', cfg['num_hidden_layers'])
"
```

Expected (values from `_build_draft_config` given Gemma4's config + Task 1 fix):
```
block_size: 16
rope_theta: 1000000.0
final_logit_softcapping: 30.0
target_layer_ids: [1, 15, 29, 44, 58]  # 5 evenly-spaced across 60 layers
num_hidden_layers: 5
```

---

### Task 5: Configure models.json and run the benchmark

Wire the trained draft into `models.json`, run `baseline` and `dflash` scenarios, compare.

**Files:** `~/.olmlx/models.json` (edited in place)

- [ ] **Step 1: Add the draft to models.json**

Edit `~/.olmlx/models.json`. Find or add the entry for `mlx-community/gemma-4-31b-it-4bit` and add these fields:

```json
"mlx-community/gemma-4-31b-it-4bit": {
    "speculative": true,
    "speculative_strategy": "dflash",
    "speculative_draft_model": "/Users/<you>/.olmlx/models/mlx-community_gemma-4-31b-it-4bit/dflash"
}
```

Replace `/Users/<you>/` with your actual home directory path (e.g. `/Users/daniel/`).

- [ ] **Step 2: Run the baseline scenario**

```bash
uv run olmlx bench run mlx-community/gemma-4-31b-it-4bit --scenarios baseline
```

Expected: bench completes, prints a results table with TTFT and tok/s, saves a timestamped result under `~/.olmlx/bench/`.

- [ ] **Step 3: Run the dflash scenario**

```bash
uv run olmlx bench run mlx-community/gemma-4-31b-it-4bit --scenarios dflash
```

Expected: same output shape; tok/s should be higher than baseline if the draft's acceptance rate is reasonable (≥ 2 tokens/step).

- [ ] **Step 4: Compare the two runs**

```bash
# List runs to get timestamps
uv run olmlx bench list

# Compare (replace timestamps with the actual ones from bench list output)
uv run olmlx bench compare <baseline-timestamp> <dflash-timestamp>
```

Expected: a table showing tok/s delta and speedup ratio per prompt. A speedup of 1.5–4× on code/structured prompts is consistent with known DFlash benchmarks on 4-bit targets. Prose prompts may show smaller gains.

---

## Self-Review Against Spec

| Spec requirement | Task |
|------------------|------|
| Fix `rope_theta` extraction for Gemma4's per-attention-type `rope_parameters` | Task 1 |
| Fix `final_logit_softcapping` propagation to `DraftConfig` | Task 2 |
| Add `dflash` bench scenario (pins `OLMLX_SPECULATIVE_STRATEGY=dflash`) | Task 3 |
| Train with `--self-generate`, default paper hyperparameters | Task 4 |
| Run baseline + dflash bench and compare | Task 5 |

All spec requirements covered. No placeholders remain. Type names (`DraftConfig`, `_build_draft_config`, `SCENARIOS`, `Scenario`, `_requires_speculative_draft`) are consistent across all tasks.
