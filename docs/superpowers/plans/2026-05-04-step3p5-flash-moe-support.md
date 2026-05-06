# Step-3.5-Flash Flash-MoE Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `olmlx flash prepare mlx-community/Step-3.5-Flash-6bit` succeed and `olmlx serve` produce correct output for the model under Flash-MoE.

**Architecture:** Three small additive changes — recognize `moe_num_experts` as an expert-count alias in MoE detection/prep, parse `moe_layers_enum` for explicit MoE layer indexing in the bundler, and probe `share_expert` (singular) as a shared-expert fallback in the runtime DeepSeek-style replacement. Tests are unit-level against synthetic configs/safetensors; end-to-end is verified manually.

**Tech Stack:** Python, mlx-lm, safetensors, pytest.

**Spec:** `docs/superpowers/specs/2026-05-04-step3p5-flash-moe-support-design.md`

---

## File Map

- Modify: `olmlx/engine/flash/moe_prepare.py` — `is_moe_model()` and expert-count read in `prepare_moe_for_flash()`.
- Modify: `olmlx/engine/flash/moe_bundler.py` — `_detect_moe_layers()` adds `moe_layers_enum` branch.
- Modify: `olmlx/engine/flash/flash_moe_model.py` — `_FlashMoEDeepSeek.__init__` falls back to `share_expert`.
- Modify: `tests/test_flash_moe_integration.py` — add `is_moe_model` test for `moe_num_experts`.
- Modify: `tests/test_flash_moe_bundler.py` — add bundler test for `moe_layers_enum`.
- Modify: `tests/test_flash_moe_model.py` — add runtime test for `share_expert` (singular) fallback.

---

## Task 1: Recognize `moe_num_experts` in `is_moe_model()`

**Files:**
- Modify: `olmlx/engine/flash/moe_prepare.py:28-32`
- Test: `tests/test_flash_moe_integration.py` (new test in `TestIsMoeModel`)

- [ ] **Step 1.1: Write the failing test**

Add to `tests/test_flash_moe_integration.py` inside `class TestIsMoeModel`:

```python
    def test_step3p5_moe_detected(self, tmp_path):
        """Step-3.5 uses 'moe_num_experts' instead of the other aliases."""
        from olmlx.engine.flash.moe_prepare import is_moe_model

        config = {
            "model_type": "step3p5",
            "moe_num_experts": 288,
            "hidden_size": 4096,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_moe_model(tmp_path) is True
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `uv run pytest tests/test_flash_moe_integration.py::TestIsMoeModel::test_step3p5_moe_detected -v`
Expected: FAIL — `assert False is True` (the existing OR-chain doesn't see `moe_num_experts`).

- [ ] **Step 1.3: Implement**

In `olmlx/engine/flash/moe_prepare.py`, change `is_moe_model`:

```python
    return (
        (config.get("n_routed_experts") or 0) > 1
        or (config.get("num_local_experts") or 0) > 1
        or (config.get("num_experts") or 0) > 1
        or (config.get("moe_num_experts") or 0) > 1
    )
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `uv run pytest tests/test_flash_moe_integration.py::TestIsMoeModel -v`
Expected: PASS (all tests in class, including the new one).

- [ ] **Step 1.5: Commit**

```bash
git add olmlx/engine/flash/moe_prepare.py tests/test_flash_moe_integration.py
git commit -m "feat(flash-moe): recognize moe_num_experts in is_moe_model"
```

---

## Task 2: Read `moe_num_experts` in `prepare_moe_for_flash()`

**Files:**
- Modify: `olmlx/engine/flash/moe_prepare.py:78-82`
- Test: `tests/test_flash_moe_integration.py` (new test in `TestPrepareMoeForFlash`)

- [ ] **Step 2.1: Write the failing test**

Add to `tests/test_flash_moe_integration.py` inside `class TestPrepareMoeForFlash`. Reuse `_make_synthetic_moe_weights` from `test_flash_moe_bundler.py` by importing it:

```python
    def test_step3p5_moe_num_experts_in_config(self, tmp_path):
        """Expert count must be read from moe_num_experts when present."""
        from tests.test_flash_moe_bundler import _make_synthetic_moe_weights

        hidden, inter, experts = 64, 32, 6
        model_dir = _make_synthetic_moe_weights(hidden, inter, experts, 2, 1, tmp_path)

        # Overwrite config to use moe_num_experts (Step-3.5 style)
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "step3p5",
            "hidden_size": hidden,
            "moe_intermediate_size": inter,
            "moe_num_experts": experts,
            "num_hidden_layers": 3,
            "moe_layers_enum": "1,2",
            "num_experts_per_tok": 2,
        }))

        from olmlx.engine.flash.moe_prepare import prepare_moe_for_flash

        output_dir = tmp_path / "flash_moe"
        prepare_moe_for_flash(str(model_dir), output_dir)

        cfg = json.loads((output_dir / "flash_moe_config.json").read_text())
        assert cfg["num_experts"] == experts
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `uv run pytest tests/test_flash_moe_integration.py::TestPrepareMoeForFlash::test_step3p5_moe_num_experts_in_config -v`
Expected: FAIL — either `num_experts` is `None` (the OR-chain returns `None`), or layer detection returns `[]` because `moe_layers_enum` isn't yet handled. Either failure is acceptable; Task 3 fixes the layer detection. Confirm the failure mentions `num_experts` being wrong (`None`/missing) before proceeding.

- [ ] **Step 2.3: Implement**

In `olmlx/engine/flash/moe_prepare.py`:

```python
    num_experts = (
        text_config.get("n_routed_experts")
        or text_config.get("num_local_experts")
        or text_config.get("num_experts")
        or text_config.get("moe_num_experts")
    )
```

- [ ] **Step 2.4: Run test (still expected to fail)**

Run: `uv run pytest tests/test_flash_moe_integration.py::TestPrepareMoeForFlash::test_step3p5_moe_num_experts_in_config -v`
Expected: still FAIL, but now with a different error — bundler raises because `_detect_moe_layers` returned `[]`. This confirms Task 2's change works; Task 3 unblocks the test.

- [ ] **Step 2.5: Commit (test stays failing until Task 3)**

```bash
git add olmlx/engine/flash/moe_prepare.py tests/test_flash_moe_integration.py
git commit -m "feat(flash-moe): read moe_num_experts in prepare_moe_for_flash"
```

---

## Task 3: Parse `moe_layers_enum` in `_detect_moe_layers()`

**Files:**
- Modify: `olmlx/engine/flash/moe_bundler.py:256-291`
- Test: `tests/test_flash_moe_bundler.py` (new test class)

- [ ] **Step 3.1: Write the failing test**

Add to `tests/test_flash_moe_bundler.py`:

```python
class TestDetectMoeLayersEnum:
    """Step-3.5-Flash declares MoE layers via explicit moe_layers_enum string."""

    def test_moe_layers_enum_parsed(self, tmp_path):
        from olmlx.engine.flash.moe_bundler import _detect_moe_layers

        config = {
            "num_hidden_layers": 6,
            "moe_layers_enum": "2,3,5",
            # Conflicting freq/offset settings should be ignored.
            "first_k_dense_replace": 0,
            "moe_layer_freq": 1,
        }
        assert _detect_moe_layers(config) == [2, 3, 5]

    def test_moe_layers_enum_empty_string_falls_through(self, tmp_path):
        """Empty moe_layers_enum falls back to freq-based detection."""
        from olmlx.engine.flash.moe_bundler import _detect_moe_layers

        config = {
            "num_hidden_layers": 4,
            "moe_layers_enum": "",
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
        }
        assert _detect_moe_layers(config) == [1, 2, 3]
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `uv run pytest tests/test_flash_moe_bundler.py::TestDetectMoeLayersEnum -v`
Expected: FAIL — `test_moe_layers_enum_parsed` returns `[0,1,2,3,4,5]` (freq path) instead of `[2,3,5]`.

- [ ] **Step 3.3: Implement**

In `olmlx/engine/flash/moe_bundler.py`, edit `_detect_moe_layers` — insert a new branch above the freq/offset logic and below the `hybrid_override_pattern` branch:

```python
    # Step-3.5: explicit comma-separated MoE layer indices.
    enum = config.get("moe_layers_enum")
    if enum:
        return sorted(int(i) for i in enum.split(","))
```

Place this immediately after the `hybrid_override_pattern` block. The empty-string check is handled by `if enum:` (falsy).

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_flash_moe_bundler.py::TestDetectMoeLayersEnum -v`
Expected: PASS (both tests).

- [ ] **Step 3.5: Run the previously-stuck Task-2 test**

Run: `uv run pytest tests/test_flash_moe_integration.py::TestPrepareMoeForFlash::test_step3p5_moe_num_experts_in_config -v`
Expected: PASS — Task 2's blocker is now resolved.

- [ ] **Step 3.6: Run the full bundler/integration suite for regressions**

Run: `uv run pytest tests/test_flash_moe_bundler.py tests/test_flash_moe_integration.py -v`
Expected: PASS — no regressions in existing freq-based / `hybrid_override_pattern` tests.

- [ ] **Step 3.7: Commit**

```bash
git add olmlx/engine/flash/moe_bundler.py tests/test_flash_moe_bundler.py
git commit -m "feat(flash-moe): support moe_layers_enum for explicit MoE layer indexing"
```

---

## Task 4: Probe `share_expert` (singular) in `_FlashMoEDeepSeek`

**Files:**
- Modify: `olmlx/engine/flash/flash_moe_model.py:53-70`
- Test: `tests/test_flash_moe_model.py` (new test)

- [ ] **Step 4.1: Inspect existing test patterns**

Run: `head -60 tests/test_flash_moe_model.py`

Note the test fixtures and how `_FlashMoEDeepSeek` is constructed in tests. Match that style.

- [ ] **Step 4.2: Write the failing test**

Add to `tests/test_flash_moe_model.py` (place near other `_FlashMoEDeepSeek` tests; if none exist, create a new class):

```python
class TestFlashMoEDeepSeekShareExpertSingular:
    """Step-3.5MoE uses share_expert (singular), not shared_experts (plural)."""

    def test_share_expert_singular_picked_up(self):
        import mlx.core as mx
        import mlx.nn as nn

        from olmlx.engine.flash.flash_moe_model import _FlashMoEDeepSeek

        class FakeGate(nn.Module):
            def __call__(self, x):
                B = x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[1]
                inds = mx.zeros((B, 1), dtype=mx.int32)
                scores = mx.ones((B, 1))
                return inds, scores

        class FakeShareExpert(nn.Module):
            def __call__(self, x):
                return x * 2.0

        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = FakeGate()
                # singular name — Step-3.5 style
                self.share_expert = FakeShareExpert()

        original = FakeMoE()

        class StubFlashMoE:
            def __call__(self, x, inds, scores):
                return mx.zeros_like(x)

        replacement = _FlashMoEDeepSeek(original, StubFlashMoE())

        x = mx.ones((1, 4))
        y = replacement(x)
        # routed output is zeros; share_expert doubles input → expect 2.0
        assert mx.allclose(y, mx.full((1, 4), 2.0))
```

- [ ] **Step 4.3: Run test to verify it fails**

Run: `uv run pytest tests/test_flash_moe_model.py::TestFlashMoEDeepSeekShareExpertSingular -v`
Expected: FAIL — `self.shared_experts` is `None` (it only checks the plural attr), so output is zeros (`0.0` ≠ `2.0`).

- [ ] **Step 4.4: Implement**

In `olmlx/engine/flash/flash_moe_model.py`, change `_FlashMoEDeepSeek.__init__`:

```python
    def __init__(self, original_moe, flash_moe: FlashMoE):
        super().__init__(original_moe, flash_moe)
        self.gate = original_moe.gate
        # Some models (Step-3.5) use the singular `share_expert` attribute.
        self.shared_experts = getattr(
            original_moe, "shared_experts", None
        ) or getattr(original_moe, "share_expert", None)
```

The `_combine` method is unchanged — it already handles `None`.

- [ ] **Step 4.5: Run test to verify it passes**

Run: `uv run pytest tests/test_flash_moe_model.py::TestFlashMoEDeepSeekShareExpertSingular -v`
Expected: PASS.

- [ ] **Step 4.6: Run full flash_moe_model tests for regressions**

Run: `uv run pytest tests/test_flash_moe_model.py -v`
Expected: PASS — DeepSeek-style models (with `shared_experts` plural) still work.

- [ ] **Step 4.7: Commit**

```bash
git add olmlx/engine/flash/flash_moe_model.py tests/test_flash_moe_model.py
git commit -m "feat(flash-moe): probe share_expert (singular) for Step-3.5 compat"
```

---

## Task 5: Lint, full-suite check, manual smoke

- [ ] **Step 5.1: Run ruff**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean (or auto-fix and re-run; see memory `feedback_ruff_before_push.md`).

- [ ] **Step 5.2: Run full Flash-MoE test suite**

Run: `uv run pytest tests/test_flash_moe.py tests/test_flash_moe_bundler.py tests/test_flash_moe_integration.py tests/test_flash_moe_model.py tests/test_flash_moe_weight_store.py -v`
Expected: all PASS.

- [ ] **Step 5.3: Manual prep run**

Run: `uv run olmlx flash prepare mlx-community/Step-3.5-Flash-6bit`
Expected: completes without error; populates `~/.olmlx/models/mlx-community_Step-3.5-Flash-6bit/flash_moe/` with `flash_moe_config.json`, `flash_moe_layout.json`, and 42 `.flashexperts` files (one per MoE layer 3..44). Verify with: `uv run olmlx flash info mlx-community/Step-3.5-Flash-6bit`.

- [ ] **Step 5.4: Manual serve smoke test**

Run: `uv run olmlx` and in another terminal `curl http://localhost:11434/api/chat -d '{"model":"mlx-community/Step-3.5-Flash-6bit:latest","messages":[{"role":"user","content":"Say hello in one word."}],"stream":false}'`
Expected: a coherent one-word response. If output is gibberish, the share_expert fallback may not be wiring correctly — debug before proceeding.

- [ ] **Step 5.5: Update CLAUDE.md if needed**

Per memory `feedback_update_claude_md.md`: this change is small and additive (new MoE config keys recognized), so no `# Key Design Decisions` entry is warranted unless something surprising came up during smoke testing. Skip unless smoke test reveals something noteworthy.

- [ ] **Step 5.6: Final commit (if anything left unstaged)**

```bash
git status
# only commit if there are leftover formatting fixes or CLAUDE.md edits
```

---

## Self-Review Notes

- **Spec coverage:** All four numbered gaps (3 prep + 1 runtime) plus tests are covered by Tasks 1-4. Manual end-to-end is Task 5.
- **Type / name consistency:** `moe_num_experts` (config key) vs `num_experts` (in-code variable name) — names match between Tasks 1, 2, and the existing `prepare_moe_for_flash` body. `share_expert` (singular, Step-3.5 attr) vs `shared_experts` (plural, internal field) — kept distinct on purpose; documented in the Task 4 comment.
- **Regression coverage:** Task 3 step 3.6 and Task 4 step 4.6 explicitly run the existing class to catch breakage.
- **No placeholders / TBDs.**
