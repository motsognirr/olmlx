# Promote Flash prefetch + Flash-speculative; tensor-only distributed — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `flash_prefetch` (toggle), `flash_speculative`, `flash_speculative_draft_model`, `flash_speculative_tokens` from `OLMLX_EXPERIMENTAL_*` to the supported `OLMLX_*` surface (#275, #276), and narrow `distributed_strategy` to tensor-only (#273).

**Architecture:** Follow the existing Flash-dense / Flash-MoE promotion pattern exactly: move fields `ExperimentalSettings`→`Settings`; add per-model fields + resolution on `ModelConfig`; register the keys in `registry.PROMOTED_EXPERIMENTAL_KEYS` (and drop from `PER_MODEL_EXPERIMENTAL_KEYS`) so the standard migration error fires; add a `surface_legacy_*_env` shim; switch read sites in `model_manager.py`/`cli.py`; add `olmlx serve` flags; update README/USER_MANUAL/CLAUDE.md; add bench scenario definitions. The 4 prefetch *tuning* knobs (`flash_prefetch_confidence_threshold`/`_min_neurons`/`_max_neurons`/`_io_threads`) stay experimental.

**Tech Stack:** Python, pydantic-settings, pydantic v2, argparse, pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-22-promote-flash-prefetch-speculative-design.md`

**Promoted keys (the 4 that move):** `flash_prefetch`, `flash_speculative`, `flash_speculative_draft_model`, `flash_speculative_tokens`.
**Stay-experimental keys:** `flash_prefetch_confidence_threshold`, `flash_prefetch_min_neurons`, `flash_prefetch_max_neurons`, `flash_prefetch_io_threads`.

---

## Task 1: Move the 4 promoted fields `ExperimentalSettings` → `Settings`

**Files:**
- Modify: `olmlx/config.py` (`Settings` ~line 147, `ExperimentalSettings` ~237-244)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
class TestFlashPrefetchSpeculativePromotion:
    def test_promoted_fields_on_settings_via_env(self, monkeypatch):
        from olmlx.config import Settings

        monkeypatch.setenv("OLMLX_FLASH_PREFETCH", "true")
        monkeypatch.setenv("OLMLX_FLASH_SPECULATIVE", "true")
        monkeypatch.setenv(
            "OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        )
        monkeypatch.setenv("OLMLX_FLASH_SPECULATIVE_TOKENS", "6")
        s = Settings()
        assert s.flash_prefetch is True
        assert s.flash_speculative is True
        assert s.flash_speculative_draft_model == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        assert s.flash_speculative_tokens == 6

    def test_flash_speculative_draft_model_rejects_blank(self):
        import pytest
        from pydantic import ValidationError
        from olmlx.config import Settings

        with pytest.raises(ValidationError):
            Settings(flash_speculative_draft_model="")
        with pytest.raises(ValidationError):
            Settings(flash_speculative_draft_model="   ")

    def test_prefetch_tuning_knobs_stay_experimental(self, monkeypatch):
        from olmlx.config import ExperimentalSettings

        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_PREFETCH_IO_THREADS", "8")
        e = ExperimentalSettings()
        assert e.flash_prefetch_io_threads == 8
        # The promoted toggle is no longer an ExperimentalSettings field.
        assert not hasattr(e, "flash_prefetch") or "flash_prefetch" not in ExperimentalSettings.model_fields
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestFlashPrefetchSpeculativePromotion -v`
Expected: FAIL (`OLMLX_FLASH_PREFETCH` etc. are not yet `Settings` fields).

- [ ] **Step 3: Add fields to `Settings`**

In `olmlx/config.py`, after the Flash-MoE block (the `flash_moe_io_threads` field, ~line 147), add:

```python
    # Flash prefetch — promoted toggle. Advanced prefetch tuning
    # (confidence_threshold, min/max_neurons, io_threads) stays on
    # ``ExperimentalSettings``. Controls both runtime prefetch and whether
    # ``olmlx flash prepare`` trains the LookaheadBank.
    flash_prefetch: bool = False

    # Flash + speculative decoding (SpeculativeFlashDecoder). Per-model
    # overrides live on ``ModelConfig`` in ``olmlx.engine.registry``.
    flash_speculative: bool = False
    flash_speculative_draft_model: Annotated[str, Field(min_length=1)] | None = None
    flash_speculative_tokens: Annotated[int, Field(gt=0)] = 4
```

- [ ] **Step 4: Add the strip-and-reject validator**

In `Settings`, next to `validate_speculative_draft_model` (~line 183), add:

```python
    @field_validator("flash_speculative_draft_model")
    @classmethod
    def validate_flash_speculative_draft_model(cls, v: str | None) -> str | None:
        if v is None:
            return v
        stripped = v.strip()
        if not stripped:
            raise ValueError(
                "flash_speculative_draft_model must be a non-empty HuggingFace path"
            )
        return stripped
```

- [ ] **Step 5: Remove the 4 promoted fields from `ExperimentalSettings`**

In `olmlx/config.py`, delete these three lines from `ExperimentalSettings` (~242-244):

```python
    flash_speculative: bool = False
    flash_speculative_draft_model: str | None = None
    flash_speculative_tokens: Annotated[int, Field(gt=0)] = 4
```

And delete the `flash_prefetch: bool = False` line (~237). **Keep** the four `flash_prefetch_*` tuning lines (~238-241). Update the `ExperimentalSettings` docstring comment to note prefetch toggle + flash_speculative were promoted in this change.

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py::TestFlashPrefetchSpeculativePromotion -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add olmlx/config.py tests/test_config.py
git commit -m "feat(config): promote flash_prefetch + flash_speculative to Settings (#275, #276)"
```

---

## Task 2: Registry — promote keys, add ModelConfig fields + resolution

**Files:**
- Modify: `olmlx/engine/registry.py` (`PER_MODEL_EXPERIMENTAL_KEYS` ~26, `PROMOTED_EXPERIMENTAL_KEYS` ~56, `ResolvedFlashConfig` ~224, `ModelConfig` ~260, `resolved_flash` ~435, `from_entry` ~599/626, `to_entry` ~668)
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_registry.py`:

```python
class TestFlashPrefetchSpeculativePromotionRegistry:
    def test_promoted_keys_in_experimental_raise(self):
        import pytest
        from olmlx.engine.registry import ModelConfig

        for key in ("flash_prefetch", "flash_speculative",
                    "flash_speculative_draft_model", "flash_speculative_tokens"):
            with pytest.raises(ValueError, match="promoted out of 'experimental'"):
                ModelConfig.from_entry(
                    {"hf_path": "Qwen/Qwen3-8B", "experimental": {key: True
                        if key in ("flash_prefetch", "flash_speculative")
                        else ("x" if "draft" in key else 4)}}
                )

    def test_prefetch_tuning_keys_still_allowed_in_experimental(self):
        from olmlx.engine.registry import ModelConfig

        mc = ModelConfig.from_entry(
            {"hf_path": "Qwen/Qwen3-8B",
             "experimental": {"flash_prefetch_io_threads": 8}}
        )
        assert mc.experimental == {"flash_prefetch_io_threads": 8}

    def test_promoted_fields_top_level_resolve(self):
        from olmlx.engine.registry import ModelConfig

        mc = ModelConfig.from_entry({
            "hf_path": "Qwen/Qwen3-8B",
            "flash_prefetch": True,
            "flash_speculative": True,
            "flash_speculative_draft_model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "flash_speculative_tokens": 5,
        })
        rf = mc.resolved_flash()
        assert rf.prefetch is True
        assert rf.flash_speculative is True
        assert rf.flash_speculative_draft_model == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        assert rf.flash_speculative_tokens == 5

    def test_to_entry_round_trips_promoted_fields(self):
        from olmlx.engine.registry import ModelConfig

        entry = {
            "hf_path": "Qwen/Qwen3-8B",
            "flash_speculative": True,
            "flash_speculative_draft_model": "d/m",
            "flash_speculative_tokens": 3,
            "flash_prefetch": True,
        }
        assert ModelConfig.from_entry(entry).to_entry() == entry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py::TestFlashPrefetchSpeculativePromotionRegistry -v`
Expected: FAIL (keys still in `PER_MODEL_EXPERIMENTAL_KEYS`; `ResolvedFlashConfig` lacks new fields).

- [ ] **Step 3: Update the key sets**

In `olmlx/engine/registry.py`, remove these 4 lines from `PER_MODEL_EXPERIMENTAL_KEYS` (~39, 45-47):

```python
        "flash_prefetch",
        "flash_speculative",
        "flash_speculative_draft_model",
        "flash_speculative_tokens",
```

(Keep the four `flash_prefetch_confidence_threshold`/`_min_neurons`/`_max_neurons`/`_io_threads` entries.)

Add to `PROMOTED_EXPERIMENTAL_KEYS` (~70, after the flash_moe entries):

```python
    # Flash prefetch toggle + Flash-speculative promoted to top-level.
    "flash_prefetch": "flash_prefetch",
    "flash_speculative": "flash_speculative",
    "flash_speculative_draft_model": "flash_speculative_draft_model",
    "flash_speculative_tokens": "flash_speculative_tokens",
```

- [ ] **Step 4: Extend `ResolvedFlashConfig`**

In `olmlx/engine/registry.py`, append 4 defaulted fields to the `ResolvedFlashConfig` NamedTuple (~240) and update its docstring to mention prefetch + flash-speculative:

```python
    enabled: bool
    sparsity_threshold: float
    min_active_neurons: int
    max_active_neurons: int | None
    memory_budget_fraction: float | None
    prefetch: bool = False
    flash_speculative: bool = False
    flash_speculative_draft_model: str | None = None
    flash_speculative_tokens: int = 4
```

- [ ] **Step 5: Add ModelConfig fields**

In `ModelConfig` (~294, after `flash_memory_budget_fraction`), add:

```python
    #: Per-model Flash prefetch + Flash-speculative overrides (promoted from
    #: ``experimental``). ``None`` means inherit from global ``Settings``.
    flash_prefetch: bool | None = None
    flash_speculative: bool | None = None
    flash_speculative_draft_model: str | None = None
    flash_speculative_tokens: int | None = None
```

- [ ] **Step 6: Add `__post_init__` validation**

In `ModelConfig.__post_init__`, after the `flash_moe` bool check (~390), add:

```python
        if self.flash_prefetch is not None and not isinstance(self.flash_prefetch, bool):
            raise ValueError(
                f"'flash_prefetch' must be a bool or None, got {self.flash_prefetch!r}"
            )
        if self.flash_speculative is not None and not isinstance(
            self.flash_speculative, bool
        ):
            raise ValueError(
                f"'flash_speculative' must be a bool or None, "
                f"got {self.flash_speculative!r}"
            )
        if (
            self.flash_speculative_draft_model is not None
            and not self.flash_speculative_draft_model.strip()
        ):
            raise ValueError(
                "'flash_speculative_draft_model' must be a non-empty HuggingFace "
                "path or None"
            )
        if self.flash_speculative_tokens is not None and (
            isinstance(self.flash_speculative_tokens, bool)
            or not isinstance(self.flash_speculative_tokens, int)
            or self.flash_speculative_tokens < 1
        ):
            raise ValueError(
                f"'flash_speculative_tokens' must be a positive integer or None, "
                f"got {self.flash_speculative_tokens!r}"
            )
```

- [ ] **Step 7: Populate the new fields in `resolved_flash()`**

In `resolved_flash()` (~467), extend the returned `ResolvedFlashConfig(...)` with:

```python
            prefetch=(
                self.flash_prefetch
                if self.flash_prefetch is not None
                else settings.flash_prefetch
            ),
            flash_speculative=(
                self.flash_speculative
                if self.flash_speculative is not None
                else settings.flash_speculative
            ),
            flash_speculative_draft_model=(
                self.flash_speculative_draft_model
                if self.flash_speculative_draft_model is not None
                else settings.flash_speculative_draft_model
            ),
            flash_speculative_tokens=(
                self.flash_speculative_tokens
                if self.flash_speculative_tokens is not None
                else settings.flash_speculative_tokens
            ),
```

- [ ] **Step 8: Wire `from_entry` and `to_entry`**

In `from_entry` (~604, after `flash_moe_io_threads = entry.get(...)`), add raw pulls:

```python
            flash_prefetch = entry.get("flash_prefetch")
            flash_speculative = entry.get("flash_speculative")
            flash_speculative_draft_model = entry.get("flash_speculative_draft_model")
            flash_speculative_tokens = entry.get("flash_speculative_tokens")
```

Add them to the `cls(...)` call (~647, after `flash_moe_io_threads=...`):

```python
                flash_prefetch=flash_prefetch,
                flash_speculative=flash_speculative,
                flash_speculative_draft_model=flash_speculative_draft_model,
                flash_speculative_tokens=flash_speculative_tokens,
```

In `to_entry`, add to the all-`None` short-circuit guard (~688) AND the serialization block (~715):

```python
            and self.flash_prefetch is None
            and self.flash_speculative is None
            and self.flash_speculative_draft_model is None
            and self.flash_speculative_tokens is None
```

```python
        if self.flash_prefetch is not None:
            result["flash_prefetch"] = self.flash_prefetch
        if self.flash_speculative is not None:
            result["flash_speculative"] = self.flash_speculative
        if self.flash_speculative_draft_model is not None:
            result["flash_speculative_draft_model"] = self.flash_speculative_draft_model
        if self.flash_speculative_tokens is not None:
            result["flash_speculative_tokens"] = self.flash_speculative_tokens
```

(`_KNOWN_CONFIG_KEYS` is derived from dataclass fields automatically — no edit needed.)

- [ ] **Step 9: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py::TestFlashPrefetchSpeculativePromotionRegistry -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add olmlx/engine/registry.py tests/test_registry.py
git commit -m "feat(registry): per-model flash_prefetch + flash_speculative, promote keys (#275, #276)"
```

---

## Task 3: Switch read sites in `model_manager.py` and `cli.py`

**Files:**
- Modify: `olmlx/engine/model_manager.py` (`_load_flash_model` body ~2832-2901, incompat guards ~3061/3121/3144)
- Modify: `olmlx/cli.py` (~2379)
- Test: `tests/test_flash_speculative.py` (integration assertion)

The promoted fields now arrive via the `flash_config: ResolvedFlashConfig` parameter that `_load_flash_model` already receives. The 4 prefetch *tuning* fields keep reading from `model_exp`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_flash_speculative.py` (a pure-resolution assertion, no model load):

```python
def test_resolved_flash_carries_speculative_and_prefetch(monkeypatch):
    from olmlx.config import settings
    from olmlx.engine.registry import ModelConfig

    monkeypatch.setattr(settings, "flash_speculative", True, raising=False)
    monkeypatch.setattr(
        settings, "flash_speculative_draft_model", "d/m", raising=False
    )
    monkeypatch.setattr(settings, "flash_speculative_tokens", 7, raising=False)
    monkeypatch.setattr(settings, "flash_prefetch", True, raising=False)

    rf = ModelConfig(hf_path="Qwen/Qwen3-8B").resolved_flash()
    assert rf.flash_speculative is True
    assert rf.flash_speculative_draft_model == "d/m"
    assert rf.flash_speculative_tokens == 7
    assert rf.prefetch is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_flash_speculative.py::test_resolved_flash_carries_speculative_and_prefetch -v`
Expected: FAIL only if Task 2 incomplete; if Task 2 done it PASSES (this guards the resolution wiring). If it already passes, proceed — it locks the contract for the read-site switch.

- [ ] **Step 3: Switch `_load_flash_model` read sites**

In `olmlx/engine/model_manager.py`, change these reads (keep tuning knobs on `model_exp`):

- ~2832: `prefetch=model_exp.flash_prefetch,` → `prefetch=flash_config.prefetch,`
- ~2850: `if model_exp.flash_prefetch and lookahead_path.exists():` → `if flash_config.prefetch and lookahead_path.exists():`
- ~2865: `if model_exp.flash_speculative:` → `if flash_config.flash_speculative:`
- ~2868: `if not model_exp.flash_speculative_draft_model:` → `if not flash_config.flash_speculative_draft_model:`
- ~2876, ~2879: `model_exp.flash_speculative_draft_model` → `flash_config.flash_speculative_draft_model`
- ~2901: `num_speculative_tokens=model_exp.flash_speculative_tokens,` → `num_speculative_tokens=flash_config.flash_speculative_tokens,`

Leave ~2833-2836 (`flash_prefetch_confidence_threshold`/`_min_neurons`/`_max_neurons`/`_io_threads`) reading from `model_exp` unchanged.

- [ ] **Step 4: Switch the incompat guards**

In `olmlx/engine/model_manager.py`, the Flash-MoE / VLM / both-speculative guards read `model_exp.flash_speculative` (~3061, ~3121, ~3144). These guards receive the resolved config too — change each `model_exp.flash_speculative` to the resolved flash-speculative value. **Verify the available variable name in each guard's scope** (it may be `flash_config` or a `resolved`/`spec` local). Use `grep -n "flash_speculative" olmlx/engine/model_manager.py` and replace each `model_exp.flash_speculative` read with the in-scope `ResolvedFlashConfig.flash_speculative`. If `flash_config` is not in scope at a guard, thread it in or resolve via the `model_config` already present.

- [ ] **Step 5: Switch the cli.py prepare read site**

In `olmlx/cli.py:2379`: `train_lookahead=experimental.flash_prefetch,` → `train_lookahead=settings.flash_prefetch,`. Confirm `settings` is imported in that scope (`grep -n "^from olmlx.config import\|import settings" olmlx/cli.py`); add to the import if missing.

- [ ] **Step 6: Run the flash test suites**

Run: `uv run pytest tests/test_flash_speculative.py tests/test_flash_prefetch.py tests/test_flash_integration.py -v`
Expected: PASS (no regressions; new assertion passes).

- [ ] **Step 7: Commit**

```bash
git add olmlx/engine/model_manager.py olmlx/cli.py tests/test_flash_speculative.py
git commit -m "refactor(flash): read promoted prefetch/speculative from resolved config (#275, #276)"
```

---

## Task 4: Legacy env-var forwarding shim

**Files:**
- Modify: `olmlx/config.py` (add shim near `surface_legacy_flash_moe_env` ~552)
- Modify: `olmlx/cli.py` (import alias ~21; call in `_apply_serve_overrides` ~595; calls at ~1775/2127/2641 if those forward flash)
- Modify: `olmlx/engine/distributed_worker.py` (~187, ~199)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
class TestLegacyFlashPrefetchSpeculativeForwarding:
    def test_legacy_prefetch_speculative_forwarded(self, monkeypatch, caplog):
        import logging
        from olmlx.config import settings, surface_legacy_flash_prefetch_speculative_env

        for k in ("flash_prefetch", "flash_speculative",
                  "flash_speculative_draft_model", "flash_speculative_tokens"):
            monkeypatch.delenv("OLMLX_" + k.upper(), raising=False)
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=False)
        monkeypatch.setattr(settings, "flash_speculative", False, raising=False)
        monkeypatch.setattr(settings, "flash_speculative_draft_model", None, raising=False)
        monkeypatch.setattr(settings, "flash_speculative_tokens", 4, raising=False)

        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_PREFETCH", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL", "d/m")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS", "6")
        with caplog.at_level(logging.WARNING):
            surface_legacy_flash_prefetch_speculative_env()
        assert settings.flash_prefetch is True
        assert settings.flash_speculative is True
        assert settings.flash_speculative_draft_model == "d/m"
        assert settings.flash_speculative_tokens == 6
        assert "OLMLX_FLASH_SPECULATIVE" in caplog.text

    def test_new_env_var_wins_over_legacy(self, monkeypatch):
        from olmlx.config import settings, surface_legacy_flash_prefetch_speculative_env

        monkeypatch.setattr(settings, "flash_speculative_tokens", 4, raising=False)
        monkeypatch.setenv("OLMLX_FLASH_SPECULATIVE_TOKENS", "9")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS", "6")
        # New var already drove Settings to 9 via env; legacy must not clobber.
        monkeypatch.setattr(settings, "flash_speculative_tokens", 9, raising=False)
        surface_legacy_flash_prefetch_speculative_env()
        assert settings.flash_speculative_tokens == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::TestLegacyFlashPrefetchSpeculativeForwarding -v`
Expected: FAIL (`surface_legacy_flash_prefetch_speculative_env` does not exist).

- [ ] **Step 3: Add the shim (mirrors `surface_legacy_flash_moe_env`)**

In `olmlx/config.py`, after `surface_legacy_flash_moe_env` (~570), add a forward table + function modeled on the flash_moe shim (which already handles shell + `.env` and the "new var wins / non-default not clobbered" semantics). Reuse `_legacy_flash_moe_values_in_dotenv`'s parsing by generalizing it, OR add a parallel `.env` reader; the simplest is to add a dedicated table and a `_forward_legacy_*` loop identical in shape to `_forward_legacy_flash_moe_env`:

```python
_DEPRECATED_FLASH_PREFETCH_SPECULATIVE_ENV_VARS = (
    "OLMLX_EXPERIMENTAL_FLASH_PREFETCH",
    "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE",
    "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL",
    "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS",
)

_LEGACY_FLASH_PREFETCH_SPECULATIVE_FORWARD: tuple[
    tuple[str, str, str, Callable[[str], Any]], ...
] = (
    (
        "OLMLX_EXPERIMENTAL_FLASH_PREFETCH",
        "OLMLX_FLASH_PREFETCH",
        "flash_prefetch",
        lambda v: v.strip().lower() in ("1", "true", "yes", "on"),
    ),
    (
        "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE",
        "OLMLX_FLASH_SPECULATIVE",
        "flash_speculative",
        lambda v: v.strip().lower() in ("1", "true", "yes", "on"),
    ),
    (
        "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL",
        "OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL",
        "flash_speculative_draft_model",
        str,
    ),
    (
        "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS",
        "OLMLX_FLASH_SPECULATIVE_TOKENS",
        "flash_speculative_tokens",
        int,
    ),
)


def surface_legacy_flash_prefetch_speculative_env() -> None:
    """Forward legacy ``OLMLX_EXPERIMENTAL_FLASH_PREFETCH`` /
    ``OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE*`` to the promoted
    ``OLMLX_FLASH_PREFETCH`` / ``OLMLX_FLASH_SPECULATIVE*`` names.

    Same "new var wins, non-default not clobbered" semantics as
    ``surface_legacy_flash_moe_env``. The four prefetch *tuning* env vars
    (``..._PREFETCH_CONFIDENCE_THRESHOLD`` etc.) keep the experimental
    prefix and pass through untouched. Lives in ``olmlx.config`` so the
    distributed worker can reuse it without importing argparse/uvicorn.
    """
    dotenv_values = _legacy_values_in_dotenv(
        _DEPRECATED_FLASH_PREFETCH_SPECULATIVE_ENV_VARS
    )
    shell_stale = [
        v for v in _DEPRECATED_FLASH_PREFETCH_SPECULATIVE_ENV_VARS if os.environ.get(v)
    ]
    stale = sorted({*shell_stale, *dotenv_values.keys()})
    if not stale:
        return
    logger.warning(
        "Deprecated env vars detected: %s. They will be honoured for this "
        "release but should be renamed to OLMLX_FLASH_PREFETCH, "
        "OLMLX_FLASH_SPECULATIVE, OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL, "
        "OLMLX_FLASH_SPECULATIVE_TOKENS.",
        ", ".join(stale),
    )
    for legacy, new, attr, parse in _LEGACY_FLASH_PREFETCH_SPECULATIVE_FORWARD:
        legacy_val = os.environ.get(legacy, dotenv_values.get(legacy))
        if legacy_val is None or os.environ.get(new) is not None:
            continue
        field_default = Settings.model_fields[attr].default
        if getattr(settings, attr) != field_default:
            continue
        try:
            setattr(settings, attr, parse(legacy_val))
        except Exception as exc:
            logger.warning(
                "Could not forward legacy env var %s=%r to %s: %s",
                legacy, legacy_val, new, exc,
            )
```

Refactor `_legacy_flash_moe_values_in_dotenv` into a generic
`_legacy_values_in_dotenv(names: tuple[str, ...])` (rename + take the
names tuple as a param) and have the flash_moe path call
`_legacy_values_in_dotenv(_DEPRECATED_FLASH_MOE_ENV_VARS)`. This avoids
duplicating the `.env` parser. Update the one caller in
`surface_legacy_flash_moe_env`.

- [ ] **Step 4: Wire call sites**

In `olmlx/cli.py`, add the import alias next to the others (~21):

```python
    surface_legacy_flash_prefetch_speculative_env as _surface_legacy_flash_prefetch_speculative_env,
```

Call it in `_apply_serve_overrides` right after `_surface_legacy_flash_moe_env()` (~595):

```python
    _surface_legacy_flash_prefetch_speculative_env()
```

In `olmlx/engine/distributed_worker.py`, add to the import block (~187) and call after `surface_legacy_flash_moe_env()` (~199):

```python
        surface_legacy_flash_prefetch_speculative_env,
```
```python
    surface_legacy_flash_prefetch_speculative_env()
```

(The other `_surface_legacy_flash_env()` call sites at cli.py ~1775/2127/2641 are `cmd_chat`/`cmd_config_show`/`cmd_flash_info`; add the new call alongside each so those commands see the promoted values too. Verify with `grep -n "_surface_legacy_flash_moe_env()" olmlx/cli.py` and add the new call next to every `_surface_legacy_flash_moe_env()`.)

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_cli.py::TestLegacyFlashPrefetchSpeculativeForwarding tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Verify nothing broke in the existing flash_moe legacy tests after the `.env` reader rename**

Run: `uv run pytest tests/test_cli.py -k legacy -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add olmlx/config.py olmlx/cli.py olmlx/engine/distributed_worker.py tests/test_cli.py
git commit -m "feat(config): legacy env-var forwarding for promoted flash prefetch/speculative (#275, #276)"
```

---

## Task 5: `olmlx serve` CLI flags

**Files:**
- Modify: `olmlx/cli.py` (argparse `serve_p` ~2785; `_apply_serve_overrides` ~620)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
class TestServeFlashPrefetchSpeculativeFlags:
    def test_flags_apply_to_settings(self, monkeypatch):
        import argparse
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings

        monkeypatch.setattr(settings, "flash_speculative", False, raising=False)
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=False)
        args = argparse.Namespace(
            flash_speculative=True,
            flash_speculative_draft_model="d/m",
            flash_speculative_tokens=5,
            flash_prefetch=True,
        )
        _apply_serve_overrides(args)
        assert settings.flash_speculative is True
        assert settings.flash_speculative_draft_model == "d/m"
        assert settings.flash_speculative_tokens == 5
        assert settings.flash_prefetch is True
```

(`_apply_serve_overrides` uses `getattr(args, ..., None)`, so the bare Namespace is fine even though it omits other serve attrs.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::TestServeFlashPrefetchSpeculativeFlags -v`
Expected: FAIL (overrides not applied).

- [ ] **Step 3: Add argparse flags**

In `olmlx/cli.py`, after the `--flash` argument (~2785), add:

```python
    serve_p.add_argument(
        "--flash-prefetch",
        dest="flash_prefetch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Flash speculative neuron prefetch (overrides OLMLX_FLASH_PREFETCH).",
    )
    serve_p.add_argument(
        "--flash-speculative",
        dest="flash_speculative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Flash + speculative decoding (overrides OLMLX_FLASH_SPECULATIVE).",
    )
    serve_p.add_argument(
        "--flash-speculative-draft-model",
        dest="flash_speculative_draft_model",
        type=_non_empty_str,
        default=None,
        help="HuggingFace path of the draft model used for Flash speculative decoding.",
    )
    serve_p.add_argument(
        "--flash-speculative-tokens",
        dest="flash_speculative_tokens",
        type=_positive_int,
        default=None,
        help="Tokens drafted per verification step for Flash speculative (default: 4).",
    )
```

- [ ] **Step 4: Apply in `_apply_serve_overrides`**

In `olmlx/cli.py`, after the `flash_flag` block (~620), add:

```python
    fs = getattr(args, "flash_speculative", None)
    fs_draft = getattr(args, "flash_speculative_draft_model", None)
    fs_tokens = getattr(args, "flash_speculative_tokens", None)
    fp = getattr(args, "flash_prefetch", None)
    if fs is not None:
        _settings.flash_speculative = fs
    if fs_draft is not None:
        _settings.flash_speculative_draft_model = fs_draft
    if fs_tokens is not None:
        _settings.flash_speculative_tokens = fs_tokens
    if fp is not None:
        _settings.flash_prefetch = fp
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_cli.py::TestServeFlashPrefetchSpeculativeFlags tests/test_cli.py -k "serve or flag" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add olmlx/cli.py tests/test_cli.py
git commit -m "feat(cli): add --flash-prefetch / --flash-speculative serve flags (#275, #276)"
```

---

## Task 6: Tensor-only distributed (#273 pipeline decision)

**Files:**
- Modify: `olmlx/config.py` (`distributed_strategy` ~92)
- Modify: `olmlx/cli.py` (hostfile strategy guard ~1240)
- Test: `tests/test_config.py`, `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
class TestDistributedTensorOnly:
    def test_pipeline_strategy_rejected(self):
        import pytest
        from pydantic import ValidationError
        from olmlx.config import Settings

        with pytest.raises(ValidationError):
            Settings(distributed_strategy="pipeline")

    def test_tensor_strategy_accepted(self):
        from olmlx.config import Settings

        assert Settings(distributed_strategy="tensor").distributed_strategy == "tensor"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestDistributedTensorOnly -v`
Expected: FAIL (`pipeline` currently accepted).

- [ ] **Step 3: Narrow the literal**

In `olmlx/config.py:92`:

```python
    distributed_strategy: Literal["tensor"] = "tensor"
```

Add a comment above it: `# Distributed inference is tensor-only. The dormant pipeline.py / pre_shard_pipeline / worker pipeline branches are unreachable and kept for a future PR (#273).`

- [ ] **Step 4: Harden the hostfile strategy guard**

In `olmlx/cli.py:~1240`, the guard currently accepts `("tensor", "pipeline")`. Change it to reject `pipeline` with an actionable message:

```python
    if strategy != "tensor":
        print(
            f"Error: distributed inference is tensor-only; hostfile strategy "
            f"must be 'tensor', got {strategy!r}. The pipeline strategy is not "
            f"supported.",
            file=sys.stderr,
        )
        sys.exit(1)
```

(Confirm the exact surrounding code with `sed -n '1236,1246p' olmlx/cli.py` before editing; match the existing `print`/`sys.exit` style.)

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_config.py::TestDistributedTensorOnly tests/test_distributed.py -v`
Expected: PASS. If a `test_distributed.py` test constructs `distributed_strategy="pipeline"` expecting success, update it to assert rejection (the dormant pipeline path is intentionally unreachable now). Flag any such test in the commit message.

- [ ] **Step 6: Commit**

```bash
git add olmlx/config.py olmlx/cli.py tests/test_config.py tests/test_distributed.py
git commit -m "feat(distributed): make distributed_strategy tensor-only (#273)"
```

---

## Task 7: Bench scenario definitions

**Files:**
- Modify: `olmlx/bench/scenarios.py` (after the `speculative` scenario ~210)
- Test: `tests/test_bench_scenarios.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bench_scenarios.py`:

```python
def test_flash_prefetch_and_speculative_scenarios_present():
    from olmlx.bench.scenarios import SCENARIOS

    names = {s.name for s in SCENARIOS}
    assert "flash+prefetch" in names
    assert "flash+spec" in names

    by_name = {s.name: s for s in SCENARIOS}
    assert by_name["flash+prefetch"].env_overrides.get("OLMLX_FLASH") == "true"
    assert by_name["flash+prefetch"].env_overrides.get("OLMLX_FLASH_PREFETCH") == "true"
    assert by_name["flash+spec"].env_overrides.get("OLMLX_FLASH") == "true"
    assert by_name["flash+spec"].env_overrides.get("OLMLX_FLASH_SPECULATIVE") == "true"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bench_scenarios.py::test_flash_prefetch_and_speculative_scenarios_present -v`
Expected: FAIL (scenarios absent).

- [ ] **Step 3: Add the scenarios**

In `olmlx/bench/scenarios.py`, after the `flash+tq4` / `speculative` scenarios (~199-208), add:

```python
    Scenario(
        name="flash+prefetch",
        description="Flash inference + speculative neuron prefetch",
        env_overrides={"OLMLX_FLASH": "true", "OLMLX_FLASH_PREFETCH": "true"},
        should_skip=_requires_flash,
    ),
    Scenario(
        name="flash+spec",
        description=(
            "Flash inference + speculative decoding "
            "(set OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL to a draft model HF path)"
        ),
        env_overrides={"OLMLX_FLASH": "true", "OLMLX_FLASH_SPECULATIVE": "true"},
        should_skip=_requires_flash,
    ),
```

(Use `_requires_flash` as the skip guard — it checks for `flash_layout.json`. The draft-model presence is surfaced at load time; matching how the existing `flash`/`speculative` scenarios are guarded.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bench_scenarios.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/bench/scenarios.py tests/test_bench_scenarios.py
git commit -m "bench: add flash+prefetch and flash+spec scenario definitions (#275, #276)"
```

---

## Task 8: Docs — README, USER_MANUAL, CLAUDE.md

**Files:**
- Modify: `README.md` (env-var table ~358-361; example ~485-486; add decision matrix + Path A/B + LookaheadBank + migration + tensor-only notes)
- Modify: `docs/USER_MANUAL.md` (same env-var renames)
- Modify: `CLAUDE.md` (Flash prefetch + Flash+speculative Key Design Decisions; distributed entry)

No automated test — this is prose. Verify with `grep`.

- [ ] **Step 1: README env-var table + example**

In `README.md`, replace the four `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE*` / `OLMLX_EXPERIMENTAL_FLASH_PREFETCH` rows (~358-361) with promoted names:

```
| `OLMLX_FLASH_SPECULATIVE` | `false` | Enable speculative decoding with draft model (requires Flash) |
| `OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL` | `None` | Draft model name or HuggingFace path |
| `OLMLX_FLASH_SPECULATIVE_TOKENS` | `4` | Candidate tokens per speculative step |
| `OLMLX_FLASH_PREFETCH` | `false` | Enable speculative neuron prefetching |
```

Update the example block (~485-486) `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE` → `OLMLX_FLASH_SPECULATIVE`, and the draft-model line similarly.

- [ ] **Step 2: README — decision matrix + prefetch docs + migration**

Add a short subsection near the Flash docs covering: when to use dense vs dense+spec vs flash vs flash+spec (4-row matrix); Path A (cross-layer) vs Path B (draft-informed) prefetch and when each fires; the `LookaheadBank` opt-in (trained when `OLMLX_FLASH_PREFETCH=true` during `olmlx flash prepare`); a migration note that `OLMLX_EXPERIMENTAL_FLASH_PREFETCH` / `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE*` are deprecated (honoured one release) and renamed to the `OLMLX_FLASH_*` names; note that the 4 prefetch tuning knobs remain under `OLMLX_EXPERIMENTAL_FLASH_PREFETCH_*`. Add/adjust the distributed section to state inference is **tensor-only**.

- [ ] **Step 3: USER_MANUAL renames**

In `docs/USER_MANUAL.md`, `grep -n "EXPERIMENTAL_FLASH_SPECULATIVE\|EXPERIMENTAL_FLASH_PREFETCH\|pipeline" docs/USER_MANUAL.md` and rename the promoted env vars; note tensor-only distributed if pipeline is mentioned.

- [ ] **Step 4: CLAUDE.md**

In `CLAUDE.md`: in the "Speculative prefetch (experimental)" entry, drop "experimental" from the promoted toggle's wording (note the toggle `OLMLX_FLASH_PREFETCH` is promoted; the 4 tuning knobs remain `OLMLX_EXPERIMENTAL_FLASH_PREFETCH_*`). In the Flash-speculative coverage (under "Speculative decoding" / `engine/flash/speculative.py`), note `OLMLX_FLASH_SPECULATIVE*` is now the supported surface. In the "Distributed inference" entry, add that it is tensor-only and the `pipeline` strategy literal was removed (dormant pipeline code retained).

- [ ] **Step 5: Verify renames landed**

Run: `grep -rn "EXPERIMENTAL_FLASH_SPECULATIVE\|EXPERIMENTAL_FLASH_PREFETCH\b" README.md docs/USER_MANUAL.md`
Expected: only migration-note mentions remain (the deprecated-name callouts), no live config rows.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/USER_MANUAL.md CLAUDE.md
git commit -m "docs: promote flash prefetch/speculative env vars; tensor-only distributed (#275, #276, #273)"
```

---

## Task 9: Full verification + ruff + push + PR

**Files:** none (verification)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS. Investigate any failure that touches config/registry/cli/flash/distributed/bench before proceeding (per superpowers:systematic-debugging).

- [ ] **Step 2: ruff (per user memory: always before push)**

Run: `uv run ruff check olmlx tests && uv run ruff format olmlx tests`
Expected: clean. Commit any formatting changes.

- [ ] **Step 3: Sanity-check the CLI help renders the new flags**

Run: `uv run olmlx serve --help`
Expected: shows `--flash-prefetch`, `--flash-speculative`, `--flash-speculative-draft-model`, `--flash-speculative-tokens`.

- [ ] **Step 4: Confirm no stray live references to old env names in code**

Run: `grep -rn "EXPERIMENTAL_FLASH_PREFETCH\b\|EXPERIMENTAL_FLASH_SPECULATIVE" olmlx/ | grep -v "config.py"`
Expected: empty (only the shim in config.py references the legacy names).

- [ ] **Step 5: Push and open PR**

Create a branch, push, open a PR referencing the issues. PR body should: summarize the promotion + tensor-only change; explicitly list the deferred items (#273 wizard/error-UX, all hardware-validation checkboxes in #275/#276) so the reviewer knows they're intentionally out of scope; note the dormant pipeline code is retained.

```bash
git checkout -b promote/flash-prefetch-speculative
git push -u origin promote/flash-prefetch-speculative
gh pr create --title "Promote Flash prefetch + speculative; tensor-only distributed (#275, #276, #273)" --body "..."
```

(End PR body with the Claude Code attribution line per repo convention.)

---

## Self-review notes

- **Spec coverage:** Task 1 (config promotion), Task 2 (registry + per-model + migration error), Task 3 (read sites), Task 4 (legacy shim), Task 5 (CLI flags), Task 6 (#273 tensor-only), Task 7 (bench defs), Task 8 (docs incl. decision matrix / Path A-B / LookaheadBank / migration), Task 9 (verify). Hardware-validation checkboxes intentionally deferred (Task 9 PR body documents this).
- **Knob-exposure decision (#275):** only `flash_prefetch` promoted; 4 tuning knobs stay experimental — implemented across Tasks 1/2/3.
- **Migration error (#276/#275):** automatic via `PROMOTED_EXPERIMENTAL_KEYS` (Task 2) + the existing `_models_with_promoted_keys_in_experimental` startup check in `_apply_serve_overrides`. Note: that error message says "speculative settings" — acceptable; the listed key set is generated from `PROMOTED_EXPERIMENTAL_KEYS` so it will name the flash keys correctly.
- **Type consistency:** `ResolvedFlashConfig` new fields named `prefetch`, `flash_speculative`, `flash_speculative_draft_model`, `flash_speculative_tokens` — referenced identically in Task 2 (definition + resolution), Task 3 (read sites), and the Task 3 test.
- **Risk:** Task 3 Step 4 requires verifying the in-scope variable name at each incompat guard — flagged as a grep-and-confirm step rather than a blind replace.
