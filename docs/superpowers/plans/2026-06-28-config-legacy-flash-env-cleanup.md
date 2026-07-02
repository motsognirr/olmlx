# Config Legacy Flash Env-Var Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three near-duplicate `OLMLX_EXPERIMENTAL_FLASH*` → `OLMLX_FLASH*` *forwarding* shims in `olmlx/config.py` (~420 lines) with a single warn-only check that detects the 12 promoted legacy env vars (shell or `.env`) and tells the user the new name without applying the value.

**Architecture:** Three staged tasks — (1) **add** the new table + `warn_legacy_flash_env()` alongside the old code, (2) **repoint** all callers to it, (3) **delete** the old shims and obsolete tests. Staging keeps the package importable and every task's test suite green; the old functions are only removed once nothing references them.

**Tech Stack:** Python 3, pydantic-settings, pytest, ruff. Apple Silicon / MLX project (no Metal needed for these tests — pure config/env logic).

## Global Constraints

- Scope is the three flash families only: `flash`, `flash_moe`, `flash_prefetch`/`flash_speculative`. The sibling `OLMLX_EXPERIMENTAL_SPECULATIVE*` / `_DFLASH*` / `_KV_CACHE_QUANT*` shims live in `olmlx/cli.py` and are **out of scope** — do not touch them.
- `_legacy_values_in_dotenv(names: tuple[str, ...]) -> dict[str, str]` is **retained** unchanged (the warning's `.env` detection reuses it).
- The new behavior is **warn-only**: never forward/apply a legacy value to `settings`. This is an intentional breaking change.
- The warning fires on **presence** of any promoted legacy name regardless of its value (even `=false`) — the name is dead, so any use warrants the nudge.
- `warn_legacy_flash_env` must live in `olmlx.config` (not `olmlx.cli`) so `olmlx/engine/distributed_worker.py` can import it without pulling in argparse/uvicorn.
- The 12 promoted names (and only these) belong in the table; still-valid experimental *tuning* knobs (`OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE`, `..._PREDICTOR_*`, `..._PREFETCH_CONFIDENCE_THRESHOLD`, `..._IO_THREADS`, etc.) must NOT be listed.
- Run `uv run ruff check olmlx tests` and `uv run ruff format olmlx tests` before each commit.
- The config module logger is `olmlx.config` (`logging.getLogger(__name__)`); tests capture under that logger name.

---

### Task 1: Add the warn-only check to `config.py` (old shims left intact)

**Files:**
- Modify: `olmlx/config.py` (add table + function near the existing legacy block, ~line 732 onward; do NOT delete anything yet)
- Test: `tests/test_config.py` (add a new `TestWarnLegacyFlashEnv` class at end of file)

**Interfaces:**
- Consumes: existing `_legacy_values_in_dotenv(names: tuple[str, ...]) -> dict[str, str]`, module-level `logger`, `os`, `settings`.
- Produces:
  - `PROMOTED_FLASH_ENV_RENAMES: dict[str, str]` — 12 entries mapping legacy name → new name.
  - `warn_legacy_flash_env() -> None` — logs one WARNING naming every detected legacy var + its replacement; mutates nothing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
class TestWarnLegacyFlashEnv:
    """warn_legacy_flash_env() detects promoted OLMLX_EXPERIMENTAL_FLASH*
    names (shell or .env) and warns WITHOUT forwarding the value."""

    def _clear_all(self, monkeypatch):
        from olmlx.config import PROMOTED_FLASH_ENV_RENAMES

        for old, new in PROMOTED_FLASH_ENV_RENAMES.items():
            monkeypatch.delenv(old, raising=False)
            monkeypatch.delenv(new, raising=False)

    def test_warns_on_shell_var_and_does_not_apply(
        self, monkeypatch, tmp_path, caplog
    ):
        import logging

        from olmlx.config import settings, warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)  # avoid the dev's real .env
        self._clear_all(monkeypatch)
        monkeypatch.setattr(settings, "flash", False, raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH", "true")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        # Value is NOT forwarded.
        assert settings.flash is False
        # Warning names the legacy var and its replacement.
        assert "OLMLX_EXPERIMENTAL_FLASH" in caplog.text
        assert "OLMLX_FLASH" in caplog.text

    def test_warns_on_dotenv_var(self, monkeypatch, tmp_path, caplog):
        import logging

        from olmlx.config import settings, warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)
        self._clear_all(monkeypatch)
        (tmp_path / ".env").write_text("OLMLX_EXPERIMENTAL_FLASH_PREFETCH=true\n")
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=False)

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        assert settings.flash_prefetch is False
        assert "OLMLX_EXPERIMENTAL_FLASH_PREFETCH" in caplog.text

    def test_no_warn_when_unset(self, monkeypatch, tmp_path, caplog):
        import logging

        from olmlx.config import warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)
        self._clear_all(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        assert not any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), "Expected no warnings but got: " + caplog.text

    def test_warns_for_each_family(self, monkeypatch, tmp_path, caplog):
        import logging

        from olmlx.config import warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)
        self._clear_all(monkeypatch)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_MOE", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE", "true")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        assert "OLMLX_EXPERIMENTAL_FLASH_MOE" in caplog.text
        assert "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE" in caplog.text

    def test_presence_warns_even_for_false_value(
        self, monkeypatch, tmp_path, caplog
    ):
        import logging

        from olmlx.config import warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)
        self._clear_all(monkeypatch)
        # A dead name is dead regardless of value — warn on presence.
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH", "false")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        assert "OLMLX_EXPERIMENTAL_FLASH" in caplog.text

    def test_valid_experimental_tuning_knob_not_warned(
        self, monkeypatch, tmp_path, caplog
    ):
        import logging

        from olmlx.config import warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)
        self._clear_all(monkeypatch)
        # Still-valid experimental tuning knob — must NOT be in the table.
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE", "8")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        assert not any(r.levelno >= logging.WARNING for r in caplog.records)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py::TestWarnLegacyFlashEnv -v`
Expected: FAIL — `ImportError: cannot import name 'PROMOTED_FLASH_ENV_RENAMES'` / `warn_legacy_flash_env`.

- [ ] **Step 3: Add the table and function to `config.py`**

Insert immediately after the `PRE_SHARDED_DIR_ENV = "OLMLX_DISTRIBUTED_PRE_SHARDED_DIR"` line (currently ~line 732), before `LEGACY_FLASH_FORWARD`. Leave all existing legacy code in place for now.

```python
#: Promoted flash env vars: legacy ``OLMLX_EXPERIMENTAL_FLASH*`` name -> the
#: bare ``OLMLX_FLASH*`` name that replaced it (PRs #327/#329/#336). These are
#: no longer honored; ``warn_legacy_flash_env`` detects and warns but does NOT
#: forward them. Only PROMOTED names belong here — the still-valid experimental
#: *tuning* knobs (``OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE``, ``..._PREDICTOR_*``,
#: ``..._PREFETCH_CONFIDENCE_THRESHOLD``, ``..._IO_THREADS``, ...) stay valid
#: and are intentionally absent.
PROMOTED_FLASH_ENV_RENAMES: dict[str, str] = {
    "OLMLX_EXPERIMENTAL_FLASH": "OLMLX_FLASH",
    "OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD": "OLMLX_FLASH_SPARSITY_THRESHOLD",
    "OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS": "OLMLX_FLASH_MIN_ACTIVE_NEURONS",
    "OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS": "OLMLX_FLASH_MAX_ACTIVE_NEURONS",
    "OLMLX_EXPERIMENTAL_FLASH_MEMORY_BUDGET_FRACTION": "OLMLX_FLASH_MEMORY_BUDGET_FRACTION",
    "OLMLX_EXPERIMENTAL_FLASH_MOE": "OLMLX_FLASH_MOE",
    "OLMLX_EXPERIMENTAL_FLASH_MOE_CACHE_BUDGET_EXPERTS": "OLMLX_FLASH_MOE_CACHE_BUDGET_EXPERTS",
    "OLMLX_EXPERIMENTAL_FLASH_MOE_IO_THREADS": "OLMLX_FLASH_MOE_IO_THREADS",
    "OLMLX_EXPERIMENTAL_FLASH_PREFETCH": "OLMLX_FLASH_PREFETCH",
    "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE": "OLMLX_FLASH_SPECULATIVE",
    "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL": "OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL",
    "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS": "OLMLX_FLASH_SPECULATIVE_TOKENS",
}


def warn_legacy_flash_env() -> None:
    """Warn that promoted ``OLMLX_EXPERIMENTAL_FLASH*`` env vars are no longer
    honored, naming the new ``OLMLX_FLASH*`` replacement for each.

    Detects names set in the shell environment OR the project ``.env``. Does
    NOT forward the value: these knobs were promoted out of the experimental
    prefix several releases ago (PRs #327/#329/#336) and the one-release
    deprecation window has long closed. Warns on *presence* of the name
    regardless of its value, since the name itself is dead.

    Lives in ``olmlx.config`` so the distributed-worker entry point can reuse
    it without importing ``olmlx.cli`` (and its argparse/uvicorn baggage).
    """
    dotenv_keys = _legacy_values_in_dotenv(tuple(PROMOTED_FLASH_ENV_RENAMES))
    detected = sorted(
        old
        for old in PROMOTED_FLASH_ENV_RENAMES
        if old in os.environ or old in dotenv_keys
    )
    if not detected:
        return
    renames = "; ".join(
        f"{old} -> {PROMOTED_FLASH_ENV_RENAMES[old]}" for old in detected
    )
    logger.warning(
        "Unsupported env vars detected and IGNORED: %s. These were renamed out "
        "of the experimental prefix and are no longer honored: %s. Update your "
        "shell environment / .env to the new names.",
        ", ".join(detected),
        renames,
    )
```

Note: `warn_legacy_flash_env` is defined textually *above* `_legacy_values_in_dotenv` (which currently sits ~line 958). That is fine — `_legacy_values_in_dotenv` is resolved at call time, not definition time, and `warn_legacy_flash_env` is only ever invoked at runtime by callers. (Task 3 will move/keep `_legacy_values_in_dotenv` so it remains defined.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py::TestWarnLegacyFlashEnv -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check olmlx tests
uv run ruff format olmlx tests
git add olmlx/config.py tests/test_config.py
git commit -m "feat(config): add warn-only check for promoted flash env vars

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Repoint all callers to `warn_legacy_flash_env`

**Files:**
- Modify: `olmlx/cli.py` (import block ~22-26; 7 trio call sites at lines ~600-602, ~2054-2056, ~2451-2453, ~2488-2490, ~2690-2692, ~3042-3044)
- Modify: `olmlx/engine/distributed_worker.py` (import ~184-189; trio call ~199-201; comment ~191-198)
- Test: `tests/test_cli.py` (rewrite `test_cmd_flash_prepare_calls_legacy_shim`, ~line 2963)

**Interfaces:**
- Consumes: `warn_legacy_flash_env` from Task 1.
- Produces: every prior call site now invokes the single function. In `cli.py` it is imported as `warn_legacy_flash_env as _warn_legacy_flash_env` and called `_warn_legacy_flash_env()` (matches the file's private-alias convention).

- [ ] **Step 1: Rewrite the structural regression test (failing first)**

Replace `test_cmd_flash_prepare_calls_legacy_shim` (currently ~2963-2983 in `tests/test_cli.py`) with:

```python
    def test_cmd_flash_prepare_calls_legacy_shim(self, monkeypatch):
        """cmd_flash_prepare must surface the legacy-flash warning before
        reading settings, so a stale OLMLX_EXPERIMENTAL_FLASH* is reported."""
        import inspect

        import olmlx.cli as cli_mod

        src = inspect.getsource(cli_mod.cmd_flash_prepare)
        assert "_warn_legacy_flash_env" in src, (
            "cmd_flash_prepare must call _warn_legacy_flash_env() so a stale "
            "OLMLX_EXPERIMENTAL_FLASH* var is surfaced before settings are read."
        )
```

Run: `uv run pytest "tests/test_cli.py::TestLegacyFlashPrefetchSpeculativeForwarding::test_cmd_flash_prepare_calls_legacy_shim" -v`
Expected: FAIL — `_warn_legacy_flash_env` not yet in `cmd_flash_prepare` source.

- [ ] **Step 2: Update the `cli.py` import block**

Replace the current import (lines ~22-26):

```python
from olmlx.config import (
    settings,
    surface_legacy_flash_env as _surface_legacy_flash_env,
    surface_legacy_flash_moe_env as _surface_legacy_flash_moe_env,
    surface_legacy_flash_prefetch_speculative_env as _surface_legacy_flash_prefetch_speculative_env,
)
```

with:

```python
from olmlx.config import (
    settings,
    warn_legacy_flash_env as _warn_legacy_flash_env,
)
```

- [ ] **Step 3: Replace each trio call site in `cli.py`**

At each of the 6 locations where these three lines appear consecutively:

```python
    _surface_legacy_flash_env()
    _surface_legacy_flash_moe_env()
    _surface_legacy_flash_prefetch_speculative_env()
```

replace all three lines with the single line:

```python
    _warn_legacy_flash_env()
```

Do this for every occurrence (originally lines ~600-602, ~2054-2056, ~2451-2453, ~2488-2490, ~2690-2692, ~3042-3044). Verify none remain:

```bash
grep -n "_surface_legacy_flash_env\|_surface_legacy_flash_moe_env\|_surface_legacy_flash_prefetch_speculative_env" olmlx/cli.py
```
Expected: no output.

- [ ] **Step 4: Update `distributed_worker.py`**

Replace the import (lines ~184-189):

```python
    from olmlx.config import (
        settings as _settings_early,
        surface_legacy_flash_env,
        surface_legacy_flash_moe_env,
        surface_legacy_flash_prefetch_speculative_env,
    )
```

with:

```python
    from olmlx.config import (
        settings as _settings_early,
        warn_legacy_flash_env,
    )
```

Replace the comment + trio call (lines ~191-201):

```python
    # Honour the one-release deprecation window for ``OLMLX_EXPERIMENTAL_FLASH*``
    # and ``OLMLX_EXPERIMENTAL_FLASH_MOE*`` on the direct-worker path. The
    # coordinator runs the same shims in its own startup and forwards resolved
    # ``OLMLX_FLASH*``/``OLMLX_FLASH_MOE*`` values to workers it launches via
    # SSH, so these matter only when ``python -m olmlx.engine.distributed_worker``
    # is invoked directly with legacy env vars set. Both helpers live in
    # ``olmlx.config`` so the worker does not need to import ``olmlx.cli``
    # (and its argparse/uvicorn baggage).
    surface_legacy_flash_env()
    surface_legacy_flash_moe_env()
    surface_legacy_flash_prefetch_speculative_env()
```

with:

```python
    # Warn (don't forward) if a stale ``OLMLX_EXPERIMENTAL_FLASH*`` var is set
    # on the direct-worker path. These knobs were promoted to ``OLMLX_FLASH*``
    # several releases ago; the value is no longer honored. Lives in
    # ``olmlx.config`` so the worker need not import ``olmlx.cli`` (argparse/
    # uvicorn baggage).
    warn_legacy_flash_env()
```

- [ ] **Step 5: Verify the structural test passes and the package imports**

Run: `uv run pytest "tests/test_cli.py::TestLegacyFlashPrefetchSpeculativeForwarding::test_cmd_flash_prepare_calls_legacy_shim" -v`
Expected: PASS.

Run: `uv run python -c "import olmlx.cli, olmlx.engine.distributed_worker; print('import ok')"`
Expected: `import ok` (old `surface_legacy_*` functions still exist in config, so nothing is broken yet).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check olmlx tests
uv run ruff format olmlx tests
git add olmlx/cli.py olmlx/engine/distributed_worker.py tests/test_cli.py
git commit -m "refactor(config): repoint flash legacy callers to warn_legacy_flash_env

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Delete the old forwarding shims and obsolete tests

**Files:**
- Modify: `olmlx/config.py` (delete the three forwarding functions + their module-level tables/tuples; keep `_legacy_values_in_dotenv`, `PROMOTED_FLASH_ENV_RENAMES`, `warn_legacy_flash_env`, `PRE_SHARDED_DIR_ENV`, `resolve_experimental`, `FlashMoeConfig`)
- Modify: `tests/test_cli.py` (delete the obsolete forwarding tests listed below)

**Interfaces:**
- Consumes: nothing new.
- Produces: `config.py` with only the warn-only path; no `surface_legacy_flash_env`, `surface_legacy_flash_moe_env`, or `surface_legacy_flash_prefetch_speculative_env`.

- [ ] **Step 1: Delete the obsolete tests from `tests/test_cli.py`**

Delete these test functions entirely (they assert forwarding/clobber/cross-field machinery that no longer exists):

Inside `TestBuildParser` (flash family, ~lines 1531-1841):
- `test_legacy_flash_enable_forwarded`
- `test_legacy_flash_numeric_knobs_forwarded`
- `test_legacy_flash_new_env_var_wins`
- `test_legacy_flash_no_op_when_unset`
- `test_legacy_flash_inverted_neuron_range_drops_both`
- `test_legacy_flash_pending_min_below_live_max_drops_both`
- `test_legacy_flash_pending_max_below_live_min_drops_both`
- `test_legacy_flash_out_of_range_value_logs_failure_only`
- `test_legacy_flash_false_value_does_not_warn`

Inside `TestLegacyFlashPrefetchSpeculativeForwarding` (~lines 2765-2961) delete every method EXCEPT `test_cmd_flash_prepare_calls_legacy_shim` (rewritten in Task 2):
- `test_legacy_prefetch_speculative_forwarded`
- `test_new_env_var_wins_over_legacy`
- `test_legacy_prefetch_speculative_no_op_when_unset`
- `test_legacy_prefetch_dotenv_forwarding`
- `test_whitespace_draft_model_becomes_none`
- `test_dotenv_new_var_opt_out_not_clobbered_by_legacy_shell`
- `test_dotenv_new_var_opt_out_not_clobbered_by_legacy_shell_flash_moe`

The behaviors these covered (warn on shell var, warn on `.env` var, no-op when unset, each family, presence-warns) are all covered by `TestWarnLegacyFlashEnv` in `tests/test_config.py` from Task 1. Keep the `TestLegacyFlashPrefetchSpeculativeForwarding` class shell holding only the rewritten `test_cmd_flash_prepare_calls_legacy_shim` (rename the class to `TestCmdFlashPrepareSurfacesLegacy` if it now reads oddly — optional, mechanical).

- [ ] **Step 2: Confirm the obsolete tests are gone and nothing else references the old names**

```bash
grep -n "surface_legacy_flash_env\|surface_legacy_flash_moe_env\|surface_legacy_flash_prefetch_speculative_env" tests/test_cli.py
```
Expected: no output.

- [ ] **Step 3: Delete the old shims from `config.py`**

Remove these top-level definitions entirely:
- `LEGACY_FLASH_FORWARD` (the tuple) and `surface_legacy_flash_env` (the function, incl. its pending-set + neuron-range pre-validation block)
- `_DEPRECATED_FLASH_MOE_ENV_VARS`, `_LEGACY_FLASH_MOE_FORWARD`, `_forward_legacy_flash_moe_env`, `surface_legacy_flash_moe_env`
- `_DEPRECATED_FLASH_PREFETCH_SPECULATIVE_ENV_VARS`, `_LEGACY_FLASH_PREFETCH_SPECULATIVE_FORWARD`, `surface_legacy_flash_prefetch_speculative_env`

**Keep**: `PRE_SHARDED_DIR_ENV`, `PROMOTED_FLASH_ENV_RENAMES`, `warn_legacy_flash_env`, `_legacy_values_in_dotenv`, `resolve_experimental`, `FlashMoeConfig`, and the entire `Settings` / `ExperimentalSettings` block.

After deletion, confirm the survivors and that `_legacy_values_in_dotenv` is still defined:

```bash
grep -n "^def \|^class \|^PROMOTED_FLASH_ENV_RENAMES\|^PRE_SHARDED_DIR_ENV\|^LEGACY_FLASH_FORWARD\|^_LEGACY_FLASH\|^_DEPRECATED_FLASH\|^def _forward_legacy_flash_moe_env\|^def surface_legacy" olmlx/config.py
```
Expected: lists `warn_legacy_flash_env`, `_legacy_values_in_dotenv`, `resolve_experimental`, the validate_* free functions, `Settings`, `ExperimentalSettings`, `FlashMoeConfig` — and NONE of `LEGACY_FLASH_FORWARD`, `_LEGACY_FLASH*`, `_DEPRECATED_FLASH*`, `_forward_legacy_flash_moe_env`, `surface_legacy_*`.

If `Callable` / `Any` imports (line 6) become unused after the deletion, drop them — `ruff check` in Step 5 will flag any unused import.

- [ ] **Step 4: Run the full config + cli suites**

Run: `uv run pytest tests/test_config.py tests/test_cli.py -q`
Expected: PASS (no collection errors, no failures). If a deleted-test reference lingers, fix it.

Run: `uv run python -c "import olmlx.cli, olmlx.engine.distributed_worker; print('import ok')"`
Expected: `import ok`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check olmlx tests
uv run ruff format olmlx tests
git add olmlx/config.py tests/test_cli.py
git commit -m "refactor(config): delete legacy flash env forwarding shims

Removes ~420 lines of OLMLX_EXPERIMENTAL_FLASH* -> OLMLX_FLASH* forwarding
(three near-duplicate shims) now superseded by the warn-only
warn_legacy_flash_env. The deprecation window (PRs #327/#329/#336) closed
~260 PRs ago; stale legacy vars now warn instead of being silently honored.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Manual verification + docs sweep

**Files:**
- Possibly modify: `README.md` (migration note, if one references the legacy-forwarding behavior); `CLAUDE.md` only if a documented invariant changed (none expected — this is a deletion).

- [ ] **Step 1: Manual smoke test — legacy var warns and is NOT honored**

```bash
OLMLX_EXPERIMENTAL_FLASH=true uv run olmlx config show 2>&1 | grep -i "no longer honored\|Unsupported env vars"
```
Expected: the warning line appears. Confirm `flash` is shown as its default (False / off) in the same output — the legacy value did NOT enable flash.

- [ ] **Step 2: Manual smoke test — new var still works**

```bash
OLMLX_FLASH=true uv run olmlx config show 2>&1 | grep -i "flash"
```
Expected: `flash` shown as enabled; no "no longer honored" warning.

- [ ] **Step 3: Docs sweep**

```bash
grep -rn "OLMLX_EXPERIMENTAL_FLASH" README.md docs/ 2>/dev/null
```
For any hit that documents the old vars as *honored/forwarded*, update it to state they are removed (renamed to `OLMLX_FLASH*`). Do not invent new docs; only correct stale claims. If nothing references them, no change.

- [ ] **Step 4: Final full-suite sanity (targeted) + commit any doc edits**

Run: `uv run pytest tests/test_config.py tests/test_cli.py -q`
Expected: PASS.

```bash
# only if Step 3 changed files:
git add README.md docs/
git commit -m "docs: note removal of legacy OLMLX_EXPERIMENTAL_FLASH* env vars

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Delete three flash shims + tables/helpers → Task 3 Step 3. ✓
- Add `PROMOTED_FLASH_ENV_RENAMES` + `warn_legacy_flash_env` (warn-only, shell+`.env`) → Task 1. ✓
- Retain `_legacy_values_in_dotenv` → Global Constraints + Task 3 "Keep" list. ✓
- Only the 12 promoted names; valid tuning knobs excluded → Task 1 table + `test_valid_experimental_tuning_knob_not_warned`. ✓
- Same call sites, one call each (7 in cli.py + 1 in worker) → Task 2 Steps 2-4. ✓
- Test rewrite buckets (forwarding-applied → warn-not-applied; delete machinery tests; `.env` detection kept) → Task 1 (new coverage) + Task 3 Step 1 (deletions). ✓
- Breaking-change verification (warns, not honored) → Task 4 Steps 1-2. ✓
- CLAUDE.md only if invariant changed → Task 4 Step 3. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type consistency:** `warn_legacy_flash_env() -> None` and `PROMOTED_FLASH_ENV_RENAMES: dict[str, str]` used identically across Tasks 1-3; cli alias `_warn_legacy_flash_env` consistent between Task 2 Step 2 (import) and the structural test (Task 2 Step 1). `_legacy_values_in_dotenv(tuple(...))` signature matches the retained function. ✓
