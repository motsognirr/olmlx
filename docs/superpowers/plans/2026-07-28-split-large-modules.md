# Split Large Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mechanically split the three largest modules — `olmlx/cli.py` (3,971 lines), `olmlx/engine/model_manager.py` (4,028), `olmlx/engine/inference.py` (5,497) — into focused files, with zero behavior change and the existing 5,575-test suite green after every task.

**Architecture:** Each original module path remains a **facade** that re-exports every name it exports today, because tests are heavily coupled to these namespaces two ways: (1) direct imports (`from olmlx.engine.inference import X` — 181 sites) and (2) monkeypatch string targets (`"olmlx.engine.inference.mx"` — 111 sites, 35 distinct symbols). The governing rule: **moving a *definition* out of a facade is safe as long as every consumer that tests exercise via facade-level patches stays in the facade** (the facade's global binding is what monkeypatch rebinds). When a *consumer* moves, the tests that patch its dependencies must be repointed to the new module in the same task.

**Tech Stack:** Python 3.12, pytest, ruff. No new dependencies.

## Global Constraints

- Pure mechanical refactor: no behavior, signature, or logic changes. Code moves verbatim (imports adjusted only).
- Facades keep every current export, including underscore names tests import/patch.
- `olmlx.cli` dispatch (`_resolve_handler`) resolves handlers via `globals()` **by design** so `monkeypatch.setattr("olmlx.cli.cmd_serve", ...)` works — dispatch and all `cmd_*` bindings must live in the package `__init__` (cli.py:3908 docstring).
- CLAUDE.md invariants are law — none of the moved code paths may be reordered (e.g. `drop_for_tokenizer` runs first in `_close_loaded_model`).
- Run `uv run ruff check . && uv run ruff format .` before every commit (user memory: feedback_ruff).
- Each task ends with its affected test files passing; the final task runs the full suite.
- Entry point `olmlx = "olmlx.cli:cli_main"` (pyproject) must keep working.

---

### Task 1: Split `olmlx/cli.py` into the `olmlx/cli/` package

**Files:**
- Create: `olmlx/cli/__init__.py`, `olmlx/cli/config_cmd.py`, `olmlx/cli/serve.py`, `olmlx/cli/distributed_launch.py`, `olmlx/cli/service.py`, `olmlx/cli/models_cmd.py`, `olmlx/cli/chat_cmd.py`, `olmlx/cli/bench_cmd.py`, `olmlx/cli/prepare_cmd.py`, `olmlx/cli/parser.py`
- Delete: `olmlx/cli.py`
- Modify: test files that patch `olmlx.cli.<symbol>` whose consumer moved (see step 4)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `olmlx.cli` package whose `__init__` re-exports every current public/underscore name; `cli_main`, `build_parser`, all `cmd_*`, `ensure_config` importable exactly as today.

Move map (source line ranges from current `olmlx/cli.py`):

| New module | Contents (current lines) |
|---|---|
| `config_cmd.py` | `ensure_config` (42–55), `_configure_logging` (1931–1951), `cmd_config_show` (2412–2447) |
| `serve.py` | `cmd_serve` + all legacy-env surfacing + `_apply_serve_overrides` + audits (56–1051) |
| `distributed_launch.py` | `_install_signal_handlers` … `_find_executable` (1052–1599) |
| `service.py` | `_is_secret_env_key`, `_build_plist`, `PLIST_PATH`, `cmd_service_*` (1600–1688) |
| `models_cmd.py` | `_create_store` … `cmd_models_delete` (1689–1930) |
| `chat_cmd.py` | voice defaults, `_check_voice_deps`, `cmd_chat` (1952–2411) |
| `bench_cmd.py` | `cmd_bench_*`, `_positive_int`, `_non_empty_str` (2448–2561) |
| `prepare_cmd.py` | `_flash_progress`, `cmd_spectral/shard/flash/dflash/eagle_prepare`, `cmd_flash_info`, `_show_flash_*` (2562–3056) |
| `parser.py` | `build_parser` (3057–3888) |
| `__init__.py` | module docstring + explicit re-imports of every name above, `_COMMAND_HANDLERS`, `_validate_command_handlers`, `_resolve_handler`, `cli_main` (3889–end) |

- [ ] **Step 1: Create the package** — move code per the map. Each submodule takes only the imports it needs from the current header (lines 1–41). `__init__.py` imports names explicitly (no `import *`) so `globals()` in `_resolve_handler` sees every `cmd_*`.
- [ ] **Step 2: Sanity-run the CLI** — `uv run olmlx --help` and `uv run python -c "from olmlx.cli import cli_main, build_parser, cmd_serve, ensure_config"`. Expected: no ImportError, help text unchanged (`diff` against `git stash`-free main output not required; eyeball).
- [ ] **Step 3: Run CLI test files** — `uv run pytest tests/test_cli.py tests/test_cli_coverage.py -q` plus any file matched by `grep -rl "olmlx\.cli" tests/ | tr '\n' ' '`.
- [ ] **Step 4: Repoint broken patch targets** — for each failure, the patched symbol's *consumer* moved; repoint the string to the consumer's new module (e.g. tests patching `olmlx.cli._audit_speculative_config` exercising `cmd_serve` → `olmlx.cli.serve._audit_speculative_config`; `olmlx.cli._create_store` → `olmlx.cli.models_cmd._create_store`; `olmlx.cli.subprocess`/`shutil`/`PLIST_PATH` for service tests → `olmlx.cli.service.*`). Patches of `cmd_*` used through dispatch stay `olmlx.cli.cmd_*` (globals()-resolution in `__init__`).
- [ ] **Step 5: Re-run step 3 until green.**
- [ ] **Step 6: ruff + commit** — `uv run ruff check . && uv run ruff format .`; `git add -A && git commit -m "refactor(cli): split cli.py into olmlx/cli/ package"`.

### Task 2: Extract module-level helpers from `model_manager.py`

**Files:**
- Create: `olmlx/engine/model_load_utils.py`, `olmlx/engine/cache_capabilities.py`, `olmlx/engine/loaded_model.py`
- Modify: `olmlx/engine/model_manager.py` (facade re-imports), affected tests

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `model_manager` facade still exports `LoadedModel`, `structural_copy`, `parse_keep_alive`, `ActiveRequestsError`, `ModelLoadTimeoutError`, `SpectralCalibrationMissingError`, `ShardCalibrationMissingError`, `_load_with_model_type_fallback`, `_materialize_module_buffers`, `_cache_supports_*`, etc.

Move map:

| New module | Contents (current lines) |
|---|---|
| `model_load_utils.py` | `_sanitize_model_config_in_place`, `_ensure_tokenizer_eos_in_stops`, `_materialize_module_buffers`, `_load_with_model_type_fallback`, `_quantize_language_tower`, `_load_gemma4_unified_text`, `_maybe_load_gemma4_unified_text` (67–448) |
| `cache_capabilities.py` | `_is_serializable_cache` … `_cache_supports_persistence` (449–673) |
| `loaded_model.py` | error classes, `structural_copy`, `LoadedModel`, `_is_cross_encoder_config`, `parse_keep_alive` (674–981) |

`ModelManager` itself stays in `model_manager.py` (splitting the class into mixins would break the 59 `model_manager.mx` / 46 `.settings` / 30 `.gc` patch sites wholesale — deliberately out of scope; noted as follow-up).

- [ ] **Step 1: Move code**, facade re-imports all names. Moved helpers referencing `mx`/`settings` now bind their own module's globals.
- [ ] **Step 2: Run** `uv run pytest tests/test_model_manager.py tests/test_thread_local_streams.py -q` plus `grep -rl "model_manager" tests/ | tr '\n' ' '`.
- [ ] **Step 3: Repoint failures** — tests patching `olmlx.engine.model_manager.mx`/`settings` whose consumer is a moved helper get repointed to the new module (e.g. `_materialize_module_buffers` mx-eval assertions → `olmlx.engine.model_load_utils.mx`).
- [ ] **Step 4: ruff + commit** — `git commit -m "refactor(engine): extract model_manager module-level helpers"`.

### Task 3: Extract leaf clusters from `inference.py`

**Files:**
- Create: `olmlx/engine/generation_options.py`, `olmlx/engine/kv_budget.py`
- Modify: `olmlx/engine/inference.py` (facade re-imports), affected tests

**Interfaces:**
- Consumes: `LoadedModel` via existing facade import.
- Produces: facade still exports `_merge_default_options`, `_apply_sampling_defaults`, `_build_generate_kwargs`, `_apply_seed`, `estimate_kv_cache_bytes`, `tokenize_for_cache`, `build_context_input_tokens`, `count_chat_tokens`.

Move map:

| New module | Contents (current lines) |
|---|---|
| `generation_options.py` | `_merge_default_options`, `_apply_sampling_defaults`, `_build_generate_kwargs`, `_apply_seed` (1191–1367) |
| `kv_budget.py` | `estimate_kv_cache_bytes`, `tokenize_for_cache`, `build_context_input_tokens` (733–968) |

Explicitly **out of scope** (documented follow-up): the batched path, the streaming/full-completion orchestration, aux `generate_*` (embeddings/rerank/transcription/speech), and cache factories — all are consumers of the facade's heavily-patched globals (`mx` 111×, `make_prompt_cache` 39×, `trim_prompt_cache` 25×, `_inference_locked` 11×); moving them requires a dedicated test-migration pass.

- [ ] **Step 1: Move code**, facade re-imports. `_apply_sampling_defaults` reads `settings` from its new module.
- [ ] **Step 2: Run** `uv run pytest tests/test_inference.py tests/test_sampling_defaults.py -q` (adjust to actual file names via `grep -rl "_apply_sampling_defaults\|estimate_kv_cache_bytes" tests/`).
- [ ] **Step 3: Repoint failures** — tests patching `olmlx.engine.inference.settings` to drive `_apply_sampling_defaults` directly → `olmlx.engine.generation_options.settings`; same for `estimate_kv_cache_bytes` internals.
- [ ] **Step 4: ruff + commit** — `git commit -m "refactor(engine): extract generation options + KV budget helpers from inference.py"`.

### Task 4: Stale-patch audit, docs, full suite, PR

**Files:**
- Create: `scripts/check_patch_targets.py` (temporary, not committed) or inline shell
- Modify: `CLAUDE.md` (Project Structure section)

- [ ] **Step 1: Stale-patch audit** — for every string `"olmlx\.(cli|engine\.(inference|model_manager))[.a-zA-Z_0-9]*"` in `tests/`, import the module and `getattr` the dotted tail; every target must resolve. This catches *vacuous* patches (patch applies to a facade name no consumer reads anymore) that tests can't catch by failing.
- [ ] **Step 2: Update CLAUDE.md** project tree: `cli.py` → `cli/` (subcommand modules), add `model_load_utils.py` / `cache_capabilities.py` / `loaded_model.py` / `generation_options.py` / `kv_budget.py` one-liners.
- [ ] **Step 3: Full suite** — `uv run pytest -q`. Expected: same pass/skip counts as the baseline run recorded at plan time.
- [ ] **Step 4: ruff, push, PR** — `git push -u origin worktree-refactor-large-modules`; `gh pr create` with a body explaining the facade strategy, the patch-repoint rule, and the explicit out-of-scope follow-ups.
