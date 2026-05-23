# Promote Flash prefetch (#275) + Flash-speculative (#276); resolve pipeline decision (#273)

Date: 2026-05-22
Issues: #275, #276, partial #273 (tracked by #279)

## Goal

Promote two experimental features from the `OLMLX_EXPERIMENTAL_*` surface to the
supported `OLMLX_*` surface, and resolve the `distributed_strategy` pipeline
decision from #273. Single PR.

Scope decisions (confirmed with maintainer):

- **Code + docs + bench-scenario definitions now**; hardware-measurement
  checkboxes (prefetch hit-rate across 3+ models, tokens/sec uplift, adaptive
  windowing stress) are deferred to the maintainer to run separately.
- **#273**: do only the pipeline-strategy decision in this PR — narrow the
  `distributed_strategy` literal to tensor-only and document it. Defer the
  setup wizard and error-UX tasks to a separate PR. (#273's env-var promotion
  was already done in #326.)
- The dead `pipeline.py` / `pre_shard_pipeline*` / worker pipeline branches stay
  **in place** (unreachable), not deleted — a full removal is a larger, riskier
  diff for a later PR.

## Non-goals

- Running benchmarks or publishing measured numbers.
- The `olmlx distributed init` wizard, distributed error-message quality,
  sideband retry progress (deferred #273 tasks).
- Deleting the dormant pipeline implementation modules.

## What gets promoted (config surface)

Follows the Flash-dense precedent (PR #329): promote the primary user-facing
toggle(s), keep fine-tuning knobs experimental.

**#276 Flash + speculative** — promote all three (mirrors standalone
`speculative` which promoted toggle + draft_model + tokens):

| Field | New location | Default |
|-------|-------------|---------|
| `flash_speculative` | `Settings` | `False` |
| `flash_speculative_draft_model` | `Settings` | `None` |
| `flash_speculative_tokens` | `Settings` | `4` |

**#275 Flash prefetch** — promote only the toggle; the 4 tuning knobs stay
experimental/advanced (as Flash-dense kept `window_size`/`io_threads`/
`cache_budget`):

| Field | New location | Default |
|-------|-------------|---------|
| `flash_prefetch` | `Settings` | `False` |
| `flash_prefetch_confidence_threshold` | **stays** `ExperimentalSettings` | `0.3` |
| `flash_prefetch_min_neurons` | **stays** `ExperimentalSettings` | `64` |
| `flash_prefetch_max_neurons` | **stays** `ExperimentalSettings` | `None` |
| `flash_prefetch_io_threads` | **stays** `ExperimentalSettings` | `16` |

`flash_speculative_draft_model` gets the same `Field(min_length=1)` +
strip-and-reject-whitespace validator as `speculative_draft_model`.

## Migration & registry wiring

For the 4 newly-promoted keys (`flash_prefetch`, `flash_speculative`,
`flash_speculative_draft_model`, `flash_speculative_tokens`):

1. **`config.py`**: move the 4 fields from `ExperimentalSettings` → `Settings`.
   Keep the 4 prefetch tuning fields in `ExperimentalSettings`.
2. **`registry.py`**:
   - Remove the 4 promoted keys from `PER_MODEL_EXPERIMENTAL_KEYS`.
   - Add them (self-mapped) to `PROMOTED_EXPERIMENTAL_KEYS`. The existing
     `_validate_experimental_overrides` then raises the standard "promoted out
     of 'experimental'… move to top level" migration error automatically if a
     `models.json` still nests them under `experimental`.
   - Keep the 4 prefetch tuning keys in `PER_MODEL_EXPERIMENTAL_KEYS`.
   - Add the 4 promoted fields to `ModelConfig` as top-level per-model
     overrides + resolution (mirroring how `flash`/`flash_sparsity_threshold`
     resolve), so `models.json` top-level `flash_speculative: true` works
     per-model.
3. **Env-var shim** (`config.py`): add
   `surface_legacy_flash_prefetch_speculative_env()` alongside
   `surface_legacy_flash_env` / `surface_legacy_flash_moe_env`. Forwards:
   - `OLMLX_EXPERIMENTAL_FLASH_PREFETCH` → `OLMLX_FLASH_PREFETCH`
   - `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE` → `OLMLX_FLASH_SPECULATIVE`
   - `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL` → `OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL`
   - `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS` → `OLMLX_FLASH_SPECULATIVE_TOKENS`

   With a one-release deprecation warning, and the same "explicit
   `false` is honoured / new var wins if both set" semantics as the existing
   shims. The 4 prefetch tuning env vars keep their `OLMLX_EXPERIMENTAL_`
   prefix. Wired into the same call sites as the existing shims:
   `cli.py` (`cmd_serve` ~594, plus 1775/2127/2641) and
   `distributed_worker.py` (~198).
4. **Read-site switch**:
   - `model_manager.py`: the promoted fields move from `model_exp.flash_*` to
     the resolved per-model flash config object (same object that serves
     `sparsity_threshold`). The 4 prefetch tuning fields keep reading from
     `model_exp`. Sites: ~2832 (`prefetch=`), ~2850 (lookahead load), ~2865+
     (`flash_speculative*`), ~3061/3121/3144 (incompat guards).
   - `cli.py:2379`: `train_lookahead=experimental.flash_prefetch` →
     `settings.flash_prefetch`.

## CLI flags on `olmlx serve`

Mirror existing `--flash` / `--speculative` flags:

- `--flash-speculative` (store_true → `OLMLX_FLASH_SPECULATIVE`)
- `--flash-speculative-draft-model <hf-path>`
- `--flash-speculative-tokens <N>`
- `--flash-prefetch` (store_true)

## #273 pipeline decision (tensor-only)

- `config.py`: `distributed_strategy: Literal["tensor", "pipeline"]` →
  `Literal["tensor"]`. Config now rejects `pipeline` at parse time.
- `cli.py:~1240`: hostfile-strategy guard rejects `pipeline` with an actionable
  "distributed inference is tensor-only" message.
- Dormant `pipeline.py` / `pre_shard_pipeline*` / `distributed_worker.py`
  pipeline branches / `model_manager.py` pipeline branch: left in place,
  unreachable. One-line note in CLAUDE.md.

## Docs

- **`README.md`**: replace the 4 `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE*` /
  `_PREFETCH` env-var rows with promoted names; add a short "when to use"
  decision matrix (dense vs dense+spec vs flash vs flash+spec); Path A vs
  Path B prefetch explanation; `LookaheadBank` opt-in note (trained when
  prefetch enabled during `olmlx flash prepare`); migration notes for renamed
  env vars; document distributed as tensor-only.
- **`docs/USER_MANUAL.md`**: same env-var renames.
- **`CLAUDE.md`**: drop "experimental" wording from the Flash prefetch &
  Flash+speculative Key Design Decisions entries; note tensor-only distributed
  + dormant pipeline code.

## Bench

`olmlx/bench/scenarios.py`: add `flash+spec` and `flash+prefetch` scenario
*definitions* with `env_overrides` and `_requires_flash` /
`_requires_speculative_draft` skip guards. Not run in this PR.

## Testing (TDD)

Write failing tests first, then implement:

1. **Config promotion**: `OLMLX_FLASH_SPECULATIVE=true` / `OLMLX_FLASH_PREFETCH=true`
   set the corresponding `Settings` fields. `OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL=""`
   and whitespace-only are rejected.
2. **Legacy shim**: `OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE=true` (with new var
   unset) forwards to the new var + warns; explicit new var wins when both set;
   `OLMLX_EXPERIMENTAL_FLASH_PREFETCH` forwards. Prefetch tuning env vars are
   untouched.
3. **Per-model migration**: a `models.json` `experimental` block containing any
   of the 4 promoted keys raises the promotion migration error; the 4 prefetch
   tuning keys are still accepted under `experimental`; the 4 promoted keys are
   accepted at the top level of a `ModelConfig` entry.
4. **Pipeline removal**: `OLMLX_DISTRIBUTED_STRATEGY=pipeline` raises a
   validation error; a hostfile with `strategy: "pipeline"` is rejected with the
   tensor-only message.
5. **CLI flags**: `olmlx serve --flash-speculative --flash-prefetch …` set the
   right env vars before app startup (follow the existing `--flash` flag test
   pattern).

## Risks

- Mixed-source resolution in the `FlashConfig` builder (promoted `prefetch` from
  ModelConfig, tuning from `model_exp`) — already the pattern for flash-dense,
  low risk but verify the resolved object is consistent.
- Whatever consumes `distributed_strategy` downstream must not assume the
  `pipeline` literal is still a valid value — the dormant branches reference the
  string literal, not the type, so they compile fine; just unreachable.
