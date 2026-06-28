# Config legacy flash env-var cleanup

## Problem

`olmlx/config.py` (1097 lines) carries ~420 lines of legacy env-var
*forwarding* machinery for three families of promoted flash knobs. Each
family was promoted out of the `OLMLX_EXPERIMENTAL_FLASH*` prefix to a
bare `OLMLX_FLASH*` name during a "one-release deprecation window":

- Flash primary knobs — PR #274/#327
- Flash-MoE — PR #277/#329
- Flash prefetch + speculative — PR #275/#276/#336

HEAD is at #593; the one-release window passed ~260 PRs ago. The
forwarding code is three near-duplicate implementations of the same idea
("if the old name is set and the new one isn't, parse and apply it"),
each with its own table, its own clobber-precedence rules, and — for the
first family — a bespoke cross-field neuron-range pre-validation pass. A
hand-rolled `.env` parser (`_legacy_values_in_dotenv`) backs two of them.

The three functions are called as a trio from 7 sites in `olmlx/cli.py`
and once in `olmlx/engine/distributed_worker.py`, and have ~39 test
references in `tests/test_cli.py`.

## Decision

Stop *forwarding* legacy flash env vars. Replace all three shims with a
single **warn-only** check: detect any of the 12 promoted legacy names
(in the shell environment **or** the project `.env`) and emit one
warning telling the user the new name. The legacy value is **not**
applied — this is an intentional breaking change, made safe by the
loud warning and by olmlx's single-user / localhost usage context.

This removes all parsing, clobber-precedence, and cross-field
pre-validation logic, since nothing is applied to `Settings` anymore.

## Scope

In scope (all in `olmlx/config.py` plus its callers):

- The three flash families only: `flash`, `flash_moe`,
  `flash_prefetch`/`flash_speculative`.

Out of scope:

- The sibling legacy shims that live in `olmlx/cli.py`
  (`_surface_legacy_speculative_env`, `_surface_legacy_dflash_env`,
  `_surface_legacy_kv_cache_quant`). They are not in `config.py` and are
  a separate cleanup.
- `PRE_SHARDED_DIR_ENV`, `resolve_experimental`, `FlashMoeConfig`,
  `ExperimentalSettings`, and the `Settings` class body — unchanged.

## What is removed from `config.py`

- `LEGACY_FLASH_FORWARD`
- `surface_legacy_flash_env` (incl. its `pending`-set logic and the
  cross-field flash-neuron-range pre-validation block)
- `_DEPRECATED_FLASH_MOE_ENV_VARS`, `_LEGACY_FLASH_MOE_FORWARD`,
  `_forward_legacy_flash_moe_env`, `surface_legacy_flash_moe_env`
- `_DEPRECATED_FLASH_PREFETCH_SPECULATIVE_ENV_VARS`,
  `_LEGACY_FLASH_PREFETCH_SPECULATIVE_FORWARD`,
  `surface_legacy_flash_prefetch_speculative_env`

## What is added to `config.py`

One rename table and one function. `_legacy_values_in_dotenv` is
**retained** (trimmed if any forwarding-only behavior is unused) so the
warning fires for vars set in `.env`, not just the shell.

```python
#: Promoted flash env vars: legacy OLMLX_EXPERIMENTAL_FLASH* name -> the
#: bare OLMLX_FLASH* name that replaced it. These are no longer honored;
#: warn_legacy_flash_env() detects and warns, but does not forward them.
#: Only PROMOTED names appear here — the still-valid experimental *tuning*
#: knobs (OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE, ..._PREDICTOR_*,
#: ..._PREFETCH_CONFIDENCE_THRESHOLD, ...) are intentionally absent.
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
    """Warn that promoted ``OLMLX_EXPERIMENTAL_FLASH*`` env vars are no
    longer honored, naming the new ``OLMLX_FLASH*`` replacements.

    Detects names set in the shell environment OR the project ``.env``.
    Does NOT forward the value — these knobs were promoted out of the
    experimental prefix several releases ago (PRs #327/#329/#336) and the
    deprecation window has closed. Lives in ``olmlx.config`` so the
    distributed-worker entry point can reuse it without importing
    ``olmlx.cli`` (argparse/uvicorn).
    """
    dotenv_keys = _legacy_values_in_dotenv(tuple(PROMOTED_FLASH_ENV_RENAMES))
    detected = sorted(
        old
        for old in PROMOTED_FLASH_ENV_RENAMES
        if old in os.environ or old in dotenv_keys
    )
    if not detected:
        return
    renames = "; ".join(f"{old} -> {PROMOTED_FLASH_ENV_RENAMES[old]}" for old in detected)
    logger.warning(
        "Unsupported env vars detected and IGNORED: %s. These were renamed "
        "out of the experimental prefix and are no longer honored: %s. "
        "Update your shell env / .env to the new names.",
        ", ".join(detected),
        renames,
    )
```

Exact final form of the warning string and whether `_legacy_values_in_dotenv`
needs any trimming are implementation details for the plan.

## Call-site changes

Replace the three-call trio with a single `warn_legacy_flash_env()` at
each existing site — no consolidation, to preserve current per-command
warn behavior and keep the change low-risk:

- `olmlx/cli.py`: update the import block (lines ~23-25) to import the
  one new name; replace each of the 7 trio call sites
  (~600-602, ~2054-2056, ~2451-2453, ~2488-2490, ~2690-2692,
  ~3042-3044) with one call. Drop the two now-deleted imports.
- `olmlx/engine/distributed_worker.py`: update the import (~186-188)
  and the trio call (~199-201) to the single function; refresh the
  surrounding comment (the "one-release deprecation window" wording no
  longer applies — it now warns, not forwards).

## Test changes (`tests/test_cli.py`)

The ~39 references fall into three buckets:

1. **Forwarding-applied assertions** (value landed in `settings.*`):
   rewrite to assert the new contract — `settings.<field>` keeps its
   default (NOT forwarded) AND a warning naming the legacy var is
   emitted.
2. **Forwarding-machinery-specific tests** (clobber precedence /
   new-var-wins / inverted-neuron-range-drops-both / "value swallowed by
   validator"): delete — that machinery no longer exists.
3. **`.env`-detection tests**: keep the detection path but rewrite
   expectations to "warned, not applied."

The `test_cmd_*` tests that only assert the trio is *called* at the top
of each subcommand get updated to assert the single function is called.

No new public behavior beyond the warning, so no new test file — changes
stay in `tests/test_cli.py`.

## Risks

- **Breaking change**: a user still on `OLMLX_EXPERIMENTAL_FLASH*` loses
  the setting. Mitigated by the loud per-invocation warning and the
  single-user/localhost context. Accepted.
- **Warning noise**: fires on every CLI invocation while the stale var
  is set. This is intended (it nags until the user renames) and matches
  the prior shims' per-invocation warning cadence.

## Verification

- `uv run pytest tests/test_cli.py` green.
- `ruff check` + `ruff format` clean.
- Manual: `OLMLX_EXPERIMENTAL_FLASH=true uv run olmlx config show` emits
  the warning and does NOT enable flash; `OLMLX_FLASH=true` still works.
- Update `CLAUDE.md` only if a documented invariant changes (none
  expected — this is a deletion, not a new design rule).
