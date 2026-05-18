import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

#: Metal sync behavior at inference-lock boundaries. Single source of truth
#: shared by ``Settings.sync_mode``, ``ModelConfig.sync_mode``, and
#: ``_lock_boundary_sync`` — keep them in lockstep.
SyncMode = Literal["full", "minimal", "none"]


def validate_kv_cache_quant_format(v: str | None) -> str | None:
    """Validate that *v* is a valid ``kv_cache_quant`` value.

    Called by both the Pydantic field validator and the per-model config
    path (``ModelConfig.from_entry``) so bad values are caught at config
    load time regardless of whether they come from an env var or a
    ``models.json`` entry.
    """
    if v is None:
        return v
    _VALID_METHODS = {"turboquant", "spectral"}
    _VALID_BITS = {"2", "4"}
    parts = v.split(":", 1)
    if len(parts) != 2 or parts[0] not in _VALID_METHODS or parts[1] not in _VALID_BITS:
        raise ValueError(
            f"Invalid kv_cache_quant={v!r}. "
            f"Expected '<method>:<bits>' where method is one of {_VALID_METHODS} "
            f"and bits is one of {_VALID_BITS}."
        )
    return v


class Settings(BaseSettings):
    # Note: ``validate_assignment=True`` applies to *all* fields, not just
    # the new speculative ones. Programmatic writes that previously
    # silently set out-of-range values (e.g. ``settings.port = 0``) now
    # raise ``ValidationError``. This is intentional — tests and CLI
    # overrides should not be able to construct invalid Settings — but
    # it does broaden the surface beyond the immediate use case.
    model_config = {
        "env_prefix": "OLMLX_",
        "env_file": ".env",
        "extra": "ignore",
        "validate_assignment": True,
    }

    host: str = "0.0.0.0"
    port: Annotated[int, Field(ge=1, le=65535)] = 11434
    models_dir: Path = Path.home() / ".olmlx" / "models"
    models_config: Path = Path.home() / ".olmlx" / "models.json"
    default_keep_alive: str = "5m"
    max_loaded_models: int = 1
    memory_limit_fraction: Annotated[float, Field(gt=0, le=1.0)] = 0.75
    model_load_timeout: Annotated[float, Field(gt=0)] | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    prompt_cache: bool = True
    prompt_cache_max_tokens: Annotated[int, Field(gt=0)] | None = 32768
    prompt_cache_max_slots: Annotated[int, Field(gt=0)] = 4
    prompt_cache_disk: bool = False
    prompt_cache_disk_path: Path = Path.home() / ".olmlx" / "cache" / "kv"
    prompt_cache_disk_max_gb: Annotated[float, Field(gt=0)] = 10.0
    inference_queue_timeout: Annotated[float, Field(gt=0)] | None = 300.0
    inference_timeout: Annotated[float, Field(gt=0)] | None = None
    sync_mode: SyncMode = "full"
    max_tokens_limit: Annotated[int, Field(gt=0)] = 131072
    cors_origins: list[str] = ["http://localhost:*", "http://127.0.0.1:*"]
    anthropic_models: dict[str, str] = {}

    # KV cache quantization (TurboQuant or SpectralQuant).
    # Format: "<method>:<bits>" where method ∈ {turboquant, spectral} and
    # bits ∈ {2, 4}. Per-model overrides live on ``ModelConfig`` in
    # ``olmlx.engine.registry``.
    kv_cache_quant: str | None = None

    # Auto-run spectral calibration when spectral quant is configured but
    # calibration data is missing. Uses the default calibration dataset
    # (c4) and 64 samples. Set to ``true`` to avoid manual ``olmlx spectral
    # prepare <model>`` on first load.
    kv_cache_auto_calibrate: bool = False

    # Distributed inference — split models across Apple Silicon machines.
    # Configured via hostfile (``distributed_hostfile``), launched by
    # ``olmlx serve``.
    distributed: bool = False
    distributed_strategy: Literal["tensor", "pipeline"] = "tensor"
    distributed_hostfile: Path = Path("~/.olmlx/hostfile.json")
    distributed_backend: str = "ring"
    distributed_port: int = 32323
    distributed_sideband_port: int = 32400
    distributed_secret: str = ""
    distributed_remote_working_dir: str = ""
    distributed_remote_python: str = "python"
    distributed_pre_shard: bool = True
    distributed_shard_dir: Path = Path("~/.olmlx/shards")
    distributed_worker_shard_dir: str = "~/.olmlx/shards"

    # Speculative decoding (works with any model, not just Flash).
    # Per-model overrides live on ``ModelConfig`` in ``olmlx.engine.registry``.
    # ``min_length=1`` rejects ``OLMLX_SPECULATIVE_DRAFT_MODEL=""`` at parse
    # time so the load process doesn't surface a misleading "draft not set"
    # error for an empty string.
    #
    # ``speculative_strategy`` selects the algorithm:
    # - ``classic``: standalone draft LM (autoregressive lambda candidates).
    # - ``dflash``: block-diffusion draft conditioned on target hidden states
    #   (mask-token parallel block prediction). Requires a draft model whose
    #   ``config.json`` carries the upstream z-lab ``dflash_config`` schema.
    # - ``eagle``: autoregressive draft head conditioned on the target's
    #   last-layer hidden state (arxiv 2401.15077). Predicts the next
    #   *hidden* in feature space; the target's lm_head (shared via
    #   ``bind()``) maps that hidden to the next-token distribution. Higher
    #   acceptance rate than DFlash for the same draft-parameter budget on
    #   most targets, at the cost of being autoregressive (one draft
    #   forward per drafted token). Requires a draft whose ``config.json``
    #   carries an ``eagle_config`` block.
    # ``speculative_tokens`` is reused as DFlash's ``block_size`` and as
    # EAGLE's per-verify draft-token count. ``None`` means "use the
    # strategy default": 4 for classic, the draft model's pre-trained
    # ``block_size`` for DFlash and EAGLE.
    speculative: bool = False
    speculative_strategy: Literal["classic", "dflash", "eagle"] = "classic"
    speculative_draft_model: Annotated[str, Field(min_length=1)] | None = None
    speculative_tokens: Annotated[int, Field(gt=0)] | None = None

    # Flash inference (LLM in a Flash). Primary, user-facing knobs.
    # Advanced tuning (window size, IO threads, cache budget, predictor
    # rank, buffer modes) lives on ``ExperimentalSettings`` and the
    # ``olmlx flash prepare`` CLI — these five fields are the ones a
    # typical user needs to touch. Per-model overrides live on
    # ``ModelConfig`` in ``olmlx.engine.registry``.
    flash: bool = False
    flash_sparsity_threshold: Annotated[float, Field(gt=0, le=1.0)] = 0.5
    flash_min_active_neurons: Annotated[int, Field(gt=0)] = 128
    flash_max_active_neurons: Annotated[int, Field(gt=0)] | None = None
    flash_memory_budget_fraction: Annotated[float, Field(gt=0, le=1.0)] | None = None

    # Flash MoE (SSD-based expert offloading for MoE models)
    flash_moe: bool = False
    flash_moe_cache_budget_experts: Annotated[int, Field(ge=0)] = 48
    flash_moe_io_threads: Annotated[int, Field(gt=0)] = 32

    @model_validator(mode="after")
    def validate_auto_calibrate(self) -> "Settings":
        if self.kv_cache_auto_calibrate and (
            self.kv_cache_quant is None
            or not self.kv_cache_quant.startswith("spectral:")
        ):
            raise ValueError(
                "OLMLX_KV_CACHE_AUTO_CALIBRATE=true requires "
                "OLMLX_KV_CACHE_QUANT=spectral:<bits>"
            )
        return self

    @model_validator(mode="after")
    def validate_flash_neuron_range(self) -> "Settings":
        # Cross-field check: an inverted min/max would produce a silently
        # broken ``FlashConfig`` at runtime (FlashMLP clamps each token's
        # active neurons into ``[min, max]``, and a min > max collapses
        # the interval). Fail at config load instead.
        if (
            self.flash_max_active_neurons is not None
            and self.flash_min_active_neurons > self.flash_max_active_neurons
        ):
            raise ValueError(
                f"flash_min_active_neurons ({self.flash_min_active_neurons}) "
                f"must be <= flash_max_active_neurons "
                f"({self.flash_max_active_neurons})"
            )
        return self

    @field_validator("kv_cache_quant")
    @classmethod
    def validate_kv_cache_quant(cls, v: str | None) -> str | None:
        return validate_kv_cache_quant_format(v)

    @field_validator("speculative_draft_model")
    @classmethod
    def validate_speculative_draft_model(cls, v: str | None) -> str | None:
        # ``Field(min_length=1)`` already rejects ``""``, but a
        # whitespace-only value (``"   "``) has length > 0 and would
        # otherwise reach the load path and surface as a misleading
        # "draft not set" error. Strip and reject empty here.
        if v is None:
            return v
        stripped = v.strip()
        if not stripped:
            raise ValueError(
                "speculative_draft_model must be a non-empty HuggingFace path"
            )
        return stripped

    @field_validator("anthropic_models")
    @classmethod
    def validate_anthropic_model_keys(cls, v: dict[str, str]) -> dict[str, str]:
        for key in v:
            if "-" in key or ":" in key:
                raise ValueError(
                    f"anthropic_models key {key!r} must be a single segment "
                    "(no dashes or colons)"
                )
        return v


settings = Settings()


class ExperimentalSettings(BaseSettings):
    model_config = {
        "env_prefix": "OLMLX_EXPERIMENTAL_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # Flash inference — advanced/tuning knobs. The primary user-facing
    # fields (``flash``, ``flash_sparsity_threshold``,
    # ``flash_min_active_neurons``, ``flash_max_active_neurons``,
    # ``flash_memory_budget_fraction``) were promoted to ``Settings``;
    # what remains here is rarely-touched tuning that most users should
    # not need to set. Per-model overrides for advanced knobs still go
    # under the ``experimental`` block in models.json. (``distributed_*``
    # was promoted to ``Settings`` in #326 and removed from here too.)
    flash_window_size: Annotated[int, Field(gt=0)] = 5
    flash_io_threads: Annotated[int, Field(gt=0)] = 32
    flash_cache_budget_neurons: Annotated[int, Field(ge=0)] = 1024
    flash_predictor_rank: Annotated[int, Field(gt=0)] = 128  # prepare-time only
    flash_predictor_sensitive_layers: Annotated[int, Field(ge=0)] = 0
    flash_predictor_sensitive_rank_multiplier: Annotated[int, Field(gt=0)] = 4
    flash_bypass_os_cache: bool = False
    flash_preallocated_buffer: bool = False
    flash_prefetch: bool = False
    flash_prefetch_confidence_threshold: Annotated[float, Field(gt=0, le=1.0)] = 0.3
    flash_prefetch_min_neurons: Annotated[int, Field(gt=0)] = 64
    flash_prefetch_max_neurons: Annotated[int, Field(gt=0)] | None = None
    flash_prefetch_io_threads: Annotated[int, Field(gt=0)] = 16
    flash_speculative: bool = False
    flash_speculative_draft_model: str | None = None
    flash_speculative_tokens: Annotated[int, Field(gt=0)] = 4


experimental = ExperimentalSettings()

PRE_SHARDED_DIR_ENV = "OLMLX_DISTRIBUTED_PRE_SHARDED_DIR"


#: Legacy ``OLMLX_EXPERIMENTAL_FLASH*`` env vars and their parsers.
#: Lives here rather than in ``olmlx.cli`` so the distributed-worker
#: entry point (which must not import argparse/uvicorn) can apply the
#: same one-release deprecation shim that ``cmd_serve`` / ``cmd_chat`` /
#: ``cmd_config_show`` / ``cmd_flash_info`` apply on the CLI side.
LEGACY_FLASH_FORWARD: tuple[tuple[str, str, str, Callable[[str], Any]], ...] = (
    (
        "OLMLX_EXPERIMENTAL_FLASH",
        "OLMLX_FLASH",
        "flash",
        lambda v: v.strip().lower() in ("1", "true", "yes", "on"),
    ),
    (
        "OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD",
        "OLMLX_FLASH_SPARSITY_THRESHOLD",
        "flash_sparsity_threshold",
        float,
    ),
    (
        "OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS",
        "OLMLX_FLASH_MIN_ACTIVE_NEURONS",
        "flash_min_active_neurons",
        int,
    ),
    (
        "OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS",
        "OLMLX_FLASH_MAX_ACTIVE_NEURONS",
        "flash_max_active_neurons",
        int,
    ),
    (
        "OLMLX_EXPERIMENTAL_FLASH_MEMORY_BUDGET_FRACTION",
        "OLMLX_FLASH_MEMORY_BUDGET_FRACTION",
        "flash_memory_budget_fraction",
        float,
    ),
)


def surface_legacy_flash_env() -> None:
    """Detect and forward legacy ``OLMLX_EXPERIMENTAL_FLASH*`` (primary
    knobs only) to the new ``OLMLX_FLASH*`` names.

    Only the five promoted primary knobs are forwarded. Advanced tuning
    fields (``OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE``,
    ``..._IO_THREADS``, ``..._CACHE_BUDGET_NEURONS``,
    ``..._BYPASS_OS_CACHE``, ``..._PREALLOCATED_BUFFER``,
    ``..._PREDICTOR_*``, ``..._PREFETCH*``, ``..._SPECULATIVE*``,
    ``..._MOE*``) remain under the experimental prefix and pass through
    unchanged.

    Updates ``settings.<field>`` in-process but does not write back to
    ``os.environ``. All callers in this codebase consume the promoted
    knobs via ``settings.*`` (the distributed-worker env-forwarding loop
    in ``_launch_distributed_workers`` mirrors ``settings.*`` into the
    worker's ``OLMLX_FLASH*`` env vars). Anything that reads
    ``os.environ.get("OLMLX_FLASH*")`` directly will miss the legacy
    value — mirror through ``settings`` instead.

    Defined in ``olmlx.config`` rather than ``olmlx.cli`` so the
    distributed-worker entry point can reuse the same logic without
    pulling in argparse/uvicorn.
    """
    # Collect per-field actions: only consider a legacy var "pending"
    # if its *parsed* value would actually change the live Settings.
    # ``OLMLX_EXPERIMENTAL_FLASH=false`` (a user explicitly disabling
    # flash via the old name) parses to the schema default ``False``
    # and has nothing to migrate — skipping it here suppresses a
    # noisy warning that would otherwise nag every invocation for a
    # variable whose only effect is "leave the default".
    pending: list[tuple[str, str, str, Any]] = []
    for legacy, new, attr, parse in LEGACY_FLASH_FORWARD:
        legacy_val = os.environ.get(legacy)
        if legacy_val is None:
            continue
        # ``os.environ`` only — pydantic-settings reads ``.env`` directly
        # into the model fields without writing to the shell env, so a
        # ``.env`` line like ``OLMLX_FLASH=false`` (explicit default)
        # combined with a legacy shell var would slip past this guard.
        # The ``getattr != field_default`` check below catches the
        # non-default case; the only remaining blind spot is a ``.env``
        # value identical to the schema default. Acceptable during the
        # one-release deprecation window.
        if os.environ.get(new) is not None:
            continue
        field_default = Settings.model_fields[attr].default
        if getattr(settings, attr) != field_default:
            # ``.env`` (or programmatic write at import time) already
            # supplied a non-default value; don't let the legacy var
            # clobber it.
            continue
        try:
            value = parse(legacy_val)
        except Exception as exc:
            logger.warning(
                "Could not forward legacy env var %s=%r to %s: %s",
                legacy,
                legacy_val,
                new,
                exc,
            )
            continue
        if value == field_default:
            # Parsed value already matches the schema default — no
            # behavioural change, nothing to migrate.
            continue
        pending.append((legacy, new, attr, value))

    if not pending:
        return

    # Pre-validate cross-field constraints by combining the *pending*
    # set with the *live* Settings for the non-pending bound. Without
    # this, three failure modes would slip through to the per-field
    # setattr — caught only by the generic "Could not forward" log,
    # with the migration banner suppressed so the user never sees the
    # rename nudge:
    #   (a) pending min + pending max, inverted (round-7 case).
    #   (b) pending min only, but live max (e.g. from ``.env``) is
    #       below the pending min.
    #   (c) pending max only, but live min is above the pending max.
    # All three are handled the same way: drop both flash neuron-range
    # entries from ``pending`` and emit one explicit warning naming
    # the effective pair.
    pending_attrs = {attr: value for _, _, attr, value in pending}
    pending_min = pending_attrs.get("flash_min_active_neurons")
    pending_max = pending_attrs.get("flash_max_active_neurons")
    effective_min = (
        pending_min if pending_min is not None else settings.flash_min_active_neurons
    )
    effective_max = (
        pending_max if pending_max is not None else settings.flash_max_active_neurons
    )
    has_neuron_pending = (
        "flash_min_active_neurons" in pending_attrs
        or "flash_max_active_neurons" in pending_attrs
    )
    if (
        has_neuron_pending
        and effective_max is not None
        and effective_min > effective_max
    ):
        logger.warning(
            "Refusing to forward legacy flash neuron-range values: the "
            "resulting pair (min=%r, max=%r) would have min > max. "
            "Dropping the legacy flash_min/flash_max forwards (any "
            "partial apply would leave one bound unset and silently "
            "remove the user's intended ceiling/floor). Rename to "
            "OLMLX_FLASH_MIN_ACTIVE_NEURONS / "
            "OLMLX_FLASH_MAX_ACTIVE_NEURONS with a consistent pair.",
            effective_min,
            effective_max,
        )
        pending = [
            (legacy, new, attr, value)
            for legacy, new, attr, value in pending
            if attr not in ("flash_min_active_neurons", "flash_max_active_neurons")
        ]
        if not pending:
            return

    # Apply each pending value; banner names only the vars that
    # actually landed in Settings. A legacy value rejected by a
    # single-field Pydantic validator (e.g. ``Field(gt=0)``) should
    # not be listed as "rename this" — the legacy value was never
    # honoured, so renaming it would just hit the same validator
    # again.
    applied: list[str] = []
    for legacy, new, attr, value in pending:
        try:
            setattr(settings, attr, value)
        except Exception as exc:
            logger.warning(
                "Could not forward legacy env var %s to %s: %s",
                legacy,
                new,
                exc,
            )
            continue
        applied.append(legacy)
    if not applied:
        return
    logger.warning(
        "Deprecated env vars detected: %s. The Flash primary knobs have "
        "been promoted out of the experimental prefix — rename to "
        "OLMLX_FLASH, OLMLX_FLASH_SPARSITY_THRESHOLD, "
        "OLMLX_FLASH_MIN_ACTIVE_NEURONS, OLMLX_FLASH_MAX_ACTIVE_NEURONS, "
        "OLMLX_FLASH_MEMORY_BUDGET_FRACTION. The legacy names will be "
        "removed in a future release. Advanced flash tuning fields "
        "(window_size, io_threads, cache_budget_neurons, predictor_*, "
        "prefetch_*, etc.) remain under OLMLX_EXPERIMENTAL_FLASH_*.",
        ", ".join(applied),
    )


def resolve_experimental(
    base: ExperimentalSettings,
    overrides: dict,
) -> ExperimentalSettings:
    """Create a new ExperimentalSettings with per-model overrides applied.

    Fields not present in *overrides* retain their value from *base*.
    Returns *base* unchanged if overrides is empty.

    Uses the pydantic core validator directly to avoid pydantic-settings
    re-reading ``OLMLX_EXPERIMENTAL_*`` env vars, which would produce
    confusing errors if an unrelated env var is malformed.
    """
    if not overrides:
        return base
    merged = base.model_dump()
    merged.update(overrides)
    # Validate field constraints and custom validators without triggering
    # BaseSettings env var resolution.  __pydantic_validator__ is the core
    # schema validator shared by __init__ and model_validate; calling it
    # directly bypasses _settings_build_values().
    return ExperimentalSettings.__pydantic_validator__.validate_python(merged)


@dataclass
class FlashMoeConfig:
    """Resolved Flash-MoE configuration: per-model overrides global Settings."""

    enabled: bool
    cache_budget_experts: int
    io_threads: int
