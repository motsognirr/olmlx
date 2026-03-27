from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "OLMLX_", "env_file": ".env", "extra": "ignore"}

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
    cors_origins: list[str] = ["http://localhost:*", "http://127.0.0.1:*"]
    anthropic_models: dict[str, str] = {}

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

    # Flash inference (LLM in a Flash)
    flash: bool = False
    flash_sparsity_threshold: Annotated[float, Field(gt=0, le=1.0)] = 0.5
    flash_min_active_neurons: Annotated[int, Field(gt=0)] = 128
    flash_max_active_neurons: Annotated[int, Field(gt=0)] | None = None
    flash_window_size: Annotated[int, Field(gt=0)] = 5
    flash_io_threads: Annotated[int, Field(gt=0)] = 32
    flash_cache_budget_neurons: Annotated[int, Field(ge=0)] = 1024
    flash_predictor_rank: Annotated[int, Field(gt=0)] = 128  # prepare-time only
    flash_predictor_sensitive_layers: Annotated[int, Field(ge=0)] = 0
    flash_predictor_sensitive_rank_multiplier: Annotated[int, Field(gt=0)] = 4
    flash_bypass_os_cache: bool = False
    flash_preallocated_buffer: bool = False
    flash_memory_budget_fraction: Annotated[float, Field(gt=0, le=1.0)] | None = None
    flash_speculative: bool = False
    flash_speculative_draft_model: str | None = None
    flash_speculative_tokens: Annotated[int, Field(gt=0)] = 4

    # TurboQuant KV cache quantization (e.g. "turboquant:4", "turboquant:2")
    kv_cache_quant: str | None = None

    @field_validator("kv_cache_quant")
    @classmethod
    def validate_kv_cache_quant(cls, v: str | None) -> str | None:
        if v is None:
            return v
        _VALID_METHODS = {"turboquant"}
        _VALID_BITS = {"2", "4"}
        parts = v.split(":", 1)
        if (
            len(parts) != 2
            or parts[0] not in _VALID_METHODS
            or parts[1] not in _VALID_BITS
        ):
            raise ValueError(
                f"Invalid kv_cache_quant={v!r}. "
                f"Expected '<method>:<bits>' where method is one of {_VALID_METHODS} "
                f"and bits is one of {_VALID_BITS}."
            )
        return v

    # Flash MoE (SSD-based expert offloading for MoE models)
    flash_moe: bool = False
    flash_moe_cache_budget_experts: Annotated[int, Field(ge=0)] = 48  # per layer
    flash_moe_io_threads: Annotated[int, Field(gt=0)] = 32


experimental = ExperimentalSettings()

PRE_SHARDED_DIR_ENV = "OLMLX_EXPERIMENTAL_DISTRIBUTED_PRE_SHARDED_DIR"
