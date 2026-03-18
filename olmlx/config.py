from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "OLMLX_", "env_file": ".env"}

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
    model_config = {"env_prefix": "OLMLX_EXPERIMENTAL_", "env_file": ".env"}

    distributed: bool = False
    distributed_hostfile: Path = Path("~/.olmlx/hostfile.json")
    distributed_backend: str = "ring"
    distributed_port: int = 32323
    distributed_sideband_port: int = 32400
    distributed_secret: str = ""


experimental = ExperimentalSettings()
