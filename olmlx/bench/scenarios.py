"""Benchmark scenario definitions — feature flag combinations to test."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_DEFAULT_HOSTFILE = Path("~/.olmlx/hostfile.json").expanduser()

# Skip functions return None (don't skip) or a reason string (skip).
SkipCheck = Callable[[Path], str | None]


def _no_skip(_model_path: Path) -> str | None:
    return None


def _requires_flash(model_path: Path) -> str | None:
    """Skip if model has no flash_layout.json (not flash-prepared)."""
    flash_dir = model_path / "flash"
    if not flash_dir.exists() or not (flash_dir / "flash_layout.json").exists():
        return f"No flash preparation found at {flash_dir}"
    return None


def _requires_spectral(model_path: Path) -> str | None:
    """Skip if model has no spectral calibration."""
    spectral_dir = model_path / "spectral"
    if (
        not spectral_dir.exists()
        or not (spectral_dir / "spectral_config.json").exists()
        or not (spectral_dir / "calibration.safetensors").exists()
    ):
        return f"No spectral calibration found at {spectral_dir}"
    return None


def _requires_moe(model_path: Path) -> str | None:
    """Skip if model is not a MoE architecture."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return f"No config.json at {model_path}"
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return f"Invalid config.json at {config_path}"
    if not isinstance(config, dict):
        return "config.json is not a JSON object"
    if "text_config" in config:
        config = config["text_config"]
    if not isinstance(config, dict):
        return "text_config is not a JSON object"
    is_moe = (
        (config.get("n_routed_experts") or 0) > 1
        or (config.get("num_local_experts") or 0) > 1
        or (config.get("num_experts") or 0) > 1
    )
    if not is_moe:
        return "Model is not MoE"
    return None


def _requires_flash_moe(model_path: Path) -> str | None:
    """Skip if model is not MoE or has no flash_moe preparation."""
    reason = _requires_moe(model_path)
    if reason is not None:
        return reason
    flash_moe_dir = model_path / "flash_moe"
    if (
        not flash_moe_dir.exists()
        or not (flash_moe_dir / "flash_moe_layout.json").exists()
    ):
        return f"No flash-MoE preparation found at {flash_moe_dir}"
    return None


def _requires_speculative_draft(_model_path: Path) -> str | None:
    """Skip if no draft model is configured for speculative decoding.

    The bench worker inherits the parent environment, so either the
    new ``OLMLX_SPECULATIVE_DRAFT_MODEL`` or the legacy
    ``OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL`` name is enough —
    ``_apply_serve_overrides`` in the worker will forward the legacy
    value during the deprecation window. Accept either so users on
    the old name during migration can still run the scenario.
    """
    if os.environ.get("OLMLX_SPECULATIVE_DRAFT_MODEL") or os.environ.get(
        "OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL"
    ):
        return None
    return (
        "Set OLMLX_SPECULATIVE_DRAFT_MODEL to a draft model HF path "
        "to run this scenario"
    )


def _requires_distributed(_model_path: Path) -> str | None:
    """Skip if no valid hostfile exists for distributed inference."""
    if not _DEFAULT_HOSTFILE.exists():
        return f"No hostfile at {_DEFAULT_HOSTFILE}"
    try:
        hostfile = json.loads(_DEFAULT_HOSTFILE.read_text())
    except (json.JSONDecodeError, OSError):
        return f"Hostfile at {_DEFAULT_HOSTFILE} is invalid"
    hosts = hostfile.get("hosts", [])
    if len(hosts) < 2:
        return "Hostfile has fewer than 2 hosts"
    if not hostfile.get("model"):
        return "Hostfile has no 'model' field"
    return None


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    env_overrides: dict[str, str] = field(default_factory=dict)
    should_skip: SkipCheck = _no_skip
    server_mode: bool = False  # True = launch olmlx serve + hit via HTTP

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "env_overrides": self.env_overrides,
            "server_mode": self.server_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        return cls(
            name=d["name"],
            description=d["description"],
            env_overrides=d.get("env_overrides", {}),
            server_mode=d.get("server_mode", False),
        )


SCENARIOS: list[Scenario] = [
    Scenario(
        name="baseline",
        description="Default settings (prompt cache on, no experimental features)",
    ),
    Scenario(
        name="no-cache",
        description="Prompt cache disabled",
        env_overrides={"OLMLX_PROMPT_CACHE": "false"},
    ),
    Scenario(
        name="turboquant-4",
        description="TurboQuant 4-bit KV cache quantization",
        env_overrides={"OLMLX_KV_CACHE_QUANT": "turboquant:4"},
    ),
    Scenario(
        name="turboquant-2",
        description="TurboQuant 2-bit KV cache quantization",
        env_overrides={"OLMLX_KV_CACHE_QUANT": "turboquant:2"},
    ),
    Scenario(
        name="spectral-4",
        description="SpectralQuant 4-bit KV cache quantization",
        env_overrides={"OLMLX_KV_CACHE_QUANT": "spectral:4"},
        should_skip=_requires_spectral,
    ),
    Scenario(
        name="spectral-2",
        description="SpectralQuant 2-bit KV cache quantization",
        env_overrides={"OLMLX_KV_CACHE_QUANT": "spectral:2"},
        should_skip=_requires_spectral,
    ),
    Scenario(
        name="cache+tq4",
        description="Prompt cache + TurboQuant 4-bit",
        env_overrides={
            "OLMLX_PROMPT_CACHE": "true",
            "OLMLX_KV_CACHE_QUANT": "turboquant:4",
        },
    ),
    Scenario(
        name="flash",
        description="Flash inference (CPU/GPU offloading)",
        env_overrides={"OLMLX_EXPERIMENTAL_FLASH": "true"},
        should_skip=_requires_flash,
    ),
    Scenario(
        name="flash+tq4",
        description="Flash inference + TurboQuant 4-bit",
        env_overrides={
            "OLMLX_EXPERIMENTAL_FLASH": "true",
            "OLMLX_KV_CACHE_QUANT": "turboquant:4",
        },
        should_skip=_requires_flash,
    ),
    Scenario(
        name="speculative",
        description=(
            "Standalone speculative decoding "
            "(set OLMLX_SPECULATIVE_DRAFT_MODEL to a draft model HF path)"
        ),
        env_overrides={"OLMLX_SPECULATIVE": "true"},
        should_skip=_requires_speculative_draft,
    ),
    Scenario(
        name="flash-moe",
        description="Flash MoE expert offloading",
        env_overrides={"OLMLX_EXPERIMENTAL_FLASH_MOE": "true"},
        should_skip=_requires_flash_moe,
    ),
    Scenario(
        name="flash-moe+tq4",
        description="Flash MoE + TurboQuant 4-bit",
        env_overrides={
            "OLMLX_EXPERIMENTAL_FLASH_MOE": "true",
            "OLMLX_KV_CACHE_QUANT": "turboquant:4",
        },
        should_skip=_requires_flash_moe,
    ),
    # Distributed scenarios — require a valid hostfile with 2+ hosts and SSH
    # connectivity. The model used is taken from the hostfile's "model" field,
    # not the --model CLI flag. These launch `olmlx serve` as a real server.
    Scenario(
        name="distributed",
        description="Distributed tensor-parallel inference",
        env_overrides={"OLMLX_EXPERIMENTAL_DISTRIBUTED": "true"},
        should_skip=_requires_distributed,
        server_mode=True,
    ),
    Scenario(
        name="distributed+tq4",
        description="Distributed tensor-parallel + TurboQuant 4-bit",
        env_overrides={
            "OLMLX_EXPERIMENTAL_DISTRIBUTED": "true",
            "OLMLX_KV_CACHE_QUANT": "turboquant:4",
        },
        should_skip=_requires_distributed,
        server_mode=True,
    ),
]


def get_scenarios(names: list[str] | None = None) -> list[Scenario]:
    """Return scenarios filtered by name. Returns all if names is None."""
    if names is None:
        return list(SCENARIOS)
    by_name = {s.name: s for s in SCENARIOS}
    result = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"Unknown scenario {name!r}. Available: {sorted(by_name)}")
        result.append(by_name[name])
    return result
