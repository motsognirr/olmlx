"""Benchmark scenario definitions — feature flag combinations to test."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_DEFAULT_HOSTFILE = Path("~/.olmlx/hostfile.json").expanduser()


def _no_skip(_model_path: Path) -> bool:
    return False


def _requires_flash(model_path: Path) -> bool:
    """Skip if model has no flash_layout.json (not flash-prepared)."""
    flash_dir = model_path / "flash"
    if not flash_dir.exists() or not (flash_dir / "flash_layout.json").exists():
        logger.info("Skipping: no flash preparation found at %s", flash_dir)
        return True
    return False


def _requires_moe(model_path: Path) -> bool:
    """Skip if model is not a MoE architecture."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        logger.info("Skipping: no config.json at %s", model_path)
        return True
    config = json.loads(config_path.read_text())
    if "text_config" in config:
        config = config["text_config"]
    is_moe = (
        (config.get("n_routed_experts") or 0) > 1
        or (config.get("num_local_experts") or 0) > 1
        or (config.get("num_experts") or 0) > 1
    )
    if not is_moe:
        logger.info("Skipping: model is not MoE")
        return True
    return False


def _requires_flash_and_moe(model_path: Path) -> bool:
    return _requires_flash(model_path) or _requires_moe(model_path)


def _requires_flash_or_moe(model_path: Path) -> bool:
    """Skip only if model has neither flash preparation nor MoE architecture."""
    return _requires_flash(model_path) and _requires_moe(model_path)


def _requires_distributed(_model_path: Path) -> bool:
    """Skip if no valid hostfile exists for distributed inference."""
    if not _DEFAULT_HOSTFILE.exists():
        logger.info("Skipping: no hostfile at %s", _DEFAULT_HOSTFILE)
        return True
    try:
        hostfile = json.loads(_DEFAULT_HOSTFILE.read_text())
    except (json.JSONDecodeError, OSError):
        logger.info("Skipping: hostfile at %s is invalid", _DEFAULT_HOSTFILE)
        return True
    hosts = hostfile.get("hosts", [])
    if len(hosts) < 2:
        logger.info("Skipping: hostfile has fewer than 2 hosts")
        return True
    if not hostfile.get("model"):
        logger.info("Skipping: hostfile has no 'model' field")
        return True
    return False


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    env_overrides: dict[str, str] = field(default_factory=dict)
    should_skip: Callable[[Path], bool] = _no_skip
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
        env_overrides={"OLMLX_EXPERIMENTAL_KV_CACHE_QUANT": "turboquant:4"},
    ),
    Scenario(
        name="turboquant-2",
        description="TurboQuant 2-bit KV cache quantization",
        env_overrides={"OLMLX_EXPERIMENTAL_KV_CACHE_QUANT": "turboquant:2"},
    ),
    Scenario(
        name="cache+tq4",
        description="Prompt cache + TurboQuant 4-bit",
        env_overrides={
            "OLMLX_PROMPT_CACHE": "true",
            "OLMLX_EXPERIMENTAL_KV_CACHE_QUANT": "turboquant:4",
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
            "OLMLX_EXPERIMENTAL_KV_CACHE_QUANT": "turboquant:4",
        },
        should_skip=_requires_flash,
    ),
    Scenario(
        name="flash-moe",
        description="Flash MoE expert offloading",
        env_overrides={"OLMLX_EXPERIMENTAL_FLASH_MOE": "true"},
        should_skip=_requires_moe,
    ),
    Scenario(
        name="flash-moe+tq4",
        description="Flash MoE + TurboQuant 4-bit",
        env_overrides={
            "OLMLX_EXPERIMENTAL_FLASH_MOE": "true",
            "OLMLX_EXPERIMENTAL_KV_CACHE_QUANT": "turboquant:4",
        },
        should_skip=_requires_moe,
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
            "OLMLX_EXPERIMENTAL_KV_CACHE_QUANT": "turboquant:4",
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
            raise ValueError(
                f"Unknown scenario {name!r}. Available: {sorted(by_name)}"
            )
        result.append(by_name[name])
    return result
