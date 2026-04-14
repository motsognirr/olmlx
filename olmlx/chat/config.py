"""Chat configuration and MCP config loading."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmlx.chat.tool_safety import ToolSafetyConfig

logger = logging.getLogger(__name__)

_BLOCKED_ENV_VARS = {
    "PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_FALLBACK_FRAMEWORK_PATH",
    "DYLD_IMAGE_SUFFIX",
    "DYLD_FORCE_FLAT_NAMESPACE",
    "SYSTEMROOT",
    "WINDIR",
}


@dataclass
class ChatConfig:
    model_name: str
    system_prompt: str | None = None
    max_tokens: int = 4096
    max_turns: int = 25
    thinking: bool = True
    mcp_enabled: bool = True
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    mcp_config_path: Path = field(
        default_factory=lambda: Path.home() / ".olmlx" / "mcp.json"
    )
    skills_enabled: bool = True
    skills_dir: Path = field(default_factory=lambda: Path.home() / ".olmlx" / "skills")
    builtin_tools_enabled: bool = True
    plans_dir: Path = field(default_factory=lambda: Path.home() / ".olmlx" / "plans")
    sequential_tool_execution: bool = False


def _load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON file with error handling. Returns empty dict on failure."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(
                "Expected JSON object in %s, got %s", path, type(data).__name__
            )
            return {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load JSON from %s: %s", path, exc)
        return {}


def load_mcp_config(path: Path) -> dict[str, Any]:
    """Parse MCP config file in Claude Desktop format.

    Entries with ``command`` get ``transport: "stdio"``.
    Entries with ``url`` get ``transport: "sse"``.
    Entries with neither are skipped.

    Returns a dict of server_name -> server config with transport type added.
    """
    raw = _load_json_file(path)
    if not raw:
        return {}

    servers_raw = raw.get("mcpServers", {})
    if not isinstance(servers_raw, dict):
        return {}

    result = {}
    for name, cfg in servers_raw.items():
        if not isinstance(cfg, dict):
            continue
        if "command" in cfg:
            result[name] = {**cfg, "transport": "stdio"}
        elif "url" in cfg:
            result[name] = {**cfg, "transport": "sse"}
        else:
            logger.debug("Skipping MCP server %r: no 'command' or 'url'", name)

    return result


def sanitize_mcp_env(user_env: dict[str, str] | None) -> dict[str, str] | None:
    """Sanitize environment variables for MCP server processes.

    Starts with full parent environment, removes blocked dangerous variables
    (PATH, LD_PRELOAD, etc.) from user-provided overrides only. Returns None
    when no user_env is provided to preserve inherited environment.

    Args:
        user_env: User-provided environment dict from MCP config

    Returns:
        Sanitized environment dict safe for subprocess execution, or None
        to inherit parent environment unchanged.
    """
    if user_env is None:
        return None

    allowed_env = os.environ.copy()

    allowed_env.update(
        {k: v for k, v in user_env.items() if k not in _BLOCKED_ENV_VARS}
    )

    blocked_found = [k for k in user_env if k in _BLOCKED_ENV_VARS]
    if blocked_found:
        logger.warning(
            "Blocked dangerous environment variables in MCP config: %s",
            ", ".join(blocked_found),
        )

    return allowed_env


def load_tool_safety_config(path: Path) -> ToolSafetyConfig:
    """Parse tool safety config from mcp.json.

    Reads the ``toolSafety`` section. Returns default config on missing
    file, missing section, or parse errors.
    """
    from olmlx.chat.tool_safety import ToolPolicy, ToolSafetyConfig

    raw = _load_json_file(path)
    if not raw:
        return ToolSafetyConfig()

    safety_raw = raw.get("toolSafety")
    if not isinstance(safety_raw, dict):
        return ToolSafetyConfig()

    # Parse default policy
    default_str = safety_raw.get("defaultPolicy", "confirm")
    try:
        default_policy = ToolPolicy(default_str)
    except ValueError:
        logger.warning("Invalid defaultPolicy %r, using 'confirm'", default_str)
        default_policy = ToolPolicy.CONFIRM

    # Parse per-tool policies
    tool_policies: dict[str, ToolPolicy] = {}
    tools_raw = safety_raw.get("tools", {})
    if isinstance(tools_raw, dict):
        for name, value in tools_raw.items():
            try:
                tool_policies[name] = ToolPolicy(value)
            except ValueError:
                logger.warning("Invalid policy %r for tool %r, skipping", value, name)

    return ToolSafetyConfig(default_policy=default_policy, tool_policies=tool_policies)
