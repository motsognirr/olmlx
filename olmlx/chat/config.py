"""Chat configuration and MCP config loading."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmlx.chat.tool_safety import ToolSafetyConfig

logger = logging.getLogger(__name__)


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


def _load_json_file(path: Path) -> dict:
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


def load_mcp_config(path: Path) -> dict:
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
