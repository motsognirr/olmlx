"""Chat configuration and MCP config loading."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

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


def load_mcp_config(path: Path) -> dict:
    """Parse MCP config file in Claude Desktop format.

    Entries with ``command`` get ``transport: "stdio"``.
    Entries with ``url`` get ``transport: "sse"``.
    Entries with neither are skipped.

    Returns a dict of server_name -> server config with transport type added.
    """
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load MCP config from %s: %s", path, exc)
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
