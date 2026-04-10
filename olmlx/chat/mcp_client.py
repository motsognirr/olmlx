"""MCP client manager for connecting to external tool servers."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Protocol
from typing_extensions import TypedDict

from olmlx.chat.config import sanitize_mcp_env

logger = logging.getLogger(__name__)


class MCPToolInputSchema(TypedDict, total=False):
    """TypedDict for MCP tool input schema."""

    type: str
    properties: dict[str, Any]


class MCPToolDict(TypedDict):
    """TypedDict for MCP tool in config format."""

    name: str
    description: str
    inputSchema: MCPToolInputSchema


class MCPSessionProtocol(Protocol):
    """Protocol for MCP ClientSession (for type annotation)."""

    async def initialize(self) -> Any: ...
    async def list_tools(self) -> Any: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class MCPTransport(Protocol):
    """Protocol for MCP transport context manager."""

    async def __aenter__(self) -> tuple[Any, Any]: ...
    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> bool | None: ...


class MCPClientManager:
    """Manages connections to MCP servers and routes tool calls."""

    def __init__(self):
        self._servers: dict[str, dict] = {}  # name -> server info
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name
        self._tools: list[dict] = []  # OpenAI function-calling format

    @staticmethod
    def _convert_tool(mcp_tool: MCPToolDict) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": mcp_tool["name"],
                "description": mcp_tool.get("description", ""),
                "parameters": mcp_tool.get(
                    "inputSchema",
                    {
                        "type": "object",
                        "properties": {},
                    },
                ),
            },
        }

    async def connect_all(self, config: dict[str, Any]) -> None:
        """Connect to each configured MCP server and discover tools."""
        for name, server_cfg in config.items():
            try:
                if server_cfg["transport"] == "stdio":
                    await self._connect_stdio(name, server_cfg)
                elif server_cfg["transport"] == "sse":
                    await self._connect_sse(name, server_cfg)
            except Exception as exc:
                logger.warning("Failed to connect to MCP server %r: %s", name, exc)

    async def _connect_stdio(self, name: str, cfg: dict[str, Any]) -> None:
        """Connect to a stdio MCP server."""
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        user_env = cfg.get("env")
        safe_env = sanitize_mcp_env(user_env)

        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=safe_env,
        )
        transport_cm: MCPTransport = stdio_client(params)
        await self._connect_transport(name, transport_cm)

    async def _connect_sse(self, name: str, cfg: dict[str, Any]) -> None:
        """Connect to an SSE MCP server."""
        from mcp.client.sse import sse_client

        transport_cm: MCPTransport = sse_client(cfg["url"])
        await self._connect_transport(name, transport_cm)

    async def _connect_transport(self, name: str, transport_cm: MCPTransport) -> None:
        """Connect via a transport context manager, create session, discover tools."""
        from mcp import ClientSession

        read_stream, write_stream = await transport_cm.__aenter__()
        try:
            session_cm = ClientSession(read_stream, write_stream)
            session: MCPSessionProtocol = await session_cm.__aenter__()
            try:
                await session.initialize()
            except BaseException:
                await session_cm.__aexit__(*sys.exc_info())
                raise
        except BaseException:
            await transport_cm.__aexit__(*sys.exc_info())
            raise

        try:
            await self._discover_tools(name, session)
        except BaseException:
            await session_cm.__aexit__(*sys.exc_info())
            await transport_cm.__aexit__(*sys.exc_info())
            raise
        self._servers[name] = {
            "session": session,
            "session_cm": session_cm,
            "transport_cm": transport_cm,
        }

    async def _discover_tools(
        self, server_name: str, session: MCPSessionProtocol
    ) -> None:
        """Discover tools from an MCP session and register them."""
        result = await session.list_tools()
        for tool in result.tools:
            converted = self._convert_tool(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": getattr(
                        tool,
                        "inputSchema",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                }
            )
            if tool.name in self._tool_to_server:
                logger.warning(
                    "Tool %r from server %r conflicts with existing tool from %r; skipping",
                    tool.name,
                    server_name,
                    self._tool_to_server[tool.name],
                )
                continue
            self._tools.append(converted)
            self._tool_to_server[tool.name] = server_name
            logger.info("Discovered tool %r from server %r", tool.name, server_name)

    def get_tools_for_chat(self) -> list[dict]:
        """Return tools in OpenAI function-calling format for generate_chat()."""
        return self._tools

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30.0) -> str:
        """Route a tool call to the correct server and return the result text."""
        server_name = self._tool_to_server.get(name)
        if server_name is None:
            raise ValueError(f"Unknown tool: {name!r}")

        server = self._servers.get(server_name)
        if server is None:
            raise RuntimeError(
                f"Server {server_name!r} for tool {name!r} is not connected"
            )

        session = server["session"]
        result = await asyncio.wait_for(
            session.call_tool(name, arguments), timeout=timeout
        )

        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            else:
                parts.append(str(content))
        return "\n".join(parts) if parts else ""

    async def disconnect_all(self) -> None:
        """Clean shutdown of all MCP connections."""
        for name, server in self._servers.items():
            # Close session first, then transport
            for key in ("session_cm", "transport_cm"):
                cm = server.get(key)
                if cm is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                    except BaseException as exc:
                        logger.debug("Error closing %s for %r: %s", key, name, exc)
        self._servers.clear()
        self._tool_to_server.clear()
        self._tools.clear()
