"""Tests for olmlx.chat.mcp_client."""

import pytest

from olmlx.chat.errors import ToolError
from olmlx.chat.mcp_client import MCPClientManager


class TestMCPToolConversion:
    """Test MCP tool schema to OpenAI function-calling format conversion."""

    def test_converts_simple_tool(self):
        mcp_tool = {
            "name": "read_file",
            "description": "Read a file from disk",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        }
        result = MCPClientManager._convert_tool(mcp_tool)
        assert result == {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from disk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"],
                },
            },
        }

    def test_converts_tool_without_description(self):
        mcp_tool = {
            "name": "ping",
            "inputSchema": {"type": "object", "properties": {}},
        }
        result = MCPClientManager._convert_tool(mcp_tool)
        assert result["function"]["name"] == "ping"
        assert result["function"]["description"] == ""

    def test_converts_tool_without_input_schema(self):
        mcp_tool = {"name": "get_time", "description": "Get current time"}
        result = MCPClientManager._convert_tool(mcp_tool)
        assert result["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }


class TestMCPToolRouting:
    def test_get_tools_empty_when_no_servers(self):
        mgr = MCPClientManager()
        assert mgr.get_tools_for_chat() == []

    def test_tool_to_server_mapping(self):
        mgr = MCPClientManager()
        # Simulate registered tools
        mgr._tool_to_server["read_file"] = "filesystem"
        mgr._tool_to_server["search"] = "search_api"
        mgr._tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        tools = mgr.get_tools_for_chat()
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"read_file", "search"}

    @pytest.mark.asyncio
    async def test_call_tool_unknown_returns_tool_error(self):
        mgr = MCPClientManager()
        result = await mgr.call_tool("nonexistent", {})
        assert isinstance(result, ToolError)
        assert "Unknown tool" in result.message
        assert result.tool_name == "nonexistent"
        assert result.is_user_error is True


class TestConnectAll:
    @pytest.mark.asyncio
    async def test_unknown_transport_skipped_with_warning(self, caplog):
        """An unknown transport must be logged and skipped, not silently
        treated as a successful connection with zero tools (#636)."""
        import logging

        mgr = MCPClientManager()
        with caplog.at_level(logging.WARNING, logger="olmlx.chat.mcp_client"):
            await mgr.connect_all({"srv": {"transport": "grpc"}})
        assert "unknown transport" in caplog.text.lower()
        assert mgr.get_tools_for_chat() == []
