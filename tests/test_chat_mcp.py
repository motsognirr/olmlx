"""Tests for olmlx.chat.mcp_client."""

import pytest

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
    async def test_call_tool_unknown_raises(self):
        mgr = MCPClientManager()
        with pytest.raises(ValueError, match="Unknown tool"):
            await mgr.call_tool("nonexistent", {})
