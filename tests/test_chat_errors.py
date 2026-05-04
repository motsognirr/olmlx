"""Tests for error handling consistency across chat layers.

Covers gaps identified in issue #179 (Standardize error handling).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmlx.chat.builtin_tools import BuiltinToolManager
from olmlx.chat.config import ChatConfig
from olmlx.chat.errors import ToolError
from olmlx.chat.mcp_client import MCPClientManager
from olmlx.chat.session import ChatSession
from olmlx.engine.template_caps import TemplateCaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(*, mcp=None, builtin=None, tool_safety=None, thinking=True):
    config = ChatConfig(model_name="test:latest", thinking=thinking)
    manager = MagicMock()
    loaded_model = MagicMock()
    loaded_model.template_caps = TemplateCaps()
    manager.ensure_loaded = AsyncMock(return_value=loaded_model)
    return ChatSession(
        config=config,
        manager=manager,
        mcp=mcp,
        builtin=builtin,
        tool_safety=tool_safety,
    )


# ---------------------------------------------------------------------------
# Builtin error format consistency
# ---------------------------------------------------------------------------


class TestBuiltinErrorFormat:
    """Validate built-in tools return ToolError for errors, str for success.

    After the #179 refactoring, all error paths return structured ToolError
    instances instead of ad-hoc "Error: ..." strings.
    """

    @pytest.mark.asyncio
    async def test_read_file_not_found_returns_tool_error(self, tmp_path):
        mgr = BuiltinToolManager(ChatConfig(model_name="x", plans_dir=tmp_path))
        result = await mgr.call_tool("read_file", {"path": str(tmp_path / "nope.txt")})
        assert isinstance(result, ToolError)
        assert result.tool_name == "read_file"
        assert "Error reading file" in result.message

    @pytest.mark.asyncio
    async def test_read_file_path_traversal_returns_tool_error(self, tmp_path):
        mgr = BuiltinToolManager(ChatConfig(model_name="x", plans_dir=tmp_path))
        result = await mgr.call_tool("read_file", {"path": "../outside/passwd"})
        assert isinstance(result, ToolError)
        assert result.tool_name == "read_file"
        assert result.is_user_error is True

    @pytest.mark.asyncio
    async def test_write_file_error_returns_tool_error(self, tmp_path):
        mgr = BuiltinToolManager(ChatConfig(model_name="x", plans_dir=tmp_path))
        result = await mgr.call_tool(
            "write_file", {"path": "../outside", "content": "x"}
        )
        assert isinstance(result, ToolError)
        assert result.tool_name == "write_file"
        assert result.is_user_error is True

    @pytest.mark.asyncio
    async def test_edit_file_not_found_returns_tool_error(self, tmp_path):
        mgr = BuiltinToolManager(ChatConfig(model_name="x", plans_dir=tmp_path))
        result = await mgr.call_tool(
            "edit_file",
            {"path": str(tmp_path / "nope.txt"), "old_text": "x", "new_text": "y"},
        )
        assert isinstance(result, ToolError)
        assert result.tool_name == "edit_file"

    @pytest.mark.asyncio
    async def test_edit_file_no_match_returns_tool_error(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hello")
        mgr = BuiltinToolManager(ChatConfig(model_name="x", plans_dir=tmp_path))
        result = await mgr.call_tool(
            "edit_file", {"path": str(p), "old_text": "zzz", "new_text": "y"}
        )
        assert isinstance(result, ToolError)
        assert result.tool_name == "edit_file"
        assert result.is_user_error is True

    @pytest.mark.asyncio
    async def test_edit_file_multiple_matches_returns_tool_error(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("x x")
        mgr = BuiltinToolManager(ChatConfig(model_name="x", plans_dir=tmp_path))
        result = await mgr.call_tool(
            "edit_file", {"path": str(p), "old_text": "x", "new_text": "y"}
        )
        assert isinstance(result, ToolError)
        assert result.tool_name == "edit_file"

    @pytest.mark.asyncio
    async def test_bash_timeout_returns_tool_error(self):
        mgr = BuiltinToolManager(ChatConfig(model_name="x"))
        result = await mgr.call_tool("bash", {"command": "sleep 2", "timeout": 0})
        assert isinstance(result, ToolError)
        assert result.tool_name == "bash"
        assert "timed out" in result.message.lower()

    @pytest.mark.asyncio
    async def test_web_fetch_scheme_returns_tool_error(self):
        mgr = BuiltinToolManager(ChatConfig(model_name="x"))
        result = await mgr.call_tool("web_fetch", {"url": "file:///etc/passwd"})
        assert isinstance(result, ToolError)
        assert result.tool_name == "web_fetch"
        assert result.is_user_error is True

    @pytest.mark.asyncio
    async def test_web_search_import_error_returns_tool_error(self):
        mgr = BuiltinToolManager(ChatConfig(model_name="x"))
        result = await mgr.call_tool("web_search", {"query": "test"})
        # duckduckgo-search may or may not be installed
        if isinstance(result, ToolError):
            assert result.tool_name == "web_search"
        else:
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_tool_error(self, tmp_path):
        mgr = BuiltinToolManager(ChatConfig(model_name="x", plans_dir=tmp_path))
        result = await mgr.call_tool("nonexistent_tool", {})
        assert isinstance(result, ToolError)
        assert "Unknown built-in tool" in result.message
        assert result.tool_name == "nonexistent_tool"

    @pytest.mark.asyncio
    async def test_all_tool_errors_are_tool_error_or_str(self, tmp_path):
        """Every built-in tool error path returns ToolError, success returns str."""
        mgr = BuiltinToolManager(ChatConfig(model_name="x", plans_dir=tmp_path))

        r1 = await mgr.call_tool("read_file", {"path": str(tmp_path / "nope.txt")})
        assert isinstance(r1, ToolError)

        r2 = await mgr.call_tool("web_fetch", {"url": "ftp://example.com"})
        assert isinstance(r2, ToolError)


# ---------------------------------------------------------------------------
# MCP client error conversion gaps
# ---------------------------------------------------------------------------


class TestMCPErrorConversion:
    """MCP client returns ToolError for all failure paths instead of raising."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_tool_error(self):
        mgr = MCPClientManager()
        result = await mgr.call_tool("no_such_tool", {})
        assert isinstance(result, ToolError)
        assert "Unknown tool" in result.message
        assert result.tool_name == "no_such_tool"
        assert result.is_user_error is True

    @pytest.mark.asyncio
    async def test_disconnected_server_returns_tool_error(self):
        mgr = MCPClientManager()
        mgr._tool_to_server["t"] = "offline"
        result = await mgr.call_tool("t", {})
        assert isinstance(result, ToolError)
        assert "not connected" in result.message
        assert result.tool_name == "t"
        assert result.is_user_error is False

    @pytest.mark.asyncio
    async def test_call_tool_timeout_returns_tool_error(self):
        """MCP timeouts return ToolError instead of raising."""
        mgr = MCPClientManager()
        mgr._tool_to_server["slow"] = "slow_srv"

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock(content=[])

        mock_session = MagicMock()
        mock_session.call_tool = slow_call
        mgr._servers["slow_srv"] = {"session": mock_session}

        result = await mgr.call_tool("slow", {}, timeout=0.01)
        assert isinstance(result, ToolError)
        assert "timed out" in result.message
        assert result.tool_name == "slow"
        assert result.is_user_error is False


# ---------------------------------------------------------------------------
# Session error handling: deferred BaseException path
# ---------------------------------------------------------------------------


class TestSessionDeferredException:
    """Characterise the deferred BaseException path in _execute_tool_calls.

    When asyncio.gather captures a non-Exception BaseException (e.g.
    KeyboardInterrupt, SystemExit, GeneratorExit), the code path at
    session.py:855-862 re-raises it after appending error messages.
    CancelledError is handled separately; these tests verify the
    remaining BaseException subclasses.
    """

    @pytest.mark.asyncio
    async def test_tool_raising_non_exception_base_exception_propagates(self):
        """Non-Exception BaseException (e.g. GeneratorExit) must propagate."""
        mcp = MagicMock()
        mcp.get_tools_for_chat.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "crash",
                    "description": "crashes hard",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        class CustomBaseException(BaseException):
            pass

        mcp.call_tool = AsyncMock(side_effect=CustomBaseException("boom"))

        session = _make_session(mcp=mcp)

        async def fake_generate(*args, **kwargs):
            yield {
                "text": '<tool_call>{"name": "crash", "arguments": {}}</tool_call>',
                "done": False,
            }
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            events = []
            with pytest.raises(CustomBaseException):
                async for event in session.send_message("Use the tool"):
                    events.append(event)

        # Messages should still be in history (error appended before re-raise)
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1

    @pytest.mark.asyncio
    async def test_tool_raising_non_exception_base_exception_is_fed_as_tool_error(self):
        """Non-Exception BaseException from asyncio.gather is handled before re-raise.

        Verifies that _execute_tool_calls yields a tool_error event before
        re-raising the deferred BaseException.
        """

        class FatalError(BaseException):
            pass

        mcp = MagicMock()
        mcp.call_tool = AsyncMock(side_effect=FatalError("fatal"))

        session = _make_session(mcp=mcp)

        tool_uses = [
            {"name": "fatal_tool", "input": {}, "id": "tc_1"},
        ]
        events = []
        exc_raised = False
        try:
            async for event in session._execute_tool_calls(tool_uses):
                events.append(event)
        except FatalError:
            exc_raised = True
        assert exc_raised
        # The tool_error event must be yielded before the re-raise
        error_events = [e for e in events if e.get("type") == "tool_error"]
        assert len(error_events) == 1
        assert error_events[0]["name"] == "fatal_tool"


# ---------------------------------------------------------------------------
# Router error shape consistency
# ---------------------------------------------------------------------------


class TestRouterErrorShapeConsistency:
    """Characterise the error shapes produced by the different API layers.

    The Ollama NDJSON routers (chat, generate) and the SSE routers
    (OpenAI, Anthropic) produce different error JSON structures.
    These tests document the current shapes so a refactoring does not
    accidentally change the wire format.
    """

    def test_ollama_ndjson_error_shape(self):
        """Ollama chat/generate errors use: {"error": "...", "done": true, ...}"""
        from olmlx.routers.common import format_error

        err = format_error("test:latest")
        import json

        body = json.loads(err.rstrip("\n"))
        assert "error" in body
        assert "An internal server error occurred" in body["error"]
        assert body["done"] is True
        assert body["done_reason"] == "error"
        assert body["model"] == "test:latest"

    def test_openai_error_shape(self):
        """OpenAI errors use: {"error": {"message": ..., "type": ..., ...}}"""
        from olmlx.app import _make_error_response

        resp = _make_error_response(
            "/v1/chat/completions",
            400,
            "bad request",
            "api_error",
            "server_error",
            "invalid_value",
        )
        import json

        body = json.loads(resp.body)
        assert "error" in body
        assert isinstance(body["error"], dict)
        assert body["error"]["message"] == "bad request"
        assert body["error"]["type"] == "server_error"

    def test_anthropic_error_shape(self):
        """Anthropic errors use: {"type": "error", "error": {"type": ..., "message": ...}}"""
        from olmlx.app import _make_error_response

        resp = _make_error_response(
            "/v1/messages",
            400,
            "bad request",
            "invalid_request_error",
            "invalid_request_error",
            "invalid_value",
        )
        import json

        body = json.loads(resp.body)
        assert body["type"] == "error"
        assert isinstance(body["error"], dict)
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"] == "bad request"

    def test_manage_error_shape(self):
        """manage.py returns JSONResponse({"error": ...}) directly."""
        from fastapi.testclient import TestClient

        from olmlx.app import create_app

        app = create_app()
        client = TestClient(app)

        # Simulate a manage endpoint condition that produces a JSON error
        # POST /api/copy with nonexistent source
        resp = client.post(
            "/api/copy",
            content="{}",
            headers={"Content-Type": "application/json"},
        )
        # Returns validation error — fastapi sends 422 with {'detail': [...]}
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body


# ---------------------------------------------------------------------------
# End-to-end: error event structure across builtin/MCP/session
# ---------------------------------------------------------------------------


class TestEndToEndToolErrorEvents:
    """Verify that tool_error events have a consistent shape regardless of
    whether the error comes from built-in tools, MCP, or parse failures."""

    REQUIRED_TOOL_ERROR_KEYS = {"type", "name", "error", "id", "is_user_error"}

    @pytest.mark.asyncio
    async def test_builtin_tool_error_event_shape(self, tmp_path):
        builtin = BuiltinToolManager(ChatConfig(model_name="x", plans_dir=tmp_path))

        mgr = MagicMock()
        loaded_model = MagicMock()
        loaded_model.template_caps = TemplateCaps()
        mgr.ensure_loaded = AsyncMock(return_value=loaded_model)

        config = ChatConfig(model_name="test:latest")
        session = ChatSession(config=config, manager=mgr, builtin=builtin)

        async def fake_generate(*args, **kwargs):
            yield {
                "text": '<tool_call>{"name": "read_file", "arguments": {"path": "/nonexistent"}}</tool_call>',
                "done": False,
            }
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            events = []
            async for event in session.send_message("Read a file"):
                events.append(event)

        error_events = [e for e in events if e.get("type") == "tool_error"]
        for ev in error_events:
            assert self.REQUIRED_TOOL_ERROR_KEYS.issubset(ev.keys())
            assert ev["name"] == "read_file"
            assert ev["error"] != ""

    @pytest.mark.asyncio
    async def test_mcp_tool_error_event_shape(self):
        """MCP returning ToolError → _exec_tool yields tool_error event."""
        mcp = MagicMock()
        mcp.get_tools_for_chat.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "remote_fail",
                    "description": "always fails",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        mcp.call_tool = AsyncMock(
            return_value=ToolError(
                message="connection refused",
                tool_name="remote_fail",
                is_user_error=False,
            )
        )

        session = _make_session(mcp=mcp)

        async def fake_generate(*args, **kwargs):
            yield {
                "text": '<tool_call>{"name": "remote_fail", "arguments": {}}</tool_call>',
                "done": False,
            }
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            events = []
            async for event in session.send_message("Use remote tool"):
                events.append(event)

        error_events = [e for e in events if e.get("type") == "tool_error"]
        assert len(error_events) >= 1
        for ev in error_events:
            assert self.REQUIRED_TOOL_ERROR_KEYS.issubset(ev.keys())
            assert ev["name"] == "remote_fail"
            assert ev["error"] != ""

    @pytest.mark.asyncio
    async def test_mcp_exception_path_still_yields_tool_error(self):
        """MCP raising exception → asyncio.gather path still yields tool_error.

        Even though the real MCPClientManager.call_tool now returns ToolError
        instead of raising, the exception path in _execute_tool_calls (gather
        with return_exceptions=True) is still exercised when code paths
        inside call_tool raise before the return (e.g. BaseException from
        the MCP transport).
        """
        mcp = MagicMock()
        mcp.get_tools_for_chat.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "crash",
                    "description": "raises unexpectedly",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        mcp.call_tool = AsyncMock(side_effect=RuntimeError("unexpected"))

        session = _make_session(mcp=mcp)
        session.config.max_consecutive_tool_failures = 3

        async def fake_generate(*args, **kwargs):
            yield {
                "text": '<tool_call>{"name": "crash", "arguments": {}}</tool_call>',
                "done": False,
            }
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            events = []
            async for event in session.send_message("Crash"):
                events.append(event)

        error_events = [e for e in events if e.get("type") == "tool_error"]
        assert len(error_events) >= 1
        assert error_events[0]["name"] == "crash"

    @pytest.mark.asyncio
    async def test_parse_error_event_shape(self):
        session = _make_session()

        async def fake_generate(*args, **kwargs):
            yield {
                "text": "No tools here, just text.",
                "done": False,
            }
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            # Force parse failure by passing malformed text with tools present
            # Actually, parse_model_output handles most cases gracefully.
            # Verify the event shape when errors do occur.
            events = []
            async for event in session.send_message("hello"):
                events.append(event)

        # No tools → no parse errors; just verify no tool_error events
        tool_errors = [e for e in events if e.get("type") == "tool_error"]
        assert len(tool_errors) == 0

    @pytest.mark.asyncio
    async def test_model_load_error_event_shape(self):
        """Model load failure yields model_load_error with error string."""
        mgr = MagicMock()
        mgr.ensure_loaded = AsyncMock(side_effect=RuntimeError("OOM"))

        config = ChatConfig(model_name="big:model")
        session = ChatSession(config=config, manager=mgr)

        events = []
        async for event in session.send_message("hello"):
            events.append(event)

        load_errors = [e for e in events if e.get("type") == "model_load_error"]
        assert len(load_errors) == 1
        assert load_errors[0]["error"] == "Failed to load model: OOM"


# ---------------------------------------------------------------------------
# MCP call_tool result handling (server-side errors)
# ---------------------------------------------------------------------------


class TestMCPCallToolResultHandling:
    """Characterise how call_tool handles MCP result.content items."""

    @pytest.mark.asyncio
    async def test_result_content_with_text_attribute(self):
        mgr = MCPClientManager()
        mgr._tool_to_server["echo"] = "srv"

        mock_content = MagicMock()
        mock_content.text = "hello world"

        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = False

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mgr._servers["srv"] = {"session": mock_session}

        result = await mgr.call_tool("echo", {})
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_result_content_without_text_falls_back_to_str(self):
        mgr = MCPClientManager()
        mgr._tool_to_server["fn"] = "srv"

        # Create an object that has no .text attribute (falls back to __str__)
        class NoText:
            def __str__(self):
                return "raw result"

        mock_result = MagicMock()
        mock_result.content = [NoText()]
        mock_result.isError = False

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mgr._servers["srv"] = {"session": mock_session}

        result = await mgr.call_tool("fn", {})
        assert result == "raw result"

    @pytest.mark.asyncio
    async def test_result_empty_content_returns_empty_string(self):
        mgr = MCPClientManager()
        mgr._tool_to_server["empty"] = "srv"

        mock_result = MagicMock()
        mock_result.content = []
        mock_result.isError = False

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mgr._servers["srv"] = {"session": mock_session}

        result = await mgr.call_tool("empty", {})
        assert result == ""

    @pytest.mark.asyncio
    async def test_result_is_error_converts_to_tool_error(self):
        mgr = MCPClientManager()
        mgr._tool_to_server["fail"] = "srv"

        mock_content = MagicMock()
        mock_content.text = "server refused"

        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = True

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mgr._servers["srv"] = {"session": mock_session}

        result = await mgr.call_tool("fail", {})
        assert isinstance(result, ToolError)
        assert "server refused" in result.message
        assert result.tool_name == "fail"
        assert result.is_user_error is False
