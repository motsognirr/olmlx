"""Regression coverage for olmlx.chat.mcp_client.

Focuses on the connect/retry/teardown lifecycle, tool discovery,
tool invocation routing, and uniform error propagation. All MCP
transports and sessions are mocked; no network, subprocess, or
filesystem access.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import olmlx.chat.mcp_client as mcp_client_mod
from olmlx.chat.errors import ToolError
from olmlx.chat.mcp_client import MCPClientManager


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _tool(name, description=None, input_schema=None):
    """Build a fake MCP tool object (attribute access, like the real SDK)."""
    ns = SimpleNamespace(name=name, description=description)
    if input_schema is not None:
        ns.inputSchema = input_schema
    return ns


def _list_tools_result(*tools):
    return SimpleNamespace(tools=list(tools))


def _content_text(text):
    return SimpleNamespace(text=text)


def _content_other(value):
    """Content block without a .text attribute (exercises str() fallback)."""
    return value


class FakeSessionCM:
    """Stand-in for ClientSession(...) — an async context manager whose
    __aenter__ returns a session object."""

    def __init__(self, session):
        self._session = session
        self.aexit_calls = []

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.aexit_calls.append((exc_type, exc_val, exc_tb))
        return None


class FakeTransportCM:
    """Stand-in for stdio_client()/sse_client() transport context manager."""

    def __init__(self, streams=("read", "write"), enter_exc=None):
        self._streams = streams
        self._enter_exc = enter_exc
        self.aexit_calls = []
        self.aenter_calls = 0

    async def __aenter__(self):
        self.aenter_calls += 1
        if self._enter_exc is not None:
            raise self._enter_exc
        return self._streams

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.aexit_calls.append((exc_type, exc_val, exc_tb))
        return None


def _make_session(tools=(), call_result=None, *, initialize_exc=None):
    session = MagicMock(name="MCPSession")
    session.initialize = AsyncMock(
        side_effect=initialize_exc if initialize_exc is not None else None
    )
    session.list_tools = AsyncMock(return_value=_list_tools_result(*tools))
    session.call_tool = AsyncMock(return_value=call_result)
    return session


def _patch_clientsession(monkeypatch, session_cm):
    """Patch `mcp.ClientSession` (imported lazily inside _connect_transport)."""
    import mcp

    monkeypatch.setattr(
        mcp, "ClientSession", MagicMock(return_value=session_cm), raising=True
    )


# ---------------------------------------------------------------------------
# _connect_transport: success path + tool discovery
# ---------------------------------------------------------------------------


async def test_connect_transport_registers_server_and_tools(monkeypatch):
    session = _make_session(
        tools=[
            _tool(
                "read_file",
                "Read a file",
                {"type": "object", "properties": {"path": {"type": "string"}}},
            ),
            _tool("ping"),
        ]
    )
    session_cm = FakeSessionCM(session)
    transport_cm = FakeTransportCM(streams=("r", "w"))
    _patch_clientsession(monkeypatch, session_cm)

    mgr = MCPClientManager()
    await mgr._connect_transport("srv", transport_cm)

    # Server recorded with both context managers for teardown.
    assert "srv" in mgr._servers
    assert mgr._servers["srv"]["session"] is session
    assert mgr._servers["srv"]["session_cm"] is session_cm
    assert mgr._servers["srv"]["transport_cm"] is transport_cm

    # Tools discovered and routed.
    assert mgr._tool_to_server == {"read_file": "srv", "ping": "srv"}
    names = {t["function"]["name"] for t in mgr.get_tools_for_chat()}
    assert names == {"read_file", "ping"}

    # initialize() and list_tools() actually invoked.
    session.initialize.assert_awaited_once()
    session.list_tools.assert_awaited_once()
    # No teardown on success.
    assert transport_cm.aexit_calls == []
    assert session_cm.aexit_calls == []


async def test_discover_tools_missing_input_schema_uses_default(monkeypatch):
    # Tool object without an inputSchema attribute -> default empty schema.
    session = _make_session(tools=[_tool("noschema", "desc")])
    session_cm = FakeSessionCM(session)
    _patch_clientsession(monkeypatch, session_cm)

    mgr = MCPClientManager()
    await mgr._connect_transport("srv", FakeTransportCM())

    tool = mgr.get_tools_for_chat()[0]
    assert tool["function"]["parameters"] == {"type": "object", "properties": {}}
    # None description coalesces to "".
    assert tool["function"]["description"] == "desc"


async def test_discover_tools_none_description_coalesces(monkeypatch):
    session = _make_session(tools=[_tool("t", None, {"type": "object"})])
    session_cm = FakeSessionCM(session)
    _patch_clientsession(monkeypatch, session_cm)

    mgr = MCPClientManager()
    await mgr._connect_transport("srv", FakeTransportCM())
    assert mgr.get_tools_for_chat()[0]["function"]["description"] == ""


async def test_duplicate_tool_name_across_servers_is_skipped(monkeypatch):
    # First server.
    s1 = _make_session(tools=[_tool("shared", "first", {"type": "object"})])
    _patch_clientsession(monkeypatch, FakeSessionCM(s1))
    mgr = MCPClientManager()
    await mgr._connect_transport("srvA", FakeTransportCM())

    # Second server exposing the same tool name.
    s2 = _make_session(
        tools=[
            _tool("shared", "second", {"type": "object"}),
            _tool("unique", "u", {"type": "object"}),
        ]
    )
    _patch_clientsession(monkeypatch, FakeSessionCM(s2))
    await mgr._connect_transport("srvB", FakeTransportCM())

    # The shared tool stays mapped to the first server; only unique is added.
    assert mgr._tool_to_server["shared"] == "srvA"
    assert mgr._tool_to_server["unique"] == "srvB"
    names = [t["function"]["name"] for t in mgr.get_tools_for_chat()]
    assert names.count("shared") == 1


# ---------------------------------------------------------------------------
# _connect_transport: teardown on failure
# ---------------------------------------------------------------------------


async def test_initialize_failure_tears_down_session_and_transport(monkeypatch):
    session = _make_session(initialize_exc=RuntimeError("init boom"))
    session_cm = FakeSessionCM(session)
    transport_cm = FakeTransportCM()
    _patch_clientsession(monkeypatch, session_cm)

    mgr = MCPClientManager()
    with pytest.raises(RuntimeError, match="init boom"):
        await mgr._connect_transport("srv", transport_cm)

    # Both context managers closed, server NOT registered.
    assert "srv" not in mgr._servers
    assert len(session_cm.aexit_calls) == 1
    assert len(transport_cm.aexit_calls) == 1
    # The exc_info propagated into __aexit__ carries the RuntimeError.
    assert session_cm.aexit_calls[0][0] is RuntimeError


async def test_transport_aenter_failure_propagates_without_session(monkeypatch):
    transport_cm = FakeTransportCM(enter_exc=ConnectionError("refused"))
    # ClientSession should never be constructed if transport __aenter__ fails.
    import mcp

    sentinel = MagicMock(side_effect=AssertionError("must not construct session"))
    monkeypatch.setattr(mcp, "ClientSession", sentinel, raising=True)

    mgr = MCPClientManager()
    with pytest.raises(ConnectionError, match="refused"):
        await mgr._connect_transport("srv", transport_cm)
    assert "srv" not in mgr._servers
    # Transport __aexit__ is NOT called because __aenter__ itself raised.
    assert transport_cm.aexit_calls == []


async def test_discover_tools_failure_tears_down_everything(monkeypatch):
    session = _make_session()
    session.list_tools = AsyncMock(side_effect=ValueError("list failed"))
    session_cm = FakeSessionCM(session)
    transport_cm = FakeTransportCM()
    _patch_clientsession(monkeypatch, session_cm)

    mgr = MCPClientManager()
    with pytest.raises(ValueError, match="list failed"):
        await mgr._connect_transport("srv", transport_cm)

    assert "srv" not in mgr._servers
    assert len(session_cm.aexit_calls) == 1
    assert len(transport_cm.aexit_calls) == 1


# ---------------------------------------------------------------------------
# connect_all: dispatch by transport + retry/backoff
# ---------------------------------------------------------------------------


async def test_connect_all_dispatches_stdio_and_sse(monkeypatch):
    called = []

    async def fake_stdio(self, name, cfg):
        called.append(("stdio", name, cfg))

    async def fake_sse(self, name, cfg):
        called.append(("sse", name, cfg))

    monkeypatch.setattr(MCPClientManager, "_connect_stdio", fake_stdio)
    monkeypatch.setattr(MCPClientManager, "_connect_sse", fake_sse)

    mgr = MCPClientManager()
    config = {
        "fs": {"transport": "stdio", "command": "x"},
        "web": {"transport": "sse", "url": "http://h"},
    }
    await mgr.connect_all(config)

    assert ("stdio", "fs", config["fs"]) in called
    assert ("sse", "web", config["web"]) in called


async def test_connect_all_retries_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    async def flaky_stdio(self, name, cfg):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("not yet")

    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(MCPClientManager, "_connect_stdio", flaky_stdio)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    mgr = MCPClientManager()
    await mgr.connect_all({"fs": {"transport": "stdio", "command": "x"}})

    # Succeeded on third attempt; two backoff sleeps with exponential delays.
    assert attempts["n"] == 3
    assert sleeps == [1, 2]


async def test_connect_all_gives_up_after_max_attempts(monkeypatch):
    attempts = {"n": 0}

    async def always_fail(self, name, cfg):
        attempts["n"] += 1
        raise ConnectionError("down")

    async def fake_sleep(d):
        pass

    monkeypatch.setattr(MCPClientManager, "_connect_stdio", always_fail)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    mgr = MCPClientManager()
    # Should not raise even when all attempts fail.
    await mgr.connect_all(
        {"fs": {"transport": "stdio", "command": "x"}}, max_attempts=2
    )
    assert attempts["n"] == 2
    assert mgr._servers == {}


async def test_connect_all_max_attempts_floor_is_one(monkeypatch):
    attempts = {"n": 0}

    async def always_fail(self, name, cfg):
        attempts["n"] += 1
        raise ConnectionError("down")

    monkeypatch.setattr(MCPClientManager, "_connect_stdio", always_fail)

    mgr = MCPClientManager()
    # 0 (and negatives) clamp to a single attempt; no sleep is reached.
    await mgr.connect_all(
        {"fs": {"transport": "stdio", "command": "x"}}, max_attempts=0
    )
    assert attempts["n"] == 1


async def test_connect_all_backoff_capped_at_ten(monkeypatch):
    async def always_fail(self, name, cfg):
        raise ConnectionError("down")

    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(MCPClientManager, "_connect_stdio", always_fail)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    mgr = MCPClientManager()
    await mgr.connect_all(
        {"fs": {"transport": "stdio", "command": "x"}}, max_attempts=8
    )
    # 2**attempt for attempts 0..6 (the last attempt does not sleep), capped at 10.
    assert sleeps == [1, 2, 4, 8, 10, 10, 10]


# ---------------------------------------------------------------------------
# _connect_stdio / _connect_sse: wiring to transports
# ---------------------------------------------------------------------------


async def test_connect_stdio_builds_params_and_connects(monkeypatch):
    import mcp
    import mcp.client.stdio as stdio_mod

    captured = {}

    def fake_stdio_client(params):
        captured["params"] = params
        return FakeTransportCM()

    monkeypatch.setattr(stdio_mod, "stdio_client", fake_stdio_client, raising=True)
    monkeypatch.setattr(
        mcp_client_mod, "sanitize_mcp_env", lambda env: {"SAFE": "1"}, raising=True
    )

    session = _make_session(tools=[_tool("t", "d", {"type": "object"})])
    _patch_clientsession(monkeypatch, FakeSessionCM(session))

    mgr = MCPClientManager()
    await mgr._connect_stdio(
        "fs",
        {"command": "mytool", "args": ["--flag"], "env": {"USER_VAR": "x"}},
    )

    params = captured["params"]
    assert isinstance(params, mcp.StdioServerParameters)
    assert params.command == "mytool"
    assert params.args == ["--flag"]
    assert params.env == {"SAFE": "1"}
    assert "fs" in mgr._servers


async def test_connect_stdio_defaults_args_when_missing(monkeypatch):
    import mcp.client.stdio as stdio_mod

    captured = {}

    def fake_stdio_client(params):
        captured["params"] = params
        return FakeTransportCM()

    monkeypatch.setattr(stdio_mod, "stdio_client", fake_stdio_client, raising=True)
    session = _make_session()
    _patch_clientsession(monkeypatch, FakeSessionCM(session))

    mgr = MCPClientManager()
    await mgr._connect_stdio("fs", {"command": "mytool"})
    assert captured["params"].args == []


async def test_connect_sse_uses_url(monkeypatch):
    import mcp.client.sse as sse_mod

    captured = {}

    def fake_sse_client(url):
        captured["url"] = url
        return FakeTransportCM()

    monkeypatch.setattr(sse_mod, "sse_client", fake_sse_client, raising=True)
    session = _make_session()
    _patch_clientsession(monkeypatch, FakeSessionCM(session))

    mgr = MCPClientManager()
    await mgr._connect_sse("web", {"url": "http://example/mcp"})
    assert captured["url"] == "http://example/mcp"
    assert "web" in mgr._servers


# ---------------------------------------------------------------------------
# call_tool: routing + result extraction + error mapping
# ---------------------------------------------------------------------------


async def test_call_tool_success_joins_text_content():
    mgr = MCPClientManager()
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            isError=False,
            content=[_content_text("line1"), _content_text("line2")],
        )
    )
    mgr._tool_to_server["echo"] = "srv"
    mgr._servers["srv"] = {"session": session}

    result = await mgr.call_tool("echo", {"a": 1})
    assert result == "line1\nline2"
    session.call_tool.assert_awaited_once_with("echo", {"a": 1})


async def test_call_tool_non_text_content_str_fallback():
    mgr = MCPClientManager()
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            isError=False, content=[_content_other(123), _content_text("x")]
        )
    )
    mgr._tool_to_server["t"] = "srv"
    mgr._servers["srv"] = {"session": session}

    result = await mgr.call_tool("t", {})
    assert result == "123\nx"


async def test_call_tool_empty_content_returns_empty_string():
    mgr = MCPClientManager()
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(isError=False, content=[])
    )
    mgr._tool_to_server["t"] = "srv"
    mgr._servers["srv"] = {"session": session}

    assert await mgr.call_tool("t", {}) == ""


async def test_call_tool_is_error_returns_tool_error():
    mgr = MCPClientManager()
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            isError=True, content=[_content_text("server says nope")]
        )
    )
    mgr._tool_to_server["t"] = "srv"
    mgr._servers["srv"] = {"session": session}

    result = await mgr.call_tool("t", {})
    assert isinstance(result, ToolError)
    assert result.message == "server says nope"
    assert result.is_user_error is False
    assert result.tool_name == "t"


async def test_call_tool_is_error_non_text_content_str_fallback():
    mgr = MCPClientManager()
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            isError=True, content=[_content_other(42), _content_text("detail")]
        )
    )
    mgr._tool_to_server["t"] = "srv"
    mgr._servers["srv"] = {"session": session}

    result = await mgr.call_tool("t", {})
    assert isinstance(result, ToolError)
    # Non-text content stringified, joined with the text block.
    assert result.message == "42\ndetail"
    assert result.is_user_error is False


async def test_call_tool_is_error_with_empty_content_default_message():
    mgr = MCPClientManager()
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(isError=True, content=[])
    )
    mgr._tool_to_server["t"] = "srv"
    mgr._servers["srv"] = {"session": session}

    result = await mgr.call_tool("t", {})
    assert isinstance(result, ToolError)
    assert result.message == "MCP server returned an error"


async def test_call_tool_unknown_tool_user_error():
    mgr = MCPClientManager()
    result = await mgr.call_tool("ghost", {})
    assert isinstance(result, ToolError)
    assert "Unknown tool" in result.message
    assert result.is_user_error is True


async def test_call_tool_server_disconnected_system_error():
    mgr = MCPClientManager()
    # Tool mapped, but the server entry is gone.
    mgr._tool_to_server["t"] = "vanished"
    result = await mgr.call_tool("t", {})
    assert isinstance(result, ToolError)
    assert "not connected" in result.message
    assert result.is_user_error is False


async def test_call_tool_timeout_returns_tool_error():
    mgr = MCPClientManager()
    session = MagicMock()

    async def never():
        await asyncio.Event().wait()

    session.call_tool = MagicMock(return_value=never())
    mgr._tool_to_server["slow"] = "srv"
    mgr._servers["srv"] = {"session": session}

    result = await mgr.call_tool("slow", {}, timeout=0.01)
    assert isinstance(result, ToolError)
    assert "timed out" in result.message
    assert result.is_user_error is False


async def test_call_tool_exception_returns_tool_error():
    mgr = MCPClientManager()
    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=RuntimeError("kaboom"))
    mgr._tool_to_server["t"] = "srv"
    mgr._servers["srv"] = {"session": session}

    result = await mgr.call_tool("t", {})
    assert isinstance(result, ToolError)
    assert "Error calling t" in result.message
    assert "kaboom" in result.message
    assert result.is_user_error is False


# ---------------------------------------------------------------------------
# disconnect_all: teardown ordering + error resilience
# ---------------------------------------------------------------------------


async def test_disconnect_all_closes_session_then_transport_and_clears():
    mgr = MCPClientManager()
    order = []

    session_cm = MagicMock()
    session_cm.__aexit__ = AsyncMock(side_effect=lambda *a: order.append("session"))
    transport_cm = MagicMock()
    transport_cm.__aexit__ = AsyncMock(side_effect=lambda *a: order.append("transport"))

    mgr._servers["srv"] = {
        "session": MagicMock(),
        "session_cm": session_cm,
        "transport_cm": transport_cm,
    }
    mgr._tool_to_server["t"] = "srv"
    mgr._tools = [{"function": {"name": "t"}}]

    await mgr.disconnect_all()

    # Session closed before transport.
    assert order == ["session", "transport"]
    session_cm.__aexit__.assert_awaited_once_with(None, None, None)
    transport_cm.__aexit__.assert_awaited_once_with(None, None, None)
    # State fully cleared.
    assert mgr._servers == {}
    assert mgr._tool_to_server == {}
    assert mgr._tools == []


async def test_disconnect_all_suppresses_close_errors():
    mgr = MCPClientManager()
    session_cm = MagicMock()
    session_cm.__aexit__ = AsyncMock(side_effect=RuntimeError("close fail"))
    transport_cm = MagicMock()
    transport_cm.__aexit__ = AsyncMock()

    mgr._servers["srv"] = {
        "session": MagicMock(),
        "session_cm": session_cm,
        "transport_cm": transport_cm,
    }

    # Must not raise despite session_cm failing; transport still attempted.
    await mgr.disconnect_all()
    transport_cm.__aexit__.assert_awaited_once()
    assert mgr._servers == {}


async def test_disconnect_all_handles_missing_cms():
    mgr = MCPClientManager()
    # Server dict missing the cm keys (defensive .get(...) path).
    mgr._servers["srv"] = {"session": MagicMock()}
    await mgr.disconnect_all()
    assert mgr._servers == {}
