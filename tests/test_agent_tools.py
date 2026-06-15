"""Tests for olmlx.engine.agent.tools — agent control tools."""

import pytest

from olmlx.chat.config import ChatConfig
from olmlx.engine.agent.orchestrator import AgentContext
from olmlx.engine.agent.store import AgentStore
from olmlx.engine.agent.tools import AgentToolManager


@pytest.fixture
def context(tmp_path):
    store = AgentStore(tmp_path / "agent.db")
    yield AgentContext(run_id="r1", store=store)
    store.close()


@pytest.fixture
def tools(context):
    return AgentToolManager(ChatConfig(model_name="m"), context)


class TestAgentToolManager:
    def test_finish_in_tool_names(self, tools):
        assert "finish" in tools.tool_names

    def test_inherits_builtin_tools(self, tools):
        # A representative builtin still present alongside the agent tools.
        assert "bash" in tools.tool_names
        assert "read_file" in tools.tool_names

    def test_finish_in_definitions(self, tools):
        names = {d["function"]["name"] for d in tools.get_tool_definitions()}
        assert "finish" in names
        assert "bash" in names

    async def test_finish_sets_context_and_returns_confirmation(self, tools, context):
        result = await tools.call_tool("finish", {"summary": "did the thing"})
        assert "complete" in result.lower()
        assert context.finished is True
        assert context.finish_summary == "did the thing"

    async def test_builtin_still_dispatches(self, tools):
        # edit_file with a nonexistent file returns a ToolError, proving the
        # call routed to the inherited builtin handler (not the agent branch).
        result = await tools.call_tool(
            "edit_file", {"path": "/nonexistent/xyz", "old_text": "a", "new_text": "b"}
        )
        from olmlx.chat.errors import ToolError

        assert isinstance(result, ToolError)
