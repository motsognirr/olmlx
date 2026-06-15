"""Tests for subagent delegation (issue #449)."""

import asyncio

import pytest

from olmlx.chat.config import ChatConfig
from olmlx.chat.errors import ToolError
from olmlx.engine.agent.delegate import DelegateError, DelegateRunner
from olmlx.engine.agent.service import AgentService
from olmlx.engine.agent.store import AgentStore
from olmlx.engine.agent.tools import AgentToolManager


@pytest.fixture
def store(tmp_path):
    s = AgentStore(tmp_path / "agent.db")
    yield s
    s.close()


class _FinishSession:
    """Child session that immediately finishes with its goal echoed back."""

    def __init__(self, run):
        self.goal = run["goal"]
        self.messages = [{"role": "system", "content": "sys"}]

    async def send_message(self, user_text):
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": "done"})
        yield {
            "type": "tool_call",
            "name": "finish",
            "arguments": {"summary": f"finished: {self.goal}"},
            "id": "t1",
        }
        yield {"type": "done"}


def _finish_factory(run, context, manager):
    return _FinishSession(run)


def _raising_factory(run, context, manager):
    class _Boom:
        messages = [{"role": "system", "content": "s"}]

        async def send_message(self, user_text):
            raise RuntimeError("child blew up")
            yield  # pragma: no cover  (make it an async generator)

    return _Boom()


def _service(store, factory, **settings_over):
    svc = AgentService(
        store=store, manager_getter=lambda: object(), session_factory=factory
    )
    return svc


class TestDelegateRunner:
    async def test_delegate_creates_and_runs_child(self, store):
        svc = _service(store, _finish_factory)
        await store.create_run(run_id="p", goal="parent", model="m", config={})
        result = await svc._delegate_runner.delegate(parent_id="p", goal="sub task")
        assert result["status"] == "finished"
        assert "sub task" in result["result"]

    async def test_child_linked_to_parent(self, store):
        svc = _service(store, _finish_factory)
        await store.create_run(run_id="p", goal="parent", model="qwen", config={})
        await svc._delegate_runner.delegate(parent_id="p", goal="sub")
        children = await store.list_children("p")
        assert len(children) == 1
        assert children[0]["parent_id"] == "p"
        assert children[0]["depth"] == 1
        assert children[0]["model"] == "qwen"  # inherits parent model

    async def test_depth_bound_enforced(self, store, monkeypatch):
        monkeypatch.setattr("olmlx.config.settings.agent_max_subagent_depth", 1)
        svc = _service(store, _finish_factory)
        # Parent already at the max depth → its child would exceed.
        await store.create_run(run_id="deep", goal="g", model="m", config={}, depth=1)
        with pytest.raises(DelegateError, match="depth"):
            await svc._delegate_runner.delegate(parent_id="deep", goal="sub")

    async def test_fanout_bound_enforced(self, store, monkeypatch):
        monkeypatch.setattr("olmlx.config.settings.agent_max_subagent_fanout", 2)
        svc = _service(store, _finish_factory)
        await store.create_run(run_id="p", goal="g", model="m", config={})
        await svc._delegate_runner.delegate(parent_id="p", goal="a")
        await svc._delegate_runner.delegate(parent_id="p", goal="b")
        with pytest.raises(DelegateError, match="fan-out"):
            await svc._delegate_runner.delegate(parent_id="p", goal="c")

    async def test_child_failure_returned(self, store):
        svc = _service(store, _raising_factory)
        await store.create_run(run_id="p", goal="g", model="m", config={})
        result = await svc._delegate_runner.delegate(parent_id="p", goal="sub")
        assert result["status"] == "failed"
        assert "blew up" in (result["error"] or "")

    async def test_empty_goal_rejected(self, store):
        svc = _service(store, _finish_factory)
        await store.create_run(run_id="p", goal="g", model="m", config={})
        with pytest.raises(DelegateError):
            await svc._delegate_runner.delegate(parent_id="p", goal="   ")


class TestSerializedExecution:
    async def test_children_generate_serially(self, store):
        state = {"active": 0, "max_active": 0}

        def factory(run, context, manager):
            class _TrackSession:
                def __init__(self):
                    self.messages = [{"role": "system", "content": "s"}]

                async def send_message(self, user_text):
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                    await asyncio.sleep(0.02)
                    state["active"] -= 1
                    self.messages.append({"role": "assistant", "content": "x"})
                    yield {
                        "type": "tool_call",
                        "name": "finish",
                        "arguments": {"summary": "ok"},
                        "id": "t",
                    }
                    yield {"type": "done"}

            return _TrackSession()

        svc = _service(store, factory)
        await store.create_run(run_id="p1", goal="g", model="m", config={})
        await store.create_run(run_id="p2", goal="g", model="m", config={})
        runner: DelegateRunner = svc._delegate_runner
        await asyncio.gather(
            runner.delegate(parent_id="p1", goal="a"),
            runner.delegate(parent_id="p2", goal="b"),
        )
        # The shared semaphore must have prevented overlapping generation.
        assert state["max_active"] == 1


class TestDelegateTool:
    @pytest.fixture
    def parent_tools(self, store):
        svc = _service(store, _finish_factory)
        return svc

    async def test_delegate_tool_runs_child(self, store):
        svc = _service(store, _finish_factory)
        parent = await store.create_run(
            run_id="p", goal="parent goal", model="m", config={}
        )
        context = svc._make_context(parent)
        tools = AgentToolManager(ChatConfig(model_name="m"), context)
        assert "delegate" in tools.tool_names
        result = await tools.call_tool("delegate", {"goal": "do subtask"})
        assert "subagent finished" in result.lower()
        assert len(await store.list_children("p")) == 1

    async def test_delegate_tool_surfaces_failure_as_toolerror(self, store):
        svc = _service(store, _raising_factory)
        parent = await store.create_run(run_id="p", goal="g", model="m", config={})
        context = svc._make_context(parent)
        tools = AgentToolManager(ChatConfig(model_name="m"), context)
        result = await tools.call_tool("delegate", {"goal": "sub"})
        assert isinstance(result, ToolError)
        assert "failed" in result.message.lower()

    async def test_delegate_tool_bound_violation_is_toolerror(self, store, monkeypatch):
        monkeypatch.setattr("olmlx.config.settings.agent_max_subagent_depth", 0)
        svc = _service(store, _finish_factory)
        parent = await store.create_run(run_id="p", goal="g", model="m", config={})
        context = svc._make_context(parent)
        tools = AgentToolManager(ChatConfig(model_name="m"), context)
        result = await tools.call_tool("delegate", {"goal": "sub"})
        assert isinstance(result, ToolError)
        assert result.is_user_error
