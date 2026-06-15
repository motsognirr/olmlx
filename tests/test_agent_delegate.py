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

    async def test_child_handle_cleaned_up(self, store):
        """run_child must not leak a _RunHandle after the child completes."""
        svc = _service(store, _finish_factory)
        await store.create_run(run_id="p", goal="g", model="m", config={})
        await svc._delegate_runner.delegate(parent_id="p", goal="sub")
        assert svc._handles == {}


class TestConcurrencyBounds:
    async def test_concurrent_siblings_respect_fanout(self, store, monkeypatch):
        """Concurrent delegate() from one parent (a gather'd tool turn) must not
        overshoot the fan-out cap — the check+create is atomic (TOCTOU guard)."""
        monkeypatch.setattr("olmlx.config.settings.agent_max_subagent_fanout", 2)
        svc = _service(store, _finish_factory)
        await store.create_run(run_id="p", goal="g", model="m", config={})
        runner: DelegateRunner = svc._delegate_runner
        results = await asyncio.gather(
            runner.delegate(parent_id="p", goal="a"),
            runner.delegate(parent_id="p", goal="b"),
            runner.delegate(parent_id="p", goal="c"),
            return_exceptions=True,
        )
        errs = [r for r in results if isinstance(r, DelegateError)]
        assert len(errs) == 1  # exactly one over-cap delegate rejected
        assert len(await store.list_children("p")) == 2  # no overshoot

    async def test_nested_delegation_does_not_deadlock(self, store, monkeypatch):
        """A child delegating a grandchild must not re-enter a held lock and
        deadlock (admit lock is released before run_child runs the child)."""
        monkeypatch.setattr("olmlx.config.settings.agent_max_subagent_depth", 2)

        def nesting_factory(run, context, manager):
            class _Nest:
                def __init__(self):
                    self.messages = [{"role": "system", "content": "s"}]

                async def send_message(self, user_text):
                    self.messages.append({"role": "user", "content": user_text})
                    # Each level tries to delegate one level deeper; the depth
                    # cap stops the recursion (raises, caught here).
                    try:
                        await context.delegate_runner.delegate(
                            parent_id=context.run_id, goal="deeper"
                        )
                    except DelegateError:
                        pass
                    self.messages.append({"role": "assistant", "content": "done"})
                    yield {
                        "type": "tool_call",
                        "name": "finish",
                        "arguments": {"summary": "ok"},
                        "id": "t",
                    }
                    yield {"type": "done"}

            return _Nest()

        svc = _service(store, nesting_factory)
        await store.create_run(run_id="root", goal="g", model="m", config={})
        # Would hang forever under a lock held across run_child.
        result = await asyncio.wait_for(
            svc._delegate_runner.delegate(parent_id="root", goal="child"),
            timeout=5,
        )
        assert result["status"] == "finished"
        child_id = result["id"]
        assert len(await store.list_children("root")) == 1
        # The child (depth 1) successfully delegated a grandchild (depth 2).
        assert len(await store.list_children(child_id)) == 1


class TestSubtreeCancel:
    async def test_cancel_propagates_to_children(self, store):
        from olmlx.engine.agent.orchestrator import AgentContext
        from olmlx.engine.agent.service import _RunHandle

        svc = _service(store, _finish_factory)
        await store.create_run(run_id="p", goal="g", model="m", config={})
        await store.create_run(
            run_id="c", goal="s", model="m", config={}, parent_id="p", depth=1
        )
        pctx = AgentContext(run_id="p", store=store)
        cctx = AgentContext(run_id="c", store=store)
        svc._handles["p"] = _RunHandle(cancel_event=pctx.cancel_event, context=pctx)
        svc._handles["c"] = _RunHandle(cancel_event=cctx.cancel_event, context=cctx)

        await svc.cancel("p")

        assert pctx.cancel_event.is_set()
        # Cancellation reached the in-flight child so a parent blocked awaiting
        # it can actually stop.
        assert cctx.cancel_event.is_set()
        assert (await store.get_run("c"))["status"] == "cancelled"


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
