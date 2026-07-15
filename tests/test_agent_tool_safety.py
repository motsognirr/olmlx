"""Agent tool-safety policy + workspace write confinement (issue #611).

The headless autonomous agent has no human to CONFIRM tool calls, and it can
ingest untrusted web content via ``web_fetch``. Before this change it built its
``ChatSession`` with no ``tool_safety`` policy, so every builtin tool — ``bash``,
``write_file``, ``edit_file`` — was auto-allowed, letting prompt-injected page
content drive arbitrary shell/file actions. It also wrote arbitrary absolute
paths (``_resolve_path``'s traversal guard only covered relative paths).
"""

from pathlib import Path

import pytest

from olmlx.chat.builtin_tools import BuiltinToolManager, _resolve_path
from olmlx.chat.config import ChatConfig
from olmlx.chat.errors import ToolError
from olmlx.chat.tool_safety import ToolPolicy
from olmlx.config import Settings
from olmlx.engine.agent.orchestrator import AgentContext
from olmlx.engine.agent.service import AgentService
from olmlx.engine.agent.store import AgentStore


# --------------------------------------------------------------------------
# _resolve_path workspace confinement (the absolute-path bypass fix)
# --------------------------------------------------------------------------
class TestResolvePathConfinement:
    def test_absolute_path_outside_root_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="workspace"):
            _resolve_path("/etc/passwd", confine_root=tmp_path)

    def test_absolute_path_inside_root_ok(self, tmp_path):
        target = tmp_path / "sub" / "f.txt"
        assert _resolve_path(str(target), confine_root=tmp_path) == target.resolve()

    def test_relative_path_confined_to_root(self, tmp_path):
        assert (
            _resolve_path("a/b.txt", confine_root=tmp_path)
            == (tmp_path / "a" / "b.txt").resolve()
        )

    def test_relative_traversal_escape_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            _resolve_path("../../etc/passwd", confine_root=tmp_path)

    def test_no_confine_root_absolute_passes_through(self, tmp_path):
        # Unchanged legacy behavior for interactive chat (write_root=None).
        p = tmp_path / "x.txt"
        assert _resolve_path(str(p)) == p.resolve()


# --------------------------------------------------------------------------
# write_file / edit_file honor config.write_root
# --------------------------------------------------------------------------
class TestWriteConfinement:
    def _mgr(self, tmp_path, write_root):
        cfg = ChatConfig(
            model_name="m", plans_dir=tmp_path / "plans", write_root=write_root
        )
        return BuiltinToolManager(cfg)

    @pytest.mark.asyncio
    async def test_write_file_outside_workspace_blocked(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside"
        mgr = self._mgr(tmp_path, workspace)
        res = await mgr.call_tool(
            "write_file", {"path": str(outside / "evil.txt"), "content": "x"}
        )
        assert isinstance(res, ToolError)
        assert not (outside / "evil.txt").exists()

    @pytest.mark.asyncio
    async def test_write_file_inside_workspace_ok(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        mgr = self._mgr(tmp_path, workspace)
        res = await mgr.call_tool(
            "write_file", {"path": str(workspace / "ok.txt"), "content": "hi"}
        )
        assert not isinstance(res, ToolError)
        assert (workspace / "ok.txt").read_text() == "hi"

    @pytest.mark.asyncio
    async def test_write_file_no_workspace_allows_absolute(self, tmp_path):
        # write_root=None (interactive default) — behavior unchanged.
        mgr = self._mgr(tmp_path, None)
        target = tmp_path / "abs.txt"
        res = await mgr.call_tool("write_file", {"path": str(target), "content": "hi"})
        assert not isinstance(res, ToolError)
        assert target.read_text() == "hi"

    @pytest.mark.asyncio
    async def test_edit_file_outside_workspace_blocked(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("orig")
        mgr = self._mgr(tmp_path, workspace)
        res = await mgr.call_tool(
            "edit_file",
            {"path": str(outside), "old_text": "orig", "new_text": "pwned"},
        )
        assert isinstance(res, ToolError)
        assert outside.read_text() == "orig"


# --------------------------------------------------------------------------
# Agent session policy wiring
# --------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path):
    s = AgentStore(tmp_path / "agent.db")
    yield s
    s.close()


def _service(store, tmp_path, **over):
    over.setdefault("agent_skills_dir", tmp_path / "skills")
    return AgentService(
        store=store,
        manager_getter=lambda: object(),
        settings=Settings(**over),
    )


def _session(svc, store):
    run = {"model": "m", "goal": "do the thing"}
    ctx = AgentContext(run_id="r1", store=store)
    return svc._default_session(run, ctx)


class TestAgentSessionPolicy:
    def test_default_gates_dangerous_tools_as_auto(self, store, tmp_path):
        ts = _session(_service(store, tmp_path), store).tool_safety
        assert ts is not None
        assert ts.get_policy("bash") == ToolPolicy.AUTO
        assert ts.get_policy("write_file") == ToolPolicy.AUTO
        assert ts.get_policy("edit_file") == ToolPolicy.AUTO
        # Safe tools stay ALLOW so the agent still runs autonomously.
        assert ts.get_policy("read_file") == ToolPolicy.ALLOW
        assert ts.get_policy("web_fetch") == ToolPolicy.ALLOW
        assert ts.get_policy("finish") == ToolPolicy.ALLOW

    def test_local_tool_safety_enabled(self, store, tmp_path):
        # Without this, builtin (local) tools bypass the policy entirely.
        sess = _session(_service(store, tmp_path), store)
        assert sess.config.local_tool_safety is True

    def test_deny_policy_override(self, store, tmp_path):
        svc = _service(
            store,
            tmp_path,
            agent_shell_policy="deny",
            agent_file_write_policy="deny",
        )
        ts = _session(svc, store).tool_safety
        assert ts.get_policy("bash") == ToolPolicy.DENY
        assert ts.get_policy("write_file") == ToolPolicy.DENY
        assert ts.get_policy("edit_file") == ToolPolicy.DENY

    def test_allow_policy_override(self, store, tmp_path):
        svc = _service(
            store,
            tmp_path,
            agent_shell_policy="allow",
            agent_file_write_policy="allow",
        )
        ts = _session(svc, store).tool_safety
        assert ts.get_policy("bash") == ToolPolicy.ALLOW
        assert ts.get_policy("write_file") == ToolPolicy.ALLOW

    def test_workspace_wired_into_config(self, store, tmp_path):
        ws = tmp_path / "ws"
        svc = _service(store, tmp_path, agent_workspace_dir=ws)
        assert _session(svc, store).config.write_root == ws

    def test_workspace_defaults_to_cwd(self, store, tmp_path):
        sess = _session(_service(store, tmp_path), store)
        assert sess.config.write_root == Path.cwd()

    def test_classify_routes_bash_to_judge_not_allow(self, store, tmp_path):
        # End-to-end wiring: the session's own classifier must send bash to the
        # AUTO (judged) bucket, not straight to allow — this is the actual gate
        # (local_tool_safety + policy together). read_file stays allowed.
        sess = _session(_service(store, tmp_path), store)
        uses = [
            {"name": "bash", "input": {"command": "ls"}, "id": "1"},
            {"name": "read_file", "input": {"path": "x"}, "id": "2"},
        ]
        allow, confirm, auto, deny = sess._classify_tool_calls(uses)
        assert [tu["name"] for tu in auto] == ["bash"]
        assert "bash" not in [tu["name"] for tu in allow]
        assert "read_file" in [tu["name"] for tu in allow]

    def test_classify_denies_bash_under_deny_policy(self, store, tmp_path):
        sess = _session(_service(store, tmp_path, agent_shell_policy="deny"), store)
        uses = [{"name": "bash", "input": {"command": "ls"}, "id": "1"}]
        allow, confirm, auto, deny = sess._classify_tool_calls(uses)
        assert [tu["name"] for tu in deny] == ["bash"]


# --------------------------------------------------------------------------
# LLM safety judge (AUTO path). Fail-closed on anything but an explicit ALLOW.
# --------------------------------------------------------------------------
def _fake_generate_chat(verdict, *, raise_exc=False):
    async def fake(manager, model, messages, **kw):
        if raise_exc:
            raise RuntimeError("boom")

        async def gen():
            yield {"text": verdict}
            yield {"done": True}

        return gen()

    return fake


class TestToolSafetyJudge:
    @pytest.mark.asyncio
    async def test_allows_on_explicit_allow(self, store, tmp_path, monkeypatch):
        svc = _service(store, tmp_path)
        judge = svc._make_tool_safety_judge("m", "goal")
        monkeypatch.setattr(
            "olmlx.engine.inference.generate_chat", _fake_generate_chat("ALLOW")
        )
        assert await judge("bash", {"command": "ls"}, None) is True

    @pytest.mark.asyncio
    async def test_denies_on_deny(self, store, tmp_path, monkeypatch):
        svc = _service(store, tmp_path)
        judge = svc._make_tool_safety_judge("m", "goal")
        monkeypatch.setattr(
            "olmlx.engine.inference.generate_chat", _fake_generate_chat("DENY")
        )
        assert await judge("bash", {"command": "rm -rf /"}, None) is False

    @pytest.mark.asyncio
    async def test_denies_on_unparseable(self, store, tmp_path, monkeypatch):
        svc = _service(store, tmp_path)
        judge = svc._make_tool_safety_judge("m", "goal")
        monkeypatch.setattr(
            "olmlx.engine.inference.generate_chat", _fake_generate_chat("maybe?")
        )
        assert await judge("bash", {"command": "ls"}, None) is False

    @pytest.mark.asyncio
    async def test_denies_on_exception(self, store, tmp_path, monkeypatch):
        svc = _service(store, tmp_path)
        judge = svc._make_tool_safety_judge("m", "goal")
        monkeypatch.setattr(
            "olmlx.engine.inference.generate_chat",
            _fake_generate_chat("ALLOW", raise_exc=True),
        )
        assert await judge("bash", {"command": "ls"}, None) is False
