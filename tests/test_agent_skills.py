"""Tests for self-improving skills (issue #448).

Covers the writable skill path (chat/skills.py), the create_skill tool, SQLite
persistence, and a learned skill being offered to a later run.
"""

import pytest

from olmlx.chat.config import ChatConfig
from olmlx.chat.errors import ToolError
from olmlx.chat.skills import (
    SkillManager,
    load_skills_from_dir,
    validate_skill_name,
    write_skill_file,
)
from olmlx.engine.agent.orchestrator import AgentContext
from olmlx.engine.agent.store import AgentStore
from olmlx.engine.agent.tools import AgentToolManager


@pytest.fixture
def context(tmp_path):
    store = AgentStore(tmp_path / "agent.db")
    yield AgentContext(run_id="r1", store=store)
    store.close()


class TestWriteSkillFile:
    def test_writes_parseable_file(self, tmp_path):
        path = write_skill_file(
            tmp_path / "skills", "deploy-helper", "deploy things", "Step 1. Do X."
        )
        assert path.exists()
        skills = load_skills_from_dir(tmp_path / "skills")
        assert "deploy-helper" in skills
        assert skills["deploy-helper"].description == "deploy things"
        assert "Step 1" in skills["deploy-helper"].content

    def test_rejects_bad_name(self, tmp_path):
        for bad in ["../escape", "has/slash", "", "has space", "-leading"]:
            with pytest.raises(ValueError):
                write_skill_file(tmp_path / "skills", bad, "d", "body")

    def test_rejects_empty_body(self, tmp_path):
        with pytest.raises(ValueError):
            write_skill_file(tmp_path / "skills", "ok", "d", "   ")

    def test_validate_skill_name_accepts_good(self):
        for good in ["a", "deploy-helper", "snake_case", "Mixed123"]:
            assert validate_skill_name(good) == good


class TestSkillManagerCreate:
    def test_create_registers_and_persists(self, tmp_path):
        mgr = SkillManager(tmp_path / "skills")
        mgr.load()
        mgr.create_skill("new-skill", "does a thing", "Instructions here.")
        assert mgr.get_skill("new-skill") is not None
        # Reloading from disk finds it too.
        mgr2 = SkillManager(tmp_path / "skills")
        mgr2.load()
        assert mgr2.get_skill("new-skill") is not None


class TestStoreSkills:
    async def test_upsert_and_get(self, tmp_path):
        store = AgentStore(tmp_path / "agent.db")
        try:
            await store.upsert_skill("s1", "desc", "body", source_run="r1")
            got = await store.get_skill("s1")
            assert got["body"] == "body"
            assert got["source_run"] == "r1"
            # Upsert overwrites.
            await store.upsert_skill("s1", "desc2", "body2", source_run="r2")
            got = await store.get_skill("s1")
            assert got["body"] == "body2"
            assert len(await store.list_skills()) == 1
        finally:
            store.close()


class TestCreateSkillTool:
    @pytest.fixture
    def tools(self, context, tmp_path):
        config = ChatConfig(model_name="m", skills_dir=tmp_path / "skills")
        return AgentToolManager(config, context)

    def test_in_tool_names(self, tools):
        assert "create_skill" in tools.tool_names

    async def test_create_skill_persists_to_dir_and_sqlite(
        self, tools, context, tmp_path
    ):
        await context.store.create_run(run_id="r1", goal="g", model="m", config={})
        result = await tools.call_tool(
            "create_skill",
            {
                "name": "git-bisect",
                "description": "find a regression",
                "body": "Run git bisect start ...",
            },
        )
        assert "created skill" in result.lower()
        # On disk + loadable.
        skills = load_skills_from_dir(tmp_path / "skills")
        assert "git-bisect" in skills
        # In SQLite with source run.
        persisted = await context.store.get_skill("git-bisect")
        assert persisted["source_run"] == "r1"

    async def test_malformed_name_rejected(self, tools, context):
        await context.store.create_run(run_id="r1", goal="g", model="m", config={})
        result = await tools.call_tool(
            "create_skill",
            {"name": "../evil", "description": "d", "body": "b"},
        )
        assert isinstance(result, ToolError)
        assert result.is_user_error
        # Nothing persisted.
        assert await context.store.get_skill("../evil") is None


class TestOfferedToLaterRun:
    async def test_learned_skill_in_next_run_index(self, tmp_path, monkeypatch):
        """A skill authored by one run is offered to a later run's session."""
        from olmlx.engine.agent.service import AgentService

        monkeypatch.setattr(
            "olmlx.config.settings.agent_skills_dir", tmp_path / "skills"
        )
        store = AgentStore(tmp_path / "agent.db")
        service = AgentService(store=store, manager_getter=lambda: object())
        try:
            # Simulate a prior run having authored a skill.
            write_skill_file(
                tmp_path / "skills",
                "learned-skill",
                "a learned capability",
                "Detailed steps.",
            )
            run = await store.create_run(
                run_id="later", goal="use prior knowledge", model="m", config={}
            )
            context = AgentContext(run_id="later", store=store)
            session = service._default_session(run, context)
            sys_prompt = session.messages[0]["content"]
            assert "learned-skill" in sys_prompt
        finally:
            store.close()

    async def test_materialize_skills_from_sqlite(self, tmp_path, monkeypatch):
        from olmlx.engine.agent.service import AgentService

        monkeypatch.setattr(
            "olmlx.config.settings.agent_skills_dir", tmp_path / "skills"
        )
        store = AgentStore(tmp_path / "agent.db")
        service = AgentService(store=store, manager_getter=lambda: object())
        try:
            await store.upsert_skill("persisted", "from db", "Body from the database.")
            await service.startup()  # should write the skill file to disk
            skills = load_skills_from_dir(tmp_path / "skills")
            assert "persisted" in skills
        finally:
            store.close()
