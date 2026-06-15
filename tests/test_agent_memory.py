"""Tests for olmlx.engine.agent.memory and the store's FTS5 memory table."""

import pytest

from olmlx.engine.agent.memory import MemoryManager
from olmlx.engine.agent.store import AgentStore


@pytest.fixture
def store(tmp_path):
    s = AgentStore(tmp_path / "agent.db")
    yield s
    s.close()


class TestStoreMemory:
    async def test_add_and_search(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        await store.add_memory("r1", "the capital of France is Paris")
        await store.add_memory("r1", "bananas are yellow fruit")
        hits = await store.search_memory("r1", "France capital", limit=5)
        assert len(hits) == 1
        assert "Paris" in hits[0]["text"]

    async def test_search_scoped_to_run(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        await store.create_run(run_id="r2", goal="g", model="m", config={})
        await store.add_memory("r1", "secret alpha token")
        await store.add_memory("r2", "secret beta token")
        hits = await store.search_memory("r1", "secret token", limit=5)
        assert [h["text"] for h in hits] == ["secret alpha token"]

    async def test_search_punctuation_no_crash(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        await store.add_memory("r1", "ran tests: all green (100%)")
        # A query full of FTS5 syntax chars must not raise.
        hits = await store.search_memory("r1", 'tests: "all" (green)* -bad', limit=5)
        assert any("green" in h["text"] for h in hits)

    async def test_recent_and_count(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        for i in range(3):
            await store.add_memory("r1", f"note {i}")
        assert await store.count_memory("r1") == 3
        recent = await store.recent_memory("r1", limit=2)
        assert [r["text"] for r in recent] == ["note 2", "note 1"]

    async def test_delete(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        rid = await store.add_memory("r1", "delete me")
        await store.add_memory("r1", "keep me")
        await store.delete_memory([rid])
        remaining = [m["text"] for m in await store.list_memory("r1")]
        assert remaining == ["keep me"]

    async def test_survives_reopen(self, tmp_path):
        path = tmp_path / "agent.db"
        s1 = AgentStore(path)
        await s1.create_run(run_id="r1", goal="g", model="m", config={})
        await s1.add_memory("r1", "durable knowledge about widgets")
        s1.close()
        s2 = AgentStore(path)
        try:
            hits = await s2.search_memory("r1", "widgets", limit=5)
            assert len(hits) == 1
        finally:
            s2.close()


class TestMemoryManager:
    async def test_record_and_recall(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        mem = MemoryManager(store, "r1", recall_k=3)
        await mem.record("the deploy key lives in vault path kv/deploy")
        await mem.record("unrelated chatter about lunch")
        results = await mem.recall("deploy key vault")
        assert any("vault" in r for r in results)

    async def test_summarization_on_overflow(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        seen: list[list[str]] = []

        async def fake_summarizer(texts):
            seen.append(list(texts))
            return "SUMMARY"

        mem = MemoryManager(store, "r1", max_entries=3, summarizer=fake_summarizer)
        for i in range(4):
            await mem.record(f"fact number {i}")
        # Overflowed at the 4th: oldest entries summarized, count back <= max.
        assert seen, "summarizer should have been called"
        assert await store.count_memory("r1") <= 3
        scopes = [m["scope"] for m in await store.list_memory("r1")]
        assert "summary" in scopes

    async def test_summarization_fallback_without_summarizer(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        mem = MemoryManager(store, "r1", max_entries=2)
        for i in range(3):
            await mem.record(f"item {i}")
        texts = [m["text"] for m in await store.list_memory("r1")]
        assert any(t.startswith("Summary of earlier progress:") for t in texts)


class TestInjectContext:
    class FakeSession:
        def __init__(self, messages):
            self.messages = messages

    async def test_inject_appends_block_to_system(self, store):
        await store.create_run(
            run_id="r1", goal="ship the feature", model="m", config={}
        )
        mem = MemoryManager(store, "r1")
        await mem.record("decided to use approach A")
        session = self.FakeSession([{"role": "system", "content": "base prompt"}])
        await mem.inject_context(session, "ship the feature")
        content = session.messages[0]["content"]
        assert "base prompt" in content
        assert "Agent memory" in content
        assert "approach A" in content

    async def test_inject_is_idempotent(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        mem = MemoryManager(store, "r1")
        await mem.record("first note")
        session = self.FakeSession([{"role": "system", "content": "base"}])
        await mem.inject_context(session, "G")
        await mem.record("second note")
        await mem.inject_context(session, "G")
        content = session.messages[0]["content"]
        # Only one memory block (not accreting), and it has the latest note.
        assert content.count("## Agent memory") == 1
        assert "second note" in content

    async def test_inject_with_backslash_note_does_not_raise(self, store):
        # A remembered note with backslash sequences (Windows path, regex) must
        # not be interpreted as a regex replacement backreference on re-inject.
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        mem = MemoryManager(store, "r1")
        await mem.record(r"deploy key at C:\1users\secret and pattern \g<0>")
        session = self.FakeSession([{"role": "system", "content": "base"}])
        await mem.inject_context(session, "G")  # first inject creates the block
        await mem.inject_context(session, "G")  # second inject re-subs the block
        content = session.messages[0]["content"]
        assert r"C:\1users\secret" in content
        assert content.count("## Agent memory") == 1

    async def test_inject_without_system_message(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        mem = MemoryManager(store, "r1")
        await mem.record("a note")
        session = self.FakeSession([{"role": "user", "content": "hi"}])
        await mem.inject_context(session, "G")
        assert session.messages[0]["role"] == "system"
        assert "Agent memory" in session.messages[0]["content"]
