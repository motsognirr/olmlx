"""Tests for olmlx.engine.agent.store — SQLite run lifecycle + checkpoints."""

import pytest

from olmlx.engine.agent.store import AgentStore


@pytest.fixture
def store(tmp_path):
    s = AgentStore(tmp_path / "agent.db")
    yield s
    s.close()


class TestRunLifecycle:
    async def test_create_and_get_run(self, store):
        run = await store.create_run(
            run_id="r1", goal="do a thing", model="qwen3:latest", config={"a": 1}
        )
        assert run["id"] == "r1"
        assert run["goal"] == "do a thing"
        assert run["status"] == "queued"
        assert run["model"] == "qwen3:latest"
        assert run["config"] == {"a": 1}
        assert run["parent_id"] is None
        assert run["depth"] == 0
        assert run["iterations"] == 0
        assert run["tokens"] == 0

        fetched = await store.get_run("r1")
        assert fetched == run

    async def test_get_missing_run_returns_none(self, store):
        assert await store.get_run("nope") is None

    async def test_update_run(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        await store.update_run(
            "r1", status="running", iterations=3, tokens=120, result="done"
        )
        run = await store.get_run("r1")
        assert run["status"] == "running"
        assert run["iterations"] == 3
        assert run["tokens"] == 120
        assert run["result"] == "done"

    async def test_list_runs_orders_newest_first(self, store):
        await store.create_run(run_id="r1", goal="g1", model="m", config={})
        await store.create_run(run_id="r2", goal="g2", model="m", config={})
        runs = await store.list_runs()
        ids = [r["id"] for r in runs]
        assert set(ids) == {"r1", "r2"}

    async def test_config_json_roundtrip(self, store):
        cfg = {"max_iterations": 5, "nested": {"x": [1, 2, 3]}}
        await store.create_run(run_id="r1", goal="g", model="m", config=cfg)
        run = await store.get_run("r1")
        assert run["config"] == cfg

    async def test_parent_and_depth_persist(self, store):
        await store.create_run(run_id="root", goal="g", model="m", config={})
        child = await store.create_run(
            run_id="c1", goal="sub", model="m", config={}, parent_id="root", depth=1
        )
        assert child["parent_id"] == "root"
        assert child["depth"] == 1

    async def test_list_children(self, store):
        await store.create_run(run_id="root", goal="g", model="m", config={})
        await store.create_run(
            run_id="c1", goal="s1", model="m", config={}, parent_id="root", depth=1
        )
        await store.create_run(
            run_id="c2", goal="s2", model="m", config={}, parent_id="root", depth=1
        )
        children = await store.list_children("root")
        assert {c["id"] for c in children} == {"c1", "c2"}


class TestCheckpoints:
    async def test_checkpoint_roundtrip(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        seq = await store.append_checkpoint("r1", msgs, iterations=1, tokens=10)
        assert seq == 0
        cp = await store.latest_checkpoint("r1")
        assert cp["messages"] == msgs
        assert cp["iterations"] == 1
        assert cp["tokens"] == 10

    async def test_latest_checkpoint_returns_most_recent(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        await store.append_checkpoint("r1", [{"role": "user", "content": "1"}], 1, 5)
        await store.append_checkpoint("r1", [{"role": "user", "content": "2"}], 2, 9)
        cp = await store.latest_checkpoint("r1")
        assert cp["iterations"] == 2
        assert cp["messages"] == [{"role": "user", "content": "2"}]

    async def test_latest_checkpoint_none_when_empty(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        assert await store.latest_checkpoint("r1") is None


class TestEvents:
    async def test_append_and_replay_events(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        await store.append_event("r1", {"type": "token", "text": "a"})
        await store.append_event("r1", {"type": "token", "text": "b"})
        events = await store.get_events("r1")
        assert [e["data"]["text"] for e in events] == ["a", "b"]
        assert [e["seq"] for e in events] == [0, 1]

    async def test_get_events_after_seq(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        for t in ("a", "b", "c"):
            await store.append_event("r1", {"type": "token", "text": t})
        events = await store.get_events("r1", after_seq=0)
        assert [e["data"]["text"] for e in events] == ["b", "c"]


class TestResumeOnRestart:
    async def test_mark_interrupted_runs(self, store):
        await store.create_run(run_id="r1", goal="g", model="m", config={})
        await store.update_run("r1", status="running")
        await store.create_run(run_id="r2", goal="g", model="m", config={})
        await store.update_run("r2", status="finished")

        interrupted = await store.mark_interrupted_runs()
        assert interrupted == ["r1"]
        assert (await store.get_run("r1"))["status"] == "interrupted"
        assert (await store.get_run("r2"))["status"] == "finished"

    async def test_persistence_across_reopen(self, tmp_path):
        path = tmp_path / "agent.db"
        s1 = AgentStore(path)
        await s1.create_run(run_id="r1", goal="survive", model="m", config={"k": 1})
        await s1.append_checkpoint("r1", [{"role": "user", "content": "x"}], 1, 3)
        s1.close()

        s2 = AgentStore(path)
        try:
            run = await s2.get_run("r1")
            assert run["goal"] == "survive"
            assert run["config"] == {"k": 1}
            cp = await s2.latest_checkpoint("r1")
            assert cp["messages"] == [{"role": "user", "content": "x"}]
        finally:
            s2.close()
