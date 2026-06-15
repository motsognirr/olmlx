"""Tests for routers/agent.py — the autonomous agent HTTP surface."""

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient


class FakeSession:
    """Goal-driven fake ChatSession for router tests (no real inference).

    Behavior keys off the goal text so a single factory drives every case:
      - "loop"   → never calls finish (runs until cancelled / budget)
      - anything → one token then a ``finish`` tool call
    """

    def __init__(self, run):
        self.goal = run["goal"]
        self.messages = [{"role": "system", "content": "sys"}]
        self._n = 0

    async def send_message(self, user_text):
        self.messages.append({"role": "user", "content": user_text})
        self._n += 1
        yield {"type": "token", "text": "thinking"}
        if "loop" in self.goal:
            self.messages.append({"role": "assistant", "content": f"step {self._n}"})
            await asyncio.sleep(0.01)  # yield control so cancel can land
            yield {"type": "done"}
            return
        self.messages.append(
            {
                "role": "assistant",
                "content": "done",
                "tool_calls": [{"function": {"name": "finish", "arguments": "{}"}}],
            }
        )
        yield {
            "type": "tool_call",
            "name": "finish",
            "arguments": {"summary": "completed the goal"},
            "id": "t1",
        }
        yield {"type": "done"}


def _fake_factory(run, context, manager):
    return FakeSession(run)


@pytest.fixture
async def agent_client(mock_manager, monkeypatch, tmp_path):
    """A client against an app with the agent enabled and a fake session."""
    monkeypatch.setattr("olmlx.config.settings.agent_enabled", True)
    monkeypatch.setattr("olmlx.config.settings.agent_db_path", tmp_path / "agent.db")
    monkeypatch.setattr("olmlx.config.settings.agent_model", "qwen3:latest")
    monkeypatch.setattr("olmlx.app.settings.agent_enabled", True)

    from olmlx.app import create_app

    app = create_app()
    app.state.model_manager = mock_manager
    service = app.state.agent_service
    service._session_factory = _fake_factory

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, service
    await service.aclose()


async def _poll_status(client, run_id, target, timeout=2.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/v1/agent/runs/{run_id}")
        if resp.status_code == 200 and resp.json()["status"] == target:
            return resp.json()
        await asyncio.sleep(0.02)
    resp = await client.get(f"/v1/agent/runs/{run_id}")
    raise AssertionError(f"run {run_id} did not reach {target!r}; last={resp.json()}")


class TestCreateRun:
    async def test_create_runs_to_finish(self, agent_client):
        client, service = agent_client
        resp = await client.post("/v1/agent/runs", json={"goal": "do a thing"})
        assert resp.status_code == 200
        run_id = resp.json()["id"]
        assert resp.json()["goal"] == "do a thing"
        run = await _poll_status(client, run_id, "finished")
        assert run["result"] == "completed the goal"
        assert run["iterations"] == 1

    async def test_create_uses_default_model(self, agent_client):
        client, service = agent_client
        resp = await client.post("/v1/agent/runs", json={"goal": "x"})
        assert resp.json()["model"] == "qwen3:latest"

    async def test_create_empty_goal_422(self, agent_client):
        client, service = agent_client
        resp = await client.post("/v1/agent/runs", json={"goal": "   "})
        assert resp.status_code == 422

    async def test_create_missing_goal_422(self, agent_client):
        client, service = agent_client
        resp = await client.post("/v1/agent/runs", json={})
        assert resp.status_code == 422


class TestGetAndList:
    async def test_get_unknown_run_404(self, agent_client):
        client, service = agent_client
        resp = await client.get("/v1/agent/runs/nope")
        assert resp.status_code == 404

    async def test_list_runs(self, agent_client):
        client, service = agent_client
        await client.post("/v1/agent/runs", json={"goal": "first"})
        await client.post("/v1/agent/runs", json={"goal": "second"})
        resp = await client.get("/v1/agent/runs")
        assert resp.status_code == 200
        goals = {r["goal"] for r in resp.json()["runs"]}
        assert goals == {"first", "second"}


class TestCancel:
    async def test_cancel_running_run(self, agent_client):
        client, service = agent_client
        resp = await client.post("/v1/agent/runs", json={"goal": "loop forever"})
        run_id = resp.json()["id"]
        await client.post(f"/v1/agent/runs/{run_id}/cancel")
        run = await _poll_status(client, run_id, "cancelled")
        assert run["status"] == "cancelled"

    async def test_cancel_unknown_404(self, agent_client):
        client, service = agent_client
        resp = await client.post("/v1/agent/runs/nope/cancel")
        assert resp.status_code == 404


class TestResume:
    async def test_resume_interrupted_run(self, agent_client):
        client, service = agent_client
        # Seed an interrupted run directly in the store.
        await service.store.create_run(
            run_id="r-resume", goal="finish me", model="qwen3:latest", config={}
        )
        await service.store.update_run("r-resume", status="interrupted")
        resp = await client.post("/v1/agent/runs/r-resume/resume")
        assert resp.status_code == 200
        run = await _poll_status(client, "r-resume", "finished")
        assert run["status"] == "finished"

    async def test_resume_finished_run_400(self, agent_client):
        client, service = agent_client
        await service.store.create_run(run_id="r-done", goal="g", model="m", config={})
        await service.store.update_run("r-done", status="finished")
        resp = await client.post("/v1/agent/runs/r-done/resume")
        assert resp.status_code == 400

    async def test_resume_unknown_404(self, agent_client):
        client, service = agent_client
        resp = await client.post("/v1/agent/runs/nope/resume")
        assert resp.status_code == 404


class TestSSE:
    async def test_events_replayed(self, agent_client):
        client, service = agent_client
        resp = await client.post("/v1/agent/runs", json={"goal": "stream me"})
        run_id = resp.json()["id"]
        await _poll_status(client, run_id, "finished")

        resp = await client.get(f"/v1/agent/runs/{run_id}/events")
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        types = [e["type"] for e in events]
        assert "token" in types
        assert "tool_call" in types
        assert any(
            e.get("type") == "run_status" and e.get("status") == "finished"
            for e in events
        )

    async def test_events_unknown_404(self, agent_client):
        client, service = agent_client
        resp = await client.get("/v1/agent/runs/nope/events")
        assert resp.status_code == 404


class TestGatedOff:
    async def test_routes_absent_when_disabled(self, app_client):
        # Default app_client fixture has agent disabled → routes not registered.
        resp = await app_client.post("/v1/agent/runs", json={"goal": "x"})
        assert resp.status_code == 404
