"""Tests for olmlx.engine.agent.orchestrator — the goal-pursuit loop."""

from typing import Any

import pytest

from olmlx.engine.agent.orchestrator import (
    CONTINUATION_NUDGE,
    AgentContext,
    Budgets,
    Orchestrator,
)
from olmlx.engine.agent.store import AgentStore


@pytest.fixture
def store(tmp_path):
    s = AgentStore(tmp_path / "agent.db")
    yield s
    s.close()


class FakeSession:
    """Duck-typed stand-in for ChatSession.

    ``turns`` is a list of per-iteration scripts. Each script is a dict with:
      - ``events``: list of event dicts to yield from send_message
      - ``messages``: list of messages appended to ``self.messages`` this turn
    If send_message is called more times than there are scripts, the last
    script repeats (useful for "never finishes" budget tests).
    """

    def __init__(self, turns: list[dict[str, Any]]):
        self.turns = turns
        self.messages: list[dict] = [{"role": "system", "content": "sys"}]
        self.prompts: list[str] = []
        self._i = 0

    async def send_message(self, user_text: str):
        self.prompts.append(user_text)
        script = self.turns[min(self._i, len(self.turns) - 1)]
        self._i += 1
        self.messages.append({"role": "user", "content": user_text})
        for msg in script.get("messages", []):
            # Append a fresh copy each turn — a real ChatSession produces new
            # message objects per turn, and the orchestrator's stall signature
            # now diffs by object identity (#622), so reusing one object would
            # mask a legitimate identical-output stall.
            self.messages.append(dict(msg))
        for event in script.get("events", []):
            yield event
        yield {"type": "done"}


def _clock(values):
    """A fake monotonic clock that yields the given values, then clamps to the
    last one — so a test need not predict the exact number of clock() calls."""
    seq = list(values)
    i = {"n": 0}

    def clock():
        v = seq[min(i["n"], len(seq) - 1)]
        i["n"] += 1
        return v

    return clock


def _ctx(store, run_id="r1", **kw):
    return AgentContext(run_id=run_id, store=store, **kw)


def _assistant(content="working", tool_calls=None):
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _finish_turn(summary="all done"):
    return {
        "events": [
            {
                "type": "tool_call",
                "name": "finish",
                "arguments": {"summary": summary},
                "id": "t1",
            },
            {"type": "tool_result", "name": "finish", "result": "ok", "id": "t1"},
        ],
        "messages": [_assistant("calling finish", [{"function": {"name": "finish"}}])],
    }


class TestFinish:
    async def test_finish_tool_terminates_with_success(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        session = FakeSession([_finish_turn("the summary")])
        orch = Orchestrator(
            session=session, context=_ctx(store), budgets=Budgets(max_iterations=10)
        )
        result = await orch.run()
        assert result["status"] == "finished"
        assert result["result"] == "the summary"
        run = await store.get_run("r1")
        assert run["status"] == "finished"
        assert run["result"] == "the summary"
        assert run["iterations"] == 1

    async def test_finish_summary_is_stripped(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        turn = {
            "events": [
                {
                    "type": "tool_call",
                    "name": "finish",
                    "arguments": {"summary": "  padded summary \n"},
                    "id": "t1",
                },
            ],
            "messages": [_assistant("done", [{"function": {"name": "finish"}}])],
        }
        session = FakeSession([turn])
        orch = Orchestrator(session=session, context=_ctx(store), budgets=Budgets())
        result = await orch.run()
        assert result["result"] == "padded summary"
        assert (await store.get_run("r1"))["result"] == "padded summary"

    async def test_first_prompt_is_goal(self, store):
        await store.create_run(run_id="r1", goal="MY GOAL", model="m", config={})
        session = FakeSession([_finish_turn()])
        orch = Orchestrator(session=session, context=_ctx(store), budgets=Budgets())
        await orch.run()
        assert session.prompts[0] == "MY GOAL"


class TestContinuation:
    async def test_continuation_nudge_on_stop_without_finish(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        # First turn: no finish (just an assistant msg). Second turn: finish.
        session = FakeSession(
            [
                {"messages": [_assistant("step 1")]},
                _finish_turn(),
            ]
        )
        orch = Orchestrator(
            session=session, context=_ctx(store), budgets=Budgets(max_iterations=10)
        )
        result = await orch.run()
        assert result["status"] == "finished"
        assert session.prompts[0] == "G"
        assert session.prompts[1] == CONTINUATION_NUDGE


class TestBudgets:
    async def test_max_iterations_trips(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        # Each turn produces a *different* assistant msg so stall never fires.
        turns = [{"messages": [_assistant(f"step {i}")]} for i in range(5)]
        session = FakeSession(turns)
        orch = Orchestrator(
            session=session,
            context=_ctx(store),
            budgets=Budgets(max_iterations=2, stall_max_no_progress=99),
        )
        result = await orch.run()
        assert result["status"] == "failed"
        assert result["reason"] == "max_iterations"
        run = await store.get_run("r1")
        assert run["iterations"] == 2
        assert run["status"] == "failed"

    async def test_token_budget_trips(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        # 10 token events per turn; budget 5 → trips at the boundary after iter 1.
        tok_events = [{"type": "token", "text": "x"} for _ in range(10)]
        turns = [
            {"events": tok_events, "messages": [_assistant(f"s{i}")]} for i in range(5)
        ]
        session = FakeSession(turns)
        orch = Orchestrator(
            session=session,
            context=_ctx(store),
            budgets=Budgets(
                max_iterations=99, token_budget=5, stall_max_no_progress=99
            ),
        )
        result = await orch.run()
        assert result["status"] == "failed"
        assert result["reason"] == "token_budget"
        assert (await store.get_run("r1"))["iterations"] == 1
        assert (await store.get_run("r1"))["tokens"] == 10

    async def test_wallclock_timeout_trips(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        turns = [{"messages": [_assistant(f"s{i}")]} for i in range(5)]
        session = FakeSession(turns)
        # Clock stays at 0 for the first iteration's checks, then jumps past the
        # timeout. Clamps to the last value so the exact number of clock() calls
        # doesn't matter.
        orch = Orchestrator(
            session=session,
            context=_ctx(store),
            budgets=Budgets(
                max_iterations=99, wallclock_timeout=10.0, stall_max_no_progress=99
            ),
            clock=_clock([0.0, 0.0, 0.0, 100.0]),
        )
        result = await orch.run()
        assert result["status"] == "failed"
        assert result["reason"] == "wallclock_timeout"

    async def test_wallclock_is_cumulative_across_resume(self, store):
        """Prior active runtime counts toward the timeout on resume, so a run
        can't extend past its limit by resuming."""
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        # Pretend a prior session already used 8s of a 10s budget.
        await store.append_checkpoint(
            "r1", [{"role": "system", "content": "s"}], iterations=1, tokens=0
        )
        await store.update_run("r1", status="interrupted", runtime_seconds=8.0)
        session = FakeSession([{"messages": [_assistant("s")]}])
        orch = Orchestrator(
            session=session,
            context=_ctx(store),
            budgets=Budgets(
                max_iterations=99, wallclock_timeout=10.0, stall_max_no_progress=99
            ),
            clock=_clock([0.0, 5.0]),  # +5s this session → 8+5=13 ≥ 10
        )
        result = await orch.run(resume=True)
        assert result["status"] == "failed"
        assert result["reason"] == "wallclock_timeout"

    async def test_budget_independent_of_model(self, store):
        """Budgets trip even if the model never stops calling tools."""
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        turns = [
            {
                "events": [
                    {
                        "type": "tool_call",
                        "name": "bash",
                        "arguments": {"command": f"echo {i}"},
                        "id": f"t{i}",
                    }
                ],
                "messages": [
                    _assistant(f"running {i}", [{"function": {"name": "bash"}}])
                ],
            }
            for i in range(20)
        ]
        session = FakeSession(turns)
        orch = Orchestrator(
            session=session,
            context=_ctx(store),
            budgets=Budgets(max_iterations=3, stall_max_no_progress=99),
        )
        result = await orch.run()
        assert result["status"] == "failed"
        assert result["reason"] == "max_iterations"


class TestStall:
    async def test_stall_detection(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        # Identical assistant output every turn → stall.
        session = FakeSession([{"messages": [_assistant("same thing")]}])
        orch = Orchestrator(
            session=session,
            context=_ctx(store),
            budgets=Budgets(max_iterations=99, stall_max_no_progress=2),
        )
        result = await orch.run()
        assert result["status"] == "failed"
        assert result["reason"] == "stall"
        # iter1 sets baseline, iter2 +1, iter3 +2 == threshold.
        assert (await store.get_run("r1"))["iterations"] == 3

    async def test_no_false_stall_when_history_truncated(self, store):
        """When ``send_message`` rebinds ``messages`` to a shorter list
        (memory truncation), a positional ``before`` slice would yield an
        empty ``"[]"`` signature every iteration and falsely stall a run that
        is making real progress. The identity-based diff must still see each
        turn's distinct assistant output (#622).
        """
        await store.create_run(run_id="r1", goal="G", model="m", config={})

        class TruncatingSession(FakeSession):
            async def send_message(self, user_text):
                self.prompts.append(user_text)
                i = self._i
                self._i += 1
                # Simulate _check_memory_and_truncate: rebind messages to a
                # fresh, shorter list — but with DISTINCT assistant content.
                self.messages = [{"role": "system", "content": "summary"}]
                self.messages.append({"role": "user", "content": user_text})
                self.messages.append(_assistant(f"progress step {i}"))
                yield {"type": "done"}

        session = TruncatingSession([])
        orch = Orchestrator(
            session=session,
            context=_ctx(store),
            budgets=Budgets(max_iterations=5, stall_max_no_progress=2),
        )
        result = await orch.run()
        # Must run to the iteration cap, NOT be killed as a stall.
        assert result["reason"] != "stall"
        assert (await store.get_run("r1"))["iterations"] == 5


class TestTokenBatching:
    async def test_token_events_batched_and_ordered(self, store):
        """Token events are buffered and flushed in a batch, but must still be
        persisted before the following ordered event so SSE replay preserves
        order (#636)."""
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        turn = {
            "events": [
                {"type": "token", "text": "a"},
                {"type": "token", "text": "b"},
                {
                    "type": "tool_call",
                    "name": "finish",
                    "arguments": {"summary": "done"},
                    "id": "t1",
                },
            ],
            "messages": [_assistant("x")],
        }
        session = FakeSession([turn])
        orch = Orchestrator(session=session, context=_ctx(store), budgets=Budgets())
        await orch.run()
        events = await store.get_events("r1")
        types = [e["type"] for e in events]
        assert types.count("token") == 2
        # Both tokens land before the tool_call.
        assert max(i for i, t in enumerate(types) if t == "token") < types.index(
            "tool_call"
        )


class TestResume:
    async def test_resume_rehydrates_messages_and_counters(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        prior_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "G"},
            {"role": "assistant", "content": "did some work"},
        ]
        await store.append_checkpoint("r1", prior_messages, iterations=2, tokens=40)
        await store.update_run("r1", status="interrupted", iterations=2, tokens=40)

        session = FakeSession([_finish_turn()])
        orch = Orchestrator(
            session=session, context=_ctx(store), budgets=Budgets(max_iterations=10)
        )
        result = await orch.run(resume=True)
        assert result["status"] == "finished"
        # Rehydrated prior history then continued (nudge, not the goal).
        assert session.messages[:3] == prior_messages
        assert session.prompts[0] == CONTINUATION_NUDGE
        # Counters continued from the checkpoint (2 prior + 1 here).
        assert (await store.get_run("r1"))["iterations"] == 3


class TestCancellation:
    async def test_cooperative_cancel_at_boundary(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        ctx = _ctx(store)
        ctx.cancel_event.set()  # cancel before the first iteration
        session = FakeSession([_finish_turn()])
        orch = Orchestrator(session=session, context=ctx, budgets=Budgets())
        result = await orch.run()
        assert result["status"] == "cancelled"
        assert (await store.get_run("r1"))["status"] == "cancelled"


class TestEventPersistence:
    async def test_events_persisted_for_replay(self, store):
        await store.create_run(run_id="r1", goal="G", model="m", config={})
        session = FakeSession(
            [
                {
                    "events": [{"type": "token", "text": "hello"}],
                    "messages": [_assistant("hi")],
                },
                _finish_turn(),
            ]
        )
        orch = Orchestrator(
            session=session, context=_ctx(store), budgets=Budgets(max_iterations=10)
        )
        await orch.run()
        events = await store.get_events("r1")
        types = [e["type"] for e in events]
        assert "token" in types
        assert "tool_call" in types
        # A terminal status event is recorded.
        assert any(e["type"] == "run_status" for e in events)
