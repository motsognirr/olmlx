"""Run registry + lifecycle service for autonomous agent runs (issue #446).

``AgentService`` is the seam between the HTTP router and the orchestrator. It
owns the SQLite store, an in-memory registry of in-flight ``asyncio`` tasks, and
the policy for building a real ``ChatSession`` per run. Runs execute as
background tasks (decoupled from the request, OpenClaw-style) and ride the
existing global inference lock for natural backpressure.

The session is built through ``session_factory`` so tests inject a fake (no real
inference) while production builds a ``ChatSession`` wrapping the agent's
control tools. The model manager is resolved lazily via ``manager_getter`` so
the service can be constructed in ``create_app`` before the manager exists.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from olmlx.config import Settings
from olmlx.config import settings as global_settings
from olmlx.engine.agent.delegate import DelegateRunner
from olmlx.engine.agent.memory import MemoryManager
from olmlx.engine.agent.orchestrator import AgentContext, Budgets, Orchestrator
from olmlx.engine.agent.store import AgentStore

if TYPE_CHECKING:
    from olmlx.engine.model_manager import ModelManager

logger = logging.getLogger(__name__)

#: Statuses from which a run may be resumed.
RESUMABLE_STATUSES = frozenset({"interrupted", "paused", "failed", "cancelled"})
#: Statuses still in flight (cancellable).
ACTIVE_STATUSES = frozenset({"queued", "running", "paused", "interrupted"})

_AGENT_SYSTEM_PROMPT = """You are an autonomous agent pursuing a goal without a \
human in the loop. You run headless: do not ask the user questions — make \
reasonable assumptions and proceed.

GOAL:
{goal}

Work step by step using the available tools, taking one concrete action at a \
time and observing the result before the next. When — and only when — the goal \
is fully accomplished, call the `finish` tool with a short summary of what you \
did. If an approach fails repeatedly, try a different one."""


@dataclass
class _RunHandle:
    cancel_event: asyncio.Event
    context: AgentContext
    task: asyncio.Task | None = None


#: ``(run, context, manager) -> session`` (duck-typed ChatSession).
SessionFactory = Callable[[dict, AgentContext, Any], Any]


class AgentService:
    def __init__(
        self,
        *,
        store: AgentStore,
        manager_getter: Callable[[], "ModelManager"],
        settings: Settings | None = None,
        session_factory: SessionFactory | None = None,
        id_factory: Callable[[], str] | None = None,
    ):
        self.store = store
        self._manager_getter = manager_getter
        self._settings = settings or global_settings
        self._session_factory = session_factory
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._handles: dict[str, _RunHandle] = {}
        self._delegate_runner = DelegateRunner(self)

    async def startup(self) -> None:
        """Recover crash-orphaned runs and re-materialize learned skills."""
        interrupted = await self.store.mark_interrupted_runs()
        if interrupted:
            logger.info(
                "Marked %d orphaned agent run(s) interrupted: %s",
                len(interrupted),
                ", ".join(interrupted),
            )
        await self._materialize_skills()

    async def _materialize_skills(self) -> None:
        """Write SQLite-persisted learned skills to the skills dir.

        SQLite is the durable record; the on-disk markdown files are what the
        loader reads. Re-materializing at startup means learned skills survive a
        wiped skills dir and are available to new runs.
        """
        from olmlx.chat.skills import write_skill_file

        skills = await self.store.list_skills()
        if not skills:
            return
        skills_dir = self._settings.agent_skills_dir
        for skill in skills:
            try:
                await asyncio.to_thread(
                    write_skill_file,
                    skills_dir,
                    skill["name"],
                    skill["description"],
                    skill["body"],
                )
            except ValueError:
                logger.warning(
                    "skipping malformed persisted skill %r", skill.get("name")
                )

    # -- create / resume -------------------------------------------------

    async def create_and_start(
        self,
        *,
        goal: str,
        model: str | None = None,
        max_iterations: int | None = None,
        token_budget: int | None = None,
        wallclock_timeout: float | None = None,
        parent_id: str | None = None,
        depth: int = 0,
    ) -> dict[str, Any]:
        resolved_model = (model or self._settings.agent_model or "").strip()
        if not resolved_model:
            raise ValueError(
                "no model specified and OLMLX_AGENT_MODEL is unset; "
                "provide 'model' in the request or set OLMLX_AGENT_MODEL"
            )
        run_id = self._id_factory()
        config = {
            "max_iterations": max_iterations,
            "token_budget": token_budget,
            "wallclock_timeout": wallclock_timeout,
        }
        run = await self.store.create_run(
            run_id=run_id,
            goal=goal,
            model=resolved_model,
            config=config,
            parent_id=parent_id,
            depth=depth,
        )
        self._start(run, resume=False)
        return run

    async def resume(self, run_id: str) -> dict[str, Any] | None:
        run = await self.store.get_run(run_id)
        if run is None:
            return None
        if run["status"] not in RESUMABLE_STATUSES:
            raise ValueError(
                f"run {run_id} has status {run['status']!r}, not resumable "
                f"(must be one of {sorted(RESUMABLE_STATUSES)})"
            )
        self._start(run, resume=True)
        return await self.store.get_run(run_id)

    def _make_context(self, run: dict[str, Any]) -> AgentContext:
        context = AgentContext(run_id=run["id"], store=self.store, depth=run["depth"])
        context.memory = MemoryManager(
            self.store,
            run["id"],
            max_entries=self._settings.agent_memory_max_entries,
            recall_k=self._settings.agent_memory_recall_k,
            summarizer=self._make_summarizer(run["model"]),
        )
        # Shared runner; bounded delegation (Phase 4). Children get their own
        # context via this same path, so the tree nests up to the depth cap.
        context.delegate_runner = self._delegate_runner
        return context

    def _build_orchestrator(
        self, run: dict[str, Any], context: AgentContext
    ) -> Orchestrator:
        session = self._make_session(run, context)
        return Orchestrator(
            session=session, context=context, budgets=self._budgets_for(run)
        )

    def _start(self, run: dict[str, Any], *, resume: bool) -> None:
        context = self._make_context(run)
        orch = self._build_orchestrator(run, context)
        handle = _RunHandle(cancel_event=context.cancel_event, context=context)
        self._handles[run["id"]] = handle
        handle.task = asyncio.create_task(self._run_wrapper(orch, run["id"], resume))

    async def run_child(self, run: dict[str, Any]) -> dict[str, Any]:
        """Run a child (delegated) run to completion inline and return its
        final state. Used by ``DelegateRunner``; generation still serializes on
        the global inference lock."""
        run_id = run["id"]
        context = self._make_context(run)
        orch = self._build_orchestrator(run, context)
        self._handles[run_id] = _RunHandle(
            cancel_event=context.cancel_event, context=context
        )
        try:
            await orch.run()
        except asyncio.CancelledError:
            # Cancellation must propagate (the parent run is being torn down),
            # but finalize the child row so it isn't left dangling 'running'.
            await self.store.update_run(run_id, status="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 — surface to the parent
            logger.exception("delegated run %s crashed", run_id)
            await self.store.update_run(
                run_id, status="failed", error=f"{type(exc).__name__}: {exc}"
            )
        finally:
            self._handles.pop(run_id, None)
        result = await self.store.get_run(run_id)
        return result or {}

    async def _run_wrapper(self, orch: Orchestrator, run_id: str, resume: bool) -> None:
        try:
            await orch.run(resume=resume)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface any orchestrator failure
            logger.exception("agent run %s crashed", run_id)
            # Event before status, so an SSE consumer can't stop on the terminal
            # status before the terminal event is written (see _finalize).
            await self.store.append_event(
                run_id,
                {"type": "run_status", "status": "failed", "reason": "exception"},
            )
            await self.store.update_run(
                run_id, status="failed", error=f"{type(exc).__name__}: {exc}"
            )
        finally:
            # Drop the handle once the run is terminal — queries read terminal
            # state from the store, so keeping it would only leak memory across
            # many runs. (Done in the wrapper, not _start, so a still-running
            # task remains cancellable via its handle.)
            self._handles.pop(run_id, None)

    def _budgets_for(self, run: dict[str, Any]) -> Budgets:
        cfg = run.get("config") or {}
        s = self._settings
        return Budgets(
            max_iterations=cfg.get("max_iterations") or s.agent_max_iterations,
            token_budget=cfg.get("token_budget") or s.agent_token_budget,
            wallclock_timeout=cfg.get("wallclock_timeout") or s.agent_wallclock_timeout,
            stall_max_no_progress=s.agent_stall_max_no_progress,
        )

    def _make_session(self, run: dict[str, Any], context: AgentContext) -> Any:
        if self._session_factory is not None:
            return self._session_factory(run, context, self._manager_getter())
        return self._default_session(run, context)

    def _default_session(self, run: dict[str, Any], context: AgentContext) -> Any:
        from olmlx.chat.config import ChatConfig
        from olmlx.chat.session import ChatSession
        from olmlx.chat.skills import SkillManager
        from olmlx.engine.agent.tools import AgentToolManager

        config = ChatConfig(
            model_name=run["model"],
            system_prompt=_AGENT_SYSTEM_PROMPT.format(goal=run["goal"]),
            max_turns=self._settings.agent_inner_max_turns,
            skills_dir=self._settings.agent_skills_dir,
        )
        # Load the learned-skill library so skills authored by earlier runs are
        # offered to this one (the self-improving loop's read side).
        skills = SkillManager(config.skills_dir)
        skills.load()
        builtin = AgentToolManager(config, context)
        return ChatSession(
            config=config,
            manager=self._manager_getter(),
            skills=skills,
            builtin=builtin,
        )

    def _make_summarizer(self, model: str):
        """An LLM-backed memory summarizer (best-effort; MemoryManager falls
        back to a concat summary if this raises)."""

        async def summarize(texts: list[str]) -> str:
            from olmlx.engine.inference import generate_chat

            joined = "\n".join(f"- {t}" for t in texts)
            messages = [
                {
                    "role": "system",
                    "content": "Compress the notes into a concise summary that "
                    "preserves decisions, facts, and progress. Be brief.",
                },
                {"role": "user", "content": f"Notes:\n{joined}"},
            ]
            parts: list[str] = []
            async for chunk in await generate_chat(
                self._manager_getter(),
                model,
                messages,
                stream=True,
                max_tokens=256,
                enable_thinking=False,
            ):
                if (
                    chunk.get("done")
                    or chunk.get("cache_info")
                    or "thinking_expected" in chunk
                ):
                    continue
                text = chunk.get("text", "")
                if text:
                    parts.append(text)
            return "".join(parts).strip() or "Summary unavailable."

        return summarize

    # -- queries ---------------------------------------------------------

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        return await self.store.get_run(run_id)

    async def list_runs(self) -> list[dict[str, Any]]:
        return await self.store.list_runs()

    async def cancel(self, run_id: str) -> dict[str, Any] | None:
        run = await self.store.get_run(run_id)
        if run is None:
            return None
        # Cancel the whole subtree: a parent blocked awaiting a delegated child
        # only reaches its own cancel check after the child returns, so the
        # cancel must also reach in-flight descendants (each has its own
        # cancel_event) for it to take effect promptly.
        await self._cancel_subtree(run_id, run)
        return await self.store.get_run(run_id)

    async def _cancel_subtree(
        self, run_id: str, run: dict[str, Any] | None = None
    ) -> None:
        if run is None:
            run = await self.store.get_run(run_id)
            if run is None:
                return
        handle = self._handles.get(run_id)
        if handle is not None:
            handle.cancel_event.set()
        # No live task to cooperatively notice the cancel (queued-only, or a
        # run from a previous process): mark it cancelled directly.
        no_live_task = handle is None or handle.task is None or handle.task.done()
        if no_live_task and run["status"] in ACTIVE_STATUSES:
            await self.store.update_run(run_id, status="cancelled")
        for child in await self.store.list_children(run_id):
            await self._cancel_subtree(child["id"], child)

    async def wait(self, run_id: str) -> None:
        """Await the background task for a run (test/shutdown helper)."""
        handle = self._handles.get(run_id)
        if handle is not None and handle.task is not None:
            try:
                await handle.task
            except asyncio.CancelledError:
                pass

    # -- SSE -------------------------------------------------------------

    async def stream_events(
        self, run_id: str, *, poll_interval: float = 0.05
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield persisted events then tail live until the run is terminal.

        Replays the event log (so a reconnecting client catches up), then polls
        for new events until the run reaches a terminal status and the log is
        fully drained. Polling (rather than a pub/sub) keeps it decoupled from
        the run's task and trivially correct across reconnects.
        """
        last_seq = -1
        while True:
            events = await self.store.get_events(run_id, after_seq=last_seq)
            for event in events:
                last_seq = event["seq"]
                yield event["data"]
            run = await self.store.get_run(run_id)
            terminal = run is None or run["status"] in (
                "finished",
                "failed",
                "cancelled",
            )
            if terminal and not events:
                return
            if not events:
                await asyncio.sleep(poll_interval)

    async def aclose(self) -> None:
        for handle in list(self._handles.values()):
            if handle.task is not None and not handle.task.done():
                handle.task.cancel()
        for handle in list(self._handles.values()):
            if handle.task is not None:
                try:
                    await handle.task
                except (asyncio.CancelledError, Exception):
                    pass
        self.store.close()
