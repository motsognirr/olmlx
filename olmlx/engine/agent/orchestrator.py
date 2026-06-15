"""Goal-pursuit loop around a ``ChatSession`` (issue #446).

The orchestrator wraps the existing ``ChatSession`` ReAct loop (Approach A in
the design) and adds exactly one behavioral extension: **continuation
semantics** — a stop without the ``finish`` tool is not the end of the run; the
orchestrator injects a continuation nudge and keeps going until the model calls
``finish`` (success) or a hard guard trips (budget / stall / cancel).

Each outer iteration runs one ``ChatSession.send_message`` (which itself runs
the inner tool loop), then: accounts tokens, checkpoints the full message
history, persists events for SSE replay, and checks termination. Budgets are
enforced by the orchestrator regardless of model behavior.

The wrapped session is duck-typed — anything with a ``messages: list`` attribute
and an ``async send_message(text)`` generator works — so tests inject a fake and
the router injects a real ``ChatSession``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from olmlx.engine.agent.memory import MemoryManager
    from olmlx.engine.agent.store import AgentStore

logger = logging.getLogger(__name__)

#: Injected as the user message on every iteration after the first (and on the
#: first iteration of a resumed run). The single behavioral change layered over
#: ``ChatSession`` — see the design's "continuation semantics".
CONTINUATION_NUDGE = (
    "You stopped without calling the `finish` tool. If the goal is fully "
    "complete, call `finish` with a short summary of what you accomplished. "
    "Otherwise, keep working toward the goal — take the next concrete step."
)

#: Token-bearing event types counted against the token budget. Approximate: one
#: streamed chunk ≈ one token under mlx-lm streaming.
_TOKEN_EVENT_TYPES = frozenset({"token", "thinking_token"})


class _Session(Protocol):
    messages: list[dict]

    def send_message(self, user_text: str) -> Any: ...


@dataclass
class Budgets:
    """Hard guards enforced by the orchestrator independent of model output."""

    max_iterations: int = 50
    token_budget: int | None = None
    wallclock_timeout: float | None = None
    stall_max_no_progress: int = 3


@dataclass
class AgentContext:
    """Shared mutable state passed to agent tools and the orchestrator.

    ``cancel_event`` is the cooperative-cancel seam (checked at iteration
    boundaries). ``memory`` / ``skills`` / ``delegate_runner`` are wired by the
    later phases; Phase 1 leaves them ``None``.
    """

    run_id: str
    store: "AgentStore"
    depth: int = 0
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    memory: "MemoryManager | None" = None
    skills: Any = None
    delegate_runner: Any = None
    #: Set by the ``finish`` tool handler (bookkeeping; the orchestrator's
    #: authoritative finish signal is the ``tool_call`` event stream).
    finished: bool = False
    finish_summary: str = ""


class Orchestrator:
    def __init__(
        self,
        *,
        session: _Session,
        context: AgentContext,
        budgets: Budgets,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.session = session
        self.context = context
        self.budgets = budgets
        self.clock = clock
        self.run_id = context.run_id
        self.store = context.store

    async def run(self, *, resume: bool = False) -> dict[str, Any]:
        run = await self.store.get_run(self.run_id)
        if run is None:
            raise ValueError(f"run {self.run_id!r} does not exist")
        goal = run["goal"]
        iterations = run["iterations"]
        tokens = run["tokens"]

        if resume:
            checkpoint = await self.store.latest_checkpoint(self.run_id)
            if checkpoint is not None:
                self.session.messages = list(checkpoint["messages"])
                iterations = checkpoint["iterations"]
                tokens = checkpoint["tokens"]

        await self.store.update_run(self.run_id, status="running")
        await self.store.append_event(
            self.run_id, {"type": "run_status", "status": "running"}
        )

        start = self.clock()
        first = iterations == 0 and not resume
        last_sig: str | None = None
        no_progress = 0

        # Inject memory/skill context into the system prompt before the loop
        # (Phase 2 hook; a no-op when memory is unset).
        if self.context.memory is not None:
            await self.context.memory.inject_context(self.session, goal)

        try:
            while True:
                boundary = self._check_boundary(iterations, tokens, start)
                if boundary is not None:
                    return await self._finalize(
                        boundary[0], iterations, tokens, reason=boundary[1]
                    )

                prompt = goal if first else CONTINUATION_NUDGE
                first = False
                before = len(self.session.messages)

                finished = False
                summary = ""
                async for event in self.session.send_message(prompt):
                    await self.store.append_event(self.run_id, event)
                    etype = event.get("type")
                    if etype in _TOKEN_EVENT_TYPES:
                        tokens += 1
                    elif etype == "tool_call" and event.get("name") == "finish":
                        finished = True
                        args = event.get("arguments")
                        if isinstance(args, dict):
                            summary = str(args.get("summary", ""))

                iterations += 1
                new_messages = self.session.messages[before:]
                await self.store.append_checkpoint(
                    self.run_id, self.session.messages, iterations, tokens
                )
                await self.store.update_run(
                    self.run_id, iterations=iterations, tokens=tokens
                )

                if finished:
                    return await self._finalize(
                        "finished", iterations, tokens, result=summary
                    )

                sig = _iteration_signature(new_messages)
                if last_sig is not None and sig == last_sig:
                    no_progress += 1
                else:
                    no_progress = 0
                last_sig = sig
                if no_progress >= self.budgets.stall_max_no_progress:
                    return await self._finalize(
                        "failed", iterations, tokens, reason="stall"
                    )
        except asyncio.CancelledError:
            await self._finalize("cancelled", iterations, tokens, reason="cancelled")
            raise

    def _check_boundary(
        self, iterations: int, tokens: int, start: float
    ) -> tuple[str, str] | None:
        """Return ``(status, reason)`` if a guard trips at this boundary, else None."""
        if self.context.cancel_event.is_set():
            return ("cancelled", "cancelled")
        if iterations >= self.budgets.max_iterations:
            return ("failed", "max_iterations")
        if (
            self.budgets.token_budget is not None
            and tokens >= self.budgets.token_budget
        ):
            return ("failed", "token_budget")
        if (
            self.budgets.wallclock_timeout is not None
            and (self.clock() - start) >= self.budgets.wallclock_timeout
        ):
            return ("failed", "wallclock_timeout")
        return None

    async def _finalize(
        self,
        status: str,
        iterations: int,
        tokens: int,
        *,
        reason: str | None = None,
        result: str | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "status": status,
            "iterations": iterations,
            "tokens": tokens,
        }
        if result is not None:
            fields["result"] = result
        if reason is not None and status != "finished":
            fields["error"] = reason
        await self.store.update_run(self.run_id, **fields)
        await self.store.append_event(
            self.run_id,
            {
                "type": "run_status",
                "status": status,
                "reason": reason,
                "result": result,
            },
        )
        logger.info(
            "agent run %s -> %s (reason=%s, iterations=%d, tokens=%d)",
            self.run_id,
            status,
            reason,
            iterations,
            tokens,
        )
        return {
            "status": status,
            "reason": reason,
            "result": result,
            "iterations": iterations,
            "tokens": tokens,
        }


def _iteration_signature(messages: list[dict]) -> str:
    """Stable signature of an iteration's assistant output for stall detection.

    Keyed on assistant message content + tool-call (name, args) only — tool
    *results* are excluded so legitimately-varying results (timestamps, fresh
    reads) don't mask a model that keeps proposing the identical action.
    """
    parts: list[Any] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            calls.append([fn.get("name", ""), fn.get("arguments", "")])
        parts.append([msg.get("content", ""), calls])
    return json.dumps(parts, sort_keys=True, default=str)
