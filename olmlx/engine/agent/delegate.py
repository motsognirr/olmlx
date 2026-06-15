"""Subagent delegation for autonomous agent runs (issue #449).

A ``delegate`` tool lets a run spawn a child run for a sub-task. Children are
persisted runs linked to the parent (``parent_id``), so the whole tree is
resumable. Given the single global inference lock, children orchestrate
concurrently but **generate serially** — modeled here as a concurrency-limited
queue (a semaphore), not true wall-clock parallelism. Depth and fan-out are
bounded; a child failure surfaces to the parent as a tool error so the parent
decides whether to continue or ``finish``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmlx.engine.agent.service import AgentService

logger = logging.getLogger(__name__)


class DelegateError(Exception):
    """A delegation request that violates a bound (depth / fan-out)."""


class DelegateRunner:
    """Spawns and runs bounded child runs, serializing their generation."""

    def __init__(self, service: "AgentService"):
        self._service = service
        # Serialize child *generation* (belt-and-suspenders alongside the global
        # inference lock); the queue this implies is the design's model.
        self._sem = asyncio.Semaphore(1)

    async def delegate(self, *, parent_id: str, goal: str) -> dict[str, Any]:
        goal = (goal or "").strip()
        if not goal:
            raise DelegateError("delegate requires a non-empty goal")

        parent = await self._service.store.get_run(parent_id)
        if parent is None:
            raise DelegateError(f"parent run {parent_id!r} not found")

        settings = self._service._settings
        child_depth = parent["depth"] + 1
        if child_depth > settings.agent_max_subagent_depth:
            raise DelegateError(
                f"max subagent depth ({settings.agent_max_subagent_depth}) "
                f"exceeded; cannot delegate at depth {child_depth}"
            )
        existing = await self._service.store.list_children(parent_id)
        if len(existing) >= settings.agent_max_subagent_fanout:
            raise DelegateError(
                f"max subagent fan-out ({settings.agent_max_subagent_fanout}) "
                f"reached for run {parent_id}"
            )

        child = await self._service.store.create_run(
            run_id=self._service._id_factory(),
            goal=goal,
            model=parent["model"],
            config=parent.get("config") or {},
            parent_id=parent_id,
            depth=child_depth,
        )
        logger.info(
            "run %s delegating child %s (depth %d): %s",
            parent_id,
            child["id"],
            child_depth,
            goal,
        )
        async with self._sem:
            return await self._service.run_child(child)
