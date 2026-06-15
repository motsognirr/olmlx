"""Hermes-style cross-session memory for agent runs (issue #447).

Record/recall over the FTS5 ``memory`` table in the agent store, with LLM
summarization when the working set overflows ``max_entries`` (old detail is
compressed into a ``summary`` entry rather than dropped). Memory is injected
into the wrapped ``ChatSession``'s system prompt as a tiered block
(stable goal → recalled context → recent progress) that survives restart and
context truncation, since it is re-derived from the store each iteration rather
than living only in the (truncatable) message history.

The summarizer is injected (``summarizer(texts) -> str``); the service supplies
one backed by ``generate_chat``, and tests supply a fake. With no summarizer a
deterministic concat fallback keeps the module inference-free for unit tests.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from olmlx.engine.agent.store import AgentStore

logger = logging.getLogger(__name__)

_BLOCK_START = "<!--AGENT_MEMORY_START-->"
_BLOCK_END = "<!--AGENT_MEMORY_END-->"
_BLOCK_RE = re.compile(
    re.escape(_BLOCK_START) + r".*?" + re.escape(_BLOCK_END),
    re.DOTALL,
)

Summarizer = Callable[[list[str]], Awaitable[str]]


class MemoryManager:
    def __init__(
        self,
        store: "AgentStore",
        run_id: str,
        *,
        max_entries: int = 1000,
        recall_k: int = 5,
        summarizer: Summarizer | None = None,
    ):
        self.store = store
        self.run_id = run_id
        self.max_entries = max_entries
        self.recall_k = recall_k
        self._summarizer = summarizer

    async def record(self, text: str, scope: str = "note") -> None:
        text = (text or "").strip()
        if not text:
            return
        await self.store.add_memory(self.run_id, text, scope)
        await self._maybe_summarize()

    async def recall(self, query: str, k: int | None = None) -> list[str]:
        rows = await self.store.search_memory(self.run_id, query, k or self.recall_k)
        return [r["text"] for r in rows]

    async def _maybe_summarize(self) -> None:
        count = await self.store.count_memory(self.run_id)
        if count <= self.max_entries:
            return
        entries = await self.store.list_memory(self.run_id)  # oldest first
        # Drop enough oldest entries to get back under the cap, compressing them
        # into a single summary entry (net change is negative when >= 2 dropped).
        n_drop = count - self.max_entries + 1
        overflow = entries[:n_drop]
        texts = [e["text"] for e in overflow]
        summary = await self._summarize(texts)
        await self.store.add_memory(self.run_id, summary, scope="summary")
        await self.store.delete_memory([e["id"] for e in overflow])
        logger.info(
            "agent run %s memory summarized %d entries -> 1 summary",
            self.run_id,
            len(overflow),
        )

    async def _summarize(self, texts: list[str]) -> str:
        if self._summarizer is not None:
            try:
                return await self._summarizer(texts)
            except Exception:
                logger.warning(
                    "memory summarizer failed; using concat fallback", exc_info=True
                )
        joined = " | ".join(t.replace("\n", " ")[:200] for t in texts)
        return f"Summary of earlier progress: {joined}"

    async def inject_context(self, session, goal: str) -> None:
        """Refresh the tiered memory block in the session's system prompt.

        Idempotent: an existing block (delimited by markers) is replaced, so
        calling this each iteration keeps recalled context fresh without the
        block accreting.
        """
        recalled = await self.recall(goal, self.recall_k)
        recent = await self.store.recent_memory(self.run_id, self.recall_k)
        block = _render_block(goal, recalled, [r["text"] for r in recent])
        _apply_block(session, block)


def _render_block(goal: str, recalled: list[str], recent: list[str]) -> str:
    lines = [_BLOCK_START, "## Agent memory", f"Goal: {goal}"]
    if recalled:
        lines.append("")
        lines.append("Relevant recalled notes:")
        lines.extend(f"- {t}" for t in recalled)
    if recent:
        lines.append("")
        lines.append("Recent progress:")
        lines.extend(f"- {t}" for t in recent)
    lines.append(_BLOCK_END)
    return "\n".join(lines)


def _apply_block(session, block: str) -> None:
    messages = session.messages
    sys_idx = next(
        (i for i, m in enumerate(messages) if m.get("role") == "system"), None
    )
    if sys_idx is None:
        messages.insert(0, {"role": "system", "content": block})
        return
    content = messages[sys_idx].get("content") or ""
    if _BLOCK_RE.search(content):
        # Function replacement, not a string: the block is model/user-derived
        # (remembered notes, goal) and may contain backslash sequences that
        # re.sub would interpret as group backreferences (\1, \g<...>) —
        # raising re.error or corrupting the prompt.
        content = _BLOCK_RE.sub(lambda _m: block, content)
    else:
        content = f"{content}\n\n{block}" if content else block
    messages[sys_idx] = {**messages[sys_idx], "content": content}
