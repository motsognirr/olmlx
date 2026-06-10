"""Shared buffer-and-parse streaming plumbing for the chat routers (issue #471).

When tools are declared, every surface (OpenAI SSE, Anthropic SSE, Ollama
NDJSON) must buffer the full model output before emitting anything — see the
rationale on ``_stream_buffered_with_tools`` in ``routers/anthropic.py``
(format priority, brace balancing, upfront tool ids).  The drain/aggregate
loop and the parse step live here; the per-surface emission stays in each
router.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from olmlx.engine.tool_parser import (
    fill_missing_required_args,
    parse_model_output,
    resolve_tool_names,
)

logger = logging.getLogger(__name__)


@dataclass
class BufferedModelOutput:
    """Aggregate of a fully drained ``generate_chat`` stream."""

    full_text: str = ""
    raw_text: str = ""
    done_reason: str | None = None
    stats: Any | None = None
    thinking_expected: bool = False

    @property
    def parse_text(self) -> str:
        """Text to feed ``parse_model_output``.

        ``raw_text`` (gpt-oss channel format, set only on the done chunk)
        supersedes the channel-filtered ``full_text``; ``or`` rather than a
        sentinel default so an empty-string ``raw_text`` also falls back.
        """
        return self.raw_text or self.full_text


async def with_keepalive_pings(
    aiter: AsyncIterator[Any],
    interval: float,
    ping: Any,
) -> AsyncIterator[Any]:
    """Yield items from *aiter*; yield *ping* when no item arrives within
    *interval* seconds.

    The caller supplies the ready-to-send ping event (e.g. a formatted SSE
    string), so downstream consumers never have to filter a sentinel — model
    chunks are dicts, pings are whatever the caller passed.
    """
    ait = aiter.__aiter__()
    next_item_task = None
    try:
        while True:
            if next_item_task is None:
                next_item_task = asyncio.ensure_future(ait.__anext__())
            done, _ = await asyncio.wait({next_item_task}, timeout=interval)
            if done:
                try:
                    item = next_item_task.result()
                except StopAsyncIteration:
                    return
                next_item_task = None
                yield item
            else:
                yield ping
    finally:
        if next_item_task is not None and not next_item_task.done():
            next_item_task.cancel()
            try:
                await next_item_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass


async def buffer_stream(
    result: AsyncIterator[dict[str, Any]],
    *,
    keepalive_interval: float | None = None,
    ping: Any = None,
) -> AsyncIterator[Any]:
    """Drain *result* to completion, yielding passthrough events along the way.

    Yields, in order: any *ping* events (when ``keepalive_interval`` is set)
    and ``cache_info`` chunks as they arrive, then exactly one final
    :class:`BufferedModelOutput`.  Callers that need neither passthrough
    should use :func:`collect_stream` instead.

    Does NOT close *result* — lifecycle stays with the caller (the routers'
    ``finally`` blocks own ``aclose``).
    """
    out = BufferedModelOutput()
    chunks = (
        result
        if keepalive_interval is None
        else with_keepalive_pings(result, keepalive_interval, ping)
    )
    parts: list[str] = []
    async for chunk in chunks:
        if ping is not None and chunk is ping:
            # Keepalive ping — forward to the consumer.  Identity check
            # (like the pre-#471 sentinel): anything else that isn't a dict
            # is a protocol violation and surfaces as an error below.
            yield chunk
            continue
        if chunk.get("cache_info"):
            yield chunk
            continue
        if "thinking_expected" in chunk:
            # Engine meta chunk, emitted before any text: gates the orphan
            # `</think>` heuristic in `parse_model_output` (issue #307).
            out.thinking_expected = bool(chunk["thinking_expected"])
            continue
        if chunk.get("done"):
            # gpt-oss models put the unfiltered channel output here so tool
            # calls survive the visible-text filter.
            out.raw_text = chunk.get("raw_text", "") or ""
            out.done_reason = chunk.get("done_reason")
            out.stats = chunk.get("stats")
            break
        parts.append(chunk.get("text", ""))
    out.full_text = "".join(parts)
    yield out


async def collect_stream(result: AsyncIterator[dict[str, Any]]) -> BufferedModelOutput:
    """Drain *result* and return the aggregate, discarding cache_info chunks."""
    out = BufferedModelOutput()
    async for item in buffer_stream(result):
        if isinstance(item, BufferedModelOutput):
            out = item
    return out


def parse_buffered_output(
    out: BufferedModelOutput,
    declared_tools: list[dict[str, Any]] | None,
    *,
    has_tools: bool = True,
    fill_missing_args: bool = True,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Parse a buffered output into ``(thinking, visible_text, tool_uses)``.

    Runs ``parse_model_output`` on :attr:`BufferedModelOutput.parse_text`,
    resolves parsed tool names against *declared_tools*, and (unless
    ``fill_missing_args`` is False — the Anthropic surface leaves model
    omissions visible to the client) injects empty strings for required
    string args the model omitted.
    """
    thinking, visible_text, tool_uses = parse_model_output(
        out.parse_text,
        has_tools,
        thinking_expected=out.thinking_expected,
    )
    resolve_tool_names(tool_uses, declared_tools)
    if fill_missing_args:
        fill_missing_required_args(tool_uses, declared_tools)
    return thinking, visible_text, tool_uses
