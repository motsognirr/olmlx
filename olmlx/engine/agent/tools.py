"""Agent control tools layered over the chat builtin tools (issue #446+).

``AgentToolManager`` subclasses ``BuiltinToolManager`` so the wrapped
``ChatSession`` sees the agent's control tools transparently — ``tool_names``,
``get_tool_definitions``, and ``call_tool`` all fall through to the agent tools
first, then to the inherited file/shell/web/plan tools.

Phase 1 adds only ``finish`` (the self-judged success terminator). Later phases
register ``remember`` / ``recall`` (Phase 2), ``create_skill`` (Phase 3), and
``delegate`` (Phase 4) by extending ``_agent_handlers`` / ``_agent_defs``.

``finish`` itself does no control-flow magic: the orchestrator detects it from
the ``tool_call`` event ``ChatSession`` emits, so the handler only needs to
return a confirmation string (and record the summary on the context for
bookkeeping).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from olmlx.chat.builtin_tools import BuiltinToolManager
from olmlx.chat.config import ChatConfig
from olmlx.chat.errors import ToolError

if TYPE_CHECKING:
    from olmlx.engine.agent.orchestrator import AgentContext

logger = logging.getLogger(__name__)


_FINISH_DEF = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Call this when the goal is fully complete to end the autonomous "
            "run. Provide a short summary of what was accomplished. This is the "
            "only clean way to stop — otherwise the run continues."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short summary of what was accomplished.",
                },
            },
            "required": ["summary"],
        },
    },
}

_REMEMBER_DEF = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Save a durable note to long-term memory (survives restart and "
            "context truncation). Use for decisions, facts learned, and "
            "progress worth recalling later."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The note to remember.",
                },
            },
            "required": ["text"],
        },
    },
}

_RECALL_DEF = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Search long-term memory for notes relevant to a query. Returns "
            "the matching notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search memory for.",
                },
            },
            "required": ["query"],
        },
    },
}

_CREATE_SKILL_DEF = {
    "type": "function",
    "function": {
        "name": "create_skill",
        "description": (
            "Author a reusable skill (markdown instructions) after solving a "
            "non-trivial task, so future runs can load it on demand. Use a "
            "short kebab/snake-case name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name (letters, digits, '-' or '_').",
                },
                "description": {
                    "type": "string",
                    "description": "One-line summary of when to use the skill.",
                },
                "body": {
                    "type": "string",
                    "description": "The skill's full markdown instructions.",
                },
            },
            "required": ["name", "description", "body"],
        },
    },
}


class AgentToolManager(BuiltinToolManager):
    """Builtin tools plus the agent's control tools, bound to an AgentContext."""

    def __init__(self, config: ChatConfig, context: "AgentContext"):
        super().__init__(config)
        self._context = context
        self._agent_defs: list[dict] = [
            _FINISH_DEF,
            _REMEMBER_DEF,
            _RECALL_DEF,
            _CREATE_SKILL_DEF,
        ]

    @property
    def tool_names(self) -> set[str]:
        names = super().tool_names
        return names | {d["function"]["name"] for d in self._agent_defs}

    def get_tool_definitions(self) -> list[dict]:
        return super().get_tool_definitions() + list(self._agent_defs)

    async def call_tool(self, name: str, arguments: dict) -> str | ToolError:
        if name == "finish":
            return self._handle_finish(arguments)
        if name == "remember":
            return await self._handle_remember(arguments)
        if name == "recall":
            return await self._handle_recall(arguments)
        if name == "create_skill":
            return await self._handle_create_skill(arguments)
        return await super().call_tool(name, arguments)

    def _handle_finish(self, arguments: dict) -> str:
        summary = str(arguments.get("summary", "")).strip()
        self._context.finish_summary = summary
        self._context.finished = True
        return f"Run marked complete. Summary recorded: {summary or '(none)'}"

    async def _handle_remember(self, arguments: dict) -> str | ToolError:
        if self._context.memory is None:
            return ToolError(
                message="Memory is not available for this run.",
                tool_name="remember",
                is_user_error=False,
            )
        text = str(arguments.get("text", "")).strip()
        if not text:
            return ToolError(
                message="remember requires non-empty 'text'.",
                tool_name="remember",
                is_user_error=True,
            )
        await self._context.memory.record(text)
        return "Saved to memory."

    async def _handle_recall(self, arguments: dict) -> str | ToolError:
        if self._context.memory is None:
            return ToolError(
                message="Memory is not available for this run.",
                tool_name="recall",
                is_user_error=False,
            )
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolError(
                message="recall requires a non-empty 'query'.",
                tool_name="recall",
                is_user_error=True,
            )
        results = await self._context.memory.recall(query)
        if not results:
            return "No relevant memories found."
        return "\n".join(f"- {r}" for r in results)

    async def _handle_create_skill(self, arguments: dict) -> str | ToolError:
        from olmlx.chat.skills import write_skill_file

        name = str(arguments.get("name", "")).strip()
        description = str(arguments.get("description", "")).strip()
        body = str(arguments.get("body", ""))
        try:
            path = await asyncio.to_thread(
                write_skill_file,
                self._config.skills_dir,
                name,
                description,
                body,
            )
        except ValueError as exc:
            return ToolError(
                message=f"Invalid skill: {exc}",
                tool_name="create_skill",
                is_user_error=True,
            )
        await self._context.store.upsert_skill(
            name, description, body.strip(), source_run=self._context.run_id
        )
        return f"Created skill {name!r} at {path}."
