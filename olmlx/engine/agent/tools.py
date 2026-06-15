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


class AgentToolManager(BuiltinToolManager):
    """Builtin tools plus the agent's control tools, bound to an AgentContext."""

    def __init__(self, config: ChatConfig, context: "AgentContext"):
        super().__init__(config)
        self._context = context
        self._agent_defs: list[dict] = [_FINISH_DEF]

    @property
    def tool_names(self) -> set[str]:
        names = super().tool_names
        return names | {d["function"]["name"] for d in self._agent_defs}

    def get_tool_definitions(self) -> list[dict]:
        return super().get_tool_definitions() + list(self._agent_defs)

    async def call_tool(self, name: str, arguments: dict) -> str | ToolError:
        if name == "finish":
            return self._handle_finish(arguments)
        return await super().call_tool(name, arguments)

    def _handle_finish(self, arguments: dict) -> str:
        summary = str(arguments.get("summary", "")).strip()
        self._context.finish_summary = summary
        self._context.finished = True
        return f"Run marked complete. Summary recorded: {summary or '(none)'}"
