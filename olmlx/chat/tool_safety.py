"""Tool safety policy for gating tool execution."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolPolicy(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass
class ToolSafetyConfig:
    default_policy: ToolPolicy = ToolPolicy.CONFIRM
    tool_policies: dict[str, ToolPolicy] = field(default_factory=dict)


class ToolSafetyPolicy:
    """Classifies tools by safety policy and gates execution.

    Policy is determined solely from user config (``ToolSafetyConfig``).
    Tool source awareness (e.g. which tools are local vs MCP) belongs
    in the caller — see ``ChatSession`` which separates local tools
    before calling ``classify_batch``.
    """

    def __init__(
        self,
        config: ToolSafetyConfig,
        decider: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None,
    ):
        self.config = config
        self.decider = decider

    def get_policy(self, tool_name: str) -> ToolPolicy:
        if tool_name in self.config.tool_policies:
            return self.config.tool_policies[tool_name]
        return self.config.default_policy

    def classify_batch(
        self,
        tool_uses: list[dict],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        allow, confirm, deny = [], [], []
        for tu in tool_uses:
            policy = self.get_policy(tu["name"])
            if policy == ToolPolicy.ALLOW:
                allow.append(tu)
            elif policy == ToolPolicy.CONFIRM:
                confirm.append(tu)
            else:
                deny.append(tu)
        return allow, confirm, deny

    async def check_and_confirm(self, name: str, arguments: dict) -> bool:
        """Check policy and prompt for confirmation if needed.

        Intended for CONFIRM-classified tools but safe to call for any
        policy — ALLOW returns True, DENY returns False without calling
        the decider.
        """
        policy = self.get_policy(name)
        if policy == ToolPolicy.ALLOW:
            return True
        if policy == ToolPolicy.DENY:
            return False
        if self.decider:
            return await self.decider(name, arguments)
        return False
