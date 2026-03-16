"""Tool safety policy for gating tool execution."""

from collections.abc import Callable
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
    """Classifies tools by safety policy and gates execution."""

    _BUILTIN_SAFE = {"use_skill"}

    def __init__(
        self,
        config: ToolSafetyConfig,
        decider: Callable[[str, dict[str, Any]], Any] | None = None,
    ):
        self.config = config
        self.decider = decider  # async (name, args) -> bool

    @property
    def builtin_safe_tools(self) -> frozenset[str]:
        """Tools that are safe by default (no side effects)."""
        return frozenset(self._BUILTIN_SAFE)

    def get_policy(self, tool_name: str) -> ToolPolicy:
        if tool_name in self.config.tool_policies:
            return self.config.tool_policies[tool_name]
        if tool_name in self._BUILTIN_SAFE:
            return ToolPolicy.ALLOW
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
        policy = self.get_policy(name)
        if policy == ToolPolicy.ALLOW:
            return True
        if policy == ToolPolicy.DENY:
            return False
        if self.decider:
            return await self.decider(name, arguments)
        return False
