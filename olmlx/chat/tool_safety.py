"""Tool safety policy for gating tool execution."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ToolPolicy(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"
    AUTO = "auto"


@dataclass
class ToolSafetyConfig:
    default_policy: ToolPolicy = ToolPolicy.CONFIRM
    tool_policies: dict[str, ToolPolicy] = field(default_factory=dict)
    judge_model: str | None = None


class ToolSafetyPolicy:
    """Classifies tools by safety policy and gates execution.

    Policy is determined from user config (``ToolSafetyConfig``).
    Tool source awareness (e.g. which tools are local vs MCP) belongs
    in the caller — see ``ChatSession`` which separates local tools
    before calling ``classify_batch``.
    """

    def __init__(
        self,
        config: ToolSafetyConfig,
        decider: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None,
        llm_judge: Callable[[str, dict[str, Any], list[dict] | None], Awaitable[bool]]
        | None = None,
    ):
        self.config = config
        self.decider = decider
        self.llm_judge = llm_judge

    def get_policy(self, tool_name: str) -> ToolPolicy:
        if tool_name in self.config.tool_policies:
            return self.config.tool_policies[tool_name]
        return self.config.default_policy

    def classify_batch(
        self,
        tool_uses: list[dict],
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        allow, confirm, auto, deny = [], [], [], []
        for tu in tool_uses:
            policy = self.get_policy(tu["name"])
            if policy == ToolPolicy.ALLOW:
                allow.append(tu)
            elif policy == ToolPolicy.CONFIRM:
                confirm.append(tu)
            elif policy == ToolPolicy.AUTO:
                auto.append(tu)
            else:
                deny.append(tu)
        return allow, confirm, auto, deny

    async def check_and_confirm(
        self,
        name: str,
        arguments: dict,
        context: list[dict] | None = None,
    ) -> bool:
        """Check policy and prompt for confirmation if needed.

        Intended for CONFIRM or AUTO-classified tools but safe to call
        for any policy — ALLOW returns True, DENY returns False without
        calling the decider or LLM judge. AUTO tools use the LLM judge
        first, falling back to the user decider if no LLM judge is set.
        """
        policy = self.get_policy(name)
        if policy == ToolPolicy.ALLOW:
            return True
        if policy == ToolPolicy.DENY:
            return False
        if policy == ToolPolicy.AUTO:
            if self.llm_judge:
                return await self.llm_judge(name, arguments, context)
            logger.warning(
                "Tool %r classified as AUTO but no LLM judge configured — "
                "falling back to user confirmation",
                name,
            )
        if self.decider:
            return await self.decider(name, arguments)
        logger.warning(
            "Tool %r requires confirmation but no decider is configured — denying",
            name,
        )
        return False
