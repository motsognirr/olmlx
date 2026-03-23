"""Detect chat template capabilities by inspecting the Jinja2 template string."""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TemplateCaps:
    supports_tools: bool = False
    supports_enable_thinking: bool = False
    has_thinking_tags: bool = False
    has_channel_format: bool = False


def _find_template_variables(tpl: str) -> set[str] | None:
    """Parse a Jinja2 template and return its undeclared variables.

    Returns None if parsing fails (caller should fall back to substring matching).
    """
    try:
        import jinja2
        import jinja2.meta

        env = jinja2.Environment()
        ast = env.parse(tpl)
        return jinja2.meta.find_undeclared_variables(ast)
    except Exception:
        return None


def detect_caps(tokenizer: Any) -> TemplateCaps:
    """Inspect the tokenizer's chat_template to determine supported features."""
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl is None:
        return TemplateCaps()

    # Handle list-of-dicts format (named templates)
    if isinstance(tpl, list):
        tpl = " ".join(t.get("template", "") for t in tpl if isinstance(t, dict))

    variables = _find_template_variables(tpl)

    if variables is not None:
        # AST-based detection: only match actual template variables
        supports_tools = "tools" in variables
        supports_enable_thinking = "enable_thinking" in variables
    else:
        # Fallback: substring matching (for malformed templates)
        logger.debug("Jinja2 parsing failed, falling back to substring matching")
        supports_tools = "tools" in tpl
        supports_enable_thinking = "enable_thinking" in tpl

    # has_thinking_tags checks for literal output, not a variable — keep string check
    has_thinking_tags = "<think>" in tpl or "thinking" in tpl.lower()

    has_channel_format = "<|channel|>" in tpl

    return TemplateCaps(
        supports_tools=supports_tools,
        supports_enable_thinking=supports_enable_thinking,
        has_thinking_tags=has_thinking_tags,
        has_channel_format=has_channel_format,
    )
