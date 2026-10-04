"""Detect chat template capabilities by inspecting the Jinja2 template string."""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TemplateCaps:
    supports_tools: bool = False
    supports_enable_thinking: bool = False
    #: Channel-format reasoners (gpt-oss / Harmony) take a ``reasoning_effort``
    #: kwarg ("low"/"medium"/"high") instead of a boolean ``enable_thinking``.
    supports_reasoning_effort: bool = False
    has_thinking_tags: bool = False
    has_channel_format: bool = False
    uses_tool_responses: bool = False
    # Whether the template can render tool turns at all — via a ``tool`` role,
    # ``tool_calls`` iteration, ``[TOOL_RESULTS]`` markers, or ``tool_responses``.
    # Minimal templates (e.g. Devstral/Mistral) only allow user/system/assistant
    # and raise on anything else; for those we fold tool turns into user text.
    handles_tool_role: bool = False
    # Whether the template raises on a system turn that isn't the first message
    # (Qwen3.5/3.6: "System message must be at the beginning."). For those,
    # late system/developer turns are folded into the leading one (#740);
    # templates that render a positional system turn keep it in place.
    rejects_positional_system: bool = False


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


_PROBE_LEADING_SYSTEM = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "u"},
    {"role": "assistant", "content": "a"},
    {"role": "user", "content": "u"},
]
_PROBE_LATE_SYSTEM = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "u"},
    {"role": "assistant", "content": "a"},
    {"role": "system", "content": "s"},
    {"role": "user", "content": "u"},
]


def _probe_rejects_positional_system(tokenizer: Any, tpl: str) -> bool:
    """Render a late-system probe and report whether only it is rejected.

    A static scan can't tell "raise on a non-first system turn" from other
    ``raise_exception`` guards, so render the template: the conversation with
    a leading system turn must render and the same conversation with an extra
    late system turn must raise. A template that rejects both (no system role
    at all) or neither is not flagged — folding wouldn't change the outcome.
    """
    if "raise_exception" not in tpl:
        return False
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        return False
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        apply(_PROBE_LEADING_SYSTEM, **kwargs)
    except Exception:
        return False
    try:
        apply(_PROBE_LATE_SYSTEM, **kwargs)
    except Exception as exc:
        logger.debug("Template rejects a non-leading system turn: %s", exc)
        return True
    return False


def detect_caps(tokenizer: Any) -> TemplateCaps:
    """Inspect the tokenizer's chat_template to determine supported features."""
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl is None:
        # VLM processors may keep the template (and its renderer) on the
        # wrapped tokenizer — same lookup as ``_get_chat_template_text``.
        inner = getattr(tokenizer, "tokenizer", None)
        inner_tpl = getattr(inner, "chat_template", None)
        if isinstance(inner_tpl, (str, list)):
            tokenizer, tpl = inner, inner_tpl
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
        supports_reasoning_effort = "reasoning_effort" in variables
    else:
        # Fallback: substring matching (for malformed templates)
        logger.debug("Jinja2 parsing failed, falling back to substring matching")
        supports_tools = "tools" in tpl
        supports_enable_thinking = "enable_thinking" in tpl
        supports_reasoning_effort = "reasoning_effort" in tpl

    # has_thinking_tags checks for literal output, not a variable — keep string check
    has_thinking_tags = "<think>" in tpl or "thinking" in tpl.lower()

    has_channel_format = "<|channel|>" in tpl

    # tool_responses is accessed as message.tool_responses (dot notation) or
    # message['tool_responses'] (bracket notation, e.g. Gemma 4).  Use dot
    # prefix and bracket patterns to avoid false-matching comments or literals.
    uses_tool_responses = (
        ".tool_responses" in tpl
        or "['tool_responses']" in tpl
        or '["tool_responses"]' in tpl
    )

    # The template can natively render a tool-result turn only if it branches on
    # the tool role itself — a quoted ``'tool'``/``"tool"`` role literal, Mistral's
    # ``[TOOL_RESULTS]`` block, or Gemma's ``tool_responses``.  A bare
    # ``tool_calls`` reference is NOT sufficient: templates iterate assistant
    # ``tool_calls`` while still rejecting a separate ``role: "tool"`` message, so
    # keying on it would skip the rewrite and reintroduce the crash.  Absent all
    # of these (e.g. the minimal Devstral template), ``role: "tool"`` raises.
    handles_tool_role = (
        uses_tool_responses
        or "'tool'" in tpl
        or '"tool"' in tpl
        or "[TOOL_RESULTS]" in tpl
    )

    return TemplateCaps(
        supports_tools=supports_tools,
        supports_enable_thinking=supports_enable_thinking,
        supports_reasoning_effort=supports_reasoning_effort,
        has_thinking_tags=has_thinking_tags,
        has_channel_format=has_channel_format,
        uses_tool_responses=uses_tool_responses,
        handles_tool_role=handles_tool_role,
        rejects_positional_system=_probe_rejects_positional_system(tokenizer, tpl),
    )
