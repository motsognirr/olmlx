"""Parse tool calls from model output across multiple formats.

Supported formats:
- Qwen: <tool_call>{"name": ..., "arguments": ...}</tool_call>
- XML-style: <function=Name><parameter=key>value</parameter></function>
- Mistral: [TOOL_CALLS] [{"name": ..., "arguments": ...}]
- Llama 3.x: <|python_tag|>{"name": ..., "parameters": ...}
- DeepSeek: <|tool_calls_begin|>...<|tool_calls_end|>
- Bare JSON: {"name": "...", "arguments": {...}} (on its own line or at text start)
"""

import json
import logging
import re
import uuid

logger = logging.getLogger(__name__)


def _make_tool_use_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


# --- Regex patterns ---

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC_TAG_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAM_TAG_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

# Mistral: [TOOL_CALLS] followed by a JSON array
_MISTRAL_TOOL_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)

# Llama 3.x: <|python_tag|> followed by JSON
_LLAMA_TOOL_RE = re.compile(
    r"<\|python_tag\|>\s*(\{.*?\})\s*(?:<\|eom_id\|>|$)", re.DOTALL
)

# DeepSeek: <|tool_calls_begin|>...<|tool_calls_end|>
_DEEPSEEK_TOOL_RE = re.compile(
    r"<\|tool_calls_begin\|>(.*?)<\|tool_calls_end\|>", re.DOTALL
)
_DEEPSEEK_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(?:function)?\s*\n?\s*(\w+)\s*\n(.*?)<\|tool_call_end\|>",
    re.DOTALL,
)

# Bare JSON: find `{` at line start (possibly followed by whitespace/newline then "name")
_BARE_JSON_START_RE = re.compile(r'(?:^|\n)\s*(\{)\s*"name"', re.MULTILINE)


def _parse_json_call(data: dict) -> dict | None:
    """Parse a JSON dict into a tool_use block, or None if invalid."""
    name = data.get("name", "")
    if not name:
        return None
    arguments = data.get("arguments") or data.get("parameters") or {}
    if isinstance(arguments, str):
        if arguments.strip():  # Only parse non-empty strings
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        else:
            arguments = {}
    return {
        "type": "tool_use",
        "id": _make_tool_use_id(),
        "name": name,
        "input": arguments,
    }


def _try_qwen(text: str) -> tuple[list[dict], str]:
    """Parse Qwen-style <tool_call>...</tool_call> blocks."""
    tool_uses = []
    for match in _TOOL_CALL_RE.finditer(text):
        inner = match.group(1)
        # Try JSON
        try:
            call = json.loads(inner)
            result = _parse_json_call(call)
            if result:
                tool_uses.append(result)
                continue
        except (json.JSONDecodeError, AttributeError):
            pass
        # Try XML-style: <function=Name><parameter=key>value</parameter></function>
        func_match = _FUNC_TAG_RE.search(inner)
        if func_match:
            name = func_match.group(1).strip()
            params = {}
            for pm in _PARAM_TAG_RE.finditer(func_match.group(2)):
                pval = pm.group(2).strip()
                try:
                    pval = json.loads(pval)
                except (json.JSONDecodeError, ValueError):
                    pass
                params[pm.group(1).strip()] = pval
            tool_uses.append(
                {
                    "type": "tool_use",
                    "id": _make_tool_use_id(),
                    "name": name,
                    "input": params,
                }
            )
        else:
            logger.warning("Failed to parse <tool_call> block: %r", inner[:500])

    if tool_uses:
        text = _TOOL_CALL_RE.sub("", text)
    return tool_uses, text


def _try_mistral(text: str) -> tuple[list[dict], str]:
    """Parse Mistral-style [TOOL_CALLS] blocks."""
    tool_uses = []
    match = _MISTRAL_TOOL_RE.search(text)
    if not match:
        return [], text
    try:
        calls = json.loads(match.group(1))
        for call in calls:
            result = _parse_json_call(call)
            if result:
                tool_uses.append(result)
        if tool_uses:
            text = _MISTRAL_TOOL_RE.sub("", text)
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("Failed to parse [TOOL_CALLS] block: %s", e)
    return tool_uses, text


def _try_llama(text: str) -> tuple[list[dict], str]:
    """Parse Llama 3.x <|python_tag|> blocks."""
    tool_uses = []
    for match in _LLAMA_TOOL_RE.finditer(text):
        try:
            call = json.loads(match.group(1))
            result = _parse_json_call(call)
            if result:
                tool_uses.append(result)
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "Failed to parse <|python_tag|> block: %r", match.group(1)[:500]
            )
    if tool_uses:
        text = _LLAMA_TOOL_RE.sub("", text)
    return tool_uses, text


def _try_deepseek(text: str) -> tuple[list[dict], str]:
    """Parse DeepSeek <|tool_calls_begin|>...<|tool_calls_end|> blocks."""
    tool_uses = []
    ds_match = _DEEPSEEK_TOOL_RE.search(text)
    if not ds_match:
        return [], text
    inner = ds_match.group(1)
    for call_match in _DEEPSEEK_CALL_RE.finditer(inner):
        name = call_match.group(1).strip()
        args_str = call_match.group(2).strip()
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            arguments = {}
        tool_uses.append(
            {
                "type": "tool_use",
                "id": _make_tool_use_id(),
                "name": name,
                "input": arguments,
            }
        )
    if tool_uses:
        text = _DEEPSEEK_TOOL_RE.sub("", text)
    return tool_uses, text


def _extract_json_object(text: str, start: int) -> str | None:
    """Extract a JSON object from text starting at a '{', counting braces.

    Respects string literals so braces inside quoted strings are ignored.
    Returns the substring from start to the matching '}', or None if unbalanced.
    """
    depth = 0
    in_string = False
    escape = False
    i = start
    while i < len(text):
        ch = text[i]
        if escape:
            escape = False
        elif ch == "\\" and in_string:
            escape = True
        elif ch == '"' and not escape:
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None


def _try_bare_json(text: str) -> tuple[list[dict], str]:
    """Parse bare JSON tool calls (must be on own line or at text start)."""
    tool_uses = []
    spans = []
    for match in _BARE_JSON_START_RE.finditer(text):
        brace_pos = match.start(1)
        obj_str = _extract_json_object(text, brace_pos)
        if obj_str is None:
            continue
        try:
            call = json.loads(obj_str)
            result = _parse_json_call(call)
            if result:
                tool_uses.append(result)
                spans.append((brace_pos, brace_pos + len(obj_str)))
        except (json.JSONDecodeError, AttributeError):
            continue
    if tool_uses:
        # Remove matched spans from text (in reverse order to preserve indices)
        for start, end in reversed(spans):
            text = text[:start] + text[end:]
    return tool_uses, text


def parse_model_output(
    text: str,
    has_tools: bool,
    tool_names: set[str] | None = None,
) -> tuple[str, str, list[dict]]:
    """Parse raw model output into (thinking_text, visible_text, tool_use_blocks).

    Args:
        text: Raw model output text.
        has_tools: Whether tools were provided in the request.
        tool_names: Set of valid tool names for validation. If provided, parsed
            tool calls with unknown names will generate a warning.
    """
    thinking = ""

    # Extract thinking blocks
    think_matches = _THINK_RE.findall(text)
    if think_matches:
        thinking = "\n".join(m.strip() for m in think_matches)
        text = _THINK_RE.sub("", text)

    tool_uses: list[dict] = []
    if has_tools:
        parsers = [_try_qwen, _try_mistral, _try_llama, _try_deepseek, _try_bare_json]

        for parser in parsers:
            tool_uses, text = parser(text)
            if tool_uses:
                break

        # Validate tool names if provided
        if tool_uses and tool_names:
            for tu in tool_uses:
                if tu["name"] not in tool_names:
                    logger.warning(
                        "Parsed tool call '%s' not in provided tool set: %s",
                        tu["name"],
                        tool_names,
                    )

    visible_text = text.strip()
    return thinking, visible_text, tool_uses
