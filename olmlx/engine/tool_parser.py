"""Parse tool calls from model output across multiple formats.

Supported formats:
- Qwen: <tool_call>{"name": ..., "arguments": ...}</tool_call>
  - Also handles XML-style inside <tool_call>: <function=Name><parameter=key>value</parameter></function>
- Standalone XML: <function=Name><parameter=key>value</parameter></function> (without <tool_call> wrapper)
- Mistral: [TOOL_CALLS] [{"name": ..., "arguments": ...}]
- Llama 3.x: <|python_tag|>{"name": ..., "parameters": ...}
- DeepSeek: <|tool_calls_begin|>...<|tool_calls_end|>
- MiniMax: <minimax:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></minimax:tool_call>
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

_THINK_RE = re.compile(r"<think>([^<]*)</think>")

# gpt-oss channel format (harmony):
# <|start|>assistant<|channel|>analysis<|message|>thinking<|end|>
# <|start|>assistant<|channel|>final<|message|>visible<|return|>  (or end-of-string — mlx-lm strips EOS)
# <|start|>assistant<|channel|>commentary to=functions.NAME<|constrain|>json<|message|>{"args"}<|call|>
#
# The header (between channel name and <|message|>) may contain:
# - to=functions.TOOL_NAME (appears AFTER <|channel|>)
# - <|constrain|>json
#
# Match full assistant message blocks. We consume the end marker (<|end|>, <|call|>, <|return|>)
# so that finditer can find subsequent blocks that start with <|start|>.
# Group 1: role (e.g. "assistant")
# Group 2: channel name (e.g. "commentary", "analysis", "final")
# Group 3: header content between channel and <|message|> (may contain to=functions.*, <|constrain|>, etc.)
# Group 4: message content
_GPT_OSS_CHANNEL_RE = re.compile(r"<\|channel\|>\s*(\w+)")
_GPT_OSS_END_RE = re.compile(r"<\|(?:end|call|return)\|>")
_GPT_OSS_TOOL_NAME_RE = re.compile(r"to=functions\.(\w+)")
_GPT_OSS_DETECT = "<|channel|>"
# Gemma4 tool call: <|tool_call>call:Name{key:<|"|>val<|"|>}<tool_call|>
_GEMMA4_TOOL_CALL_RE = re.compile(r"<\|tool_call>(.*?)<tool_call\|>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC_TAG_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
# GLM-4.5/4.6 arg pairs inside <tool_call>: <arg_key>k</arg_key><arg_value>v</arg_value>
_GLM_ARG_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)
_PARAM_TAG_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
# GLM zero-argument call: the body is just the bare function name (no arg tags).
# A strict single-identifier match distinguishes a real tool name from prose.
_GLM_BARE_NAME_RE = re.compile(r"[A-Za-z_][\w.-]*")
# JSON literals match the identifier pattern but are never tool names — a bare
# "true"/"false"/"null" block is noise, not a zero-arg call.
_JSON_LITERALS = frozenset({"true", "false", "null"})

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

# MiniMax: <minimax:tool_call><invoke name="Name"><parameter name="key">value</parameter></invoke></minimax:tool_call>
_MINIMAX_TOOL_RE = re.compile(
    r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
)
_MINIMAX_INVOKE_RE = re.compile(
    r'<invoke\b[^>]*\bname="([^"]*)"[^>]*>(.*?)</invoke>', re.DOTALL
)
_MINIMAX_PARAM_RE = re.compile(
    r'<parameter\s+name="([^"]*)">(.*?)</parameter>', re.DOTALL
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


def _extract_params(text: str, pattern: re.Pattern) -> dict:
    """Extract key-value parameters from text using the given regex pattern.

    The pattern must have group(1) = key name and group(2) = value.
    Values are parsed as JSON where possible, falling back to raw strings.
    """
    params = {}
    for pm in pattern.finditer(text):
        pval = pm.group(2).strip()
        try:
            pval = json.loads(pval)
        except (json.JSONDecodeError, ValueError):
            pass
        params[pm.group(1).strip()] = pval
    return params


def _parse_func_tag(func_match: re.Match) -> dict | None:
    """Parse a <function=Name>...</function> match into a tool_use dict (without _span).

    Returns None if the function name is empty.
    """
    name = func_match.group(1).strip()
    if not name:
        return None
    return {
        "type": "tool_use",
        "id": _make_tool_use_id(),
        "name": name,
        "input": _extract_params(func_match.group(2), _PARAM_TAG_RE),
    }


def _parse_glm_tool_call(inner: str) -> dict | None:
    """Parse a GLM-4.5/4.6 tool call body.

    Format: ``name\\n<arg_key>k</arg_key>\\n<arg_value>v</arg_value>...``.
    Values follow the GLM chat template (strings raw, non-strings
    ``tojson``-encoded), so ``_extract_params`` (JSON-with-raw-fallback) parses
    them the same way it does for the Mistral/MiniMax paths. Returns None if no
    function name precedes the arg pairs.
    """
    arg_start = inner.find("<arg_key>")
    name = (inner[:arg_start] if arg_start != -1 else inner).strip()
    if not name:
        return None
    return {
        "type": "tool_use",
        "id": _make_tool_use_id(),
        "name": name,
        "input": _extract_params(inner, _GLM_ARG_RE),
    }


def _try_qwen(text: str) -> tuple[list[dict], str]:
    """Parse Qwen-style <tool_call>...</tool_call> blocks.

    Also handles the GLM-4.5/4.6 body inside the same wrapper:
    ``name\\n<arg_key>k</arg_key><arg_value>v</arg_value>...``.
    """
    tool_uses = []
    for match in _TOOL_CALL_RE.finditer(text):
        span = (match.start(), match.end())
        inner = match.group(1)
        # Try JSON
        try:
            call = json.loads(inner)
            result = _parse_json_call(call)
            if result:
                result["_span"] = span
                tool_uses.append(result)
                continue
        except (json.JSONDecodeError, AttributeError):
            pass
        # Try XML-style: <function=Name><parameter=key>value</parameter></function>
        func_matches = list(_FUNC_TAG_RE.finditer(inner))
        if func_matches:
            for func_match in func_matches:
                result = _parse_func_tag(func_match)
                if result:
                    result["_span"] = span
                    tool_uses.append(result)
                else:
                    logger.warning(
                        "Skipping <function> tag with empty name inside <tool_call>"
                    )
        elif "<arg_key>" in inner and "<arg_value>" in inner:
            # GLM-4.5/4.6 style: name\n<arg_key>k</arg_key><arg_value>v</arg_value>.
            # Require both tag types so a stray "</arg_value>" in prose can't
            # hijack the block. A bare-identifier body (no arg tags) is left to
            # the existing "unparseable block" path — indistinguishable from
            # garbage.
            result = _parse_glm_tool_call(inner)
            if result:
                result["_span"] = span
                tool_uses.append(result)
            else:
                logger.warning(
                    "Failed to parse GLM <tool_call> block (no name): %r", inner[:500]
                )
        elif (
            _GLM_BARE_NAME_RE.fullmatch(inner.strip())
            and inner.strip() not in _JSON_LITERALS
        ):
            # GLM renders a no-argument call as just the bare function name
            # ("<tool_call>ls\n</tool_call>"). Require a strict single identifier
            # so prose (which contains spaces/punctuation) still falls through to
            # the "unparseable block" path below.
            tool_uses.append(
                {
                    "type": "tool_use",
                    "id": _make_tool_use_id(),
                    "name": inner.strip(),
                    "input": {},
                    "_span": span,
                }
            )
        else:
            logger.warning("Failed to parse <tool_call> block: %r", inner[:500])

    return tool_uses, text


def _try_mistral(text: str) -> tuple[list[dict], str]:
    """Parse Mistral-style [TOOL_CALLS] blocks."""
    tool_uses = []
    match = _MISTRAL_TOOL_RE.search(text)
    if not match:
        return [], text
    span = (match.start(), match.end())
    try:
        calls = json.loads(match.group(1))
        for call in calls:
            result = _parse_json_call(call)
            if result:
                result["_span"] = span
                tool_uses.append(result)
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
                result["_span"] = (match.start(), match.end())
                tool_uses.append(result)
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "Failed to parse <|python_tag|> block: %r", match.group(1)[:500]
            )
    return tool_uses, text


def _try_deepseek(text: str) -> tuple[list[dict], str]:
    """Parse DeepSeek <|tool_calls_begin|>...<|tool_calls_end|> blocks."""
    tool_uses = []
    ds_match = _DEEPSEEK_TOOL_RE.search(text)
    if not ds_match:
        return [], text
    span = (ds_match.start(), ds_match.end())
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
                "_span": span,
            }
        )
    return tool_uses, text


def _try_minimax(text: str) -> tuple[list[dict], str]:
    """Parse MiniMax <minimax:tool_call>...</minimax:tool_call> blocks."""
    tool_uses = []
    for block_match in _MINIMAX_TOOL_RE.finditer(text):
        span = (block_match.start(), block_match.end())
        inner = block_match.group(1)
        for invoke_match in _MINIMAX_INVOKE_RE.finditer(inner):
            name = invoke_match.group(1).strip()
            if not name:
                logger.warning("Skipping <invoke> tag with empty name")
                continue
            tool_uses.append(
                {
                    "type": "tool_use",
                    "id": _make_tool_use_id(),
                    "name": name,
                    "input": _extract_params(invoke_match.group(2), _MINIMAX_PARAM_RE),
                    "_span": span,
                }
            )
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


def _try_xml_func(text: str) -> tuple[list[dict], str]:
    """Parse standalone <function=Name>...</function> blocks (without <tool_call> wrapper).

    Note: _FUNC_TAG_RE matches anywhere in text, including inside prose. This is
    accepted because models using this format (Qwen 3.5) emit it only as actual
    tool calls, not in explanatory text. If false positives become an issue,
    the regex could be anchored to line boundaries.
    """
    tool_uses = []
    for match in _FUNC_TAG_RE.finditer(text):
        result = _parse_func_tag(match)
        if result:
            result["_span"] = (match.start(), match.end())
            tool_uses.append(result)
        else:
            logger.warning("Skipping <function> tag with empty name")
    return tool_uses, text


def _try_gemma4(text: str) -> tuple[list[dict], str]:
    """Parse Gemma4-style <|tool_call>call:Name{params}<tool_call|> blocks."""
    tool_uses = []
    for match in _GEMMA4_TOOL_CALL_RE.finditer(text):
        span = (match.start(), match.end())
        inner = match.group(1).strip()
        # Format: call:Name{key1:<|"|>val<|"|>,key2:<|"|>val<|"|>}
        if not inner.startswith("call:"):
            continue
        rest = inner[5:]  # strip "call:"
        brace = rest.find("{")
        if brace == -1:
            name = rest
            args = {}
        else:
            name = rest[:brace]
            # Strip only the outermost closing brace (rfind to preserve nested ones)
            raw = rest[brace + 1 :]
            last_brace = raw.rfind("}")
            params_str = raw[:last_brace] if last_brace >= 0 else raw
            # Parse key:value pairs. Values use <|"|> as string delimiters.
            # Replace <|"|> delimiters with regular quotes for JSON-like parsing
            params_str = params_str.replace('<|"|>', '"')
            args = _parse_gemma4_params(params_str)
        if name:
            tool_uses.append(
                {
                    "type": "tool_use",
                    "id": _make_tool_use_id(),
                    "name": name,
                    "input": args,
                    "_span": span,
                }
            )
    return tool_uses, text


def _parse_gemma4_value(v: str):
    """Parse a single gemma4 parameter value, handling nested objects."""
    v = v.strip()
    if v.startswith("{"):
        # Nested object — strip outer braces and recurse
        inner = v[1:]
        last = inner.rfind("}")
        if last >= 0:
            inner = inner[:last]
        return _parse_gemma4_params(inner)
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        # json.loads failed — likely unescaped quotes inside a string value.
        # Strip the outer <|"|>-derived quotes so the raw string doesn't
        # carry spurious delimiters (e.g. '"find . -name "*.py"..."' → the
        # inner content without surrounding quotes).
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            return v[1:-1]
        return v


def _parse_gemma4_params(params_str: str) -> dict:
    """Parse gemma4 key:value parameter string into a dict."""
    args = {}
    for part in _split_gemma4_params(params_str):
        colon = part.find(":")
        if colon > 0:
            k = part[:colon].strip()
            v = part[colon + 1 :].strip()
            args[k] = _parse_gemma4_value(v)
    return args


def _split_gemma4_params(s: str) -> list[str]:
    """Split gemma4 parameter string on commas, respecting braces and quotes."""
    parts = []
    depth = 0
    in_str = False
    start = 0
    for i, c in enumerate(s):
        if c == '"' and (i == 0 or s[i - 1] != "\\"):
            in_str = not in_str
        elif not in_str:
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
            elif c == "," and depth == 0:
                parts.append(s[start:i])
                start = i + 1
    if start < len(s):
        parts.append(s[start:])
    return parts


def _try_bare_json(text: str) -> tuple[list[dict], str]:
    """Parse bare JSON tool calls (must be on own line or at text start)."""
    tool_uses = []
    for match in _BARE_JSON_START_RE.finditer(text):
        brace_pos = match.start(1)
        obj_str = _extract_json_object(text, brace_pos)
        if obj_str is None:
            continue
        try:
            call = json.loads(obj_str)
            result = _parse_json_call(call)
            if result:
                result["_span"] = (brace_pos, brace_pos + len(obj_str))
                tool_uses.append(result)
        except (json.JSONDecodeError, AttributeError):
            continue
    return tool_uses, text


def _parse_gpt_oss_channels(
    text: str, has_tools: bool
) -> tuple[str, str, list[dict]] | None:
    """Parse gpt-oss channel-formatted output (harmony format).

    Returns (thinking, visible_text, tool_uses) or None if not gpt-oss format.

    Harmony format example:
        <|start|>assistant<|channel|>commentary to=functions.get_current_weather \\
            <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>

    The header (between channel name and <|message|>) may contain:
    - to=functions.TOOL_NAME
    - <|constrain|>json
    """
    if _GPT_OSS_DETECT not in text:
        return None

    thinking_parts: list[str] = []
    visible_parts: list[str] = []
    tool_uses: list[dict] = []

    for block in text.split("<|start|>"):
        if not block:
            continue

        channel_match = _GPT_OSS_CHANNEL_RE.search(block)
        if not channel_match:
            continue

        channel = channel_match.group(1).strip()
        after_channel = block[channel_match.end() :]

        msg_pos = after_channel.find("<|message|>")
        if msg_pos < 0:
            continue

        header = after_channel[:msg_pos]
        after_msg = after_channel[msg_pos + len("<|message|>") :]

        end_match = _GPT_OSS_END_RE.search(after_msg)
        content = (after_msg[: end_match.start()] if end_match else after_msg).strip()

        if channel == "analysis":
            thinking_parts.append(content)
        elif channel == "final":
            visible_parts.append(content)
        elif channel == "commentary" and has_tools and content:
            tool_name_match = _GPT_OSS_TOOL_NAME_RE.search(header)
            if not tool_name_match:
                logger.warning(
                    "gpt-oss tool call missing to=functions.NAME in header: %s",
                    header[:200],
                )
                tool_name = "unknown"
            else:
                tool_name = tool_name_match.group(1)
            try:
                args = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                args = {}
            tool_uses.append(
                {
                    "type": "tool_use",
                    "id": _make_tool_use_id(),
                    "name": tool_name,
                    "input": args,
                }
            )

    thinking = "\n".join(thinking_parts)
    visible = " ".join(visible_parts) if visible_parts else ""

    if not visible and not tool_uses and thinking:
        visible = thinking
        thinking = ""

    return thinking, visible, tool_uses


def _extract_gemma4_blocks(text: str) -> tuple[str, list[str]]:
    """Extract gemma4 channel thinking blocks using string operations.

    Recognizes ``<|channel>thought\n...<channel|>`` and ``thought\n...<channel|>``
    (when ``skip_special_tokens`` strips the opener).  Returns ``(cleaned_text, blocks)``
    where *cleaned_text* has all matched blocks removed.
    """
    blocks: list[str] = []
    out_parts: list[str] = []
    pos = 0

    while True:
        prefixed = text.find("<|channel>thought\n", pos)
        plain = text.startswith("thought\n", pos)

        if prefixed < 0 and not plain:
            out_parts.append(text[pos:])
            break

        if prefixed >= 0 and (not plain or prefixed <= pos):
            marker_end = prefixed + 18  # len("<|channel>thought\n")
            out_parts.append(text[pos:prefixed])
        else:
            marker_end = pos + 8  # len("thought\n")
            out_parts.append(text[pos:pos])

        close = text.find("<channel|>", marker_end)
        if close < 0:
            # Truncated block: stream ended before the closer was emitted.
            tail = text[marker_end:].strip()
            if tail:
                blocks.append(tail)
            break

        content = text[marker_end:close].strip()
        if content:
            blocks.append(content)

        pos = close + 10  # len("<channel|>")

    return "".join(out_parts), blocks


def parse_model_output(
    text: str,
    has_tools: bool,
    *,
    thinking_expected: bool = False,
) -> tuple[str, str, list[dict]]:
    """Parse raw model output into (thinking_text, visible_text, tool_use_blocks).

    Args:
        text: Raw model output text.
        has_tools: Whether tools were provided in the request.
        thinking_expected: When True, fire the orphan `</think>` heuristic
            that strips any leading text preceding a lone `</think>` token
            into the thinking channel.  Standard `<think>...</think>` pairs
            and Gemma4 channel format are always extracted — those tags are
            unambiguous — so this only controls the orphan path.  Defaults
            to False (conservative): a new caller that forgets to plumb the
            flag will preserve the literal token in visible content rather
            than silently misclassifying it (issue #307 review).
    """
    # Try gpt-oss channel format first
    gpt_oss_result = _parse_gpt_oss_channels(text, has_tools)
    if gpt_oss_result is not None:
        return gpt_oss_result

    thinking = ""

    # Extract thinking blocks — gemma4 channel format and standard <think> tags
    text, gemma4_blocks = _extract_gemma4_blocks(text)
    if gemma4_blocks:
        thinking = "\n".join(gemma4_blocks)

    think_matches = _THINK_RE.findall(text)
    if think_matches:
        think_text = "\n".join(m.strip() for m in think_matches)
        thinking = f"{thinking}\n{think_text}".strip() if thinking else think_text
        text = _THINK_RE.sub("", text)

    # Handle orphaned </think> — the chat template may open <think> in the
    # prompt so the generated text starts mid-think with only a closing tag.
    # Gated on `thinking_expected` so a non-thinking model that mentions the
    # literal token is not misclassified (issue #307).
    if thinking_expected:
        orphan_idx = text.find("</think>")
        if orphan_idx != -1:
            orphan_thinking = text[:orphan_idx].strip()
            if orphan_thinking:
                thinking = (
                    f"{thinking}\n{orphan_thinking}".strip()
                    if thinking
                    else orphan_thinking
                )
            text = text[orphan_idx + len("</think>") :].lstrip("\n")

    # Detect truncated thinking block: opener present but stream ended before
    # the matching </think> could be emitted (e.g. max_tokens hit mid-reasoning).
    open_pos = text.find("<think>")
    if open_pos != -1 and text.find("</think>", open_pos) == -1:
        pre = text[:open_pos].strip()
        partial = text[open_pos + len("<think>") :].strip()
        if partial:
            thinking = f"{thinking}\n{partial}".strip() if thinking else partial
        text = pre

    tool_uses: list[dict] = []
    if has_tools:
        parsers = [
            _try_gemma4,
            _try_qwen,
            _try_mistral,
            _try_llama,
            _try_deepseek,
            _try_minimax,
            _try_xml_func,
            _try_bare_json,
        ]

        for parser in parsers:
            tool_uses, text = parser(text)
            if tool_uses:
                break

        # Strip matched spans from text for tool calls.
        # For formats where multiple calls share one span (Mistral/DeepSeek),
        # the span is stripped if any call was kept — the dropped call's raw
        # text is unavoidably lost since it's embedded in the same block.
        kept_spans: set[tuple[int, int]] = set()
        for tu in tool_uses:
            span = tu.pop("_span", None)
            if span is not None:
                kept_spans.add(span)
        for start, end in sorted(kept_spans, reverse=True):
            text = text[:start] + text[end:]

    visible_text = text.strip()
    return thinking, visible_text, tool_uses


def fill_missing_required_args(
    tool_uses: list[dict],
    declared_tools: list[dict] | None,
) -> None:
    """Fill missing required string arguments in tool calls from the client's schema.

    Models sometimes omit fields the client marks as required (e.g. opencode's
    ``description`` on bash). Walk the declared tool schemas and inject an
    empty string for any required string parameter the model left out, so
    downstream Pydantic constructors (``ToolCallFunction.arguments``) don't
    blow up on the omission. Mutates *tool_uses* in place.

    Trade-off: injecting ``""`` prevents a Pydantic crash but produces a
    semantically degraded tool call.  A ``bash`` tool receiving
    ``command: ""`` runs nothing useful; a ``write_file`` tool receiving
    ``path: ""`` may behave dangerously.  We log a warning per injection,
    but callers should treat empty injected strings as "model failed to
    populate this field" rather than a meaningful value.

    Post-condition for any tool present in *declared_tools*: ``tu["input"]``
    is a freshly-constructed ``dict`` (never ``None``), even when no
    injection runs.  Tools NOT in *declared_tools* are left untouched, so
    ``_build_tool_calls`` in ``routers/chat.py`` still needs an ``or {}``
    guard — narrowed to the unknown-tool case, which this function cannot
    normalize because it has no schema to reason about.
    """
    # ``not declared_tools`` covers both ``None`` (no tools field on the
    # request) and ``[]`` (caller passed an explicit empty list).  Both
    # mean "no schemas available to normalize against", so we leave
    # ``tu["input"]`` untouched in either case.  The ``is None`` change
    # at the per-tool loop below only affects how each individual tool's
    # ``required_params`` is interpreted; this top-level guard is
    # deliberately broader.
    if not declared_tools:
        return

    # Build lookup: lowercase tool name -> {param: type} for required params
    schema_by_tool: dict[str, dict[str, str]] = {}
    for tool in declared_tools:
        func = tool.get("function") or {}
        name = (func.get("name") or "").lower()
        params = func.get("parameters") or {}
        required = set(params.get("required", []))
        properties = params.get("properties") or {}
        schema_by_tool[name] = {
            k: v.get("type", "") for k, v in properties.items() if k in required
        }

    for tu in tool_uses:
        required_params = schema_by_tool.get((tu.get("name") or "").lower())
        # Intentionally ``is None`` rather than the old ``not required_params``
        # check: the old guard skipped declared tools with zero required
        # params too, leaving their ``input`` as-is (e.g. ``None`` from a
        # gpt-oss ``null`` payload).  The new contract is unconditional
        # normalization for *any* declared tool — only unknown tools (truly
        # absent from the declared list) are skipped here.
        if required_params is None:
            continue
        # Always materialize our own dict and write it back, even when
        # ``required_params`` is empty (the tool exists in the declared
        # list but declares no required params).  Two observable changes
        # relative to the old version, both benign for current callers
        # (which all build a ``ToolCall`` from the dict immediately):
        #
        # - ``tu["input"]`` is replaced with a fresh dict, so callers must
        #   not rely on dict identity.
        # - ``input is None`` is normalized to ``input == {}`` regardless
        #   of whether any required-string injection runs.  An earlier
        #   draft made this promotion conditional on the tool having ≥1
        #   required param, which produced a footgun-y contract where
        #   ``input is None`` meant something different depending on the
        #   declared schema.
        inp = dict(tu.get("input") or {})
        tu["input"] = inp
        for param, param_type in required_params.items():
            if param not in inp or inp[param] is None:
                if param_type == "string":
                    logger.warning(
                        "Tool '%s' missing required string param '%s', injecting empty string",
                        tu.get("name"),
                        param,
                    )
                    inp[param] = ""
                else:
                    # No safe default to inject for a non-string param;
                    # the call WILL fail downstream (any executor that
                    # validates required fields will reject the missing
                    # key).  ``error`` rather than ``warning`` so the
                    # severity matches the certainty of failure.
                    logger.error(
                        "Tool '%s' missing required param '%s' (type %r); "
                        "forwarding call with the field absent — will fail "
                        "in any executor that validates required fields",
                        tu.get("name"),
                        param,
                        param_type,
                    )


def resolve_tool_names(
    tool_uses: list[dict], declared_tools: list[dict] | None
) -> None:
    """Resolve parsed tool names to declared tool names.

    Some models (e.g. Gemma 4) generate tool names that differ from the
    declared name — e.g. ``bash:run_command`` instead of ``Bash``.  This
    function maps parsed names back to declared names using:
      1. Exact match
      2. Case-insensitive match
      3. Case-insensitive prefix match (before first ``:``)
    """
    if not declared_tools or not tool_uses:
        return
    declared_names = []
    for t in declared_tools:
        # Support both OpenAI format {function: {name}} and flat {name}
        fn = t.get("function", t)
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            declared_names.append(name)
    if not declared_names:
        return
    lower_map = {n.lower(): n for n in declared_names}
    for tu in tool_uses:
        name = tu.get("name", "")
        if name in declared_names:
            continue
        # Case-insensitive
        if name.lower() in lower_map:
            tu["name"] = lower_map[name.lower()]
            continue
        # Prefix before first ':'
        prefix = name.split(":")[0]
        if prefix.lower() in lower_map:
            resolved = lower_map[prefix.lower()]
            logger.debug("Resolved tool name %r → %r via prefix match", name, resolved)
            tu["name"] = resolved
