"""Chat-template application and message normalization for inference.

Extracted from ``engine/inference.py`` (#454). Holds the pure helpers that
turn an OpenAI/Ollama/Anthropic-style message list into a model prompt:
tool injection, native-tool-format hints, thinking resolution, the text and
VLM ``apply_chat_template`` wrappers, segmented tokenization for the prompt
cache checkpoint path, and the tool-message rewrites for templates with
varying tool-role support. ``inference.py`` re-exports these names so call
sites and tests are unchanged.
"""

import collections.abc
import json
import logging
from typing import TYPE_CHECKING, Any

from olmlx.engine.template_caps import TemplateCaps

if TYPE_CHECKING:
    from olmlx.engine.prompt_cache.checkpoint import SegmentedPrompt

logger = logging.getLogger(__name__)


def _inject_tools_into_system(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Inject tool descriptions into the system message when the template doesn't support tools natively."""
    tool_desc_parts = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        tool_desc_parts.append(
            f"- {name}: {desc}\n  Parameters: {json.dumps(params, indent=2)}"
        )
    tool_block = (
        "You have access to the following tools. To call a tool, output a JSON object "
        'with "name" and "arguments" keys.\n\n'
        "Available tools:\n" + "\n".join(tool_desc_parts)
    )

    messages = list(messages)  # shallow copy
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            **messages[0],
            "content": messages[0]["content"] + "\n\n" + tool_block,
        }
    else:
        messages.insert(0, {"role": "system", "content": tool_block})
    return messages


_NATIVE_TOOL_HINT = (
    "Disregard any tool call format instructions above. "
    "You MUST use only the native tool call format provided by the system."
)

# Substrings that indicate the system message contains client-injected
# tool-format instructions targeting a non-native format.  These are formats
# clients embed as a fallback for models without native tool support, but
# they conflict with templates that DO have native tool support and confuse
# the model at long prompt lengths.
_CLIENT_TOOL_FORMAT_PATTERNS = (
    "<function=",  # Llama 3.x style — used by opencode, Claude Code
    "[TOOL_CALLS]",  # Mistral style
    "<|python_tag|>",  # Llama 3.x JSON style
    # NB: `<tool_call>` is intentionally absent — it's Qwen's *native* tool
    # call format token, so it would false-positive on Qwen models where the
    # client's instructions match the native format.  The opencode/Claude
    # Code case is still caught by `<function=`, which appears alongside
    # `<tool_call>` in their format examples.
)


def _add_native_tool_hint(
    messages: list[dict], native_template_text: str = ""
) -> list[dict]:
    """Append a hint to the system message to use native tool call format.

    Clients like opencode and Claude Code embed their own tool-format
    instructions (e.g. ``<function=Name>``) in the system message.  These
    conflict with the model's native tool call format (e.g. Gemma 4's
    ``<|tool_call>call:Name{...}<tool_call|>``).  At long prompt lengths
    the model follows the client's text instructions instead of using native
    tokens, producing unparseable output.

    A short override at the end of the system message steers the model back
    to the native format without modifying the client's original content.

    **Scope: this is intentionally general, not Gemma 4-specific.**
    Any model with native tool support (Qwen3, Mistral, Llama 3.x, Gemma 4,
    etc.) can be confused by client-injected non-native format instructions
    at long prompt lengths.  Gemma 4 is just where the symptom was first
    observed.  The patterns matched are the formats clients inject as a
    fallback for models *without* native tool support — when the model DOES
    have native support, the template format is authoritative.

    Only applied when the system message contains a conflict pattern that
    is NOT also present in the model's own chat template.  This prevents
    false positives for models like Mistral whose template natively contains
    ``[TOOL_CALLS]``: in that case the client's instructions match the
    model's native format and the "Disregard" override would suppress
    legitimate guidance.  Pass ``native_template_text`` from the call site;
    omit it (or pass empty) to skip the suppression check.
    """
    if not messages or messages[0].get("role") != "system":
        return messages
    content = messages[0].get("content", "")
    # Multimodal content (list of parts) is not handled — the conflict pattern
    # is text-only and the hint targets text-only system messages.
    if not isinstance(content, str):
        return messages
    if _NATIVE_TOOL_HINT in content:
        return messages
    # A pattern is only a "conflict" if it appears in the system message
    # AND the model's own template doesn't use it natively.
    triggered = [
        p
        for p in _CLIENT_TOOL_FORMAT_PATTERNS
        if p in content and p not in native_template_text
    ]
    if not triggered:
        return messages
    messages = list(messages)  # shallow copy
    messages[0] = {
        **messages[0],
        "content": content + "\n\n" + _NATIVE_TOOL_HINT,
    }
    return messages


def _get_chat_template_text(tokenizer: Any) -> str:
    """Extract the chat template as a single string for substring matching.

    Handles both text tokenizers (chat_template directly) and VLM processors
    (chat_template on the wrapped tokenizer).  Lists of named templates are
    flattened into a single space-joined string.
    """
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl is None:
        sub = getattr(tokenizer, "tokenizer", None)
        if sub is not None:
            tpl = getattr(sub, "chat_template", None)
    if tpl is None:
        return ""
    if isinstance(tpl, list):
        return " ".join(t.get("template", "") for t in tpl if isinstance(t, dict))
    return tpl if isinstance(tpl, str) else ""


def _resolve_thinking_active(
    caps: TemplateCaps,
    tools: list[dict] | None,
    enable_thinking: bool | None,
) -> bool:
    """Return whether the chat template will request thinking for this call.

    Centralises the resolution rules used both inside ``_apply_chat_template``
    (to set the ``enable_thinking`` kwarg) and by ``generate_chat`` (to tell
    streaming routers whether to wait for an orphan `</think>` — issue #307).
    """
    if not caps.supports_enable_thinking:
        return False
    if enable_thinking is not None:
        return enable_thinking
    # Default: think unless tools were declared (backward compat for
    # non-Anthropic callers that expect tool calls without thinking).
    return not bool(tools)


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    caps: TemplateCaps | None = None,
    *,
    tokenize: bool = False,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
) -> Any:
    """Core chat template application.

    Uses TemplateCaps to decide which kwargs to pass, avoiding blind try/except.
    Returns str when tokenize=False, token list/dict when tokenize=True.
    """
    if caps is None:
        caps = TemplateCaps()

    kwargs: dict[str, Any] = {"tokenize": tokenize, "add_generation_prompt": True}

    if tools and caps.supports_tools:
        kwargs["tools"] = tools
    elif tools and not caps.supports_tools:
        logger.info(
            "Template lacks tool support, injecting tool descriptions into system message"
        )
        messages = _inject_tools_into_system(messages, tools)

    if caps.supports_enable_thinking:
        kwargs["enable_thinking"] = _resolve_thinking_active(
            caps, tools, enable_thinking
        )

    # Channel-format reasoners (gpt-oss / Harmony) use ``reasoning_effort``
    # instead of the boolean ``enable_thinking``. Only pass it when the template
    # declares the variable and a level was requested.
    if caps.supports_reasoning_effort and reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as exc:
        # If tools kwarg caused the error, retry without it (injecting instead)
        if tools and "tools" in kwargs:
            logger.warning(
                "apply_chat_template failed with tools kwarg (%s), retrying with injection",
                exc,
            )
            del kwargs["tools"]
            # Keep enable_thinking — it's independent of the tools kwarg failure
            messages = _inject_tools_into_system(messages, tools)
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except Exception as exc2:
                raise RuntimeError(
                    f"Chat template failed even without tools: {exc2}"
                ) from exc2
        raise RuntimeError(f"Chat template failed: {exc}") from exc


def apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    caps: TemplateCaps | None = None,
    *,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Apply chat template for text-only models (mlx-lm), returning prompt text."""
    return _apply_chat_template(
        tokenizer,
        messages,
        tools,
        caps,
        tokenize=False,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )


_KNOWN_EOM_TOKEN_STRINGS: tuple[str, ...] = (
    # Harmony format (gpt-oss): ``eos_token_id`` is ``<|return|>``
    # (end-of-generation, never appears in input), but the actual
    # end-of-message marker is ``<|end|>``.
    "<|end|>",
    # Gemma chat template: ``eos_token_id`` is the base ``<eos>``;
    # the per-message terminator is ``<end_of_turn>``.
    "<end_of_turn>",
)


def _message_boundary_token_ids(tokenizer: Any) -> set[int]:
    """Return the set of token IDs that mark end-of-message in chat templates.

    Combines two signals:

    1. ``tokenizer.eos_token_id`` — works for templates where the EOS IS
       the message-end marker: Qwen3 ``<|im_end|>``, Llama 3 ``<|eot_id|>``,
       Nemotron-style ``<|im_end|>``.

    2. Known per-template message-end strings looked up via
       ``convert_tokens_to_ids`` — for templates where the EOS is *not* the
       message-end marker (gpt-oss Harmony format, where eos is
       ``<|return|>`` but messages end with ``<|end|>``; Gemma chat where
       the per-turn marker is ``<end_of_turn>`` not the base ``<eos>``).

    For tokenizers like Llama 2's where ``eos_token_id`` (``</s>``) can in
    principle appear inside content, ``tokenize_segmented_chat`` falls back
    to a single-segment prompt via the ``len(boundaries) != len(messages)``
    guard; ``_setup_via_checkpoint_path`` then skips the checkpoint path
    for the request.
    """
    out: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple, set)):
            out.update(int(x) for x in eos if x is not None)
        else:
            out.add(int(eos))
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk = getattr(tokenizer, "unk_token_id", None)
    if callable(convert):
        for tok_str in _KNOWN_EOM_TOKEN_STRINGS:
            try:
                tok_id = convert(tok_str)
            except Exception:
                continue
            if tok_id is None or tok_id == unk:
                continue
            out.add(int(tok_id))
    return out


def tokenize_segmented_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    full_tokens: list[int] | None = None,
    **template_kwargs: Any,
) -> "SegmentedPrompt":
    """Split a tokenized chat into per-message segments by scanning for
    end-of-message marker tokens.

    Two call modes:

    1. ``full_tokens`` provided — the caller already has the authoritative
       tokenization (typically from ``tokenize_for_cache`` which replicates
       ``stream_generate``'s BOS heuristic). The function uses these tokens
       directly. This is the path that ``_setup_via_checkpoint_path`` takes
       so that ``segmented.flatten() == prompt_tokens`` holds by
       construction — without it, a re-tokenization via
       ``apply_chat_template(tokenize=True)`` and the caller's
       ``tokenize_for_cache(apply_chat_template_text(...))`` can differ by
       a leading BOS token on tokenizers like Llama 3's, silently
       disabling the checkpoint path for every request on those models.

    2. ``full_tokens`` omitted — the function calls
       ``apply_chat_template(messages, **template_kwargs)`` itself. Used by
       unit tests with synthetic tokenizers; not exercised on the production
       request path.

    EOM detection: scan ``full`` for ``tokenizer.eos_token_id`` — for Qwen3
    this is ``<|im_end|>``, for Llama 3 ``<|eot_id|>``, for Gemma
    ``<end_of_turn>``. Each EOM occurrence ends one message. The position
    *after* each EOM is a segment boundary; the last segment absorbs the
    chat template's trailing assistant-prompt suffix so
    ``segmented.flatten()`` equals ``full``.

    Falls back to a single-segment prompt (logged at debug) when the
    tokenizer exposes no EOM token or the detected EOM count differs from
    ``len(messages)``. Callers detect this via ``len(segmented.segments)
    <= 1`` and skip the checkpoint path for the request, since a
    single-segment prompt has no usable mid-prompt checkpoint boundary.
    """
    from olmlx.engine.prompt_cache.checkpoint import Segment, SegmentedPrompt

    if not messages:
        return SegmentedPrompt(segments=[])

    if full_tokens is not None:
        full = list(full_tokens)
    else:
        # Normalise apply_chat_template's varied return types — see
        # _apply_chat_template_text. Possible shapes: flat list[int],
        # batch-of-1 list[list[int]], BatchEncoding mapping with
        # ``input_ids``, or array-like sequences (numpy ndarray, torch
        # tensor, mlx array) whose elements are ints.
        raw = tokenizer.apply_chat_template(messages, **template_kwargs)
        if isinstance(raw, collections.abc.Mapping):
            token_seq: Any = raw.get("input_ids")
        else:
            token_seq = raw
        # Unwrap a batch-of-1 nested sequence.
        if (
            token_seq is not None
            and not isinstance(token_seq, (str, bytes))
            and len(token_seq) > 0
            and isinstance(token_seq[0], (list, tuple))
        ):
            token_seq = token_seq[0]
        try:
            full = [int(t) for t in token_seq]
        except (TypeError, ValueError):
            logger.debug(
                "tokenize_segmented_chat: cannot coerce apply_chat_template "
                "result (%s) to list[int]; falling back to empty segment",
                type(raw).__name__,
            )
            return SegmentedPrompt(
                segments=[Segment(tokens=[], role=messages[-1]["role"])]
            )

    eom_ids = _message_boundary_token_ids(tokenizer)
    if not eom_ids:
        return SegmentedPrompt(
            segments=[Segment(tokens=full, role=messages[-1]["role"])]
        )

    # Find positions immediately after each EOM marker.
    boundaries: list[int] = []
    for i, tok in enumerate(full):
        if tok in eom_ids:
            boundaries.append(i + 1)

    # Most chat templates emit exactly one EOM per message. Some
    # (Harmony / gpt-oss) emit extras because the template injects a
    # baseline preamble (system message about output channels) that the
    # caller didn't supply. Accept extras by treating them as preamble
    # segments at the start with the first caller-supplied role; reject
    # only when boundaries < messages (the template emitted FEWER EOMs
    # than expected, which means we can't safely align roles).
    if len(boundaries) < len(messages):
        return SegmentedPrompt(
            segments=[Segment(tokens=full, role=messages[-1]["role"])]
        )

    # Extend the last boundary to include any trailing template tokens
    # (e.g. ``\n<|im_start|>assistant\n``) so flatten() == full.
    if boundaries[-1] < len(full):
        boundaries[-1] = len(full)

    # Build the role list, padding the front with extras using
    # ``messages[0]["role"]`` so the LRU eviction tier still picks a
    # sensible (typically "system") priority for the injected preamble.
    extras = len(boundaries) - len(messages)
    roles: list[Any] = [messages[0]["role"]] * extras + [m["role"] for m in messages]

    segments: list[Segment] = []
    start = 0
    for k, end in enumerate(boundaries):
        segments.append(Segment(tokens=full[start:end], role=roles[k]))
        start = end
    return SegmentedPrompt(segments=segments)


def _normalize_tool_calls_in_messages(messages: list[dict]) -> list[dict]:
    """Normalise tool_calls in assistant messages for chat templates.

    Different chat templates expect different tool_call layouts:

    * **Qwen / Llama**: flat ``{name, arguments: dict}``
    * **Gemma 4**: nested ``{function: {name, arguments}, id, type}``

    Rather than guessing which layout a template needs, this function
    produces a *union* dict that satisfies both::

        {
            "name": "read",
            "arguments": {"path": "/foo"},
            "function": {"name": "read", "arguments": {"path": "/foo"}},
            "id": "call_x",
            "type": "function",
        }

    It also ensures ``arguments`` is always a parsed dict (never a JSON
    string), which is what both Qwen's ``|items`` filter and Gemma's
    ``is mapping`` test require.
    """
    result = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m = dict(m)
            # Templates (Qwen3, Llama3, ...) access message.content directly;
            # OpenAI clients send content=null on tool-call assistant messages
            # and the router strips None via exclude_none=True, leaving the
            # key absent. Coerce to "" so the template can render.
            if m.get("content") is None:
                m["content"] = ""
            normalised = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                name = fn.get("name", tc.get("name", ""))
                args = fn.get("arguments", tc.get("arguments", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                normalised.append(
                    {
                        "name": name,
                        "arguments": args,
                        "function": {"name": name, "arguments": args},
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                    }
                )
            m["tool_calls"] = normalised
            result.append(m)
        else:
            result.append(m)
    return result


def _convert_tool_messages_to_responses(messages: list[dict]) -> list[dict]:
    """Convert ``role: "tool"`` messages to ``tool_responses`` format.

    Some models (e.g. Gemma 4) don't support OpenAI-style ``role: "tool"``
    messages.  Instead they expect a ``tool_responses`` array merged into the
    preceding assistant message that made the ``tool_calls``.  This keeps
    tool responses inside the model turn, which is critical — the template
    omits ``<turn|>`` after tool_responses so the model continues in the same
    turn.  Placing them on a separate user message would put the model's
    generation inside a user turn, producing degenerate output.

    For *intermediate* assistant messages (followed by more messages), a
    newline content placeholder ensures the template emits ``<turn|>`` to
    properly close the model turn before the next turn opens.  The last
    assistant message with tool_responses keeps empty content so the model
    continues generating in the same turn.
    """
    if not any(m.get("role") == "tool" for m in messages):
        return messages

    # Build a mapping from tool_call_id → function name across all assistant messages.
    id_to_name: dict[str, str] = {}
    for m in messages:
        for tc in m.get("tool_calls", []):
            tc_id = tc.get("id", "")
            fn = tc.get("function", {})
            name = fn.get("name", "") if isinstance(fn, dict) else ""
            if tc_id and name:
                id_to_name[tc_id] = name

    result: list[dict] = []
    for m in messages:
        if m.get("role") == "tool":
            tc_id = m.get("tool_call_id", "")
            name = id_to_name.get(tc_id, "unknown")
            resp = {"name": name, "response": m.get("content", "")}
            # Merge into the preceding assistant message.
            prev = result[-1] if result else None
            if prev and prev.get("role") == "assistant":
                prev.setdefault("tool_responses", []).append(resp)
            else:
                # No preceding assistant — shouldn't happen, but create a
                # model-role message to keep the turn correct.
                result.append(
                    {"role": "assistant", "content": "", "tool_responses": [resp]}
                )
        else:
            result.append(dict(m))  # shallow copy to avoid mutating input

    # Ensure intermediate assistant messages with tool_responses get their
    # model turn closed.  The template omits <turn|> when tool_responses is
    # present and content is falsy.  Setting content to "\n" makes it truthy
    # (triggering <turn|>) while rendering as empty after strip_thinking().
    # The *last* such message keeps empty content so the model can continue
    # generating in the same turn.
    for i in range(len(result) - 1):
        m = result[i]
        if (
            m.get("role") == "assistant"
            and m.get("tool_responses")
            and not m.get("content")
        ):
            m["content"] = "\n"

    return result


def _convert_tool_messages_to_user_text(messages: list[dict]) -> list[dict]:
    """Fold tool turns into plain text for templates that only support the
    user/system/assistant roles.

    The minimal Devstral/Mistral template has no native tool support and
    ``raise_exception``\\ s on any other role, so an OpenAI-style
    ``role: "tool"`` result message crashes ``apply_chat_template`` with
    "Only user, system and assistant roles are supported!".  For such templates
    we rewrite the conversation using only the accepted roles:

    * An assistant message carrying ``tool_calls`` gets a readable rendering of
      those calls appended to its content, so the model still sees what it
      called and the turn isn't empty.
    * Each ``role: "tool"`` message becomes ``user`` text.  Consecutive tool
      results are merged into one user message to avoid back-to-back user
      turns.
    """
    if not any(m.get("role") == "tool" for m in messages):
        return messages

    # tool_call_id -> function name, for labelling results.
    id_to_name: dict[str, str] = {}
    for m in messages:
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            name = fn.get("name", tc.get("name", "")) if isinstance(fn, dict) else ""
            tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
            if tc_id and name:
                id_to_name[tc_id] = name

    result: list[dict] = []
    pending: list[str] = []  # buffered tool-result lines, flushed as one user msg

    def flush() -> None:
        if pending:
            result.append({"role": "user", "content": "\n".join(pending)})
            pending.clear()

    for m in messages:
        role = m.get("role")
        if role == "tool":
            name = id_to_name.get(m.get("tool_call_id", ""), "tool")
            pending.append(f"[tool_result] {name}: {m.get('content', '')}")
            continue

        flush()
        if role == "assistant" and m.get("tool_calls"):
            m = dict(m)
            lines = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                name = fn.get("name", tc.get("name", ""))
                args = fn.get("arguments", tc.get("arguments", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                lines.append(f"[tool_call] {name}({json.dumps(args, default=str)})")
            existing = (m.get("content") or "").strip()
            m["content"] = (existing + "\n" if existing else "") + "\n".join(lines)
            m.pop("tool_calls", None)  # rendered into content; template ignores it
            result.append(m)
        else:
            result.append(dict(m))

    flush()
    return result


def _vlm_image_token(processor: Any) -> str:
    """Best-effort image placeholder token for a VLM processor/tokenizer."""
    for obj in (processor, getattr(processor, "tokenizer", None)):
        token = getattr(obj, "image_token", None)
        if isinstance(token, str) and token:
            return token
    return "<image>"


def _inject_image_markers(messages: list[dict], num_images: int) -> list[dict]:
    """Rewrite the last user message's content into a parts list with
    ``num_images`` image markers so the chat template emits image placeholders
    while ``tools=`` still renders natively (issue #428).

    Single-turn scope: all images attach to the last user turn.
    """
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            text = m.get("content")
            parts: list[dict[str, Any]] = [{"type": "image"} for _ in range(num_images)]
            if isinstance(text, list):
                parts.extend(text)
            elif isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
            m["content"] = parts
            return out
    logger.warning(
        "VLM tools+images: no user message to attach %d image(s) to; "
        "rendering text-only",
        num_images,
    )
    return out


def _apply_chat_template_vlm(
    processor: Any,
    model: Any,
    messages: list[dict],
    images: list[str] | None = None,
    tools: list[dict] | None = None,
    enable_thinking: bool | None = None,
    audio: list[str] | None = None,
) -> str:
    """Apply chat template for vision-language models (mlx-vlm).

    When tools are provided, bypasses mlx_vlm.apply_chat_template and calls
    the processor's tokenizer directly.  mlx_vlm's message processing wraps
    text content in ``[{type: text, text: ..., content: ...}]`` dicts, which
    the Jinja template renders as Python list repr — garbling the prompt.
    """
    if audio and tools:
        raise ValueError(
            "tools + audio is not supported in this version: combining native "
            "tool calling with audio input is out of scope (#426). Send the "
            "audio without tools, or the tools without audio."
        )
    if tools:
        # Use the tokenizer directly to get clean native tool formatting.
        # mlx_vlm.apply_chat_template wraps text content in dicts that some
        # Jinja templates render as Python list repr — garbling the prompt.
        tok = (
            processor.tokenizer
            if hasattr(processor, "tokenizer")
            and hasattr(processor.tokenizer, "apply_chat_template")
            else processor
        )
        # Images are carried separately (msg["images"]); inject content-part
        # markers so the template emits image placeholders alongside the native
        # tool tags (#428).  The image data is threaded to mlx_vlm.generate.
        if images:
            messages = _inject_image_markers(messages, len(images))
        kwargs: dict = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        prompt = tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        if images:
            # Assumes one placeholder token per image — true for the Gemma-family
            # native-tools targets in scope (#428). Templates that expand each
            # image into many patch tokens (e.g. Qwen2-VL's repeated
            # ``<|image_pad|>``) would trip this guard; revisit if such a model
            # gains native-tools support.
            image_token = _vlm_image_token(processor)
            found = prompt.count(image_token)
            if found != len(images):
                raise ValueError(
                    f"VLM tools+images: expected {len(images)} image placeholder(s) "
                    f"({image_token!r}) in the rendered prompt but found {found}; "
                    "the model's chat template may not support image content parts."
                )
        return prompt

    import mlx_vlm

    config = model.config if hasattr(model, "config") else {}
    num_images = len(images) if images else 0
    num_audios = len(audio) if audio else 0
    # mlx_vlm.apply_chat_template forwards **kwargs to the tokenizer's
    # apply_chat_template, so enable_thinking reaches the Jinja template
    # (templates that don't declare the variable ignore it).  Only forward
    # when explicitly set so the template's own default is preserved otherwise.
    extra_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        extra_kwargs["enable_thinking"] = enable_thinking
    # Pass the full message list so the model gets proper conversation context
    result = mlx_vlm.apply_chat_template(
        processor,
        config,
        messages,
        num_images=num_images,
        num_audios=num_audios,
        **extra_kwargs,
    )
    if not isinstance(result, str):
        raise TypeError(
            f"mlx_vlm.apply_chat_template returned non-str ({type(result).__name__}); "
            "expected tokenize=False default"
        )
    return result
