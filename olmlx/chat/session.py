"""Chat session with agent loop for tool use."""

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, Literal, TypedDict, cast

from olmlx.chat.builtin_tools import BuiltinToolManager
from olmlx.chat.config import ChatConfig
from olmlx.chat.errors import ToolError
from olmlx.chat.mcp_client import MCPClientManager
from olmlx.chat.skills import SkillManager
from olmlx.chat.tool_safety import ToolSafetyPolicy
from olmlx.config import settings
from olmlx.engine.inference import (
    apply_chat_template_text,
    estimate_kv_cache_bytes,
    tokenize_for_cache,
    generate_chat,
)
from olmlx.engine.model_manager import LoadedModel, ModelManager
from olmlx.engine.tool_parser import parse_model_output
from olmlx.routers.thinking_split import (
    _THINKING_PAIRS,
    flush_split_thinking,
    split_thinking_parts,
)
from olmlx.utils import memory as memory_utils
from olmlx.utils import tracing as _tracing

logger = logging.getLogger(__name__)

# Maximum messages to keep when truncating (system message + N recent pairs)
_MIN_MESSAGES_AFTER_TRUNCATE = 6

# Emergency truncation target when first pass still exceeds memory
_MIN_MESSAGES_EMERGENCY_TRUNCATE = 2


def _truncate_keep_recent(
    messages: list[dict], system_idx: int, target_count: int
) -> list[dict]:
    """Keep the system message (if any) plus the most recent messages.

    Walk backwards collecting messages until ``target_count`` is reached,
    then continue back to the next ``user`` message so a turn boundary is
    never split (no orphan ``tool_result`` or dangling ``tool_call``).
    ``system_idx`` is 1 when ``messages[0]`` is a system message, else 0.
    """
    kept: list[dict] = []
    if system_idx > 0:
        kept.append(messages[0])

    recent: list[dict] = []
    idx = len(messages) - 1
    while idx >= system_idx:
        recent.insert(0, messages[idx])
        idx -= 1
        if len(recent) >= target_count:
            # Walk back to the next user message to avoid splitting a turn.
            while idx >= system_idx:
                msg = messages[idx]
                recent.insert(0, msg)
                idx -= 1
                if msg.get("role") == "user":
                    break
            break

    kept.extend(recent)
    return kept


# Thinking tag constants
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_CONTENT_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Gemma-4 native tool-call markup (plain text, not special tokens). Display
# is suppressed during streaming; the raw markup stays in the tracker's
# accumulated text so parse_model_output still extracts the call at turn end.
_TOOL_CALL_OPEN = "<|tool_call>"
_TOOL_CALL_CLOSE = "<tool_call|>"


def _longest_partial_tag_suffix(buf: str, tag: str) -> int:
    """Largest ``k`` (``0 < k < len(tag)``) such that ``buf[-k:] == tag[:k]``.

    Used to hold back the trailing bytes of *buf* that might be the start of
    *tag* straddling a chunk boundary.
    """
    for k in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf[-k:] == tag[:k]:
            return k
    return 0


class _ToolMarkupStripper:
    """Remove ``<|tool_call>…<tool_call|>`` spans from a stream of text.

    Holds partial open/close delimiters across chunk boundaries. Display-only:
    callers keep the raw text elsewhere for parsing.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while self._buf:
            if not self._inside:
                idx = self._buf.find(_TOOL_CALL_OPEN)
                if idx != -1:
                    out.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(_TOOL_CALL_OPEN) :]
                    self._inside = True
                    continue
                keep = _longest_partial_tag_suffix(self._buf, _TOOL_CALL_OPEN)
                out.append(self._buf[: len(self._buf) - keep] if keep else self._buf)
                self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                break
            idx = self._buf.find(_TOOL_CALL_CLOSE)
            if idx != -1:
                self._buf = self._buf[idx + len(_TOOL_CALL_CLOSE) :]
                self._inside = False
                continue
            keep = _longest_partial_tag_suffix(self._buf, _TOOL_CALL_CLOSE)
            self._buf = self._buf[len(self._buf) - keep :] if keep else ""
            break
        return "".join(out)

    def flush(self) -> str:
        """Emit any held-back bytes at stream end.

        A partial open-tag held while *outside* a call was literal text. Bytes
        held while *inside* an unterminated call are dropped (no close arrived).
        """
        if self._inside:
            self._buf = ""
            return ""
        out = self._buf
        self._buf = ""
        return out


class _TokenEvent(TypedDict):
    type: Literal["token"]
    text: str


class _ThinkingTokenEvent(TypedDict):
    type: Literal["thinking_token"]
    text: str


class _ToolCallEvent(TypedDict):
    type: Literal["tool_call"]
    name: str
    arguments: dict[str, Any]
    id: str


class _ToolResultEvent(TypedDict):
    type: Literal["tool_result"]
    name: str
    result: str
    id: str


class _ToolErrorEvent(TypedDict):
    type: Literal["tool_error"]
    name: str
    error: str
    id: str
    is_user_error: bool


class _RepetitionDetectedEvent(TypedDict):
    type: Literal["repetition_detected"]


class _MemoryTruncatedEvent(TypedDict):
    type: Literal["memory_truncated"]
    message: str


class _ModelLoadErrorEvent(TypedDict):
    type: Literal["model_load_error"]
    error: str


class _ThinkingStartEvent(TypedDict):
    type: Literal["thinking_start"]


class _ThinkingEndEvent(TypedDict):
    type: Literal["thinking_end"]


class _DoneEvent(TypedDict):
    type: Literal["done"]


class _MaxTurnsExceededEvent(TypedDict):
    type: Literal["max_turns_exceeded"]


class _ToolConfirmationNeededEvent(TypedDict):
    type: Literal["tool_confirmation_needed"]
    name: str
    arguments: dict[str, Any]
    id: str


class _ToolApprovedEvent(TypedDict):
    type: Literal["tool_approved"]
    name: str
    id: str


class _ToolDeniedEvent(TypedDict):
    type: Literal["tool_denied"]
    name: str
    arguments: dict[str, Any]
    id: str
    reason: str


class _ToolAutoJudgingEvent(TypedDict):
    type: Literal["tool_auto_judging"]
    name: str
    arguments: dict[str, Any]
    id: str


class _QuestionEvent(TypedDict):
    type: Literal["question"]
    header: str
    question: str
    options: list[str] | None
    multiple: bool
    id: str


class _ToolFailuresExceededEvent(TypedDict):
    type: Literal["tool_failures_exceeded"]
    message: str
    consecutive_failures: int


# Union type for all events yielded by send_message
ChatEvent = (
    _TokenEvent
    | _ThinkingTokenEvent
    | _ToolCallEvent
    | _ToolResultEvent
    | _ToolErrorEvent
    | _RepetitionDetectedEvent
    | _MemoryTruncatedEvent
    | _ModelLoadErrorEvent
    | _ThinkingStartEvent
    | _ThinkingEndEvent
    | _DoneEvent
    | _MaxTurnsExceededEvent
    | _ToolConfirmationNeededEvent
    | _ToolApprovedEvent
    | _ToolDeniedEvent
    | _ToolAutoJudgingEvent
    | _QuestionEvent
    | _ToolFailuresExceededEvent
)


class _TurnStreamResult(TypedDict):
    """Out-of-band results from ``_stream_one_turn``.

    Filled in place by the helper because an async generator cannot
    return values.
    """

    full_text: str
    thinking_expected: bool
    repetition_stopped: bool


class ThinkingTracker:
    """Splits streaming output into thinking vs visible, with tool markup
    suppressed from the visible channel.

    Delegates thinking-tag detection to the shared
    ``routers.thinking_split`` state machine (``<think>…</think>`` AND
    Gemma-4 ``<|channel>thought\\n…<channel|>``), and runs the visible
    channel through ``_ToolMarkupStripper`` so Gemma-4's
    ``<|tool_call>…<tool_call|>`` markup never reaches the display. The raw
    text is kept verbatim in ``accumulated`` for the turn-end parse step.
    """

    def __init__(
        self,
        implicit_mode: bool = False,
        thinking_disabled: bool = False,
        template_has_thinking: bool = False,
    ):
        self._thinking_disabled = thinking_disabled
        self._accumulated = ""
        self._think_emitted = 0
        self._visible_emitted = 0
        self._in_thinking = False
        self._just_started = False
        # A template that injects <think> (implicit) or otherwise advertises
        # thinking widens the splitter's orphan-detection window and enables
        # the orphan-</think> heuristic; pre-tag content is then held as
        # potential thinking. When thinking is NOT expected, drive the detect
        # window to 0 so plain content streams token-by-token (an explicit
        # <think>/channel opener — full or split across chunks — is still
        # caught via find() + the partial-tag hold-back).
        expected = bool(implicit_mode or template_has_thinking)
        self._split_state: dict = {"thinking_expected": expected}
        if not expected:
            self._split_state["detect_limit"] = 0
        self._tool_stripper = _ToolMarkupStripper()

    @property
    def accumulated(self) -> str:
        return self._accumulated

    @property
    def in_thinking(self) -> bool:
        return self._in_thinking

    @property
    def just_started(self) -> bool:
        """True if thinking block just started on this chunk."""
        return self._just_started

    @property
    def think_emitted(self) -> int:
        return self._think_emitted

    @property
    def visible_emitted(self) -> int:
        return self._visible_emitted

    def feed(self, text: str) -> tuple[str | None, str | None, bool, bool]:
        """Ingest a chunk; return (think_delta, visible_delta, thinking_ended,
        thinking_started)."""
        self._accumulated += text
        parts = split_thinking_parts(text, self._split_state)

        think_parts: list[str] = []
        visible_parts: list[str] = []
        thinking_started = False
        thinking_ended = False

        for channel, fragment in parts:
            if channel == "thinking":
                if self._thinking_disabled:
                    continue
                if not self._in_thinking:
                    thinking_started = True
                    self._in_thinking = True
                think_parts.append(fragment)
                self._think_emitted += len(fragment)
            else:  # content
                if self._in_thinking:
                    thinking_ended = True
                    self._in_thinking = False
                visible = self._tool_stripper.feed(fragment)
                if visible:
                    visible_parts.append(visible)
                    self._visible_emitted += len(visible)

        self._just_started = thinking_started
        think_delta = "".join(think_parts) or None
        visible_delta = "".join(visible_parts) or None
        return think_delta, visible_delta, thinking_ended, thinking_started

    def flush(self) -> tuple[str | None, str | None, bool]:
        """Flush the splitter's held buffer at stream end.

        Returns ``(think_delta, visible_delta, thinking_started)``. Classifies
        the leftover by phase:

        - thinking disabled → everything surfaces as visible (thinking is
          never shown; an unconfirmed/implicit block can't be hidden);
        - still in ``detect`` with thinking expected → implicit thinking that
          never saw a close tag, surfaced as thinking (the old implicit-mode
          contract);
        - ``in_think`` (unterminated explicit block) → thinking;
        - otherwise (plain/passthrough leftover) → visible.
        """
        phase = self._split_state.get("phase", "detect")
        thinking_expected = bool(self._split_state.get("thinking_expected"))
        thinking_buf, content_buf = flush_split_thinking(self._split_state)

        if self._thinking_disabled:
            leftover = (thinking_buf or "") + (content_buf or "")
            visible = self._tool_stripper.feed(leftover) + self._tool_stripper.flush()
            if visible:
                self._visible_emitted += len(visible)
                return None, visible, False
            return None, None, False

        think_text = thinking_buf or ""
        if phase == "detect" and thinking_expected and content_buf:
            think_text += content_buf
            content_buf = ""

        think_delta: str | None = None
        thinking_started = False
        if think_text:
            if not self._in_thinking:
                thinking_started = True
                self._in_thinking = True
            self._think_emitted += len(think_text)
            think_delta = think_text

        # Always flush the tool stripper, even when the splitter left no
        # content: it may hold a partial ``<|tool_call>`` open-tag from an
        # earlier streamed fragment that turned out to be literal visible text
        # (dropping it would lose visible bytes at stream end).
        # ``flush_split_thinking`` only yields content in the ``detect``/
        # ``passthrough`` phases, both of which leave ``_in_thinking`` False —
        # so flush returns thinking XOR visible, never a think→visible
        # transition needing a ``thinking_end`` between them.
        visible = (
            self._tool_stripper.feed(content_buf or "") + self._tool_stripper.flush()
        )
        visible_delta: str | None = None
        if visible:
            self._visible_emitted += len(visible)
            visible_delta = visible

        return think_delta, visible_delta, thinking_started

    def strip_on_repetition(self) -> int | None:
        """Truncate accumulated text at the start of the open thinking block.

        Removes an incomplete thinking block so the turn-end parse sees a
        clean prefix. Searches for the latest open tag of any known pair.
        """
        if not self._in_thinking:
            return None
        cut = max(
            (self._accumulated.rfind(open_tag) for open_tag, _ in _THINKING_PAIRS),
            default=-1,
        )
        if cut >= 0:
            self._accumulated = self._accumulated[:cut]
            self._visible_emitted = len(self._accumulated)
            self._in_thinking = False
            return self._visible_emitted
        return None


class ChatSession:
    """Manages conversation history and the agent loop."""

    def __init__(
        self,
        config: ChatConfig,
        manager: ModelManager,
        mcp: MCPClientManager | None = None,
        skills: SkillManager | None = None,
        builtin: BuiltinToolManager | None = None,
        tool_safety: ToolSafetyPolicy | None = None,
    ):
        self.config = config
        self.manager = manager
        self.mcp = mcp
        self.skills = skills
        self.builtin = builtin
        self.tool_safety = tool_safety
        self.messages: list[dict] = []

        system_prompt = self._build_system_prompt()
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _build_system_prompt(self) -> str | None:
        """Combine user system prompt with skill index."""
        parts = []
        if self.config.system_prompt:
            parts.append(self.config.system_prompt)
        if self.skills:
            index = self.skills.get_skill_index_text()
            if index:
                parts.append(index)
        return "\n\n".join(parts) if parts else None

    def clear_history(self) -> None:
        """Clear conversation history and prompt cache, re-adding current system prompt if set."""
        self.messages.clear()
        system_prompt = self._build_system_prompt()
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.manager.invalidate_prompt_cache(self.config.model_name, "chat")

    async def _estimate_conversation_tokens(self, lm: LoadedModel) -> int:
        """Estimate total tokens for current conversation history."""
        try:
            tokenizer = lm.text_tokenizer
            msgs = list(self.messages)
            caps = lm.template_caps

            def _estimate() -> int:
                prompt_text = apply_chat_template_text(
                    tokenizer, msgs, tools=None, caps=caps
                )
                return len(tokenize_for_cache(tokenizer, prompt_text))

            return await asyncio.to_thread(_estimate)
        except Exception as exc:
            logger.debug("Token estimation failed: %s", exc)
            return 0

    async def _check_memory_and_truncate(
        self, lm: LoadedModel, max_tokens: int
    ) -> bool:
        """Check if conversation + generation would exceed memory, truncate if needed.

        Returns True if truncation occurred.
        """
        total_physical = memory_utils.get_system_memory_bytes()
        if total_physical <= 0:
            return False

        memory_limit = int(total_physical * settings.memory_limit_fraction)
        current_metal = memory_utils.get_metal_memory()

        # Estimate tokens for current conversation
        conv_tokens = await self._estimate_conversation_tokens(lm)
        if conv_tokens <= 0:
            return False

        # Estimate KV cache for full generation
        kv_bytes = estimate_kv_cache_bytes(
            lm.model,
            conv_tokens + max_tokens,
            kv_cache_quant=lm.kv_cache_quant,
        )

        if current_metal + kv_bytes <= memory_limit:
            return False

        # Need to truncate — keep system message + recent message pairs
        logger.warning(
            "Chat memory check: %d tokens (%.1f GB KV) would exceed limit. Truncating history.",
            conv_tokens,
            kv_bytes / 1024**3,
        )

        # Find system message index
        system_idx = 0
        if self.messages and self.messages[0].get("role") == "system":
            system_idx = 1

        # Capture original length so we only report truncation if
        # messages were actually removed (not just memory pressure).
        original_len = len(self.messages)

        # Keep system + N most recent (user, assistant) pairs
        self.messages = _truncate_keep_recent(
            self.messages, system_idx, _MIN_MESSAGES_AFTER_TRUNCATE
        )

        # Re-verify the truncated history still fits; if not, truncate again
        # but with a shallower target — keep only the last 2 messages.
        conv_tokens_after = await self._estimate_conversation_tokens(lm)
        if conv_tokens_after > 0:
            kv_bytes_after = estimate_kv_cache_bytes(
                lm.model,
                conv_tokens_after + max_tokens,
                kv_cache_quant=lm.kv_cache_quant,
            )
            if current_metal + kv_bytes_after > memory_limit:
                logger.warning(
                    "Truncated history still exceeds memory limit; reducing further"
                )
                self.messages = _truncate_keep_recent(
                    self.messages, system_idx, _MIN_MESSAGES_EMERGENCY_TRUNCATE
                )

        if len(self.messages) < original_len:
            self.manager.invalidate_prompt_cache(self.config.model_name, "chat")
            logger.info("Truncated chat history to %d messages", len(self.messages))
            return True

        return False

    async def _exec_tool(self, tu: dict) -> dict:
        """Execute a single tool call and return the event + message.

        ToolError returns from builtin tools and MCP are converted to
        tool_error events so the model always sees a uniform error format
        regardless of whether the failure came from a builtin, MCP, or
        exception path.
        """
        tool_name = tu["name"]
        tool_input = tu["input"]
        tool_id = tu["id"]
        try:
            with _tracing.span("mcp.tool_call", **{"tool.name": tool_name}) as _sp:
                if tool_name == "use_skill" and self.skills:
                    result = self.skills.handle_use_skill(tool_input)
                elif self.builtin and tool_name in self.builtin.tool_names:
                    result = await self.builtin.call_tool(tool_name, tool_input)
                elif self.mcp is not None:
                    _sp.set_attribute("mcp.server", getattr(self.mcp, "name", "mcp"))
                    # tool_timeout=None → let the MCP client's own default
                    # apply (single source of that default).
                    if self.config.tool_timeout is not None:
                        result = await self.mcp.call_tool(
                            tool_name, tool_input, timeout=self.config.tool_timeout
                        )
                    else:
                        result = await self.mcp.call_tool(tool_name, tool_input)
                else:
                    result = ToolError(
                        message=f"No handler for tool: {tool_name!r}",
                        tool_name=tool_name,
                        is_user_error=True,
                    )

            if isinstance(result, ToolError):
                return {
                    "call_event": {
                        "type": "tool_call",
                        "name": tool_name,
                        "arguments": tool_input,
                        "id": tool_id,
                    },
                    "result_event": {
                        "type": "tool_error",
                        "name": tool_name,
                        "error": result.message,
                        "id": tool_id,
                        "is_user_error": result.is_user_error,
                    },
                    "message": {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": result.message,
                    },
                }

            return {
                "call_event": {
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": tool_input,
                    "id": tool_id,
                },
                "result_event": {
                    "type": "tool_result",
                    "name": tool_name,
                    "result": result,
                    "id": tool_id,
                },
                "message": {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result,
                },
            }
        except Exception as exc:
            error_msg = f"Error calling {tool_name}: {exc}"
            return {
                "call_event": {
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": tool_input,
                    "id": tool_id,
                },
                "result_event": {
                    "type": "tool_error",
                    "name": tool_name,
                    "error": str(exc),
                    "id": tool_id,
                    "is_user_error": False,
                },
                "message": {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": error_msg,
                },
            }

    def _prepare_tools(self) -> list[dict] | None:
        """Merge MCP, builtin, and skill tool definitions."""
        tools = None
        if self.mcp is not None:
            tools = self.mcp.get_tools_for_chat() or None

        # Merge built-in tool definitions, skipping collisions with MCP tools
        if self.builtin:
            builtin_defs = self.builtin.get_tool_definitions()
            if tools:
                mcp_names = {t["function"]["name"] for t in tools}
                filtered = []
                for d in builtin_defs:
                    n = d["function"]["name"]
                    if n in mcp_names:
                        logger.warning(
                            "Built-in tool %r skipped: MCP tool with same name takes precedence",
                            n,
                        )
                    else:
                        filtered.append(d)
                builtin_defs = filtered
            tools = (tools or []) + builtin_defs

        # Merge skill tool into the tools list
        skill_tool = self.skills.get_tool_definition() if self.skills else None
        if skill_tool:
            tools = (tools or []) + [skill_tool]

        return tools

    def _classify_tool_calls(
        self, tool_uses: list[dict]
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        """Split tool calls into (allow, confirm, auto, deny) per safety policy.

        Local tools (skills, builtins) bypass the safety policy by default
        (local_tool_safety=False) because they run in-process and were
        already trusted by the user when configured. Set
        local_tool_safety=True to apply the safety policy to local tools
        as well.
        """
        local_tools = []
        remote_tools = []
        for tu in tool_uses:
            is_local = (tu["name"] == "use_skill" and self.skills) or (
                self.builtin and tu["name"] in self.builtin.tool_names
            )
            if is_local and self.config.local_tool_safety:
                remote_tools.append(tu)
            elif is_local:
                local_tools.append(tu)
            else:
                remote_tools.append(tu)

        if self.tool_safety:
            allow, confirm, auto, deny = self.tool_safety.classify_batch(remote_tools)
            allow = local_tools + allow
        else:
            allow, confirm, auto, deny = local_tools + remote_tools, [], [], []
        return allow, confirm, auto, deny

    async def _confirm_tools(
        self,
        tools: list[dict],
        *,
        request_event: Literal["tool_confirmation_needed", "tool_auto_judging"],
        context: list[dict] | None,
        denial_reason: str,
        denial_content: str,
        approved: list[dict],
        results_by_id: dict[str, dict],
    ) -> AsyncGenerator[ChatEvent, None]:
        """Run one confirmation phase (user confirm or LLM auto-judge).

        Yields a `request_event` per tool, then `tool_approved` or
        `tool_denied` based on `check_and_confirm`. Approved tools are
        appended to `approved`; denials record a tool-result message in
        `results_by_id` with `denial_content` (formatted with the tool
        name).
        """
        for tu in tools:
            # Both request-event TypedDicts share this shape; only the
            # Literal tag differs, which request_event's type pins down.
            yield cast(
                "_ToolConfirmationNeededEvent | _ToolAutoJudgingEvent",
                {
                    "type": request_event,
                    "name": tu["name"],
                    "arguments": tu["input"],
                    "id": tu["id"],
                },
            )
            if await self.tool_safety.check_and_confirm(
                tu["name"], tu["input"], context=context
            ):
                approved.append(tu)
                yield {
                    "type": "tool_approved",
                    "name": tu["name"],
                    "id": tu["id"],
                }
            else:
                yield {
                    "type": "tool_denied",
                    "name": tu["name"],
                    "arguments": tu["input"],
                    "id": tu["id"],
                    "reason": denial_reason,
                }
                results_by_id[tu["id"]] = {
                    "message": {
                        "role": "tool",
                        "tool_call_id": tu["id"],
                        "name": tu["name"],
                        "content": denial_content.format(name=tu["name"]),
                    },
                }

    async def _execute_tool_calls(
        self, tool_uses: list[dict]
    ) -> AsyncGenerator[ChatEvent, None]:
        """Classify, confirm, and execute tool calls. Yields event dicts.

        Appends tool result messages to self.messages in original call order.
        Tool errors are fed back to the model for recovery via tool_error
        events and error messages appended to the conversation history.
        """
        allow, confirm, auto, deny = self._classify_tool_calls(tool_uses)

        # Collect results by tool_call_id for call-order output.
        results_by_id: dict[str, dict] = {}

        # Handle denied tools
        for tu in deny:
            yield {
                "type": "tool_denied",
                "name": tu["name"],
                "arguments": tu["input"],
                "id": tu["id"],
                "reason": "policy",
            }
            results_by_id[tu["id"]] = {
                "message": {
                    "role": "tool",
                    "tool_call_id": tu["id"],
                    "name": tu["name"],
                    "content": f"Tool '{tu['name']}' is blocked by safety policy",
                },
            }

        # Prompt for confirmation on confirm tools
        approved: list[dict] = []
        async for event in self._confirm_tools(
            confirm,
            request_event="tool_confirmation_needed",
            context=None,
            denial_reason="user",
            denial_content="Tool '{name}' was not approved",
            approved=approved,
            results_by_id=results_by_id,
        ):
            yield event

        # Auto-judge tools via LLM
        async for event in self._confirm_tools(
            auto,
            request_event="tool_auto_judging",
            context=self.messages,
            denial_reason="auto",
            denial_content="Tool '{name}' was not approved by safety check",
            approved=approved,
            results_by_id=results_by_id,
        ):
            yield event

        # Execute allowed + approved tools
        to_execute = allow + approved
        deferred_exc = None
        pending_cancelled: asyncio.CancelledError | None = None
        if to_execute:
            if self.config.sequential_tool_execution:
                exec_results = []
                for tu in to_execute:
                    try:
                        result = await self._exec_tool(tu)
                        exec_results.append(result)
                    except asyncio.CancelledError as e:
                        pending_cancelled = e
                        exec_results.append(e)
                        # Pad remaining slots so zip maps correctly
                        for _ in range(len(exec_results), len(to_execute)):
                            exec_results.append(
                                RuntimeError(
                                    f"Execution skipped: task cancelled after "
                                    f"{tu['name']}"
                                )
                            )
                        break
                    except KeyboardInterrupt:
                        raise
                    except SystemExit:
                        raise
                    except BaseException as e:
                        exec_results.append(e)
            else:
                exec_results = await asyncio.gather(
                    *(self._exec_tool(tu) for tu in to_execute),
                    return_exceptions=True,
                )
                for r in exec_results:
                    if isinstance(r, asyncio.CancelledError):
                        pending_cancelled = r
                        break
            for tu, r in zip(to_execute, exec_results):
                if isinstance(r, asyncio.CancelledError):
                    if pending_cancelled is None:
                        pending_cancelled = r
                    name = tu["name"]
                    error_content = f"Error calling {name}: task cancelled"
                    yield {
                        "type": "tool_call",
                        "name": name,
                        "arguments": tu["input"],
                        "id": tu["id"],
                    }
                    yield {
                        "type": "tool_error",
                        "name": name,
                        "error": "task cancelled",
                        "id": tu["id"],
                        "is_user_error": False,
                    }
                    results_by_id[tu["id"]] = {
                        "message": {
                            "role": "tool",
                            "tool_call_id": tu["id"],
                            "name": name,
                            "content": error_content,
                        },
                    }
                    continue
                if isinstance(r, BaseException):
                    if deferred_exc is None:
                        deferred_exc = r
                    name = tu["name"]
                    error_content = f"Error calling {name}: {r}"
                    yield {
                        "type": "tool_call",
                        "name": name,
                        "arguments": tu["input"],
                        "id": tu["id"],
                    }
                    yield {
                        "type": "tool_error",
                        "name": name,
                        "error": str(r),
                        "id": tu["id"],
                        "is_user_error": False,
                    }
                    results_by_id[tu["id"]] = {
                        "message": {
                            "role": "tool",
                            "tool_call_id": tu["id"],
                            "name": name,
                            "content": error_content,
                        },
                    }
                else:
                    yield r["call_event"]
                    yield r["result_event"]
                    results_by_id[r["message"]["tool_call_id"]] = r

            if pending_cancelled is not None:
                # Append messages before raising to maintain history consistency
                for tu in tool_uses:
                    r = results_by_id.get(tu["id"])
                    if r:
                        self.messages.append(r["message"])
                raise pending_cancelled

        # Append messages in original call order
        for tu in tool_uses:
            r = results_by_id.get(tu["id"])
            if r:
                result_msg = r["message"]
                if tu["name"] == "question":
                    import json

                    content = result_msg.get("content", "")
                    if not isinstance(content, str) or not content.startswith(
                        "__question__:"
                    ):
                        continue
                    try:
                        payload = json.loads(content[len("__question__:") :])
                        yield {
                            "type": "question",
                            "header": payload.get("header", ""),
                            "question": payload.get("question", ""),
                            "options": payload.get("options"),
                            "multiple": payload.get("multiple", False),
                            "id": tu["id"],
                        }
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
                self.messages.append(result_msg)
            else:
                error_detail = "no result received"
                error_content = f"Error calling {tu['name']}: {error_detail}"
                yield {
                    "type": "tool_error",
                    "name": tu["name"],
                    "error": error_detail,
                    "id": tu["id"],
                    "is_user_error": False,
                }
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tu["id"],
                        "name": tu["name"],
                        "content": error_content,
                    }
                )

        if deferred_exc is not None:
            # Non-Exception BaseException (e.g. KeyboardInterrupt, SystemExit,
            # GeneratorExit) captured by asyncio.gather in the parallel path —
            # must be re-raised. Note: the tool error message was already
            # appended to self.messages above; if the session is reused the
            # history will contain a dangling tool error.
            if not isinstance(deferred_exc, Exception):
                raise deferred_exc
            # Regular exceptions have already been yielded as tool_error
            # events and appended to self.messages. Let the caller
            # (agent loop) handle recovery and track consecutive failures.
            return

    async def _stream_one_turn(
        self,
        *,
        options: dict[str, Any],
        mcp_tools: list[dict] | None,
        implicit_thinking: bool,
        template_has_thinking: bool,
        result: _TurnStreamResult,
    ) -> AsyncGenerator[ChatEvent, None]:
        """Consume one generate_chat turn, splitting thinking from visible text.

        Feeds each chunk through a ThinkingTracker, yields streaming events
        (token / thinking_* / repetition_detected) and stops early on
        repetitive output. Fills ``result`` in place: ``full_text`` is the
        raw accumulated output for the parse step (with any incomplete
        <think> block stripped on repetition), ``thinking_expected`` mirrors
        the engine's meta chunk, and ``repetition_stopped`` is True when
        generation was cut short.
        """
        repetition_stopped = False
        token_count = 0
        tracker = ThinkingTracker(
            implicit_mode=implicit_thinking,
            thinking_disabled=not self.config.thinking,
            template_has_thinking=template_has_thinking,
        )

        async for chunk in await generate_chat(
            self.manager,
            self.config.model_name,
            self.messages,
            options=options,
            tools=mcp_tools,
            stream=True,
            keep_alive="-1",
            max_tokens=self.config.max_tokens,
            cache_id="chat",
            enable_thinking=self.config.thinking,
        ):
            if chunk.get("cache_info"):
                continue
            if "thinking_expected" in chunk:
                # Captured so `parse_model_output` only fires the orphan-
                # `</think>` heuristic when thinking was actually
                # requested (issue #307).
                result["thinking_expected"] = bool(chunk["thinking_expected"])
                continue
            if chunk.get("done"):
                break
            text = chunk.get("text", "")
            if text:
                token_count += 1
                think_delta, visible_delta, thinking_ended, thinking_started = (
                    tracker.feed(text)
                )

                if thinking_started:
                    yield {"type": "thinking_start"}
                if think_delta:
                    yield {"type": "thinking_token", "text": think_delta}
                if thinking_ended:
                    yield {"type": "thinking_end"}
                if visible_delta:
                    yield {"type": "token", "text": visible_delta}

                if token_count % 10 == 0 and _detect_repetition(
                    tracker.accumulated,
                    min_phrase_len=self.config.repetition_min_phrase_len,
                    min_repeats=self.config.repetition_min_repeats,
                ):
                    logger.warning("Repetitive output detected, stopping generation")
                    repetition_stopped = True
                    yield {"type": "repetition_detected"}
                    break

        # Flush the splitter's held buffer at stream end (skip on repetition —
        # that output is being discarded by strip_on_repetition below).
        if not repetition_stopped:
            flush_think, flush_visible, flush_started = tracker.flush()
            if flush_started:
                yield {"type": "thinking_start"}
            if flush_think:
                yield {"type": "thinking_token", "text": flush_think}
            if flush_visible:
                yield {"type": "token", "text": flush_visible}

        # Strip incomplete thinking on repetition BEFORE yielding final events
        was_in_thinking = tracker.in_thinking
        if repetition_stopped:
            tracker.strip_on_repetition()

        # Close any open thinking block (unclosed <think>/channel block, or
        # content stripped mid-think on repetition)
        if was_in_thinking:
            yield {"type": "thinking_end"}

        result["full_text"] = tracker.accumulated
        result["repetition_stopped"] = repetition_stopped

    async def send_message(self, user_text: str) -> AsyncGenerator[ChatEvent, None]:
        """Send a user message and run the agent loop.

        Yields event dicts:
        - {"type": "token", "text": str} — streaming response token
        - {"type": "thinking_start"} — thinking block begins
        - {"type": "thinking_token", "text": str} — streaming thinking token
        - {"type": "thinking_end"} — thinking block ends
        - {"type": "tool_call", "name": str, "arguments": dict, "id": str}
        - {"type": "tool_result", "name": str, "result": str, "id": str}
        - {"type": "tool_error", "name": str, "error": str, "id": str, "is_user_error": bool}
        - {"type": "tool_confirmation_needed", "name": str, "arguments": dict, "id": str}
        - {"type": "tool_approved", "name": str, "id": str}
        - {"type": "tool_denied", "name": str, "arguments": dict, "id": str, "reason": str}
        - {"type": "tool_auto_judging", "name": str, "arguments": dict, "id": str}
        - {"type": "question", "header": str, "question": str, "options": list|None, "multiple": bool, "id": str}
        - {"type": "memory_truncated", "message": str} — history was truncated
        - {"type": "repetition_detected"} — repetitive output detected
        - {"type": "model_load_error", "error": str} — model load failed
        - {"type": "max_turns_exceeded"} — agent loop hit turn limit
        - {"type": "tool_failures_exceeded", "message": str, "consecutive_failures": int}
        - {"type": "done"} — end of response
        """
        self.messages.append({"role": "user", "content": user_text})

        mcp_tools = self._prepare_tools()

        options: dict[str, Any] = {
            "repeat_penalty": self.config.repeat_penalty,
            "repeat_last_n": self.config.repeat_last_n,
        }
        if self.config.temperature is not None:
            options["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            options["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            options["top_k"] = self.config.top_k

        # Load model once and extract template caps + do memory check
        try:
            lm = await self.manager.ensure_loaded(self.config.model_name)
            template_has_thinking = lm.template_caps.has_thinking_tags
        except Exception as exc:
            logger.error("Failed to load model %r: %s", self.config.model_name, exc)
            # Roll back the user message already appended at the top of send_message
            self.messages.pop()
            error_msg = f"Failed to load model: {exc}"
            yield {"type": "model_load_error", "error": error_msg}
            yield {"type": "done"}
            return

        # Implicit thinking: model injects <think> into the template prompt,
        # so generated text starts with thinking content (no <think> prefix).
        # Buffer content before </think> regardless of the thinking display
        # flag — when disabled, we still need to strip thinking from output.
        assume_implicit_thinking = template_has_thinking

        # Pre-check memory before starting agent loop
        try:
            truncated = await self._check_memory_and_truncate(
                lm, self.config.max_tokens
            )
            if truncated:
                yield {
                    "type": "memory_truncated",
                    "message": f"History truncated to {len(self.messages)} messages to fit memory",
                }
        except Exception:
            logger.debug("Memory check failed, proceeding anyway", exc_info=True)

        consecutive_failures = 0
        for turn in range(self.config.max_turns):
            turn_result: _TurnStreamResult = {
                "full_text": "",
                "thinking_expected": False,
                "repetition_stopped": False,
            }
            async for event in self._stream_one_turn(
                options=options,
                mcp_tools=mcp_tools,
                implicit_thinking=assume_implicit_thinking and turn == 0,
                template_has_thinking=template_has_thinking,
                result=turn_result,
            ):
                yield event
            repetition_stopped = turn_result["repetition_stopped"]

            visible_text, tool_uses, parse_error = _parse_turn_output(
                turn_result["full_text"],
                has_tools=(mcp_tools is not None),
                thinking_expected=turn_result["thinking_expected"],
            )
            if parse_error is not None:
                yield parse_error

            # Build assistant message
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": visible_text,
            }
            if tool_uses:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tu["id"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu["input"]),
                        },
                    }
                    for tu in tool_uses
                ]
            self.messages.append(assistant_msg)

            if not tool_uses or repetition_stopped:
                break

            turn_had_success = False
            turn_had_failure = False
            try:
                async for event in self._execute_tool_calls(tool_uses):
                    yield event
                    if event["type"] == "tool_error":
                        turn_had_failure = True
                    elif event["type"] == "tool_result":
                        turn_had_success = True
            except asyncio.CancelledError:
                raise
            # Defence-in-depth: _execute_tool_calls no longer raises Exception
            # (errors are fed back as tool_error events), but guard against
            # future regressions.
            except Exception:
                logger.warning(
                    "Unexpected exception during tool execution", exc_info=True
                )
                turn_had_failure = True

            # Any successful tool call in a turn resets the failure counter
            # so that the agent loop continues as long as the model gets
            # useful results. Mixed-result turns (parallel success + failure)
            # also reset — the model has new information to work with.
            if turn_had_success:
                consecutive_failures = 0
            elif turn_had_failure:
                consecutive_failures += 1

            if (
                self.config.max_consecutive_tool_failures > 0
                and consecutive_failures >= self.config.max_consecutive_tool_failures
            ):
                yield {
                    "type": "tool_failures_exceeded",
                    "message": (
                        f"Too many consecutive turns with tool failures "
                        f"({consecutive_failures}). Stopping agent loop."
                    ),
                    "consecutive_failures": consecutive_failures,
                }
                break
        else:
            # max_turns reached
            yield {"type": "max_turns_exceeded"}

        yield {"type": "done"}


def _parse_turn_output(
    full_text: str, *, has_tools: bool, thinking_expected: bool
) -> tuple[str, list[dict[str, Any]], _ToolErrorEvent | None]:
    """Parse one turn's raw model output into visible text and tool calls.

    Returns ``(visible_text, tool_uses, parse_error)``. On parse failure,
    falls back to thinking-stripped text with no tool calls and returns a
    ``tool_error`` event for the caller to yield. (Named to avoid colliding
    with the ``parse_model_output`` import from ``engine.tool_parser``.)
    """
    try:
        _, visible_text, tool_uses = parse_model_output(
            full_text,
            has_tools=has_tools,
            thinking_expected=thinking_expected,
        )
        return visible_text, tool_uses, None
    except Exception as exc:
        logger.error(
            "Failed to parse model output: %s\ntext: %.200s",
            exc,
            full_text,
        )
        error: _ToolErrorEvent = {
            "type": "tool_error",
            "name": "(parse error)",
            "error": f"Failed to parse model output; tools were dropped: {exc}",
            "id": "",
            "is_user_error": False,
        }
        return _strip_thinking(full_text), [], error


def _strip_thinking(text: str) -> str:
    """Remove complete and in-progress <think> blocks from text.

    Strips closed ``<think>...</think>`` blocks and truncates at any
    unclosed ``<think>`` tag so thinking content is never shown.
    Also handles implicit thinking where the template injects ``<think>``
    so only ``</think>`` appears in the output.
    """
    result = _THINK_BLOCK_RE.sub("", text)
    if _THINK_OPEN not in result:
        close_idx = result.find(_THINK_CLOSE)
        if close_idx != -1:
            return result[close_idx + len(_THINK_CLOSE) :]
    idx = result.find(_THINK_OPEN)
    if idx != -1:
        result = result[:idx]
    return result


def _extract_thinking_content(text: str) -> str:
    """Extract content from <think> blocks, including unclosed trailing blocks.

    Returns the concatenated thinking content from all closed blocks
    plus any content after an unclosed trailing ``<think>`` tag.
    Also handles implicit thinking where ``<think>`` is template-injected
    and only ``</think>`` appears in the output.
    """
    parts = []
    for m in _THINK_CONTENT_RE.finditer(text):
        parts.append(m.group(1))
    cleaned = _THINK_BLOCK_RE.sub("", text)
    idx = cleaned.find(_THINK_OPEN)
    if idx != -1:
        parts.append(cleaned[idx + len(_THINK_OPEN) :])
    elif not parts:
        close_idx = cleaned.find(_THINK_CLOSE)
        if close_idx != -1:
            parts.append(cleaned[:close_idx])
    return "".join(parts)


def _detect_repetition(
    text: str, min_phrase_len: int = 20, min_repeats: int = 4
) -> bool:
    """Detect if the accumulated text contains a repeating phrase.

    Searches the tail (last 1000 chars) of the text for any short
    substring that repeats consecutively at least min_repeats times.
    The phrase length is capped at 100 chars for performance; this
    is a deliberate trade-off — very long repetitive blocks (e.g.
    repeated function bodies >100 chars) won't be detected at the
    default config values.  Lowering ``repetition_min_phrase_len``
    or ``repetition_min_repeats`` increases detection surface at
    the cost of more checks per step.
    """
    if len(text) < min_phrase_len * min_repeats:
        return False

    if min_repeats < 1 or min_phrase_len < 1:
        return False

    tail = text[-1000:] if len(text) > 1000 else text
    max_phrase_len = max(min_phrase_len, min(100, len(tail) // min_repeats))

    for phrase_len in range(min_phrase_len, max_phrase_len + 1):
        candidate = tail[-phrase_len:]
        if not candidate.strip():
            continue
        count = 0
        pos = len(tail)
        while pos >= phrase_len:
            segment = tail[pos - phrase_len : pos]
            if segment == candidate:
                count += 1
                pos -= phrase_len
            else:
                break
        if count >= min_repeats:
            return True

    return False
