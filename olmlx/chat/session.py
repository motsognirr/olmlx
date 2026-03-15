"""Chat session with agent loop for tool use."""

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from olmlx.chat.config import ChatConfig
from olmlx.chat.mcp_client import MCPClientManager
from olmlx.chat.skills import SkillManager
from olmlx.engine.inference import generate_chat
from olmlx.engine.model_manager import ModelManager
from olmlx.engine.tool_parser import parse_model_output

logger = logging.getLogger(__name__)


class ChatSession:
    """Manages conversation history and the agent loop."""

    def __init__(
        self,
        config: ChatConfig,
        manager: ModelManager,
        mcp: MCPClientManager | None = None,
        skills: SkillManager | None = None,
    ):
        self.config = config
        self.manager = manager
        self.mcp = mcp
        self.skills = skills
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

    async def send_message(self, user_text: str) -> AsyncGenerator[dict, None]:
        """Send a user message and run the agent loop.

        Yields event dicts:
        - {"type": "token", "text": str} — streaming response token
        - {"type": "thinking_start"} — thinking block begins
        - {"type": "thinking_token", "text": str} — streaming thinking token
        - {"type": "thinking_end"} — thinking block ends
        - {"type": "tool_call", "name": str, "arguments": dict, "id": str}
        - {"type": "tool_result", "name": str, "result": str, "id": str}
        - {"type": "tool_error", "name": str, "error": str, "id": str}
        - {"type": "max_turns_exceeded"}
        - {"type": "done"}
        """
        self.messages.append({"role": "user", "content": user_text})

        mcp_tools = None
        if self.mcp is not None:
            mcp_tools = self.mcp.get_tools_for_chat() or None

        # Merge skill tool into the tools list
        skill_tool = self.skills.get_tool_definition() if self.skills else None
        if skill_tool:
            mcp_tools = (mcp_tools or []) + [skill_tool]

        options = {
            "repeat_penalty": self.config.repeat_penalty,
            "repeat_last_n": self.config.repeat_last_n,
        }

        # Check template caps for implicit thinking detection.
        # generate_chat() calls ensure_loaded too; the model stays cached.
        # Only has_thinking_tags implies implicit thinking (template injects
        # <think>); supports_enable_thinking alone means explicit tags.
        try:
            lm = await self.manager.ensure_loaded(self.config.model_name)
            template_has_thinking = lm.template_caps.has_thinking_tags
        except Exception:
            logger.debug(
                "Could not detect template caps, assuming no implicit thinking",
                exc_info=True,
            )
            template_has_thinking = False

        # Implicit thinking: model injects <think> into the template prompt,
        # so generated text starts with thinking content (no <think> prefix).
        assume_implicit_thinking = self.config.thinking and template_has_thinking

        for turn in range(self.config.max_turns):
            accumulated = ""
            think_emitted = 0
            visible_emitted = 0
            in_thinking = False
            token_count = 0
            repetition_stopped = False

            # Incremental tag tracking — positions only move forward.
            # Only assume implicit thinking on the first turn; subsequent
            # tool-call follow-up turns may not produce thinking at all.
            implicit_mode = assume_implicit_thinking and turn == 0
            open_pos = -1  # position of <think>, -1 = not found
            close_pos = -1  # position of </think>, -1 = not found
            scan_pos = 0  # how far we've scanned for tags

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
                if chunk.get("done"):
                    break
                text = chunk.get("text", "")
                if text:
                    accumulated += text
                    token_count += 1

                    # Scan only new text for tag boundaries (with overlap
                    # for partial tags that span chunk boundaries).
                    new_scan = max(0, scan_pos - len(_THINK_CLOSE) + 1)
                    scan_pos = len(accumulated)

                    if open_pos == -1:
                        pos = accumulated.find(_THINK_OPEN, new_scan)
                        if pos != -1:
                            open_pos = pos
                            implicit_mode = False  # explicit tag found

                    # Always scan for </think> — needed for both thinking
                    # display and thinking-disabled stripping.
                    if close_pos == -1:
                        search_from = max(
                            (open_pos + len(_THINK_OPEN)) if open_pos >= 0 else 0,
                            new_scan,
                        )
                        pos = accumulated.find(_THINK_CLOSE, search_from)
                        if pos != -1:
                            close_pos = pos

                    # Derive thinking content and visible text from positions
                    has_thinking = open_pos >= 0 or implicit_mode
                    # Also detect implicit thinking when disabled (model still
                    # produced thinking despite the flag — strip it from output).
                    implicit_strip = (
                        not self.config.thinking
                        and template_has_thinking
                        and open_pos == -1
                        and close_pos >= 0
                    )
                    if has_thinking:
                        # Content starts after <think> tag, or at 0 for implicit
                        cs = (open_pos + len(_THINK_OPEN)) if open_pos >= 0 else 0
                        if close_pos >= 0:
                            think_content = accumulated[cs:close_pos]
                            visible = accumulated[close_pos + len(_THINK_CLOSE) :]
                        else:
                            think_content = accumulated[cs:]
                            visible = ""
                    elif implicit_strip:
                        # Model produced thinking despite being disabled;
                        # strip everything before </think> from visible.
                        think_content = ""
                        visible = accumulated[close_pos + len(_THINK_CLOSE) :]
                    else:
                        think_content = ""
                        visible = accumulated

                    # When thinking is disabled, suppress thinking events
                    # but still strip thinking content from visible output.
                    if not self.config.thinking:
                        think_content = ""

                    # Emit thinking delta
                    if len(think_content) > think_emitted:
                        if not in_thinking:
                            yield {"type": "thinking_start"}
                            in_thinking = True
                        yield {
                            "type": "thinking_token",
                            "text": think_content[think_emitted:],
                        }
                        think_emitted = len(think_content)

                    # Emit visible delta
                    if len(visible) > visible_emitted:
                        if in_thinking:
                            yield {"type": "thinking_end"}
                            in_thinking = False
                        yield {"type": "token", "text": visible[visible_emitted:]}
                        visible_emitted = len(visible)

                    # Throttle: only check every 10 tokens to reduce overhead
                    if token_count % 10 == 0 and _detect_repetition(accumulated):
                        logger.warning(
                            "Repetitive output detected, stopping generation"
                        )
                        repetition_stopped = True
                        break

            # Close any open thinking block (unclosed <think> or implicit)
            if in_thinking:
                yield {"type": "thinking_end"}

            full_text = accumulated

            thinking, visible_text, tool_uses = parse_model_output(
                full_text, has_tools=(mcp_tools is not None)
            )

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

            # Execute tool calls concurrently
            async def _exec_tool(tu: dict) -> dict:
                """Execute a single tool call and return the event + message."""
                tool_name = tu["name"]
                tool_input = tu["input"]
                tool_id = tu["id"]
                try:
                    if tool_name == "use_skill" and self.skills:
                        result = self.skills.handle_use_skill(tool_input)
                    elif self.mcp is not None:
                        result = await self.mcp.call_tool(tool_name, tool_input)
                    else:
                        raise ValueError(f"No handler for tool: {tool_name!r}")
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
                        },
                        "message": {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": error_msg,
                        },
                    }

            results = await asyncio.gather(*(_exec_tool(tu) for tu in tool_uses))
            for r in results:
                yield r["call_event"]
                yield r["result_event"]
                self.messages.append(r["message"])
        else:
            # max_turns reached
            yield {"type": "max_turns_exceeded"}

        yield {"type": "done"}


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_CONTENT_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove complete and in-progress <think> blocks from text.

    Strips closed ``<think>...</think>`` blocks and truncates at any
    unclosed ``<think>`` tag so thinking content is never shown.
    Also handles implicit thinking where the template injects ``<think>``
    so only ``</think>`` appears in the output.
    """
    # Remove complete blocks
    result = _THINK_BLOCK_RE.sub("", text)
    # Handle implicit thinking: </think> without matching <think>
    if _THINK_OPEN not in result:
        close_idx = result.find(_THINK_CLOSE)
        if close_idx != -1:
            return result[close_idx + len(_THINK_CLOSE) :]
    # Truncate at any unclosed <think>
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
    # Closed blocks
    for m in _THINK_CONTENT_RE.finditer(text):
        parts.append(m.group(1))
    # Check for unclosed trailing <think> after removing closed blocks
    cleaned = _THINK_BLOCK_RE.sub("", text)
    idx = cleaned.find(_THINK_OPEN)
    if idx != -1:
        parts.append(cleaned[idx + len(_THINK_OPEN) :])
    # Implicit thinking: no <think> tag but </think> present
    elif not parts:
        close_idx = cleaned.find(_THINK_CLOSE)
        if close_idx != -1:
            parts.append(cleaned[:close_idx])
    return "".join(parts)


def _detect_repetition(
    text: str, min_phrase_len: int = 20, min_repeats: int = 4
) -> bool:
    """Detect if the accumulated text contains a repeating phrase.

    Checks if any substring of length >= min_phrase_len repeats
    min_repeats or more times consecutively in the recent text.
    """
    if len(text) < min_phrase_len * min_repeats:
        return False

    # Only check the tail to keep it fast
    tail = text[-1000:] if len(text) > 1000 else text

    # Try different phrase lengths from short to long
    for phrase_len in range(min_phrase_len, len(tail) // min_repeats + 1):
        # Take the last phrase_len chars as candidate
        candidate = tail[-phrase_len:]
        if not candidate.strip():
            continue
        # Count consecutive occurrences from the end
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
