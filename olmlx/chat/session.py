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
        - {"type": "token", "text": str} — streaming token
        - {"type": "thinking", "text": str} — thinking block
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

        for turn in range(self.config.max_turns):
            accumulated = ""
            visible_len = 0
            token_count = 0
            repetition_stopped = False
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
                    # Suppress <think>...</think> from streaming output
                    visible = _strip_thinking(accumulated)
                    if len(visible) > visible_len:
                        delta = visible[visible_len:]
                        visible_len = len(visible)
                        yield {"type": "token", "text": delta}
                    # Throttle: only check every 10 tokens to reduce overhead
                    if token_count % 10 == 0 and _detect_repetition(accumulated):
                        logger.warning(
                            "Repetitive output detected, stopping generation"
                        )
                        repetition_stopped = True
                        break

            full_text = accumulated

            thinking, visible_text, tool_uses = parse_model_output(
                full_text, has_tools=(mcp_tools is not None)
            )

            if thinking:
                yield {"type": "thinking", "text": thinking}

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


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove complete and in-progress <think> blocks from text.

    Strips closed ``<think>...</think>`` blocks and truncates at any
    unclosed ``<think>`` tag so thinking content is never shown.
    """
    # Remove complete blocks
    result = _THINK_BLOCK_RE.sub("", text)
    # Truncate at any unclosed <think>
    idx = result.find("<think>")
    if idx != -1:
        result = result[:idx]
    return result


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
