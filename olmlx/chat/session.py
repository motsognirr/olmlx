"""Chat session with agent loop for tool use."""

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from olmlx.chat.builtin_tools import BuiltinToolManager
from olmlx.chat.config import ChatConfig
from olmlx.chat.mcp_client import MCPClientManager
from olmlx.chat.skills import SkillManager
from olmlx.chat.tool_safety import ToolSafetyPolicy
from olmlx.engine.inference import generate_chat
from olmlx.engine.model_manager import ModelManager
from olmlx.engine.tool_parser import parse_model_output

logger = logging.getLogger(__name__)


class ThinkingTracker:
    """Tracks <think> tag boundaries during streaming generation.

    Handles both explicit ``<think>...</think>`` tags and implicit
    thinking (template-injected ``<think>`` at position 0). State
    mutations are deterministic — callers check ``has_content()``
    after each ``feed()`` to emit events.
    """

    def __init__(
        self,
        implicit_mode: bool = False,
        thinking_disabled: bool = False,
        template_has_thinking: bool = False,
    ):
        self._implicit_mode = implicit_mode
        self._thinking_disabled = thinking_disabled
        self._template_has_thinking = template_has_thinking
        self._accumulated = ""
        self._open_pos = -1
        self._close_pos = -1
        self._scan_pos = 0
        self._think_emitted = 0
        self._visible_emitted = 0
        self._in_thinking = False
        self._just_started = False
        self._thinking_start_emitted = False

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
        """Ingest a chunk of text.

        Returns ``(think_delta, visible_delta, thinking_ended, thinking_started)`` where
        each delta is a new substring to emit (or None) and
        ``thinking_ended`` is True when a visible delta transitions out
        of thinking mode, ``thinking_started`` on the first thinking chunk.
        """
        self._accumulated += text

        # Scan for tag boundaries
        new_scan = max(0, self._scan_pos - len(_THINK_CLOSE) + 1)
        self._scan_pos = len(self._accumulated)

        if self._open_pos == -1:
            pos = self._accumulated.find(_THINK_OPEN, new_scan)
            if pos != -1:
                self._open_pos = pos
                self._implicit_mode = False

        if self._close_pos == -1:
            search_from = max(
                (self._open_pos + len(_THINK_OPEN)) if self._open_pos >= 0 else 0,
                new_scan,
            )
            pos = self._accumulated.find(_THINK_CLOSE, search_from)
            if pos != -1:
                self._close_pos = pos

        think_content, visible = self._classify()
        self._just_started = bool(not self._in_thinking and think_content)
        think_delta = self._emit_think(think_content)
        visible_delta, thinking_ended = self._emit_visible(visible)
        return think_delta, visible_delta, thinking_ended, self._just_started

    def flush_disabled(self) -> str | None:
        """Flush buffered content as visible when thinking is disabled."""
        if self._implicit_mode and self._close_pos == -1 and self._thinking_disabled:
            if len(self._accumulated) > self._visible_emitted:
                text = self._accumulated[self._visible_emitted :]
                self._visible_emitted = len(self._accumulated)
                return text
        return None

    def strip_on_repetition(self) -> int | None:
        """Truncate accumulated text at the open tag position.

        Cuts before the opening <think> tag to fully remove the incomplete block.
        """
        if self._in_thinking and self._open_pos >= 0:
            self._accumulated = self._accumulated[: self._open_pos]
            self._visible_emitted = len(self._accumulated)
            self._in_thinking = False
            return self._visible_emitted
        return None

    def _classify(self) -> tuple[str, str]:
        has_thinking = self._open_pos >= 0 or self._implicit_mode
        implicit_strip = (
            self._thinking_disabled
            and self._template_has_thinking
            and self._open_pos == -1
            and self._close_pos >= 0
        )
        if has_thinking:
            cs = (self._open_pos + len(_THINK_OPEN)) if self._open_pos >= 0 else 0
            if self._close_pos >= 0:
                think_content = self._accumulated[cs : self._close_pos]
                visible = self._accumulated[self._close_pos + len(_THINK_CLOSE) :]
            else:
                think_content = self._accumulated[cs:]
                visible = ""
            if self._thinking_disabled:
                think_content = ""
        elif implicit_strip:
            think_content = ""
            visible = self._accumulated[self._close_pos + len(_THINK_CLOSE) :]
        else:
            think_content = ""
            visible = self._accumulated
        return think_content, visible

    def _emit_think(self, think_content: str) -> str | None:
        if len(think_content) > self._think_emitted:
            self._in_thinking = True
            delta = think_content[self._think_emitted :]
            self._think_emitted = len(think_content)
            return delta
        return None

    def _emit_visible(self, visible: str) -> tuple[str | None, bool]:
        if len(visible) > self._visible_emitted:
            thinking_ended = self._in_thinking
            if thinking_ended:
                self._in_thinking = False
            delta = visible[self._visible_emitted :]
            self._visible_emitted = len(visible)
            return delta, thinking_ended
        return None, False


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

    async def _exec_tool(self, tu: dict) -> dict:
        """Execute a single tool call and return the event + message."""
        tool_name = tu["name"]
        tool_input = tu["input"]
        tool_id = tu["id"]
        try:
            if tool_name == "use_skill" and self.skills:
                result = self.skills.handle_use_skill(tool_input)
            elif self.builtin and tool_name in self.builtin.tool_names:
                result = await self.builtin.call_tool(tool_name, tool_input)
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

    async def _execute_tool_calls(
        self, tool_uses: list[dict]
    ) -> AsyncGenerator[dict, None]:
        """Classify, confirm, and execute tool calls. Yields event dicts.

        Appends tool result messages to self.messages in original call order.
        Raises the first tool execution exception after all events are yielded.
        """
        # Classify tools by safety policy.
        # Local tools (skills, builtins) bypass the safety policy
        # because they run in-process and were already trusted by
        # the user when configured. Note: tool names come from model
        # output (not MCP directly), so a prompt injection could
        # cause the model to emit a local tool name — but local
        # tools are no more dangerous than the model calling them
        # without the safety layer.
        local_tools = []
        remote_tools = []
        for tu in tool_uses:
            if tu["name"] == "use_skill" and self.skills:
                local_tools.append(tu)
            elif self.builtin and tu["name"] in self.builtin.tool_names:
                local_tools.append(tu)
            else:
                remote_tools.append(tu)

        if self.tool_safety:
            allow, confirm, auto, deny = self.tool_safety.classify_batch(remote_tools)
            allow = local_tools + allow
        else:
            allow, confirm, auto, deny = local_tools + remote_tools, [], [], []

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
        approved = []
        for tu in confirm:
            yield {
                "type": "tool_confirmation_needed",
                "name": tu["name"],
                "arguments": tu["input"],
                "id": tu["id"],
            }
            if await self.tool_safety.check_and_confirm(tu["name"], tu["input"]):
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
                    "reason": "user",
                }
                results_by_id[tu["id"]] = {
                    "message": {
                        "role": "tool",
                        "tool_call_id": tu["id"],
                        "name": tu["name"],
                        "content": f"Tool '{tu['name']}' was not approved",
                    },
                }

        # Auto-judge tools via LLM
        for tu in auto:
            yield {
                "type": "tool_auto_judging",
                "name": tu["name"],
                "arguments": tu["input"],
                "id": tu["id"],
            }
            if await self.tool_safety.check_and_confirm(
                tu["name"], tu["input"], context=self.messages
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
                    "reason": "auto",
                }
                results_by_id[tu["id"]] = {
                    "message": {
                        "role": "tool",
                        "tool_call_id": tu["id"],
                        "name": tu["name"],
                        "content": f"Tool '{tu['name']}' was not approved by safety check",
                    },
                }

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
                        payload = json.loads(content[len("__question__") :])
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
            raise deferred_exc

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

        mcp_tools = self._prepare_tools()

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
        # Buffer content before </think> regardless of the thinking display
        # flag — when disabled, we still need to strip thinking from output.
        assume_implicit_thinking = template_has_thinking

        for turn in range(self.config.max_turns):
            repetition_stopped = False
            token_count = 0
            tracker = ThinkingTracker(
                implicit_mode=assume_implicit_thinking and turn == 0,
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
                        tracker.accumulated
                    ):
                        logger.warning(
                            "Repetitive output detected, stopping generation"
                        )
                        repetition_stopped = True
                        break

            # Flush disabled-thinking content
            if flush_text := tracker.flush_disabled():
                yield {"type": "token", "text": flush_text}

            # Strip incomplete thinking on repetition BEFORE yielding final events
            was_in_thinking = tracker.in_thinking
            if repetition_stopped:
                tracker.strip_on_repetition()

            # Close any open thinking block (only if content was stripped mid-think)
            if was_in_thinking:
                yield {"type": "thinking_end"}

            full_text = tracker.accumulated

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

            async for event in self._execute_tool_calls(tool_uses):
                yield event
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
