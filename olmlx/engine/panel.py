"""Multi-model panel + judge coordinator (sequential, single-box).

A ``type: "panel"`` model answers a request with a per-task-routed panel
of models plus a judge that synthesizes one reconciled answer, while
behaving as a drop-in tool-calling model: the *client* still executes
tools. The coordinator is a per-turn reconciler riding the client's
existing tool loop — see docs/superpowers/specs/2026-06-14-multi-model-panel-judge-design.md.

INVARIANT: every model call goes through ``generate_chat`` so the
Metal-stream / inference-lock handling is reused. Never touch MLX here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, cast

from olmlx.engine.grammar import GrammarSpec
from olmlx.engine.inference import generate_chat
from olmlx.engine.tool_parser import parse_model_output
from olmlx.utils.timing import TimingStats

if TYPE_CHECKING:
    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import PanelConfig

logger = logging.getLogger(__name__)


def _message_text(content: object) -> str:
    """Extract plain text from a message ``content`` (str or content-parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


def first_user_text(messages: list[dict]) -> str:
    """Return the first user message's text (stable routing key).

    The conversation's task is fixed by the first user turn, so routing
    on it is deterministically re-derived every stateless HTTP call with
    no stored server state.
    """
    for msg in messages:
        if msg.get("role") == "user":
            return _message_text(msg.get("content"))
    return ""


def route_grammar(panel: "PanelConfig") -> GrammarSpec:
    """A JSON-schema grammar constraining the classifier to one route key."""
    return GrammarSpec(
        kind="json_schema",
        schema={
            "type": "object",
            "properties": {"route": {"type": "string", "enum": sorted(panel.routes)}},
            "required": ["route"],
            "additionalProperties": False,
        },
    )


def select_members(route_key: str, panel: "PanelConfig") -> list[str]:
    """Members for *route_key*, falling back to the 'default' route."""
    return panel.routes.get(route_key, panel.routes["default"])


def _tool_key(tool_use: dict) -> str:
    """Stable dedup key: name + canonicalized arguments."""
    args = tool_use.get("input") or {}
    return tool_use["name"] + "\0" + json.dumps(args, sort_keys=True)


def merge_tool_calls(per_panelist: list[list[dict]]) -> list[dict]:
    """Deduped union of every panelist's proposed tool calls.

    Identical ``(name, arguments)`` collapse to one execution; different
    arguments both run. Insertion order is preserved (first panelist
    first). The internal ``_span`` key from ``parse_model_output`` is
    stripped. Tool-call IDs are assigned downstream by the routers.
    """
    merged: list[dict] = []
    seen: set[str] = set()
    for panelist_calls in per_panelist:
        for call in panelist_calls:
            key = _tool_key(call)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"name": call["name"], "input": call.get("input") or {}})
    return merged


_JUDGE_INSTRUCTION = (
    "You are the judge for a panel of models that answered a user request. "
    "Below are the user's request, any tool results gathered while answering, "
    "and each panelist's candidate answer. Synthesize ONE best final answer "
    "for the user, grounded in the tool results. Do NOT call tools, do NOT "
    "continue any tool sequence, and do NOT invent facts. Reconcile "
    "disagreements and prefer grounded, specific answers. Output ONLY the "
    "final answer prose for the user."
)


def _flatten_tool_results(messages: list[dict]) -> list[str]:
    """Pair each tool-result message with the call that produced it.

    Returns ``["name(args) -> result", ...]`` in conversation order, so the
    judge sees the gathered evidence without the agentic turn structure.
    """
    labels: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for call in msg.get("tool_calls") or []:
                cid = call.get("id")
                fn = call.get("function") or {}
                if cid:
                    labels[cid] = f"{fn.get('name', '?')}({fn.get('arguments', '')})"
    results = []
    for msg in messages:
        if msg.get("role") == "tool":
            label = labels.get(msg.get("tool_call_id") or "", "tool")
            results.append(f"{label} -> {_message_text(msg.get('content'))}")
    return results


def build_judge_messages(
    original_messages: list[dict],
    member_names: list[str],
    answers: list[str],
) -> list[dict]:
    """Flatten the panel run into a clean summarize-and-answer prompt.

    Returns a short ``[system, user]`` message list carrying the user's
    request, the tool results gathered during the loop, and the panelists'
    candidate answers — NOT a replay of the agentic assistant/tool turns.
    Replaying the raw conversation primes agentic models (notably gpt-oss) to
    keep calling tools and emit reasoning instead of a final answer. The judge
    still sees every tool result, so groundedness is preserved.
    ``original_messages`` is not mutated.
    """
    request = "\n\n".join(
        t
        for t in (
            _message_text(m.get("content"))
            for m in original_messages
            if m.get("role") == "user"
        )
        if t
    )
    tool_results = _flatten_tool_results(original_messages)
    candidates = "\n\n".join(
        f"--- Candidate from {name} ---\n{answer.strip()}"
        for name, answer in zip(member_names, answers)
    )
    sections = [f"## User request\n{request}"]
    if tool_results:
        sections.append("## Tool results gathered\n" + "\n\n".join(tool_results))
    sections.append("## Panel candidate answers\n" + candidates)
    return [
        {"role": "system", "content": _JUDGE_INSTRUCTION},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def serialize_tool_calls_qwen(tool_uses: list[dict]) -> str:
    """Render tool calls as canonical Qwen ``<tool_call>`` blocks.

    The routers re-parse this text via ``parse_model_output`` (the Qwen
    parser maps ``arguments`` -> ``input``), so the panel's tool turn is
    transparent to both the OpenAI and Ollama routers.
    """
    blocks = []
    for tu in tool_uses:
        payload = json.dumps({"name": tu["name"], "arguments": tu.get("input") or {}})
        # A literal "</tool_call>" inside an argument value would make the
        # non-greedy Qwen parser regex close the block early and drop the
        # call. Escaping the slash keeps the substring out of the text while
        # remaining valid JSON: json.loads decodes "<\/tool_call>" back to
        # "</tool_call>", so the round-trip is exact.
        payload = payload.replace("</tool_call>", "<\\/tool_call>")
        blocks.append(f"<tool_call>\n{payload}\n</tool_call>")
    return "\n".join(blocks)


_CLASSIFIER_SYSTEM = (
    "You are a request router. Classify the user's request into exactly "
    'one category. Respond ONLY with JSON of the form {"route": "<category>"}.'
)


async def classify(
    manager: "ModelManager",
    panel: "PanelConfig",
    user_text: str,
    keep_alive: int | str | None = None,
) -> list[str]:
    """Route the request to a member list via the classifier model.

    The classifier output is grammar-constrained to the route keys; any
    parse failure falls back to the 'default' route.
    """
    categories = ", ".join(sorted(panel.routes))
    messages = [
        {
            "role": "system",
            "content": f"{_CLASSIFIER_SYSTEM} Categories: {categories}.",
        },
        {"role": "user", "content": user_text},
    ]
    result = await generate_chat(
        manager,
        panel.classifier,
        messages,
        tools=None,
        stream=False,
        keep_alive=keep_alive,
        max_tokens=32,
        # Routing is a constrained JSON task — never spend tokens on reasoning,
        # regardless of the request's or classifier model's thinking default.
        enable_thinking=False,
        grammar_spec=route_grammar(panel),
    )
    text = (result.get("text") or "").strip()
    try:
        route = json.loads(text).get("route", "default")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Panel classifier returned non-JSON %r; using default", text)
        route = "default"
    return select_members(route, panel)


async def _run_panel(
    manager: "ModelManager",
    panel: "PanelConfig",
    messages: list[dict],
    tools: list[dict] | None,
    options: dict | None,
    keep_alive: int | str | None,
    max_tokens: int,
    enable_thinking: bool | None,
) -> tuple[tuple[list[str], list[str]], list[dict]]:
    """Route, run each panelist once, and reconcile this turn.

    Returns ``((member_names, answers), merged_tool_uses)``. When any
    panelist proposes tool calls, ``merged_tool_uses`` is their deduped
    union and ``answers`` is unused by the caller (the turn is a tool
    turn). When none do, ``merged_tool_uses`` is empty and the caller
    runs the judge over ``answers``.

    Any panelist/classifier failure propagates (fail the request).
    """
    user_text = first_user_text(messages)
    members = await classify(manager, panel, user_text, keep_alive)

    has_tools = bool(tools)
    answers: list[str] = []
    per_panelist_tools: list[list[dict]] = []
    for member in members:
        result = await generate_chat(
            manager,
            member,
            messages,
            options=options,
            tools=tools,
            stream=False,
            keep_alive=keep_alive,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        parse_text = result.get("raw_text") or result.get("text") or ""
        _thinking, visible, tool_uses = parse_model_output(
            parse_text,
            has_tools,
            thinking_expected=bool(result.get("thinking_expected")),
        )
        answers.append(visible)
        per_panelist_tools.append(tool_uses)

    merged = await _resolve_tool_turn(
        manager,
        panel,
        messages,
        members,
        answers,
        per_panelist_tools,
        options,
        keep_alive,
    )
    return (members, answers), merged


def _decision_grammar() -> GrammarSpec:
    """Constrain the judge's stop decision to ``{"action": "answer"|"gather"}``."""
    return GrammarSpec(
        kind="json_schema",
        schema={
            "type": "object",
            "properties": {"action": {"type": "string", "enum": ["answer", "gather"]}},
            "required": ["action"],
            "additionalProperties": False,
        },
    )


_JUDGE_DECISION_SYSTEM = (
    "You are the controller for a panel of models answering the conversation "
    "above. Some panelists proposed tool calls to gather more information. "
    "Decide whether there is already enough grounded information to write a "
    "final answer, or whether the proposed tools should be run first. Respond "
    'ONLY with JSON {"action": "answer"} or {"action": "gather"}.'
)


async def _judge_wants_gather(
    manager: "ModelManager",
    panel: "PanelConfig",
    messages: list[dict],
    members: list[str],
    answers: list[str],
    per_panelist_tools: list[list[dict]],
    options: dict | None,
    keep_alive: int | str | None,
) -> bool:
    """Ask the judge whether to gather more via tools (True) or finalize.

    Any parse failure defaults to finalize (``False``) — the ``"judge"`` stop
    condition exists to curb runaway tool loops, so an unreadable decision
    should stop rather than keep gathering.
    """
    states = []
    for name, answer, tools in zip(members, answers, per_panelist_tools):
        if tools:
            names = ", ".join(t["name"] for t in tools)
            states.append(f"- {name}: proposes tool calls [{names}]")
        else:
            states.append(f"- {name}: ready to answer: {answer.strip()[:200]}")
    content = _JUDGE_DECISION_SYSTEM + "\n\nPanelist states:\n" + "\n".join(states)
    result = await generate_chat(
        manager,
        panel.judge,
        [*messages, {"role": "user", "content": content}],
        tools=None,
        stream=False,
        options=options,
        keep_alive=keep_alive,
        max_tokens=16,
        enable_thinking=False,
        grammar_spec=_decision_grammar(),
    )
    text = (result.get("text") or "").strip()
    try:
        action = json.loads(text).get("action", "answer")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Panel judge decision returned non-JSON %r; finalizing", text)
        action = "answer"
    return action == "gather"


async def _resolve_tool_turn(
    manager: "ModelManager",
    panel: "PanelConfig",
    messages: list[dict],
    members: list[str],
    answers: list[str],
    per_panelist_tools: list[list[dict]],
    options: dict | None,
    keep_alive: int | str | None,
) -> list[dict]:
    """Apply the panel's ``stop_condition`` to decide this turn's output.

    Returns the deduped union of proposed tool calls to emit a tool turn, or
    ``[]`` to finalize (let the judge synthesize). When no panelist proposes
    tools, every condition finalizes.
    """
    wanting = sum(1 for tools in per_panelist_tools if tools)
    if wanting == 0:
        return []
    cond = panel.stop_condition
    if cond == "majority":
        ready = len(per_panelist_tools) - wanting
        if ready > len(per_panelist_tools) / 2:
            return []
        return merge_tool_calls(per_panelist_tools)
    if cond == "judge":
        if await _judge_wants_gather(
            manager,
            panel,
            messages,
            members,
            answers,
            per_panelist_tools,
            options,
            keep_alive,
        ):
            return merge_tool_calls(per_panelist_tools)
        return []
    # "all" (default): emit the union whenever any panelist wants tools.
    return merge_tool_calls(per_panelist_tools)


def _resolve_panel(manager: "ModelManager", model_name: str) -> "PanelConfig":
    """Resolve *model_name* to its PanelConfig or raise (fail the request)."""
    panel = manager.registry.resolve_panel(model_name)
    if panel is None:
        raise ValueError(f"{model_name!r} is not a configured panel model")
    return panel


async def _judge_answer(
    manager: "ModelManager",
    panel: "PanelConfig",
    messages: list[dict],
    member_names: list[str],
    answers: list[str],
    options: dict | None,
    keep_alive: int | str | None,
    max_tokens: int,
    enable_thinking: bool | None,
    stream: bool,
):
    """Run the judge (no tools) to synthesize the final answer.

    Returns the judge's ``generate_chat`` result verbatim — an async
    generator when ``stream`` else a dict — so the routers stream/format
    it exactly as a single model's output.
    """
    judge_messages = build_judge_messages(messages, member_names, answers)
    return await generate_chat(
        manager,
        panel.judge,
        judge_messages,
        options=options,
        tools=None,  # judge must not redo work
        stream=stream,
        keep_alive=keep_alive,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )


async def panel_generate_chat(
    manager: "ModelManager",
    model_name: str,
    messages: list[dict],
    options: dict | None = None,
    tools: list[dict] | None = None,
    stream: bool = True,
    keep_alive: int | str | None = None,
    max_tokens: int = 512,
    cache_id: str = "",
    enable_thinking: bool | None = None,
    grammar_spec: GrammarSpec | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Drop-in, ``generate_chat``-compatible entry point for a panel model.

    ``cache_id`` and ``grammar_spec`` are accepted for signature parity
    but not applied to the panel as a whole (the judge/panelists manage
    their own caching).
    """
    panel = _resolve_panel(manager, model_name)
    if stream:
        return _panel_stream(
            manager,
            panel,
            messages,
            options,
            tools,
            keep_alive,
            max_tokens,
            enable_thinking,
        )

    (member_names, answers), merged = await _run_panel(
        manager,
        panel,
        messages,
        tools,
        options,
        keep_alive,
        max_tokens,
        enable_thinking,
    )
    if merged:
        raw = serialize_tool_calls_qwen(merged)
        return {"text": "", "raw_text": raw, "done": True, "stats": TimingStats()}
    return await _judge_answer(
        manager,
        panel,
        messages,
        member_names,
        answers,
        options,
        keep_alive,
        max_tokens,
        enable_thinking,
        stream=False,
    )


async def _panel_stream(
    manager: "ModelManager",
    panel: "PanelConfig",
    messages: list[dict],
    options: dict | None,
    tools: list[dict] | None,
    keep_alive: int | str | None,
    max_tokens: int,
    enable_thinking: bool | None,
) -> AsyncGenerator[dict, None]:
    """Streaming coordinator.

    The panel compute runs *inside* the generator (during iteration) so
    the router's keepalive wrapper stays active. A tool turn yields the
    Qwen blocks as one text chunk; a final turn proxies the judge's token
    stream straight through.
    """
    (member_names, answers), merged = await _run_panel(
        manager,
        panel,
        messages,
        tools,
        options,
        keep_alive,
        max_tokens,
        enable_thinking,
    )
    if merged:
        raw = serialize_tool_calls_qwen(merged)
        yield {"text": raw}
        yield {
            "text": "",
            "done": True,
            "raw_text": raw,
            "done_reason": "stop",
            "stats": TimingStats(),
        }
        return

    judge_stream = await _judge_answer(
        manager,
        panel,
        messages,
        member_names,
        answers,
        options,
        keep_alive,
        max_tokens,
        enable_thinking,
        stream=True,
    )
    # ``_judge_answer`` calls ``generate_chat`` with a runtime ``stream`` bool,
    # so the overload can't narrow the return to the async-generator branch.
    # We pass ``stream=True`` here, so the result is always an async generator.
    async for chunk in cast("AsyncGenerator[dict, None]", judge_stream):
        yield chunk
