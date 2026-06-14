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
from typing import TYPE_CHECKING

from olmlx.engine.grammar import GrammarSpec
from olmlx.engine.inference import generate_chat
from olmlx.engine.tool_parser import parse_model_output

if TYPE_CHECKING:
    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import PanelConfig

logger = logging.getLogger(__name__)


def first_user_text(messages: list[dict]) -> str:
    """Return the first user message's text (stable routing key).

    The conversation's task is fixed by the first user turn, so routing
    on it is deterministically re-derived every stateless HTTP call with
    no stored server state.
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
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
    "You are the judge for a panel of models that all answered the "
    "conversation above. Several candidate answers are listed below. "
    "Synthesize ONE best final answer for the user. Ground every claim in "
    "the tool results already present in this conversation; do not call "
    "tools or invent new facts. Reconcile disagreements and prefer "
    "grounded, specific answers. Output only the final answer."
)


def build_judge_messages(
    original_messages: list[dict],
    member_names: list[str],
    answers: list[str],
) -> list[dict]:
    """Original conversation + a final user turn carrying the candidates.

    ``original_messages`` is not mutated (a new list is returned). The
    judge sees the full conversation — including any tool results the
    client executed — so it can verify groundedness.
    """
    candidates = []
    for name, answer in zip(member_names, answers):
        candidates.append(f"--- Candidate from {name} ---\n{answer.strip()}")
    content = _JUDGE_INSTRUCTION + "\n\n" + "\n\n".join(candidates)
    return [*original_messages, {"role": "user", "content": content}]


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

    merged = merge_tool_calls(per_panelist_tools)
    return (members, answers), merged
