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

import logging
from typing import TYPE_CHECKING

from olmlx.engine.grammar import GrammarSpec

if TYPE_CHECKING:
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
