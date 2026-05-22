"""Shared utilities for NDJSON streaming routers."""

import json
from datetime import datetime, timezone
from typing import Any

# Likely-mistake strings that mean "thinking off".  Shared by both resolvers
# so the Ollama `think` and OpenAI `reasoning_effort` routes agree (a client
# sending "none" must never get thinking silently enabled on either).
_DISABLE_WORDS = ("none", "off", "disabled")


def resolve_think_flag(value: bool | str | None) -> bool | None:
    """Map an Ollama-style ``think`` value to the engine's ``enable_thinking``.

    ``None`` preserves the engine default; a bool passes through.  Strings are
    handled defensively: a stringified bool (``"true"``/``"false"``, any case),
    an empty string, or a disable word (``"none"/"off"/"disabled"``) maps to
    the corresponding bool so a weakly-typed client sending ``"false"`` or
    ``"none"`` is not silently inverted to *on*.  Any other non-empty string is
    treated as a gpt-oss thinking level (``"low"/"medium"/"high"``) and
    collapses to ``True`` because the engine toggle is bool-only.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ("", "false", *_DISABLE_WORDS):
        return False
    if normalized == "true":
        return True
    return True


def resolve_openai_think(
    reasoning_effort: str | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> bool | None:
    """Resolve ``enable_thinking`` from OpenAI-compatible request fields.

    Precedence: an explicit ``chat_template_kwargs["enable_thinking"]``
    (vLLM/SGLang convention, the only clean OFF switch) wins; otherwise the
    presence of ``reasoning_effort`` means on; otherwise ``None`` (default).

    Note: only the ``enable_thinking`` key of ``chat_template_kwargs`` is
    consumed — other keys are intentionally ignored (this server uses the dict
    solely as the thinking switch, not a general template-kwargs passthrough).
    """
    if chat_template_kwargs and "enable_thinking" in chat_template_kwargs:
        return bool(chat_template_kwargs["enable_thinking"])
    if reasoning_effort:
        # OpenAI defines only "low"/"medium"/"high" (presence -> on).  Map the
        # likely-mistake disable words to an explicit OFF (False) rather than
        # None: returning None would fall through to the engine default
        # ("think unless tools"), silently inverting a caller who sent
        # reasoning_effort="none" expecting thinking off.
        if reasoning_effort.strip().lower() in _DISABLE_WORDS:
            return False
        return True
    return None


def format_error(model: str) -> str:
    """Format a streaming error as an NDJSON line."""
    return (
        json.dumps(
            {
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "error": "An internal server error occurred during streaming.",
                "done": True,
                "done_reason": "error",
            }
        )
        + "\n"
    )


def build_inference_options(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> dict:
    """Build inference options dict from a superset of generation params.

    Accepts all known inference parameters and drops None/missing values.
    Normalizes ``stop`` to a list of strings (Anthropic expects list[str],
    OpenAI accepts str | list[str]).
    """
    opts: dict = {}
    if temperature is not None:
        opts["temperature"] = temperature
    if top_p is not None:
        opts["top_p"] = top_p
    if top_k is not None:
        opts["top_k"] = top_k
    if seed is not None:
        opts["seed"] = seed
    if stop:
        opts["stop"] = stop if isinstance(stop, list) else [stop]
    if frequency_penalty is not None:
        opts["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        opts["presence_penalty"] = presence_penalty

    return opts
