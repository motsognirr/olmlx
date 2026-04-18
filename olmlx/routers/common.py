"""Shared utilities for NDJSON streaming routers."""

import json
from datetime import datetime, timezone


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
    if stop is not None:
        opts["stop"] = stop if isinstance(stop, list) else [stop]
    if frequency_penalty is not None:
        opts["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        opts["presence_penalty"] = presence_penalty
    return opts
