"""Shared request-shaping utilities for the chat-surface routers."""

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from olmlx.engine.chat_templating import _merge_system_turns
from olmlx.utils.audio_input import normalize_audio_block
from olmlx.utils.images import normalize_image_block

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

    Asymmetry with :func:`resolve_think_flag`: an empty ``reasoning_effort``
    returns ``None`` (engine default) here, whereas an empty Ollama ``think``
    string returns ``False``.  This is deliberate — the OpenAI chat route has
    no off-by-default contract, so an empty/absent value should fall through to
    the engine default rather than force thinking off.
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


def resolve_tool_choice(tool_choice: str | dict[str, Any] | None) -> bool:
    """Map a request ``tool_choice`` to whether tools are honored (issue #620).

    Collapses the OpenAI/Responses form (a string ``"auto"``/``"none"``/…, or a
    ``{"type": "function", …}`` forced selection) and the Anthropic form (a
    ``{"type": "auto"|"none"|"any"|"tool"}`` object) into a single boolean:

    - ``None`` / ``"auto"`` / ``{"type": "auto"}`` → ``True``: the model may
      call tools and they are parsed out of the output (the prior, default
      behavior).
    - ``"none"`` / ``{"type": "none"}`` → ``False``: suppress tools — the
      request runs as if none were declared, guaranteeing a text answer (the
      standard way to force prose mid tool-loop).

    Any other value — ``"required"``, Anthropic ``"any"``, or a forced
    ``{"type": "function"|"tool", …}`` selection — cannot be honored (there is
    no forced-tool decoding), so it raises :class:`ValueError` (→ 400 via the
    app's handler) instead of being silently ignored, which is exactly the
    divergence issue #620 reports.
    """
    if tool_choice is None:
        return True
    if isinstance(tool_choice, str):
        value = tool_choice.strip().lower()
    elif isinstance(tool_choice, dict):
        value = tool_choice.get("type")
    else:
        raise ValueError("tool_choice must be a string or an object")
    if value == "auto":
        return True
    if value == "none":
        return False
    raise ValueError(
        f"tool_choice {value!r} is not supported; only 'auto' and 'none' are honored"
    )


def collect_content_parts(
    parts: Iterable[Any],
) -> tuple[list[str], list[str], list[str]]:
    """Collect ``(texts, images, audio)`` from multimodal content parts.

    The one place that recognizes every surface's spelling (issue #471):
    OpenAI Chat ``text``/``image_url``/``input_audio``, Responses-API
    ``input_text``, and Anthropic ``text``/``image``/``audio``.  Images and
    audio are normalized to engine-loadable strings (URL, path, or data URI)
    via ``normalize_image_block``/``normalize_audio_block``, which raise
    ``ValueError`` on malformed blocks — callers surface that as a 422.

    Non-dict, unknown-type, and empty-text parts are skipped; per-channel
    order is preserved.
    """
    texts: list[str] = []
    images: list[str] = []
    audio: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("text", "input_text"):
            text = part.get("text") or ""
            if text:
                texts.append(text)
        elif ptype in ("image_url", "image"):
            images.append(normalize_image_block(part))
        elif ptype in ("input_audio", "audio"):
            audio.append(normalize_audio_block(part))
    return texts, images, audio


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


async def load_or_unload_response(
    request: Any, model: str, keep_alive: int | str | None, *, chat: bool
) -> Any:
    """Answer Ollama's preload/unload request (#760).

    An ``/api/generate`` with no prompt (or ``/api/chat`` with no messages)
    loads the model and returns ``done_reason: "load"``; with ``keep_alive: 0``
    it unloads instead (``done_reason: "unload"``). Open WebUI and the ollama
    CLI preload this way. Returns a JSON body (never a stream), like Ollama.
    """
    from fastapi.responses import JSONResponse

    from olmlx.engine.loaded_model import parse_keep_alive
    from olmlx.engine.model_manager import ActiveRequestsError

    manager = request.app.state.model_manager
    registry = request.app.state.registry
    if keep_alive is not None and parse_keep_alive(keep_alive) == 0.0:
        try:
            await manager.unload(model)  # not loaded → nothing to do
        except ActiveRequestsError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        reason = "unload"
    else:
        # A panel has no single model to load; its members load on first use.
        if not registry.is_panel(model):
            await manager.ensure_loaded(model, keep_alive)
        reason = "load"
    body: dict[str, Any] = {
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if chat:
        body["message"] = {"role": "assistant", "content": ""}
    else:
        body["response"] = ""
    body.update(done=True, done_reason=reason)
    return body


def _merge_leading_system_messages(messages: list[dict]) -> list[dict]:
    """Fold the leading run of system messages into a single system message.

    ``developer`` is normalized to ``system`` by the schema (#710), so a client
    sending ``system`` + ``developer`` produces two leading system turns (on
    ``/v1/responses``, ``instructions`` + a ``developer`` item does too, #739).
    Strict chat templates (Qwen3.5/3.6) raise "System message must be at the
    beginning." on the second one. Only the leading run is folded here; a
    mid-conversation system turn keeps its position, and ``generate_chat``
    folds it to the front only for templates that reject it
    (``TemplateCaps.rejects_positional_system``, #740). Callers run it after
    content normalization, so content is a string or absent.
    Metadata rules are shared with the engine fold via
    ``_merge_system_turns``.
    """
    run = 0
    while run < len(messages) and messages[run].get("role") == "system":
        run += 1
    if run < 2:
        return messages
    return [_merge_system_turns(messages[:run]), *messages[run:]]
