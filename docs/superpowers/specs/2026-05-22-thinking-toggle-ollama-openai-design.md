# Thinking on/off toggle for Ollama & OpenAI routes

**Date:** 2026-05-22
**Status:** Approved

## Problem

The engine already supports an `enable_thinking: bool | None` control end-to-end
(`generate_chat`, chat-template application, and output-side thinking
split/strip). Only the Anthropic router exposes it (mapping `req.thinking.type`
via `_THINKING_TYPE_MAP`). The Ollama (`/api/chat`, `/api/generate`) and
OpenAI (`/v1/chat/completions`) routes never parse a thinking-control parameter,
so callers cannot turn model thinking on or off.

## Scope

Expose the toggle on:

- `/api/chat` (Ollama)
- `/api/generate` (Ollama)
- `/v1/chat/completions` (OpenAI)

Out of scope: `/v1/completions` (legacy text completion, no chat template).

## Request surface

### Ollama (`/api/chat`, `/api/generate`)

Native top-level `think` field, matching upstream Ollama:

```
think: bool | str | None = None
```

Mapping (`resolve_think_flag`):
- `None` → `None` (engine default applies)
- `bool` → passthrough
- `str` level (gpt-oss `"low"/"medium"/"high"`) → `True`
  (collapses to on; mirrors Anthropic's `adaptive` → True)

### OpenAI (`/v1/chat/completions`)

Two accepted fields:

```
reasoning_effort: str | None = None        # OpenAI standard ("low"/"medium"/"high")
chat_template_kwargs: dict | None = None   # vLLM/SGLang de-facto: {"enable_thinking": bool}
```

Mapping (`resolve_openai_think`), in precedence order:
1. `chat_template_kwargs["enable_thinking"]` present → `bool(value)` (authoritative)
2. `reasoning_effort` is not None → `True`
3. otherwise → `None`

The only clean OFF switch is `chat_template_kwargs.enable_thinking=false`;
OpenAI's `reasoning_effort` has no "off" value, so presence simply means on.

## Engine change — `generate_completion`

`generate_completion` (used by `/api/generate` and `/v1/completions`) currently
hardcodes `enable_thinking=False` in its chat-template branch and does not emit
the `thinking_expected` meta chunk.

- Add `enable_thinking: bool | None = None`.
- In the chat-template branch use
  `enable_thinking if enable_thinking is not None else False` — preserves the
  current off-when-omitted default for `/api/generate` and `/v1/completions`
  (no behavior change for omitted requests).
- Surface `thinking_expected = _resolve_thinking_active(caps, None, enable_thinking)`
  the same way `generate_chat` does: `_prepend_meta(...)` for the streaming
  path, and a `thinking_expected` dict key for the non-streaming path. This
  gives the generate router correct orphan-`</think>` handling (issue #307).

## Router threading

- **`chat.py`**: `enable_thinking = resolve_think_flag(req.think)`, pass to both
  `generate_chat` call sites (stream + non-stream). Output split is already
  wired — no output changes needed.
- **`openai.py`**: `enable_thinking = resolve_openai_think(req.reasoning_effort, req.chat_template_kwargs)`,
  pass to both `generate_chat` call sites. No change to `/v1/completions`.
- **`generate.py`**: `enable_thinking = resolve_think_flag(req.think)`, pass to
  `generate_completion`. Reuse the chat splitter
  (`_split_thinking_streaming` / `_flush_split_thinking` + `parse_model_output`)
  to route thinking into a new top-level `thinking` field on `GenerateResponse`,
  consuming the `thinking_expected` meta the engine now emits.

## Schema changes

- `schemas/chat.py` `ChatRequest`: add `think: bool | str | None = None`.
- `schemas/generate.py` `GenerateRequest`: add `think: bool | str | None = None`.
- `schemas/generate.py` `GenerateResponse`: add `thinking: str | None = None`.
- `schemas/openai.py` `OpenAIChatRequest`: add `reasoning_effort: str | None = None`
  and `chat_template_kwargs: dict[str, Any] | None = None`.

## Behavior summary

| Route | `think`/effort omitted | Explicit on | Explicit off |
|-------|------------------------|-------------|--------------|
| `/api/chat`            | engine default (think unless tools) | think | no think |
| `/v1/chat/completions` | engine default (think unless tools) | think | no think (via `chat_template_kwargs`) |
| `/api/generate`        | off (unchanged)                     | think | no think |

## Testing (TDD)

- Schema parse tests (`test_schemas.py`): new fields accept their types/defaults.
- Resolver unit tests (`test_routers_common.py`): `resolve_think_flag` and
  `resolve_openai_think` precedence/edge cases.
- Router threading tests: mock `generate_chat` / `generate_completion`
  (`AsyncMock`) and assert the `enable_thinking` kwarg for each combination, in
  `test_routers_chat.py`, `test_routers_openai.py`, `test_routers_generate.py`.
- `/api/generate` thinking-split test: assert the `thinking` field is populated
  and `<think>` tags are stripped from `response`.
