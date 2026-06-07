# OpenAI `/v1/responses` Endpoint — Design

**Issue:** #368
**Date:** 2026-06-07
**Status:** Approved (design)

## Goal

Add OpenAI's newer agentic `/v1/responses` API so olmlx is a drop-in for callers
that have migrated off `/v1/chat/completions` — notably the OpenAI Agents SDK and
`client.responses.create()`. The endpoint is a pure request/response *reshaping*
layer over the existing `generate_chat` engine; the engine and other routers are
left untouched.

### In scope (v1)

- `POST /v1/responses` — non-streaming and streaming (full semantic SSE events).
- Function (custom) tool calling.
- `previous_response_id` state continuation with prompt-cache reuse.
- Reasoning output items (mapped from `<think>`).
- Image input (`input_image` content parts) for VLM models.
- Structured outputs via `text.format` (`json_object` / `json_schema`).
- `GET /v1/responses/{id}` and `DELETE /v1/responses/{id}`.
- Honor `store: false` (skip persisting the response).

### Out of scope

- Built-in OpenAI tools (`web_search`, `code_interpreter`, `computer_use`) →
  rejected with a 400.
- Disk-persisted state (in-memory only; lost on restart).
- Distributed-worker / speculative gating changes — inherited from `generate_chat`
  as-is.

## Approach

**Dedicated translation router** (issue-recommended). A new `routers/responses.py`
translates a Responses request into the engine's existing message-dict format,
calls `generate_chat` unchanged, then translates the output back into Responses
output-items / SSE events. This mirrors the proven `routers/anthropic.py`
structure (buffer → split `<think>`/tool calls into typed items → emit typed
events) and keeps each stage independently testable.

Rejected alternatives:
- *Translate through the chat router* (Responses→chat→engine→chat→Responses):
  double translation, couples to the chat router's response shape. Fragile.
- *Extract a shared conversation core* for chat/anthropic/responses: correct
  long-term but a large refactor touching two working routers — out of scope.

## Files

```
olmlx/routers/responses.py        # POST /v1/responses (+ GET/DELETE /v1/responses/{id})
olmlx/schemas/responses.py        # Pydantic request / response / item models
olmlx/engine/responses_state.py   # bounded-LRU store: response_id -> stored response
olmlx/app.py                      # app.include_router(responses.router)
tests/test_routers_responses.py   # unit tests (autouse MLX mock)
tests/live/test_responses_sdk.py  # real_model live test driving the OpenAI SDK
```

## Components

### 1. Request translation — `_build_messages(req, store)`

The engine consumes a list of `{role, content, images?, tool_calls?}` dicts (with
`role: "tool"` for tool results, parsed by `engine/tool_parser.py`). Responses
`input` maps onto that:

| Responses input                                   | → engine message                                              |
| ------------------------------------------------- | ------------------------------------------------------------- |
| `input` as a **string**                           | `{role: "user", content: <str>}`                              |
| `message` item (`role` + `content` parts)         | `{role, content: <joined input_text>}`; `input_image` parts split into `images` via `normalize_image_block` |
| `function_call` item (assistant tool call echoed) | `{role: "assistant", tool_calls: [...]}`                      |
| `function_call_output` item (`call_id`, `output`) | `{role: "tool", tool_call_id: call_id, content: output}`      |
| top-level `instructions`                          | prepended `{role: "system", content: instructions}`           |

- `content` parts already arrive in Responses spelling (`input_text` / `input_image`);
  reuse the chat router's `_normalize_multimodal_messages` (it already handles
  both spellings).
- If `previous_response_id` is set, fetch the stored entry and reconstruct its
  `input_items + output_items` into messages, prepended before the new input.
  Missing id → **404**.
- Unknown input-item types and built-in tool requests → **400** with a clear
  message.

### 2. Output construction — non-streaming `_build_response(...)`

Buffer `generate_chat`, run `parse_model_output` (same call the chat/anthropic
routers use, with `thinking_expected` threading), and build `output[]` in order:

1. `reasoning` item (id `rs_…`) — only if `<think>` text is present.
2. `message` item (id `msg_…`, `role: "assistant"`) with one `output_text`
   content part — omitted entirely if visible text is empty.
3. one `function_call` item (id `fc_…`, `call_id: call_…`, `name`, `arguments`)
   per parsed tool call.

Top-level fields:
- `status`: `"completed"`, or `"incomplete"` with
  `incomplete_details.reason: "max_output_tokens"` when `done_reason == "timeout"`.
- `usage`: `{input_tokens, output_tokens, total_tokens}` from `TimingStats`
  (reuse the `OpenAIUsage.from_stats` accounting).
- `id` (`resp_…`), `object: "response"`, `created_at`, `model`,
  `previous_response_id`.

### 3. Output construction — streaming `_stream_response(...)`

Full semantic event sequence; every event carries a monotonically increasing
`sequence_number` and its `type`:

```
response.created
response.in_progress
  per output item:
    response.output_item.added                { output_index, item }
    text item:
      response.content_part.added             { item_id, output_index, content_index, part }
      response.output_text.delta × N          { item_id, output_index, content_index, delta }
      response.output_text.done               { item_id, output_index, content_index, text }
      response.content_part.done              { ...part }
    function_call item:
      response.function_call_arguments.delta  { item_id, output_index, ... }
      response.function_call_arguments.done   { item_id, output_index, arguments }
    response.output_item.done                 { output_index, item }
response.completed                            { response: <final assembled object> }
```

- **No-tools text path:** stream `output_text.delta` live, token by token, with
  thinking stripped via the existing streaming stripper.
- **Tools path:** tool calls are only parseable from the complete model output
  (Qwen-family constraint, same as chat/anthropic). Buffer the full output, parse,
  then replay reasoning/text/function-call items through the same event envelope —
  the SDK still receives a well-formed stream.
- **Reasoning item:** when present, emitted as its own output item before the
  message item.
- **Error mid-stream:** emit `response.failed` (generic message, no internal
  detail leak beyond the local-tool norms) then close; `aclose()` the engine
  generator in `finally`.

### 4. State store — `engine/responses_state.py`

A bounded `OrderedDict` LRU keyed by `response_id`:

- Value: `{input_items, output_items, model, previous_response_id, created_at}`.
- Cap: `OLMLX_RESPONSES_STORE_MAX` (default **256**); evict oldest on overflow.
- `store: false` on the request → do not write.
- In-memory only; not counted against `OLMLX_MEMORY_LIMIT_FRACTION` (bounded,
  text-only); lost on restart.
- `GET /v1/responses/{id}` → stored response object, **404** if absent/evicted.
- `DELETE /v1/responses/{id}` → remove; **404** if absent.

**Prompt-cache reuse:** pass `cache_id = previous_response_id or <new response id>`
into `generate_chat`. Consecutive turns then share a KV prefix; the radix prefix
index (`PrefixCacheIndex`) handles sibling takeover. This satisfies the issue's
acceptance criterion that a multi-turn `previous_response_id` conversation reuses
the prompt cache.

## Error handling & edge cases

- Malformed image part → **422** (mirror chat router's `ValueError → 422`).
- Unknown input-item type → **400**.
- Built-in tool requested (`type` in `web_search`/`code_interpreter`/`computer_use`)
  → **400** with a "not supported" message.
- `previous_response_id` not found → **404**.
- `text.format` / structured-output schema error → **422** via the existing
  `parse_response_format`.
- Grammar/VLM/speculative/distributed gating is inherited from `generate_chat`;
  no new gates introduced here.

## Testing (TDD — failing tests first)

`tests/test_routers_responses.py`, mirroring `test_routers_openai.py` (autouse MLX
mock):

- text non-streaming.
- text streaming — assert event ordering and `sequence_number` monotonicity.
- function tool-call round-trip: request → `function_call` output item →
  `function_call_output` continuation request.
- `previous_response_id` reconstruction + `cache_id` threading into `generate_chat`.
- reasoning item produced from `<think>` output.
- image input part split into `images`.
- structured output via `text.format` (`json_schema`).
- error paths: 404 unknown id, 422 bad image / bad schema, 400 unknown item / built-in tool.
- `store: false` skips storage; `GET` then `DELETE` lifecycle (404 after delete).

`tests/live/test_responses_sdk.py` (`real_model`, outside `tests/integration/` to
dodge its autouse MLX mock): drive the actual OpenAI SDK
`client.responses.create()` for text, streaming text, and tool use — the issue's
top-level acceptance criterion.

## Acceptance criteria (from #368)

1. OpenAI SDK `client.responses.create()` works for text, streaming text, and
   tool use against olmlx.
2. A multi-turn conversation via `previous_response_id` reuses the prompt cache.
