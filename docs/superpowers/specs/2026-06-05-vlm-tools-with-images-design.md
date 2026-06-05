# VLM native tool calling with images (#428)

**Date:** 2026-06-05
**Issue:** [#428](https://github.com/motsognirr/olmlx/issues/428)
**Status:** Approved — ready for implementation plan

## Problem

A chat request carrying both `tools` and an image silently loses the image. In
`engine/inference.py`, `_apply_chat_template_vlm` bypasses
`mlx_vlm.apply_chat_template` when `tools` are present (to get clean native tool
formatting from the raw tokenizer) and logs:

```
VLM native-tools path does not support images; %d image(s) will be ignored
```

Gemma 4 and other modern VLMs support vision + function calling in one request
("look at this chart and call `record_metric(...)`"), but olmlx cannot serve it.

### Two root causes

1. **Prompt rendering (engine).** Images arrive in a separate `msg["images"]`
   field, not as content-parts. When the native-tools branch calls the raw
   tokenizer's `apply_chat_template(messages, tools=tools)`, the messages contain
   only text, so no `<|image|>` placeholder is emitted. The image *data* still
   flows to `mlx_vlm.generate(image=images)` — but with no placeholder in the
   prompt, it is effectively dropped.

2. **Intake (routers).** Only the Ollama `/api/chat` surface delivers images to
   the engine today (its `Message.images` field). The OpenAI router's
   `content` is `str`-only and rejects multimodal content; the Anthropic router's
   `_convert_messages` silently drops `image` blocks. So #428's literal
   acceptance criterion (`/v1/chat/completions` + image + tools) cannot even reach
   the engine.

## Findings (verified)

- The **Gemma 4 chat template** natively renders both `tools` (`<|tool>…<tool|>`)
  and image content-parts (`item['type'] == 'image'` → `<|image|>`).
- **mlx_vlm 0.4.4** `apply_chat_template` forwards `tools` to the tokenizer and
  inserts per-model image placeholders via `num_images`; the "garbling" rationale
  for the original bypass appears stale. (We do not rely on this — see Approach B.)
- Passing `[{type:image}, {type:text}]` content-parts + `tools=` through the
  **raw tokenizer** produces *both* `<|image|>` and the native `<|tool>`
  declaration. Empirically confirmed on `mlx-community/gemma-4-31b-it-4bit`.
- `mlx_vlm.utils.load_image` accepts file paths, http(s) URLs, and
  `data:image/…;base64,…` data URIs (PIL sniffs the real format, so the declared
  media type in the URI is cosmetic). A **bare** base64 string is not accepted.
- The image *data* already threads to `mlx_vlm.generate(image=images)`
  unconditionally (`inference.py` ~4058/4104) — independent of the tools branch.

## Chosen approach: B — inject image content-parts on the raw-tokenizer path

Keep the existing raw-tokenizer path (which already renders tools correctly) and
inject image content-parts into the message before templating, so the template
emits `<|image|>` while `tools=` still renders natively.

**Rejected alternatives:**

- **A — route through `mlx_vlm.apply_chat_template` with `num_images` + `tools`.**
  Reuses per-model placeholder logic, but mlx_vlm flattens each message to
  `{role, content:str}`, dropping `tool_calls`/`tool_responses` from history and
  re-flattening olmlx's tool-role rewriting. Fine for single-turn but fragile.
- **C — post-render string splice of the image token.** Brittle string surgery
  that replicates what the template already does.

## Scope

- **Conversation:** single-turn — image(s) + tools in one request → grounded tool
  call. All extracted images attach to the **last user turn**. Multi-turn image
  history is out of scope.
- **Surfaces:** engine fix + OpenAI intake + Anthropic intake (Ollama already
  works once the engine fix lands).
- **Out of scope:** structured outputs / grammar on the image path, speculative
  decoding with images, prompt-cache on the VLM path (all tracked under #429);
  video/audio modalities (#427/#426).

## Design

### 1. Engine fix — `engine/inference.py` :: `_apply_chat_template_vlm`

In the `if tools:` branch, when `images` are present:

- Rewrite the **last user message's** content to a parts list:
  `[{"type":"image"}] × len(images)` followed by
  `{"type":"text","text": <original text>}` (omit the text part if empty).
- Call `tok.apply_chat_template(messages, tools=tools, …)` as today.
- Delete the warn-and-ignore block.
- If there is no user message, log a warning and render text-only (defensive).
- **Safety guard:** after rendering, assert the prompt contains exactly
  `len(images)` image placeholders. If a model's template doesn't support
  `{type:image}` parts, raise a clear error naming the model instead of letting
  `mlx_vlm.generate` crash on a count mismatch.

No change to the generate call — `images` already threads through to
`generate(image=images)`.

### 2. OpenAI intake — `schemas/openai.py` + `routers/openai.py`

- Widen `OpenAIChatMessage.content` to `str | list[dict[str, Any]] | None`.
- In the router, after `model_dump`, normalize any list-content message: join
  `text` parts into the content string, collect `image_url.url` values into
  `msg["images"]`. URLs and data URIs pass straight to `load_image`.

### 3. Anthropic intake — `schemas/anthropic.py` + `routers/anthropic.py`

- Ensure the content-block model carries image fields (`type:"image"`,
  `source:{type, media_type, data}`; also `type:"url"`). Extend if absent.
- In `_convert_messages`' user-block loop, handle `block.type == "image"`:
  reassemble base64 → `data:{media_type};base64,{data}` (or pass a URL through),
  append to the message's `images` list, set `msg["images"]`.

### 4. Shared normalization helper — `utils/images.py` (new)

One function mapping OpenAI `image_url` / Anthropic `source` / bare base64 → a
`load_image`-acceptable string (path, URL, or data URI). Reused by both routers;
also wraps Ollama bare-base64 in a data URI if that path isn't already handled
(verify during implementation).

## Error handling

- Image placeholder / `len(images)` count mismatch → descriptive error naming the
  model.
- Unsupported image source (malformed data URI, unknown content-part type) →
  HTTP 400 identifying the offending part.
- Non-VLM model receiving images → existing behavior unchanged.

## Testing

### Fast unit tests (run in CI)

- `_apply_chat_template_vlm(tools + images)` → rendered prompt contains
  `<|image|> × N` + native tool tags; no warning logged.
- OpenAI content normalization → `{images, content}` shape; text-only and
  image-only messages both correct.
- Anthropic `_convert_messages` with an image block → `msg["images"]` populated;
  text / tool_use / tool_result paths unaffected.
- Normalization helper: base64 / data-URI / URL forms; rejects malformed input.

### Live integration test (gated, `pytest -m "not real_model"` in CI)

- Marked `real_model`. Load Gemma 4 26B-A4B (cached locally), send `/api/chat`
  *and* `/v1/chat/completions` with a small synthetic generated image (PIL, e.g. a
  large rendered number) + one tool. Assert a grounded `tool_call` is parsed and
  the image is not dropped (no warning).

## Docs

Update CLAUDE.md's VLM-gating design note:

- The native-tools path now threads images (Approach B); describe the three-surface
  image intake.
- Note that grammar / speculative / prompt-cache on the image path remain gated
  (tracked under #429).

## Files

- `olmlx/engine/inference.py` — `_apply_chat_template_vlm` injection + remove drop.
- `olmlx/schemas/openai.py` — widen `content` type.
- `olmlx/routers/openai.py` — multimodal content normalization.
- `olmlx/schemas/anthropic.py` — image block fields (if absent).
- `olmlx/routers/anthropic.py` — `_convert_messages` image handling.
- `olmlx/utils/images.py` — new shared normalization helper.
- `tests/` — unit tests + gated `real_model` integration test.
- `CLAUDE.md` — VLM-gating note.

## Acceptance criteria

- `/api/chat`, `/v1/chat/completions`, and `/v1/messages` requests with `tools` +
  an image produce a tool call grounded in the image on a Gemma 4 checkpoint; the
  image is **not** dropped and no warning is logged.
- Text-only tool calling and image-only chat are both unaffected on all three
  surfaces.
- Fast unit tests pass in CI; the gated live test passes locally on Gemma 4.
