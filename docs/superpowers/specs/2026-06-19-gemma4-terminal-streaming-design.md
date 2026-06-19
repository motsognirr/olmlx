# Gemma‑4 terminal streaming display — design

**Date:** 2026-06-19
**Status:** Approved (ready for implementation plan)
**Scope:** `olmlx chat` (in‑process terminal agent) only. HTTP routers and the
engine are out of scope — they already handle Gemma‑4.

## Background

`mlx-community/gemma-4-*` models use a bespoke **text** (non‑special‑token)
output format:

- Turns: `<|turn>role\n … <turn|>\n`
- Thinking channel: `<|channel>thought\n … <channel|>`
- Tool calls: `<|tool_call>call:NAME{key:value,…}<tool_call|>` — bare keys,
  string values wrapped in `<|"|> … <|"|>`, numbers/booleans bare, nested
  `{…}` / `[…]`.
- Tool results: `<|tool_response> … <tool_response|>`

Real captured output (verified against the loaded 12B):

```
<|channel>thought\nThe user is asking … the get_weather tool seems appropriate.<channel|><|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3,units:<|"|>c<|"|>}<tool_call|>
```

### What already works (do not rebuild)

- **Loading** (`gemma4_unified` 12B language tower) and the **eos stop set**
  (`<eos>`/`<turn|>`/`<|tool_response>` = 1/106/50) are fixed in
  `engine/model_manager.py`. Generation now stops cleanly at turn/tool
  boundaries instead of looping.
- **Backend parsing** is already implemented and verified:
  `tool_parser._try_gemma4` parses `<|tool_call>call:NAME{…}<tool_call|>`, and
  `tool_parser._extract_gemma4_blocks` + `routers/thinking_split.py` handle the
  `<|channel>thought…<channel|>` thinking channel. The terminal chat's
  `_parse_turn_output` → `parse_model_output` → `_execute_tool_calls` path runs
  all of this at turn end, so **tool calls already parse and execute and the
  final assistant message is already clean.**

### The one remaining gap

The terminal chat's **live streaming display**. `ThinkingTracker`
(`chat/session.py`) keys solely on the literal `<think>` / `</think>` tags, so
while streaming a Gemma‑4 turn it prints the raw `<|channel>thought…<channel|>`
and `<|tool_call>…<tool_call|>` markup as visible `token` events. The end
state is correct; only the live scroll is wrong.

## Goal

While streaming a Gemma‑4 turn in `olmlx chat`:

- `<|channel>thought…<channel|>` is routed to the **thinking** channel
  (`thinking_start` / `thinking_token` / `thinking_end` events), respecting the
  `--thinking`/`config.thinking` toggle.
- `<|tool_call>…<tool_call|>` markup is **suppressed** from visible `token`
  events (the `tool_call` event is still emitted later by the existing
  turn‑end parse path).
- Existing `<think>`‑family behavior (Qwen etc.), implicit thinking, the
  thinking‑disabled strip, and repetition handling are unchanged.

## Non‑goals

- No changes to `routers/thinking_split.py` (shared with the HTTP routers — it
  already recognizes Gemma‑4 channels). The Gemma‑4 work reuses it as‑is.
- No changes to tool‑execution correctness or `parse_model_output` — those are
  already verified.
- No streaming display of `<|tool_response>`/`<|turn>` markup handling beyond
  what stopping generation already gives us (the model does not emit these in a
  normal assistant turn; they are stop tokens / harness‑injected).

## Architecture

Refactor `ThinkingTracker` from a bespoke `<think>`‑only scanner into a thin
adapter over the shared `thinking_split.split_thinking_parts` state machine
(which already recognizes both `<think>…</think>` and
`<|channel>thought\n…<channel|>`), plus a small **streaming tool‑markup
suppressor** applied to the content channel.

### Component 1 — `ThinkingTracker` (rewritten internals, identical public API)

Preserved public surface (the session and tests depend on all of it):

- `feed(text) -> (think_delta, visible_delta, thinking_ended, thinking_started)`
- properties: `accumulated`, `in_thinking`, `just_started`, `think_emitted`,
  `visible_emitted`
- `flush_disabled() -> str | None`
- `strip_on_repetition() -> int | None`
- constructor: `ThinkingTracker(implicit_mode, thinking_disabled, template_has_thinking)`

Internals:

- `self._accumulated` keeps the **raw** text (all markup) — appended verbatim
  in `feed`. This is the invariant that keeps the turn‑end parse path
  untouched.
- `self._split_state` — the dict passed to `split_thinking_parts`. Seed
  `state["thinking_expected"] = implicit_mode or template_has_thinking` so the
  orphan‑`</think>` heuristic and channel detection behave as today for
  implicit‑thinking models.
- `feed`:
  1. append raw text to `_accumulated`.
  2. `parts = split_thinking_parts(text, self._split_state)`.
  3. for each `(channel, fragment)`:
     - `thinking` → if not `thinking_disabled`, append to think delta and mark
       `thinking_started` on the first thinking fragment of the turn; if
       disabled, discard (already removed from visible).
     - `content` → pass through the tool‑markup suppressor (Component 2);
       append the suppressor's output to visible delta. A transition from
       thinking→content sets `thinking_ended`.
  4. update `_think_emitted` / `_visible_emitted` counters and `_in_thinking`.
  5. return `(think_delta or None, visible_delta or None, thinking_ended, thinking_started)`.
- `flush_disabled`: when `thinking_disabled` and the stream ended without ever
  resolving (mirrors the current implicit+no‑close case), flush any buffered
  content as visible. Implemented via `flush_split_thinking(state)` + suppressor
  flush, content channel only.
- `strip_on_repetition`: truncate `_accumulated` at the start of the currently
  open thinking block. With the delegated splitter, "currently open" = split
  state `phase == "in_think"`; record the byte offset where the open tag began
  so the truncation point is known. Reset `_in_thinking`.

Mapping ordered parts → the 4‑tuple is the crux: accumulate `think_delta` and
`visible_delta` across all parts in the chunk; `thinking_started` fires once on
the first thinking fragment seen this turn; `thinking_ended` fires when a
content fragment follows thinking.

### Component 2 — streaming tool‑markup suppressor

A small stateful helper that removes `<|tool_call> … <tool_call|>` spans from a
stream of content fragments, holding back partial delimiters across chunk
boundaries (the same hold‑back‑partial‑tag technique `thinking_split` uses).

- Interface: `feed(fragment) -> str` (visible text with tool spans removed) and
  `flush() -> str` (emit any held‑back partial that turned out to be literal).
- State: `phase` ∈ {`outside`, `inside`}, `buffer` for partial delimiters.
- `outside`: scan for `<|tool_call>`; emit text before it; on a partial
  open‑tag suffix at the end of the buffer, hold it back. On full open, switch
  to `inside`.
- `inside`: scan for `<tool_call|>`; drop everything up to and including it;
  switch to `outside`. Hold partial close‑tag suffixes.
- Lives in the chat layer (`chat/session.py` or a sibling module), not in the
  shared `thinking_split.py`.

This is display‑only; the raw markup remains in `accumulated` for the
turn‑end parse.

## Data flow

```
generate_chat tokens
  → ThinkingTracker.feed(text)
      → _accumulated += text                       (raw, untouched)
      → split_thinking_parts(text, split_state)     (thinking vs content)
      → tool suppressor over content fragments       (drop <|tool_call> spans)
      → fold parts → (think_delta, visible_delta, thinking_ended, thinking_started)
  → session yields thinking_start / thinking_token / thinking_end / token
turn end:
  → flush_split_thinking + suppressor.flush          (emit any held bytes)
  → result["full_text"] = tracker.accumulated         (RAW)
  → _parse_turn_output → parse_model_output            (existing, unchanged)
  → tool_uses executed via _execute_tool_calls         (existing, unchanged)
```

## Error handling / edge cases

- **Chunk‑straddling delimiters:** both the splitter and the suppressor hold
  back partial open/close tags and resolve them on the next chunk; `flush`
  emits leftovers as content at stream end.
- **Thinking disabled (`--thinking` off):** thinking fragments are dropped from
  emission but still removed from the visible channel; the gemma channel is
  thereby stripped from display. `flush_disabled` covers the no‑close tail.
- **Repetition stop:** unchanged; `strip_on_repetition` truncates the raw
  accumulated text at the open thinking block so the parse path sees a clean
  prefix.
- **Non‑Gemma models:** `<think>` handling is byte‑for‑byte the same because the
  shared splitter already drove it; suppressor is a no‑op when no `<|tool_call>`
  appears.
- **Tool markup with no surrounding thinking** (rare): suppressor still removes
  it from content; parse path still extracts it from `accumulated`.

## Testing (TDD)

Preserve green:

- all 11 `ThinkingTracker` tests (`tests/test_chat_session_helpers.py`)
- chat‑session tests (`tests/test_chat_session.py`)

New unit tests:

- Gemma‑4 `<|channel>thought\n…<channel|>` split into thinking vs visible,
  including the open and close tags arriving in separate `feed` chunks.
- `<|tool_call>…<tool_call|>` removed from visible deltas, including the
  delimiters split across chunk boundaries; the tool call is still recoverable
  from `accumulated`.
- `accumulated` equals the concatenation of all raw fed text (invariant).
- `thinking_disabled=True` drops the Gemma channel from both thinking and
  visible emission.
- `strip_on_repetition` with a Gemma channel truncates at the channel opener.

Live (`tests/live/test_gemma4_unified_text.py`, `real_model`):

- Feed the captured real Gemma‑4 turn (thinking + tool call) through a
  `ThinkingTracker` and assert: visible delta contains no `<|channel`,
  `<channel|`, or `<|tool_call` markup; thinking delta contains the reasoning;
  `accumulated` still contains the raw `<|tool_call>` so `parse_model_output`
  extracts the call.

## Risks

- The 4‑tuple/transition mapping is the subtlest part; the existing 11 tests pin
  the contract and the new chunk‑boundary tests cover the rest.
- Keeping `thinking_split.py` untouched avoids any HTTP‑router regression.
