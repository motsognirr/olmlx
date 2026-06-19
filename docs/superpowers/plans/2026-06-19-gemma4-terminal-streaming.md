# Gemma-4 Terminal Streaming Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `olmlx chat` stream Gemma-4 output cleanly — route `<|channel>thought…<channel|>` to the thinking channel and suppress `<|tool_call>…<tool_call|>` markup from visible tokens — without changing tool-execution correctness or the HTTP routers.

**Architecture:** Refactor `ThinkingTracker` (chat/session.py) from a bespoke `<think>`-only scanner into a thin adapter over the shared `routers/thinking_split.split_thinking_parts` state machine (which already recognizes both `<think>` and the Gemma-4 channel format), plus a small streaming tool-markup suppressor for the content channel. The raw `accumulated` text is preserved verbatim so the existing turn-end parse path (`parse_model_output` → tool execution) is untouched.

**Tech Stack:** Python 3.11, pytest, mlx-lm; spec at `docs/superpowers/specs/2026-06-19-gemma4-terminal-streaming-design.md`.

---

## File Structure

- `olmlx/chat/session.py` — **Modify.** Add `_ToolMarkupStripper` (new class) and the `_longest_partial_tag_suffix` helper; rewrite `ThinkingTracker` internals to delegate to `split_thinking_parts` + the stripper. Public API of `ThinkingTracker` is unchanged.
- `olmlx/engine/model_manager.py` — **Already modified** (uncommitted): `gemma4_unified` loader + eos stop-set fix. Committed in Task 1.
- `tests/test_chat_session_helpers.py`, `tests/test_chat_session.py` — **Existing**, must stay green (11 ThinkingTracker / streaming tests).
- `tests/test_chat_session_helpers.py` — **Modify**, add the new stripper + Gemma-4 streaming unit tests.
- `tests/live/test_gemma4_unified_text.py` — **Already created** (uncommitted); extend with a session-splitter assertion in Task 4.

---

### Task 1: Commit the baseline bug fixes

The loading + eos fixes and their tests are already in the working tree and verified green (235 unit tests + 3 live tests pass, ruff clean). Commit them as the baseline before the streaming work.

**Files:**
- Modify: `olmlx/engine/model_manager.py`
- Test: `tests/test_model_manager.py`, `tests/live/test_gemma4_unified_text.py`

- [ ] **Step 1: Confirm tests pass and ruff is clean**

Run:
```bash
uv run pytest tests/test_model_manager.py -q
uv run ruff check olmlx/engine/model_manager.py tests/test_model_manager.py tests/live/test_gemma4_unified_text.py
uv run ruff format --check olmlx/engine/model_manager.py tests/test_model_manager.py tests/live/test_gemma4_unified_text.py
```
Expected: `235 passed`; `All checks passed!`; all files formatted.

- [ ] **Step 2: Commit**

```bash
git add olmlx/engine/model_manager.py tests/test_model_manager.py tests/live/test_gemma4_unified_text.py
git commit -m "fix(engine): load gemma-4 unified 12B language tower + thread eos stop set

mlx-community/gemma-4-12B-it-4bit ships model_type 'gemma4_unified' whose
vision tower (vision_embedder.*) loads in neither mlx-lm's gemma4_text nor
mlx-vlm 0.4.4's gemma4. Load the language tower via gemma4_text, dropping
the multimodal weights; never rewrite config.json on disk. Thread the full
eos_token_id list ([1,106,50] = <eos>/<turn|>/<|tool_response>) into the
tokenizer so generation stops at turn/tool boundaries instead of looping.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Streaming tool-markup suppressor

A stateful helper that removes `<|tool_call>…<tool_call|>` spans from a stream of text fragments, holding back partial delimiters across chunk boundaries. Display-only; raw text is preserved elsewhere.

**Files:**
- Modify: `olmlx/chat/session.py` (add near the other module-level helpers, after the `_THINK_*` constants around line 70-74)
- Test: `tests/test_chat_session_helpers.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_chat_session_helpers.py`:

```python
from olmlx.chat.session import _ToolMarkupStripper, _longest_partial_tag_suffix


class TestLongestPartialTagSuffix:
    def test_no_partial(self):
        assert _longest_partial_tag_suffix("hello", "<|tool_call>") == 0

    def test_full_partial(self):
        assert _longest_partial_tag_suffix("abc<|to", "<|tool_call>") == 4

    def test_does_not_count_full_tag(self):
        # A complete tag is not a "partial" — len must be < tag length.
        assert _longest_partial_tag_suffix("<|tool_call>", "<|tool_call>") < len(
            "<|tool_call>"
        )


class TestToolMarkupStripper:
    def test_passes_plain_text(self):
        s = _ToolMarkupStripper()
        assert s.feed("hello world") == "hello world"
        assert s.flush() == ""

    def test_removes_whole_tool_call_in_one_chunk(self):
        s = _ToolMarkupStripper()
        out = s.feed("before<|tool_call>call:f{a:1}<tool_call|>after")
        assert out == "beforeafter"
        assert s.flush() == ""

    def test_removes_tool_call_split_across_chunks(self):
        s = _ToolMarkupStripper()
        out = "".join(
            [
                s.feed("vis<|tool_"),
                s.feed("call>call:f{a:"),
                s.feed("1}<tool_"),
                s.feed("call|>tail"),
            ]
        )
        assert out == "vistail"
        assert s.flush() == ""

    def test_holds_partial_open_then_resolves_as_literal(self):
        s = _ToolMarkupStripper()
        # "<|too" looks like a partial open tag, held back...
        assert s.feed("x<|too") == "x"
        # ...but the next chunk shows it was literal text.
        assert s.feed("ls are fun") == "<|tools are fun"
        assert s.flush() == ""

    def test_flush_emits_held_partial_when_outside(self):
        s = _ToolMarkupStripper()
        assert s.feed("done<|tool") == "done"
        # Stream ended mid partial open-tag; it was literal.
        assert s.flush() == "<|tool"

    def test_flush_drops_unterminated_tool_call(self):
        s = _ToolMarkupStripper()
        assert s.feed("a<|tool_call>call:f{") == "a"
        # Unterminated tool call (no close) — drop it from display.
        assert s.flush() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_session_helpers.py -k "ToolMarkupStripper or LongestPartialTagSuffix" -q`
Expected: FAIL with `ImportError: cannot import name '_ToolMarkupStripper'`.

- [ ] **Step 3: Implement the helper and class**

In `olmlx/chat/session.py`, after the `_THINK_*` constants (around line 74), add:

```python
# Gemma-4 native tool-call markup (plain text, not special tokens). Display
# is suppressed during streaming; the raw markup stays in the tracker's
# accumulated text so parse_model_output still extracts the call at turn end.
_TOOL_CALL_OPEN = "<|tool_call>"
_TOOL_CALL_CLOSE = "<tool_call|>"


def _longest_partial_tag_suffix(buf: str, tag: str) -> int:
    """Largest ``k`` (``0 < k < len(tag)``) such that ``buf[-k:] == tag[:k]``.

    Used to hold back the trailing bytes of *buf* that might be the start of
    *tag* straddling a chunk boundary.
    """
    for k in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf[-k:] == tag[:k]:
            return k
    return 0


class _ToolMarkupStripper:
    """Remove ``<|tool_call>…<tool_call|>`` spans from a stream of text.

    Holds partial open/close delimiters across chunk boundaries. Display-only:
    callers keep the raw text elsewhere for parsing.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while self._buf:
            if not self._inside:
                idx = self._buf.find(_TOOL_CALL_OPEN)
                if idx != -1:
                    out.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(_TOOL_CALL_OPEN) :]
                    self._inside = True
                    continue
                keep = _longest_partial_tag_suffix(self._buf, _TOOL_CALL_OPEN)
                out.append(self._buf[: len(self._buf) - keep] if keep else self._buf)
                self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                break
            idx = self._buf.find(_TOOL_CALL_CLOSE)
            if idx != -1:
                self._buf = self._buf[idx + len(_TOOL_CALL_CLOSE) :]
                self._inside = False
                continue
            keep = _longest_partial_tag_suffix(self._buf, _TOOL_CALL_CLOSE)
            self._buf = self._buf[len(self._buf) - keep :] if keep else ""
            break
        return "".join(out)

    def flush(self) -> str:
        """Emit any held-back bytes at stream end.

        A partial open-tag held while *outside* a call was literal text. Bytes
        held while *inside* an unterminated call are dropped (no close arrived).
        """
        if self._inside:
            self._buf = ""
            return ""
        out = self._buf
        self._buf = ""
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_session_helpers.py -k "ToolMarkupStripper or LongestPartialTagSuffix" -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/chat/session.py tests/test_chat_session_helpers.py
git commit -m "feat(chat): streaming tool-markup suppressor for gemma-4

Removes <|tool_call>…<tool_call|> spans from streamed visible text,
holding partial delimiters across chunk boundaries. Display-only.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Refactor `ThinkingTracker` to delegate to `split_thinking_parts`

Rewrite the internals so thinking/content separation comes from the shared state machine (handles `<think>` AND Gemma-4 channels) and content is run through the tool-markup stripper. Keep the public API identical so existing tests and the session keep working.

**Files:**
- Modify: `olmlx/chat/session.py` (`ThinkingTracker`, lines ~217-365; import at top)
- Test: `tests/test_chat_session_helpers.py` (new Gemma-4 tests), plus all existing tests must stay green.

- [ ] **Step 1: Write the failing Gemma-4 streaming tests**

Add to `tests/test_chat_session_helpers.py`:

```python
from olmlx.chat.session import ThinkingTracker


class TestThinkingTrackerGemma4:
    def _drain(self, tracker, chunks):
        think, visible = [], []
        started = ended = False
        for c in chunks:
            td, vd, te, ts = tracker.feed(c)
            if td:
                think.append(td)
            if vd:
                visible.append(vd)
            started = started or ts
            ended = ended or te
        return "".join(think), "".join(visible), started, ended

    def test_gemma4_channel_split_single_chunk(self):
        t = ThinkingTracker()
        raw = "<|channel>thought\nreasoning here<channel|>The answer is 4."
        think, visible, started, ended = self._drain(t, [raw])
        assert think == "reasoning here"
        assert visible == "The answer is 4."
        assert started and ended

    def test_gemma4_channel_split_across_chunks(self):
        t = ThinkingTracker()
        chunks = ["<|channel>thought\nrea", "soning<chan", "nel|>visible"]
        think, visible, started, ended = self._drain(t, chunks)
        assert think == "reasoning"
        assert visible == "visible"

    def test_gemma4_tool_call_suppressed_from_visible(self):
        t = ThinkingTracker()
        raw = (
            "<|channel>thought\nI will call it.<channel|>"
            '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
        )
        think, visible, _, _ = self._drain(t, [raw])
        assert think == "I will call it."
        assert visible == ""
        # Raw markup is preserved for the turn-end parse.
        assert "<|tool_call>" in t.accumulated
        assert "<|channel>thought" in t.accumulated

    def test_accumulated_is_raw(self):
        t = ThinkingTracker()
        chunks = ["<|channel>thought\nx<channel|>", "<|tool_call>call:f{}<tool_call|>"]
        for c in chunks:
            t.feed(c)
        assert t.accumulated == "".join(chunks)

    def test_gemma4_thinking_disabled_strips_channel(self):
        t = ThinkingTracker(thinking_disabled=True)
        raw = "<|channel>thought\nhidden reasoning<channel|>visible answer"
        think, visible, started, ended = self._drain(t, [raw])
        assert think == ""
        assert visible == "visible answer"
        assert not started

    def test_gemma4_repetition_strip_truncates_at_channel(self):
        t = ThinkingTracker()
        t.feed("<|channel>thought\nlooping looping looping")
        assert t.in_thinking
        t.strip_on_repetition()
        assert "<|channel>thought" not in t.accumulated
        assert not t.in_thinking
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_session_helpers.py -k "ThinkingTrackerGemma4" -q`
Expected: FAIL — current `ThinkingTracker` only knows `<think>`, so `think` will be empty / visible will contain the raw channel markup.

- [ ] **Step 3: Add the import**

At the top of `olmlx/chat/session.py`, with the other `olmlx` imports, add:

```python
from olmlx.routers.thinking_split import (
    _THINKING_PAIRS,
    flush_split_thinking,
    split_thinking_parts,
)
```

- [ ] **Step 4: Rewrite `ThinkingTracker` internals**

Replace the entire `ThinkingTracker` class body (lines ~217-365) with:

```python
class ThinkingTracker:
    """Splits streaming output into thinking vs visible, with tool markup
    suppressed from the visible channel.

    Delegates thinking-tag detection to the shared
    ``routers.thinking_split`` state machine (``<think>…</think>`` AND
    Gemma-4 ``<|channel>thought\\n…<channel|>``), and runs the visible
    channel through ``_ToolMarkupStripper`` so Gemma-4's
    ``<|tool_call>…<tool_call|>`` markup never reaches the display. The raw
    text is kept verbatim in ``accumulated`` for the turn-end parse step.
    """

    def __init__(
        self,
        implicit_mode: bool = False,
        thinking_disabled: bool = False,
        template_has_thinking: bool = False,
    ):
        self._thinking_disabled = thinking_disabled
        self._accumulated = ""
        self._think_emitted = 0
        self._visible_emitted = 0
        self._in_thinking = False
        self._just_started = False
        # A template that injects <think> (implicit) or otherwise advertises
        # thinking widens the splitter's orphan-detection window and enables
        # the orphan-</think> heuristic.
        self._split_state: dict = {
            "thinking_expected": bool(implicit_mode or template_has_thinking),
        }
        self._tool_stripper = _ToolMarkupStripper()

    @property
    def accumulated(self) -> str:
        return self._accumulated

    @property
    def in_thinking(self) -> bool:
        return self._in_thinking

    @property
    def just_started(self) -> bool:
        return self._just_started

    @property
    def think_emitted(self) -> int:
        return self._think_emitted

    @property
    def visible_emitted(self) -> int:
        return self._visible_emitted

    def feed(self, text: str) -> tuple[str | None, str | None, bool, bool]:
        """Ingest a chunk; return (think_delta, visible_delta, thinking_ended,
        thinking_started)."""
        self._accumulated += text
        parts = split_thinking_parts(text, self._split_state)

        think_parts: list[str] = []
        visible_parts: list[str] = []
        thinking_started = False
        thinking_ended = False

        for channel, fragment in parts:
            if channel == "thinking":
                if self._thinking_disabled:
                    continue
                if not self._in_thinking:
                    thinking_started = True
                    self._in_thinking = True
                think_parts.append(fragment)
                self._think_emitted += len(fragment)
            else:  # content
                if self._in_thinking:
                    thinking_ended = True
                    self._in_thinking = False
                visible = self._tool_stripper.feed(fragment)
                if visible:
                    visible_parts.append(visible)
                    self._visible_emitted += len(visible)

        self._just_started = thinking_started
        think_delta = "".join(think_parts) or None
        visible_delta = "".join(visible_parts) or None
        return think_delta, visible_delta, thinking_ended, thinking_started

    def flush_disabled(self) -> str | None:
        """Flush buffered content as visible when thinking is disabled.

        Mirrors the old implicit+no-close+disabled case: at stream end any
        bytes the splitter still holds (never resolved to a thinking block)
        are real visible content.
        """
        if not self._thinking_disabled:
            return None
        _thinking, content = flush_split_thinking(self._split_state)
        content = self._tool_stripper.feed(content) + self._tool_stripper.flush()
        if content:
            self._visible_emitted += len(content)
            return content
        return None

    def strip_on_repetition(self) -> int | None:
        """Truncate accumulated text at the start of the open thinking block.

        Removes an incomplete thinking block so the turn-end parse sees a
        clean prefix. Searches for the latest open tag of any known pair.
        """
        if not self._in_thinking:
            return None
        cut = max(
            (self._accumulated.rfind(open_tag) for open_tag, _ in _THINKING_PAIRS),
            default=-1,
        )
        if cut >= 0:
            self._accumulated = self._accumulated[:cut]
            self._visible_emitted = len(self._accumulated)
            self._in_thinking = False
            return self._visible_emitted
        return None
```

- [ ] **Step 5: Run the new Gemma-4 tests**

Run: `uv run pytest tests/test_chat_session_helpers.py -k "ThinkingTrackerGemma4" -q`
Expected: PASS (6 tests).

- [ ] **Step 6: Run the full chat-session test suites (regression gate)**

Run:
```bash
uv run pytest tests/test_chat_session_helpers.py tests/test_chat_session.py -q
```
Expected: PASS — all existing ThinkingTracker / streaming / implicit-thinking / repetition tests plus the new ones.

If any existing test fails, the parts→4-tuple mapping diverged from the old contract. Likely culprits and fixes:
- `<think>` content not emitted: confirm `split_thinking_parts` sees `<think>`/`</think>` (it does via `_THINKING_PAIRS`); ensure `feed` appends to `_accumulated` before splitting.
- implicit-thinking (no opener, orphan `</think>`): confirm `thinking_expected` is set from `implicit_mode or template_has_thinking` in `__init__`.
- disabled-thinking flush: confirm `flush_disabled` uses `flush_split_thinking` and only fires when `_thinking_disabled`.

- [ ] **Step 7: Commit**

```bash
git add olmlx/chat/session.py tests/test_chat_session_helpers.py
git commit -m "refactor(chat): ThinkingTracker delegates to shared thinking_split

Route thinking detection through routers.thinking_split.split_thinking_parts
so the terminal chat understands gemma-4's <|channel>thought…<channel|>
(and any future tag pair) the same way the HTTP routers do, and suppress
<|tool_call> markup from visible tokens. accumulated stays raw so the
turn-end parse + tool execution are unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Live coverage + full verification

**Files:**
- Modify: `tests/live/test_gemma4_unified_text.py`

- [ ] **Step 1: Add a session-splitter live test**

Append to `tests/live/test_gemma4_unified_text.py`:

```python
def test_session_tracker_splits_real_gemma4_turn():
    """Feed a real gemma-4 thinking+tool-call turn through ThinkingTracker and
    assert the visible channel is free of channel/tool markup while the raw
    accumulated text still carries the tool call for parsing."""
    import mlx_lm

    from olmlx.chat.session import ThinkingTracker
    from olmlx.engine.model_manager import _load_with_model_type_fallback
    from olmlx.engine.tool_parser import parse_model_output

    model, tokenizer = _load_with_model_type_fallback(mlx_lm, str(_model_dir()))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    raw = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=256, verbose=False
    )

    tracker = ThinkingTracker(template_has_thinking=True)
    visible_parts = []
    # Feed token-by-token to exercise chunk-boundary handling.
    for ch in raw:
        _td, vd, _te, _ts = tracker.feed(ch)
        if vd:
            visible_parts.append(vd)
    if flush := tracker.flush_disabled():
        visible_parts.append(flush)
    visible = "".join(visible_parts)

    assert "<|channel" not in visible
    assert "<channel|" not in visible
    assert "<|tool_call" not in visible
    # Raw text still parses into a tool call.
    _thinking, _vis, tool_uses = parse_model_output(
        tracker.accumulated, has_tools=True, thinking_expected=True
    )
    assert tool_uses and tool_uses[0]["name"] == "get_weather"
```

- [ ] **Step 2: Run the live test**

Run: `uv run pytest tests/live/test_gemma4_unified_text.py -m real_model -q`
Expected: PASS (5 tests — the 4 existing + this one).

- [ ] **Step 3: Run the full hermetic suite + ruff**

Run:
```bash
uv run pytest tests/test_chat_session_helpers.py tests/test_chat_session.py tests/test_model_manager.py -q
uv run ruff check olmlx/chat/session.py tests/test_chat_session_helpers.py tests/live/test_gemma4_unified_text.py
uv run ruff format olmlx/chat/session.py tests/test_chat_session_helpers.py tests/live/test_gemma4_unified_text.py
```
Expected: all PASS; `All checks passed!`; files formatted.

- [ ] **Step 4: Commit**

```bash
git add tests/live/test_gemma4_unified_text.py olmlx/chat/session.py tests/test_chat_session_helpers.py
git commit -m "test(chat): live coverage for gemma-4 streaming split

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** Component 1 (ThinkingTracker delegation) → Task 3. Component 2 (tool suppressor) → Task 2. `accumulated`-stays-raw invariant → Task 3 (`test_accumulated_is_raw`) + Task 4. `thinking_split.py` untouched → confirmed (only imported). Testing section → Tasks 2-4.
- **Public API preserved:** constructor signature, `feed` 4-tuple, `accumulated`/`in_thinking`/`just_started`/`think_emitted`/`visible_emitted`, `flush_disabled`, `strip_on_repetition` — all retained in Task 3 Step 4.
- **Type consistency:** `_ToolMarkupStripper.feed/flush -> str`; `_longest_partial_tag_suffix -> int`; `ThinkingTracker.feed -> tuple[str|None, str|None, bool, bool]`. Names match across tasks.
- **Risk:** the parts→4-tuple mapping. Task 3 Step 6 is the regression gate against the 11 existing tests, with concrete debugging guidance.
