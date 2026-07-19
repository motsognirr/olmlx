# Prefill Measurement (#503 Phase 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real, measured prefill wall-clock plus a within-prefill breakdown (cache-covered vs fresh, draft-lane vs target-lane) and a reproducible ~69k agentic bench prompt, so the build-or-defer decision on draft-assisted prefill rests on numbers.

**Architecture:** Three additive changes. (1) In the streaming generate loop, capture a first-token timestamp so time-to-first-token — which equals the prefill wall-clock, since prefill runs lazily inside the stream's first iteration — is surfaced on the existing `decode` span's `ttft_ns` attribute and one log line. (2) Instrument `_drive_spec_prefill` to record per-lane ns / covered / fresh into a breakdown dict the speculative base surfaces on the `spec.prefill` span. (3) Add a deterministic synthetic `agentic-69k` bench prompt. No prefill algorithm changes; no client-facing `stats.prompt_eval_duration` rewire.

**Tech Stack:** Python, MLX, pytest (`pytest.mark.asyncio`), OpenTelemetry in-memory span exporter, existing `Timer`/tracing utilities.

## Global Constraints

- Platform: Apple Silicon; MLX. Tasks 1 and 3 tests are Metal-free (mocked `mx` / pure data). Task 2's test executes a real small `MockModel` prefill (like the existing `test_draft_prefill_chunks_long_prompt`) and runs in CI.
- Phase 0 does **not** change any prefill algorithm and does **not** modify client-facing `stats.prompt_eval_duration` (Ollama API / bench table stay on the existing heuristic).
- Measured prefill = time-to-first-token captured on the generation path; breakdown attributes are additive and consumed only by human readers.
- Follow existing patterns: reuse the `otel_memory_exporter` in-memory span fixture pattern from `tests/test_tracing.py`; reuse `_RecordingModel`/`MockModel` from `tests/test_speculative.py`; mirror `_LONG_CONTEXT_BODY` for deterministic prompt padding.
- Run `uv run ruff check` and `uv run ruff format` before any commit that will be pushed.

---

### Task 1: Measured TTFT on the streaming decode span + log line

**Files:**
- Modify: `olmlx/engine/inference.py` (the streaming `generate_chat` decode loop, ~3688–3790)
- Test: `tests/test_inference.py` (add to the streaming test class alongside `test_streaming`, ~2153)

**Interfaces:**
- Consumes: existing locals in scope — `full_prompt_tokens` (list[int] | None), `token` (last `StreamToken`), `_decode_span` (tracing span), `stats` (`TimingStats`), `logger`.
- Produces: no new public symbols. The `decode` span's `ttft_ns` attribute becomes the measured value (was the heuristic `stats.prompt_eval_duration`); a new `logger.info` prefill-summary line.

- [ ] **Step 1: Write the failing test**

Add this fixture + test to `tests/test_inference.py`. The fixture mirrors `otel_memory_exporter` from `tests/test_tracing.py`; place it at module scope (top of the file, after imports). Add `import asyncio` to the file's imports if not already present.

```python
@pytest.fixture
def otel_memory_exporter():
    """Install tracing with an in-memory span exporter; tear down after."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from olmlx.utils import tracing

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing._install_test_provider(provider)
    yield tracing, exporter
    tracing._uninstall_test_provider()
```

```python
    @pytest.mark.asyncio
    async def test_streaming_ttft_span_is_measured(
        self, mock_manager, otel_memory_exporter
    ):
        """decode span's ttft_ns reflects real time-to-first-token, not the
        rate-reconstructed heuristic. A 30ms pre-first-token sleep must show up
        as ttft_ns >= 20ms even though prompt_tps is huge (heuristic ~5us)."""
        _tracing, exporter = otel_memory_exporter
        mock_mx = MagicMock()

        tok = StreamToken(
            text="Hi",
            token=1,
            prompt_tokens=5,
            generation_tokens=1,
            prompt_tps=1_000_000.0,  # heuristic prefill ~= 5us
            generation_tps=1_000_000.0,
        )
        state = {"first": True}

        async def anext_impl():
            if state["first"]:
                state["first"] = False
                await asyncio.sleep(0.03)  # 30ms of "prefill"
                return tok
            raise StopAsyncIteration

        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream.drain_and_join = AsyncMock()
        mock_stream._thread = None
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.async_mlx_stream", return_value=mock_stream
            ):
                gen = await generate_completion(
                    mock_manager, "qwen3", "Hello", stream=True
                )
                async for _ in gen:
                    pass

        decode_spans = [
            s for s in exporter.get_finished_spans() if s.name == "decode"
        ]
        assert decode_spans, "decode span was not recorded"
        ttft_ns = dict(decode_spans[0].attributes)["ttft_ns"]
        assert ttft_ns >= 20_000_000, f"ttft_ns {ttft_ns} looks heuristic, not measured"
```

Check the required names are already imported at the top of `tests/test_inference.py`: `StreamToken`, `CancellableStream`, `AsyncMock`, `MagicMock`, `patch`, `generate_completion`, `pytest`. They are all used by the existing `test_streaming`; add `import asyncio` if missing.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inference.py::TestStreaming::test_streaming_ttft_span_is_measured -v`

(Use the actual class name that wraps `test_streaming` — find it with `grep -n "class Test" tests/test_inference.py` and pick the one whose body contains `async def test_streaming`.)

Expected: FAIL — `ttft_ns` on the decode span equals the ~5µs heuristic value (assert `>= 20_000_000` fails), because `ttft_ns` is currently set from `stats.prompt_eval_duration`.

- [ ] **Step 3: Implement the measured TTFT**

In `olmlx/engine/inference.py`, in the streaming decode block. First, just before the `async for token in stream:` loop (it sits inside `with Timer() as eval_timer:` around line 3688, right after `inf_start = time.monotonic()`), add the two capture locals:

```python
            with Timer() as eval_timer:
                inf_start = time.monotonic()
                prefill_start_ns = time.perf_counter_ns()
                ttft_measured_ns: int | None = None
                token = None
```

Inside the loop body, as the **first** statement of the `async for token in stream:` block, record the first-token timestamp exactly once:

```python
                async for token in stream:
                    if ttft_measured_ns is None:
                        ttft_measured_ns = time.perf_counter_ns() - prefill_start_ns
                    # Always accumulate for prompt cache (raw stream, not filtered)
                    stats.eval_count = token.generation_tokens
```

Then change the `_decode_span.set_attributes(...)` block (currently `"ttft_ns": stats.prompt_eval_duration,` ~line 3785) to prefer the measured value, and emit the summary log line immediately after the span block:

```python
            _decode_span.set_attributes(
                {
                    "eval_count": stats.eval_count,
                    "decode_tok_s": _metrics._decode_tps(stats),
                    "ttft_ns": (
                        ttft_measured_ns
                        if ttft_measured_ns is not None
                        else stats.prompt_eval_duration
                    ),
                    "ttft_measured": ttft_measured_ns is not None,
                    "cache_hit": bool(full_prompt_tokens)
                    and token is not None
                    and (token.prompt_tokens or 0) < len(full_prompt_tokens),
                }
            )
            if ttft_measured_ns is not None and token is not None:
                _fresh = token.prompt_tokens or 0
                _full = (
                    len(full_prompt_tokens)
                    if full_prompt_tokens is not None
                    else _fresh
                )
                logger.info(
                    "prefill %.2fs (fresh %d/%d tok, cache-covered %d)",
                    ttft_measured_ns / 1e9,
                    _fresh,
                    _full,
                    max(0, _full - _fresh),
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inference.py -k "ttft_span_is_measured or test_streaming" -v`

Expected: PASS (both the new test and the existing `test_streaming`, confirming no regression to the stats fields).

- [ ] **Step 5: Commit**

```bash
uv run ruff check olmlx/engine/inference.py tests/test_inference.py
uv run ruff format olmlx/engine/inference.py tests/test_inference.py
git add olmlx/engine/inference.py tests/test_inference.py
git commit -m "feat(prefill): measure time-to-first-token on decode span (#503)"
```

---

### Task 2: Per-lane prefill breakdown on the spec.prefill span

**Files:**
- Modify: `olmlx/engine/spec_decoder_base.py` (`SpecDecoderBase.__init__`, `reset`, `prefill`)
- Modify: `olmlx/engine/speculative.py` (`_drive_spec_prefill` + its two callers)
- Test: `tests/test_speculative.py` (add to `TestChunkedPrefill`, ~337)

**Interfaces:**
- Consumes: `_tracing.span("spec.prefill", ...)` (base), `_PREFILL_CHUNK`, `_eval_cache`, `mx` (speculative.py).
- Produces:
  - `SpecDecoderBase._last_prefill_breakdown: dict[str, int]` — populated during `prefill()`, readable after it returns.
  - `_drive_spec_prefill(..., breakdown: dict[str, int] | None = None)` — when a dict is passed, it is updated in place with keys `covered_tokens`, `fresh_tokens`, `target_lane_ns`, `draft_lane_ns`, `n_target_chunks`, `n_lanes` (all `int`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_speculative.py` inside `class TestChunkedPrefill`:

```python
    def test_prefill_records_lane_breakdown(self):
        """SpeculativeDecoder.prefill populates _last_prefill_breakdown with
        per-lane timings and covered/fresh token counts."""
        from olmlx.engine.speculative import _PREFILL_CHUNK, SpeculativeDecoder

        draft = _RecordingModel(MockModel(32, 16))
        target = _RecordingModel(MockModel(32, 16))
        decoder = SpeculativeDecoder(
            draft_model=draft, target_model=target, num_speculative_tokens=2
        )
        n = _PREFILL_CHUNK * 2 + 5
        decoder.prefill(mx.zeros((1, n), dtype=mx.int32))

        bd = decoder._last_prefill_breakdown
        assert bd["fresh_tokens"] == n
        assert bd["covered_tokens"] == 0
        assert bd["n_lanes"] == 2
        assert bd["target_lane_ns"] >= 0
        assert bd["draft_lane_ns"] >= 0
        # 2 full prefix chunks + a remainder chunk + the pass-2 last-token forward.
        assert bd["n_target_chunks"] >= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_speculative.py::TestChunkedPrefill::test_prefill_records_lane_breakdown -v`

Expected: FAIL — `AttributeError: 'SpeculativeDecoder' object has no attribute '_last_prefill_breakdown'`.

- [ ] **Step 3a: Add the breakdown field + reset in the base**

In `olmlx/engine/spec_decoder_base.py`, in `SpecDecoderBase.__init__` (after the `self._stats_accepted_draft: int = 0` line, ~318):

```python
        # Per-request prefill breakdown, populated by strategies that route
        # through _drive_spec_prefill; surfaced on the spec.prefill span and
        # read after _prefill_impl returns. Empty for strategies that don't.
        self._last_prefill_breakdown: dict[str, int] = {}
```

In `reset()`, alongside the other diagnostic resets (after `self._stats_accepted_draft = 0`, ~432):

```python
        self._last_prefill_breakdown = {}
```

- [ ] **Step 3b: Surface the breakdown on the span in `prefill()`**

In `olmlx/engine/spec_decoder_base.py`, replace the `prefill()` `with _tracing.span(...)` body (~354–368) so the span is bound and attributes are set after `_prefill_impl` succeeds:

```python
        with _tracing.span(
            "spec.prefill",
            strategy=self._strategy_label,
            prompt_tokens=int(prompt.shape[-1]),
        ) as _sp:
            self.reset()
            try:
                first = self._prefill_impl(
                    prompt, segmented=segmented, cancel_event=cancel_event
                )
            except Exception:
                # Best-effort cleanup; ``reset()`` swallows nested teardown
                # errors per step so the caller sees the original error.
                self.reset()
                raise
            if self._last_prefill_breakdown:
                _sp.set_attributes(
                    {k: int(v) for k, v in self._last_prefill_breakdown.items()}
                )
                logger.debug(
                    "spec prefill breakdown [%s]: %s",
                    self._strategy_label,
                    self._last_prefill_breakdown,
                )
            return first
```

- [ ] **Step 3c: Instrument `_drive_spec_prefill`**

In `olmlx/engine/speculative.py`, add `import time` to the imports (after `import threading`, ~14):

```python
import threading
import time
```

Change the `_drive_spec_prefill` signature to accept the breakdown dict (add the last param):

```python
def _drive_spec_prefill(
    *,
    flat: list[int],
    boundaries: list[int],
    already_covered: int,
    lanes: list[tuple[Any, list]],
    cancel_event: threading.Event | None,
    on_boundary: Any,
    breakdown: dict[str, int] | None = None,
) -> mx.array:
```

Replace the inner `_fill` helper and add timing accumulators. The existing `_fill` (~441–450) becomes lane-aware; the final single-token forward in `_target_last_logit` (~460) is timed too. Insert the accumulators just after `deepest` is computed (before `def _arr`), and rewrite `_fill` / `_target_last_logit`:

```python
    target_model = lanes[0][0]
    _acc = {"target_ns": 0, "draft_ns": 0, "target_chunks": 0}

    def _arr(start: int, end: int) -> mx.array:
        return mx.array(flat[start:end], dtype=mx.int32)[None, :]

    def _fill(model: Any, cache: list, start: int, end: int) -> None:
        is_target = model is target_model
        pos = start
        while pos < end:
            if cancel_event is not None and cancel_event.is_set():
                raise PrefillCancelled()
            stop = min(pos + _PREFILL_CHUNK, end)
            _t0 = time.perf_counter_ns()
            model(_arr(pos, stop), cache=cache)
            _eval_cache(cache)
            _dt = time.perf_counter_ns() - _t0
            if is_target:
                _acc["target_ns"] += _dt
                _acc["target_chunks"] += 1
            else:
                _acc["draft_ns"] += _dt
            mx.clear_cache()
            pos = stop

    def _target_last_logit(start: int, end: int) -> mx.array:
        # Fill [start, end-1] (logits discarded — keeps lm_head out of the eval
        # graph over the prefix), then forward the final token for the seeding
        # logit (``_prefill_last_logit`` semantics).
        target_model_, target_cache = lanes[0]
        _fill(target_model_, target_cache, start, end - 1)
        if cancel_event is not None and cancel_event.is_set():
            raise PrefillCancelled()
        _t0 = time.perf_counter_ns()
        _out = _logits(target_model_(_arr(end - 1, end), cache=target_cache))[0, -1, :]
        _acc["target_ns"] += time.perf_counter_ns() - _t0
        _acc["target_chunks"] += 1
        return _out
```

Then populate `breakdown` at the two `return` points. Wrap the tail of the function so both the `deepest is None` and two-chunk paths record before returning. Replace the final `if deepest is None: ... return _target_last_logit(...)` / two-chunk `return _target_last_logit(deepest, n)` structure so a single helper records the breakdown:

```python
    def _record() -> None:
        if breakdown is not None:
            breakdown.update(
                {
                    "covered_tokens": already_covered,
                    "fresh_tokens": n - already_covered,
                    "target_lane_ns": _acc["target_ns"],
                    "draft_lane_ns": _acc["draft_ns"],
                    "n_target_chunks": _acc["target_chunks"],
                    "n_lanes": len(lanes),
                }
            )

    if deepest is None:
        for model, cache in lanes[1:]:
            _fill(model, cache, already_covered, n)
        result = _target_last_logit(already_covered, n)
        _record()
        return result

    # Chunk 1: uncovered prefix up to the deepest interior boundary (all lanes).
    for model, cache in lanes:
        _fill(model, cache, already_covered, deepest)
    on_boundary(deepest)
    # Chunk 2: boundary to the end. Non-target lanes fill KV; the target
    # captures the seeding logit at the final position.
    for model, cache in lanes[1:]:
        _fill(model, cache, deepest, n)
    result = _target_last_logit(deepest, n)
    _record()
    return result
```

- [ ] **Step 3d: Pass the breakdown from both callers**

In `olmlx/engine/speculative.py`, in `SpeculativeDecoder._drive_segmented_prefill` (the `return _drive_spec_prefill(...)` at ~833), add the `breakdown` argument:

```python
        return _drive_spec_prefill(
            flat=flat,
            boundaries=boundaries,
            already_covered=already_covered,
            lanes=[
                (self._target, self._target_cache),
                (self._draft, self._draft_cache),
            ],
            cancel_event=cancel_event,
            on_boundary=lambda boundary: self._store_snapshot(flat[:boundary]),
            breakdown=self._last_prefill_breakdown,
        )
```

In `_prefill_with_reuse` (the `logit = _drive_spec_prefill(...)` at ~1610), add the same argument:

```python
        logit = _drive_spec_prefill(
            flat=flat,
            boundaries=boundaries,
            already_covered=already_covered,
            lanes=[(self._target, self._target_cache)],
            cancel_event=cancel_event,
            on_boundary=lambda boundary: self._store_snapshot(flat[:boundary]),
            breakdown=self._last_prefill_breakdown,
        )
```

(If `_prefill_with_reuse`'s existing call passes a different `on_boundary`, keep that argument exactly as it is and only append `breakdown=self._last_prefill_breakdown`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_speculative.py::TestChunkedPrefill -v`

Expected: PASS — the new test plus the existing `test_draft_prefill_chunks_long_prompt` and the chunk-invariance tests all pass (confirms the `_fill`/`_target_last_logit` rewrite is behavior-preserving).

- [ ] **Step 5: Commit**

```bash
uv run ruff check olmlx/engine/spec_decoder_base.py olmlx/engine/speculative.py tests/test_speculative.py
uv run ruff format olmlx/engine/spec_decoder_base.py olmlx/engine/speculative.py tests/test_speculative.py
git add olmlx/engine/spec_decoder_base.py olmlx/engine/speculative.py tests/test_speculative.py
git commit -m "feat(prefill): per-lane spec prefill breakdown on span (#503)"
```

---

### Task 3: Deterministic `agentic-69k` bench prompt

**Files:**
- Modify: `olmlx/bench/prompts.py` (add body constant + `BenchPrompt`)
- Test: `tests/test_bench_prompts.py` (add category + length assertions)

**Interfaces:**
- Consumes: `BenchPrompt`, `PROMPTS` (both existing in `olmlx/bench/prompts.py`).
- Produces: a `BenchPrompt(name="agentic-69k", category="agentic")` entry in `PROMPTS`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bench_prompts.py` inside the existing test class:

```python
    def test_agentic_prompt_exists_and_is_large(self):
        agentic = [p for p in PROMPTS if p.category == "agentic"]
        assert len(agentic) == 1
        p = agentic[0]
        assert p.name == "agentic-69k"
        # System message must carry tool definitions (the segment that creates
        # the message boundary #503 cares about).
        assert p.messages[0]["role"] == "system"
        assert "tool" in p.messages[0]["content"].lower()
        # Multi-turn agentic transcript.
        assert len(p.messages) >= 3
        # ~69k tokens at <= 4 chars/token -> >= 240k chars total content.
        total_chars = sum(len(m["content"]) for m in p.messages)
        assert total_chars >= 240_000, total_chars
        # Small decode budget so prefill dominates the sample.
        assert p.max_tokens <= 64
```

Also extend `test_categories_covered` to require the new category — add `"agentic"` to its expected tuple.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bench_prompts.py -k "agentic or categories_covered" -v`

Expected: FAIL — no `agentic` category / `agentic-69k` prompt exists.

- [ ] **Step 3: Add the prompt**

In `olmlx/bench/prompts.py`, after the `_LONG_CONTEXT_BODY` definition (~28), add a deterministic agentic code/log body sized for ~69k tokens (~276k chars at ~4 chars/token):

```python
# Deterministic agentic "repo context" block — a realistic mix of code, diffs,
# and tool output, repeated and truncated to a fixed length. Content variety
# does not change prefill cost; determinism across runs does. Sized so the full
# conversation is ~69k tokens (the agentic prefill case from #503), i.e.
# ~276k chars at the ~4 chars/token the long-context prompt already assumes.
_AGENTIC_CONTEXT_UNIT = (
    "def handle_request(req: Request) -> Response:\n"
    "    # Validate, dispatch, and record metrics for one inbound call.\n"
    "    ctx = build_context(req.headers, req.body)\n"
    "    if not ctx.authorized:\n"
    "        raise PermissionError(f'unauthorized: {ctx.principal!r}')\n"
    "    result = dispatch(ctx, req.route)\n"
    "    METRICS.observe('request', ctx.route, result.status)\n"
    "    return Response(status=result.status, body=result.payload)\n"
    "\n"
    "TOOL CALL: read_file(path='engine/router.py', start=1, end=40)\n"
    "TOOL RESULT: 40 lines returned; router dispatches on req.route via a\n"
    "  registry populated at import time; see register_route() below.\n"
    "\n"
)
_AGENTIC_BODY = (
    _AGENTIC_CONTEXT_UNIT * (276_000 // len(_AGENTIC_CONTEXT_UNIT) + 1)
)[:276_000]
```

Then append a `BenchPrompt` to the `PROMPTS` list (after the `long-context` entry, before the closing `]`):

```python
    BenchPrompt(
        name="agentic-69k",
        category="agentic",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a coding agent. You have these tools: "
                    "read_file(path, start, end), write_file(path, content), "
                    "run_bash(cmd), grep(pattern, path). Call one tool per turn "
                    "and wait for its result before the next.\n\n"
                    "Repository context follows:\n"
                    f"{_AGENTIC_BODY}"
                ),
            },
            {
                "role": "user",
                "content": "Trace how an inbound request reaches dispatch().",
            },
            {
                "role": "assistant",
                "content": (
                    "I'll start by reading the router.\n"
                    "TOOL CALL: read_file(path='engine/router.py', start=1, end=40)"
                ),
            },
            {
                "role": "user",
                "content": (
                    "TOOL RESULT: router dispatches on req.route via a registry "
                    "populated at import time. Now summarize the path in one line."
                ),
            },
        ],
        max_tokens=32,
    ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bench_prompts.py -v`

Expected: PASS (new test, updated `test_categories_covered`, and all existing prompt tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check olmlx/bench/prompts.py tests/test_bench_prompts.py
uv run ruff format olmlx/bench/prompts.py tests/test_bench_prompts.py
git add olmlx/bench/prompts.py tests/test_bench_prompts.py
git commit -m "feat(bench): deterministic agentic-69k prefill prompt (#503)"
```

---

### Task 4: Docs — CLAUDE.md note + findings-note scaffold

**Files:**
- Modify: `CLAUDE.md` (add a one-line invariant under the prefill/tracing notes)
- Create: `docs/superpowers/specs/2026-07-19-prefill-measurement-phase0-findings.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add the CLAUDE.md note**

In `CLAUDE.md`, add a short bold-lead paragraph near the other prefill invariants (e.g. after the "**MTP/EAGLE prefill chunking**" block):

```markdown
**Measured prefill wall-clock (#503 Phase 0)** — the `decode` span's `ttft_ns`
is a *real* time-to-first-token measurement (captured on the first stream
iteration in `generate_chat`), not the rate-reconstructed `prompt_eval_duration`
heuristic (which still backs the client-facing Ollama/bench field, unchanged).
`_drive_spec_prefill` records a per-lane breakdown (`target_lane_ns`,
`draft_lane_ns`, `covered_tokens`, `fresh_tokens`, `n_target_chunks`) into
`SpecDecoderBase._last_prefill_breakdown`, surfaced on the `spec.prefill` span.
Drive the `agentic-69k` bench prompt (`olmlx bench run --prompts agentic-69k`)
to profile the long agentic prefill case.
```

- [ ] **Step 2: Create the findings-note scaffold**

Create `docs/superpowers/specs/2026-07-19-prefill-measurement-phase0-findings.md`:

```markdown
# #503 Phase 0 — Prefill measurement findings

**Status:** awaiting numbers (fill in after running `olmlx bench run --prompts agentic-69k`).

## Setup
- Model: <agentic model, e.g. Qwen3-32B-4bit>
- Prompt: `agentic-69k` (~69k tokens, tool-defs system segment)
- Command: `olmlx bench run --prompts agentic-69k --model <model>`
- Read: server log `prefill Xs (fresh F/N tok, cache-covered C)` + `decode`/`spec.prefill` span attrs.

## Numbers (TO FILL)
| metric | value |
| --- | --- |
| measured prefill (ttft_ns) | |
| decode wall-clock | |
| prefill / total | |
| fresh vs cache-covered tokens | |
| target_lane_ns / draft_lane_ns (spec only) | |

## Decision (TO FILL)
- Is prefill the measured bottleneck on the agentic case? yes/no
- Build #503 proper / scope down / defer? <decision + rationale>
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-07-19-prefill-measurement-phase0-findings.md
git commit -m "docs: note measured prefill + phase0 findings scaffold (#503)"
```

---

## Self-Review

**Spec coverage:**
- Component 1 (measured TTFT + log) → Task 1. ✓
- Component 2 (per-lane breakdown at `_drive_spec_prefill`) → Task 2. ✓ (`_drive_segmented_prefill` in inference.py is the segmented/hybrid-GDN path; its breakdown is deferred — see note below.)
- Component 3 (`agentic-69k` bench prompt) → Task 3. ✓
- Decisions (a) client `prompt_eval_duration` untouched, (b) synthetic deterministic prompt → honored in Tasks 1 and 3. ✓
- Findings note (spec "Out of scope" mentions recording a findings note) → Task 4. ✓

**Known gap vs spec:** the spec's Component 2 also lists instrumenting `_drive_segmented_prefill` (inference.py, the non-spec hybrid-GDN segmented path). This plan instruments the speculative `_drive_spec_prefill` (which covers the draft-lane-cost question the issue centers on) but **defers** the inference.py segmented-drive breakdown, because the measured `ttft_ns` from Task 1 already gives the total prefill wall-clock for that path and the covered/fresh split is derivable from `token.prompt_tokens`. If the profiling run shows the segmented path needs a finer breakdown, add it as a follow-up task mirroring Task 2's accumulator pattern. This is a deliberate YAGNI scope trim, not an oversight.

**Placeholder scan:** the only "TO FILL" markers are inside the Task 4 findings-note *template* (intended — they are filled after the profiling run), not plan steps. No `TODO`/`TBD` in implementation steps.

**Type consistency:** `_last_prefill_breakdown: dict[str, int]` defined in Task 2 Step 3a, read in Step 3b, written by `_drive_spec_prefill(..., breakdown=...)` in Step 3c, passed by both callers in Step 3d — same name and type throughout. Breakdown keys (`covered_tokens`, `fresh_tokens`, `target_lane_ns`, `draft_lane_ns`, `n_target_chunks`, `n_lanes`) are identical in the writer (3c), the callers, and the Task 2 test.
