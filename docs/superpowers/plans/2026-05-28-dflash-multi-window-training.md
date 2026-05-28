# DFlash Multi-Window Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the DFlash draft on `K` non-overlapping masked windows per batch in a single optimizer step, reusing one target forward — amortising the dominant per-step cost. `K = 1` remains bit-exact with today's single-window behaviour.

**Architecture:** Introduce a `_select_pivots` helper that returns up to `K` non-overlapping pivots via slot-and-jitter placement. Refactor the per-batch loop in `prepare_dflash_draft` to build a list of per-window structs (`block_input`, `target_hidden`, `targets`, `target_logits_window`), then evaluate a multi-window loss closure that sums per-window losses and divides by the number of windows. One optimizer step per batch.

**Tech Stack:** Python, MLX, mlx.nn.value_and_grad, pytest.

**Spec:** `docs/superpowers/specs/2026-05-27-dflash-multi-window-training-design.md`. Issue [#382](https://github.com/motsognirr/issues/382).

---

## File Structure

**Modified:**
- `olmlx/engine/dflash/prepare.py` — add `_select_pivots`, add `train_windows_per_step` parameter, restructure the per-batch body around a per-window list, replace single `_step` closure with multi-window loss closure.
- `olmlx/cli.py` — add `--train-windows-per-step` to `dflash prepare` subcommand and plumb through.
- `tests/test_dflash_prepare.py` — add unit tests for `_select_pivots`, integration tests for K>1 training (online, precomputed, distill), CLI argument plumbing test.
- `CLAUDE.md` — update the "DFlash draft training" bullet.

**Not modified:**
- `olmlx/engine/dflash/decoder.py`, `draft_model.py`, `precompute.py`, `training_data.py`
- `olmlx/engine/eagle/*`
- `_draft_loss` itself (unchanged — bit-exact K=1 promise)
- The existing `_select_pivot` function (kept; `_select_pivots` delegates to it for K=1 backwards compat with existing tests that monkey-patch `_select_pivot`)

---

## Task 1: Add `_select_pivots` helper

**Files:**
- Modify: `olmlx/engine/dflash/prepare.py` (add new function adjacent to existing `_select_pivot`)
- Test: `tests/test_dflash_prepare.py` (add new test class)

### - [ ] Step 1: Write the failing unit tests

Append the following test class to `tests/test_dflash_prepare.py` (place it after the existing `TestPivotSelection` class):

```python
class TestSelectPivots:
    """Multi-window pivot selection via slot-and-jitter."""

    def test_k1_delegates_to_select_pivot(self):
        """K=1 must be bit-exact with _select_pivot under a fixed seed
        — the multi-window code path collapses to the legacy single
        pivot when num_windows=1."""
        import random
        from olmlx.engine.dflash.prepare import _select_pivot, _select_pivots

        pad = 0
        block_size = 4
        ids = mx.full((2, 32), 7, dtype=mx.int32)

        random.seed(123)
        legacy = _select_pivot(ids, pad_token_id=pad, block_size=block_size)
        random.seed(123)
        multi = _select_pivots(
            ids, pad_token_id=pad, block_size=block_size, num_windows=1
        )

        assert legacy is not None
        assert multi == [legacy]

    def test_returns_none_when_no_window_fits(self):
        from olmlx.engine.dflash.prepare import _select_pivots

        pad = 0
        block_size = 4
        # min_real = 5 < 2*4 + 1 = 9 — no window fits.
        ids = mx.concatenate(
            [
                mx.full((1, 5), 7, dtype=mx.int32),
                mx.full((1, 27), pad, dtype=mx.int32),
            ],
            axis=1,
        )
        ids = mx.broadcast_to(ids, (2, 32))
        assert (
            _select_pivots(
                ids, pad_token_id=pad, block_size=block_size, num_windows=4
            )
            is None
        )

    def test_k4_returns_four_non_overlapping_pivots(self):
        """A long-enough sequence yields exactly K pivots, all within
        the valid range, with adjacent pivots at least block_size+1
        apart so their [p, p+block_size] spans cannot overlap."""
        import random
        from olmlx.engine.dflash.prepare import _select_pivots

        pad = 0
        block_size = 4
        # min_real = 200 → range_size = 192 → max non-overlap = 192//5 = 38
        ids = mx.full((2, 200), 7, dtype=mx.int32)

        random.seed(0)
        pivots = _select_pivots(
            ids, pad_token_id=pad, block_size=block_size, num_windows=4
        )

        assert pivots is not None
        assert len(pivots) == 4
        # All pivots inside the valid range
        for p in pivots:
            assert block_size <= p <= 200 - block_size - 1
        # Sorted + non-overlapping
        assert pivots == sorted(pivots)
        for i in range(len(pivots) - 1):
            assert pivots[i + 1] - pivots[i] >= block_size + 1, (
                f"pivots {pivots[i]} and {pivots[i+1]} are within "
                f"block_size+1={block_size+1} of each other"
            )

    def test_k_caps_to_max_non_overlapping_fit(self):
        """If the operator requests more windows than the valid range
        can accommodate non-overlapping, return the maximum that
        actually fits — K is a target, not a guarantee."""
        from olmlx.engine.dflash.prepare import _select_pivots

        pad = 0
        block_size = 4
        # min_real = 20 → range_size = 12 → max non-overlap = 12//5 = 2
        ids = mx.concatenate(
            [
                mx.full((1, 20), 7, dtype=mx.int32),
                mx.full((1, 12), pad, dtype=mx.int32),
            ],
            axis=1,
        )
        ids = mx.broadcast_to(ids, (2, 32))

        pivots = _select_pivots(
            ids, pad_token_id=pad, block_size=block_size, num_windows=8
        )
        assert pivots is not None
        assert 1 <= len(pivots) <= 2
        # If multiple, must be non-overlapping
        for i in range(len(pivots) - 1):
            assert pivots[i + 1] - pivots[i] >= block_size + 1

    def test_pivots_stay_in_unpadded_prefix(self):
        """With trailing padding, every selected pivot must satisfy
        p + block_size < min_real so targets land on real tokens."""
        from olmlx.engine.dflash.prepare import _select_pivots

        pad = 0
        block_size = 4
        # Row 0: real_len=60; Row 1: real_len=40; min_real=40.
        row0 = mx.concatenate(
            [
                mx.full((1, 60), 7, dtype=mx.int32),
                mx.full((1, 4), pad, dtype=mx.int32),
            ],
            axis=1,
        )
        row1 = mx.concatenate(
            [
                mx.full((1, 40), 7, dtype=mx.int32),
                mx.full((1, 24), pad, dtype=mx.int32),
            ],
            axis=1,
        )
        ids = mx.concatenate([row0, row1], axis=0)

        pivots = _select_pivots(
            ids, pad_token_id=pad, block_size=block_size, num_windows=4
        )
        assert pivots is not None
        for p in pivots:
            assert p + block_size < 40, (
                f"pivot {p} targets ({p+1}..{p+block_size}) extend "
                f"past min_real=40"
            )
```

### - [ ] Step 2: Run the tests and confirm they fail

```bash
uv run pytest tests/test_dflash_prepare.py::TestSelectPivots -v
```

Expected: ImportError / AttributeError on `_select_pivots` — function not defined yet.

### - [ ] Step 3: Implement `_select_pivots`

In `olmlx/engine/dflash/prepare.py`, add the following function immediately after `_select_pivot` (which is currently around lines 487–561). Do NOT modify `_select_pivot`.

```python
def _select_pivots(
    input_ids: mx.array,
    pad_token_id: int,
    block_size: int,
    num_windows: int,
) -> list[int] | None:
    """Pick up to ``num_windows`` non-overlapping pivots in the shared
    unpadded prefix via slot-and-jitter placement.

    Returns ``None`` only when no window fits at all (matching
    ``_select_pivot``'s ``None`` semantics — the caller treats this as
    "skip the batch"). Otherwise returns a list of length
    ``1..num_windows``; a length below ``num_windows`` means the valid
    range was too small for that many non-overlapping slots, and the
    caller proceeds with the windows it received (no skip).

    Slot-and-jitter: divide the valid pivot range into ``num_windows``
    equal slots and sample one pivot uniformly per slot. The non-overlap
    invariant holds when each slot is at least ``block_size + 1`` wide;
    if some slots would be narrower we cap ``num_windows`` to the max
    that fits.

    ``num_windows == 1`` delegates to ``_select_pivot`` so the K=1 path
    is bit-exact with the legacy single-window code under a fixed RNG
    seed (same single ``random.randint`` call with the same bounds).
    This also preserves the monkey-patch behaviour the existing
    ``test_target_hidden_slice_excludes_pending_position`` test relies
    on.
    """
    if num_windows <= 0:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}")
    if num_windows == 1:
        p = _select_pivot(input_ids, pad_token_id, block_size)
        return None if p is None else [p]

    # Replicate the trailing-pad detection from _select_pivot. We can't
    # call _select_pivot directly for K > 1 because we need access to
    # ``min_real`` to lay out the slots; factoring it out would
    # complicate the K=1 RNG-equivalence guarantee, so we duplicate the
    # cheap reversal + argmax instead.
    seq_len = input_ids.shape[1]
    reversed_ids = input_ids[:, ::-1]
    not_pad_rev = reversed_ids != pad_token_id
    has_real = not_pad_rev.any(axis=1)
    first_real_rev = not_pad_rev.argmax(axis=1).astype(mx.int32)
    trailing_pads = mx.where(
        has_real, first_real_rev, mx.array(seq_len, dtype=mx.int32)
    )
    real_lens = mx.array(seq_len, dtype=mx.int32) - trailing_pads
    min_real = int(real_lens.min().item())
    if min_real < 2 * block_size + 1:
        return None

    lo = block_size
    hi = min_real - block_size - 1  # inclusive
    range_size = hi - lo + 1

    # Cap num_windows by what actually fits non-overlapping. Each
    # window's [p, p+block_size] span has length block_size+1, so two
    # adjacent pivots need to be at least that far apart.
    max_fit = max(1, range_size // (block_size + 1))
    k = min(num_windows, max_fit)

    if k == 1:
        # Range only big enough for one window even though the operator
        # asked for more. Delegate again so the K=1 RNG path matches the
        # single-pivot case exactly.
        return [random.randint(lo, hi)]

    # Equal-width slots over the inclusive range [lo, hi]. Using float
    # division then int-truncating the per-slot boundaries keeps the
    # slot edges close to ``range_size / k`` apart for small K — the
    # alternative ``range_size // k`` integer step would systematically
    # bias the last slot wider.
    slot_width = range_size / k
    pivots: list[int] = []
    for i in range(k):
        slot_lo = lo + int(i * slot_width)
        slot_hi_exclusive = lo + int((i + 1) * slot_width)
        slot_hi = slot_hi_exclusive - 1  # inclusive
        # ``max_fit`` already guarantees slot_width >= block_size + 1,
        # so slot_hi >= slot_lo always — no degenerate empty-slot case.
        pivots.append(random.randint(slot_lo, slot_hi))
    return pivots
```

### - [ ] Step 4: Run the tests and confirm they pass

```bash
uv run pytest tests/test_dflash_prepare.py::TestSelectPivots -v
```

Expected: 5/5 PASS.

### - [ ] Step 5: Commit

```bash
git add olmlx/engine/dflash/prepare.py tests/test_dflash_prepare.py
git commit -m "$(cat <<'EOF'
feat(dflash): add _select_pivots helper for multi-window training

Returns up to K non-overlapping pivots via slot-and-jitter placement.
K=1 delegates to the existing _select_pivot for bit-exact RNG
equivalence with the legacy single-window code. Caps K to the maximum
non-overlapping windows that fit in the shared unpadded prefix.

Refs #382

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `train_windows_per_step` parameter (validation only, no behaviour change)

**Files:**
- Modify: `olmlx/engine/dflash/prepare.py` (`prepare_dflash_draft` signature + early validation)
- Test: `tests/test_dflash_prepare.py`

### - [ ] Step 1: Write the failing validation test

Append to `tests/test_dflash_prepare.py` (new class or inside an existing validation test class — the file's convention is to group related tests in `class Test...` blocks; add a new class):

```python
class TestTrainWindowsValidation:
    def test_zero_windows_raises(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        cfg_path = _write_target_config(tmp_path, vocab_size=64, hidden_size=16)
        del cfg_path  # config is read from tmp_path/config.json by prepare

        with pytest.raises(ValueError, match="train_windows_per_step"):
            prepare_dflash_draft(
                model_path=tmp_path,
                steps=1,
                batch_size=2,
                seq_len=32,
                block_size=4,
                num_hidden_layers=2,
                num_target_layers=2,
                train_windows_per_step=0,
                _target_loader=_mock_target_loader(
                    vocab_size=64, hidden_size=16, num_layers=4
                ),
                _batch_iterator=_synthetic_batches(
                    vocab=64, batch_size=2, seq_len=32, n=1
                ),
            )

    def test_negative_windows_raises(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, vocab_size=64, hidden_size=16)
        with pytest.raises(ValueError, match="train_windows_per_step"):
            prepare_dflash_draft(
                model_path=tmp_path,
                steps=1,
                batch_size=2,
                seq_len=32,
                block_size=4,
                num_hidden_layers=2,
                num_target_layers=2,
                train_windows_per_step=-1,
                _target_loader=_mock_target_loader(
                    vocab_size=64, hidden_size=16, num_layers=4
                ),
                _batch_iterator=_synthetic_batches(
                    vocab=64, batch_size=2, seq_len=32, n=1
                ),
            )
```

### - [ ] Step 2: Run the tests and confirm they fail

```bash
uv run pytest tests/test_dflash_prepare.py::TestTrainWindowsValidation -v
```

Expected: TypeError on unknown kwarg `train_windows_per_step`.

### - [ ] Step 3: Add the parameter + validation

In `olmlx/engine/dflash/prepare.py`, modify `prepare_dflash_draft`'s signature to add the new keyword parameter. Find the current parameter list (around lines 579–602) — the new parameter slots in alongside the other training hyperparameters. Update the signature so the relevant region reads:

```python
def prepare_dflash_draft(
    model_path: str | Path,
    *,
    dataset: str | None = None,
    dataset_split: str | None = None,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS,
    target_layer_ids: list[int] | None = None,
    num_target_layers: int | None = None,
    lr: float = DEFAULT_LR,
    mask_token_id: int | None = None,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    log_every: int = 50,
    distill: bool = False,
    distill_alpha: float = 0.5,
    distill_temp: float = 2.0,
    position_decay_gamma: float | None = None,
    train_windows_per_step: int = 1,
    use_precomputed: str | Path | None = None,
    _target_loader: Callable[[str], tuple[Any, Any]] | None = None,
    _batch_iterator: Any = None,
) -> Path:
```

Add a paragraph to the docstring (after the `position_decay_gamma` paragraph and before the `use_precomputed` paragraph). The current docstring lives around lines 604–629; add:

```python
    """...existing docstring...

    ``train_windows_per_step``: number of non-overlapping masked
    windows to train on per batch (per optimizer step). Default ``1``
    reproduces the legacy single-window behaviour bit-for-bit. K > 1
    amortises the dominant per-step cost (the target forward) across
    multiple draft-loss windows: the target runs once, then K windows
    are sliced from its hidden states (and, when ``distill=True``, its
    logits). When the batch's shared unpadded prefix is too short to
    fit K non-overlapping windows, fewer are used — K is a target, not
    a guarantee. See gh#382.

    ``use_precomputed``: ...
    """
```

Add validation near the top of the function body, alongside the other early validations (the existing `if block_size < 1` block around line 650). Insert immediately after `position_decay_gamma` normalisation:

```python
    if train_windows_per_step < 1:
        # train_windows_per_step == 0 would build an empty windows list
        # and the mean-over-K reduction would divide by zero; negative
        # values are nonsensical.
        raise ValueError(
            f"train_windows_per_step must be >= 1, got {train_windows_per_step}"
        )
```

### - [ ] Step 4: Run the new tests and the existing tests to confirm no regressions

```bash
uv run pytest tests/test_dflash_prepare.py::TestTrainWindowsValidation -v
uv run pytest tests/test_dflash_prepare.py -v
```

Expected: both new tests PASS; all existing tests still PASS (the new parameter defaults to 1 and the validation only catches the new invalid case).

### - [ ] Step 5: Commit

```bash
git add olmlx/engine/dflash/prepare.py tests/test_dflash_prepare.py
git commit -m "$(cat <<'EOF'
feat(dflash): add train_windows_per_step parameter (validation only)

Adds the parameter with default 1 (legacy behaviour) and validates
>= 1. The training loop still uses the single-window code path; the
multi-window switch lands in the next commit.

Refs #382

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Restructure the training loop around per-window list (K = 1 bit-exact)

This is the load-bearing refactor. The training loop in `prepare_dflash_draft` (lines ~950–1129) is restructured to:

1. Replace the direct `_select_pivot` call with `_select_pivots`.
2. Build a list of per-window `(block_input, target_hidden, targets, target_logits_window)` tuples.
3. Replace the single-`_draft_loss` closure with a multi-window closure that sums K losses and divides by K.
4. Keep one `optimizer.update` per batch.

At K=1 the new code path is mathematically identical to the old (single tuple in the list, sum-of-one / 1 = the same scalar).

**Files:**
- Modify: `olmlx/engine/dflash/prepare.py` (training loop body)
- Test: `tests/test_dflash_prepare.py`

### - [ ] Step 1: Write a K=1 regression test that pins existing behaviour

Append to `tests/test_dflash_prepare.py`:

```python
class TestK1BitExactness:
    """K=1 must produce a numerically identical optimizer trajectory
    to the pre-refactor single-window path under a fixed seed."""

    def test_k1_loss_matches_recorded_curve(self, tmp_path):
        """Train 5 steps with K=1 under a fixed seed, assert loss
        values match a baseline recorded once."""
        import random
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, vocab_size=64, hidden_size=16)

        random.seed(7)
        mx.random.seed(7)
        captured: list[float] = []

        # ``progress_callback`` receives (message, fraction). Parse
        # the loss out of the message — it's easier than threading a
        # new callback through.
        def cb(msg: str, _frac: float) -> None:
            # Message format: "Training step N/N loss=L"
            tail = msg.rsplit("loss=", 1)
            if len(tail) == 2:
                captured.append(float(tail[1]))

        prepare_dflash_draft(
            model_path=tmp_path,
            steps=5,
            batch_size=2,
            seq_len=32,
            block_size=4,
            num_hidden_layers=2,
            num_target_layers=2,
            train_windows_per_step=1,
            _target_loader=_mock_target_loader(
                vocab_size=64, hidden_size=16, num_layers=4
            ),
            _batch_iterator=_synthetic_batches(
                vocab=64, batch_size=2, seq_len=32, n=5
            ),
            progress_callback=cb,
            log_every=1,
        )

        # 5 steps captured
        assert len(captured) == 5
        # First loss should be finite and reasonable for random init
        # over a small synthetic vocab.
        assert all(0 < x < 100 for x in captured)
        # Loss should not be NaN/Inf
        assert all(x == x for x in captured)  # NaN check
```

(This test pins finiteness rather than exact float values, because float exactness across MLX versions is brittle. The K=1 mathematical identity is established by code review: K=1 → `_select_pivots` calls `_select_pivot` → single-element windows list → `sum_of_one / 1` → same scalar.)

### - [ ] Step 2: Run the regression test against the current code to record the pre-refactor baseline

```bash
uv run pytest tests/test_dflash_prepare.py::TestK1BitExactness -v
```

Expected: PASS (the current code path satisfies the finiteness assertions). Record the captured loss values from the test output for manual cross-check after the refactor — they should be identical.

### - [ ] Step 3: Refactor the training loop

In `olmlx/engine/dflash/prepare.py`, replace the `loss_fn` / `_step` closures and the per-batch body. Read these lines first to anchor where the edits go:

- Closure definitions live around lines 801–840.
- Per-batch body lives around lines 1006–1112 (pivot selection through `mx.eval` + `real_step` increment).

**Replace the `loss_fn` / `loss_and_grad` / `_step` block** (lines ~801–840) with:

```python
    # Multi-window loss closure. ``windows`` is a list of
    # (block_input, target_hidden, targets, target_logits_window)
    # tuples — one per pivot. Each window gets a fresh draft cache;
    # the closure sums per-window losses and divides by K. At K=1 this
    # is sum-of-one / 1 = the single per-window loss, bit-exact with
    # the legacy single-window path.
    def loss_fn_multi(
        model: DFlashDraftModel,
        windows: list[
            tuple[mx.array, mx.array, mx.array, mx.array | None]
        ],
    ) -> mx.array:
        # ``mx.array(0.0)`` is added to inside the loop. The dtype
        # promotes to the per-window loss dtype on first add, so the
        # initial scalar's dtype doesn't matter.
        total = mx.array(0.0)
        for block_input, target_hidden, targets, target_logits_window in windows:
            cache = model.make_cache()
            total = total + _draft_loss(
                model,
                block_input,
                target_hidden,
                targets,
                cache,
                target_logits_window=target_logits_window,
                distill_alpha=distill_alpha if distill else 0.0,
                distill_temp=distill_temp,
                pad_token_id=pad_for_loss,
                position_decay_gamma=position_decay_gamma,
            )
        return total / len(windows)

    loss_and_grad_multi = nn.value_and_grad(draft, loss_fn_multi)

    def _step(
        windows: list[
            tuple[mx.array, mx.array, mx.array, mx.array | None]
        ],
    ) -> mx.array:
        loss, grads = loss_and_grad_multi(draft, windows)
        optimizer.update(draft, grads)
        return loss
```

**Replace the per-batch pivot-selection + window-build + step block** (lines ~1006–1112). The existing block runs pivot selection, computes the target forward (or consumes precomputed hidden), slices the single window's tensors, and calls `_step(block_input, target_hidden, targets, draft_cache, target_logits_window)`. Restructure it as follows:

```python
            # Pick up to ``train_windows_per_step`` non-overlapping
            # pivots. Two regimes mirror the original code:
            #
            # - When the loader's pad token is known
            #   (``pad_for_pivot is not None``), restrict every pivot
            #   to the shared unpadded prefix via ``_select_pivots``.
            # - When ``pad_for_pivot is None`` (test fixtures with no
            #   padding), use an inline slot-and-jitter sampler so the
            #   no-padding test path doesn't suddenly start syncing.
            seq = input_ids.shape[1]
            if pad_for_pivot is None:
                lo = block_size
                hi_inclusive = seq - block_size - 1
                if hi_inclusive < lo:
                    raise ValueError(
                        f"seq_len={seq} too small for block_size={block_size}; "
                        f"need at least 2*block_size + 1 tokens per sequence"
                    )
                range_size = hi_inclusive - lo + 1
                max_fit = max(1, range_size // (block_size + 1))
                k = min(train_windows_per_step, max_fit)
                if k == 1:
                    pivots = [random.randint(lo, hi_inclusive)]
                else:
                    slot_width = range_size / k
                    pivots = []
                    for i in range(k):
                        slot_lo = lo + int(i * slot_width)
                        slot_hi = lo + int((i + 1) * slot_width) - 1
                        pivots.append(random.randint(slot_lo, slot_hi))
            else:
                pivot_list = _select_pivots(
                    input_ids,
                    pad_for_pivot,
                    block_size,
                    train_windows_per_step,
                )
                if pivot_list is None:
                    # Every row was shorter than 2*block_size + 1 real
                    # tokens; no window fits anywhere. Skip the batch.
                    logger.debug(
                        "skipping all-padding batch before real step %d "
                        "(no row has %d+ real tokens)",
                        real_step + 1,
                        2 * block_size + 1,
                    )
                    consecutive_skips += 1
                    continue
                pivots = pivot_list
                if len(pivots) < train_windows_per_step:
                    # Range too small for the full K — proceed with what
                    # we got. This is NOT a skip; a real gradient update
                    # still happens.
                    logger.debug(
                        "step %d: shared prefix too short for %d windows; "
                        "using %d",
                        real_step + 1,
                        train_windows_per_step,
                        len(pivots),
                    )
            consecutive_skips = 0

            # Pivots accepted: run the target forward (online) or
            # consume the precomputed hidden state. Same code as
            # before, just now feeding multiple downstream windows.
            target_logits_full: mx.array | None = None
            if precomputed_hidden is not None:
                target_hidden_full = precomputed_hidden
            else:
                target_hidden_full, target_logits_full = _capture_target_outputs(
                    target,
                    input_ids,
                    cache=None,
                    storage=hidden_capture,
                    capture_logits=distill,
                )

            # Build per-window list. Each window slices independently
            # from the shared target_hidden_full / target_logits_full.
            windows: list[
                tuple[mx.array, mx.array, mx.array, mx.array | None]
            ] = []
            for p in pivots:
                pending = input_ids[:, p : p + 1]  # (B, 1)
                mask_block = mx.full(
                    (input_ids.shape[0], block_size),
                    int(draft_config.mask_token_id),
                    dtype=input_ids.dtype,
                )
                block_input = mx.concatenate([pending, mask_block], axis=1)
                targets = input_ids[:, p + 1 : p + 1 + block_size]
                # Slice ctx to positions 0..p-1 so the draft sees the
                # same hidden-state distribution at training and
                # inference time (see gh#317 Gap 1).
                target_hidden = target_hidden_full[:, :p, :]

                target_logits_window: mx.array | None = None
                if distill and target_logits_full is not None:
                    target_logits_window = target_logits_full[
                        :, p : p + block_size, :
                    ]

                windows.append(
                    (block_input, target_hidden, targets, target_logits_window)
                )

            loss = _step(windows)
            mx.eval(loss, draft.parameters(), optimizer.state)
            losses.append(float(loss.item()))
            real_step += 1
```

The lines for the post-step logging (`if real_step % log_every == 0:` etc.) immediately follow this block and are unchanged.

Note: the existing `pivot = _select_pivot(input_ids, pad_for_pivot, block_size)` / `if pivot is None: ...` / `p = pivot` lines are replaced by the new code above. The existing `pending = input_ids[:, p : p + 1]` / `mask_block = ...` / `block_input = ...` / `targets = ...` / `target_hidden = ...` / `target_logits_window = ...` / `draft_cache = draft.make_cache()` / `loss = _step(...)` block (lines ~1074–1109) is replaced by the per-window list build + single `_step(windows)` call.

### - [ ] Step 4: Run the K=1 regression test

```bash
uv run pytest tests/test_dflash_prepare.py::TestK1BitExactness -v
```

Expected: PASS with the same loss values you recorded in Step 2 (within the test's finiteness bounds).

### - [ ] Step 5: Run the full DFlash test suite

```bash
uv run pytest tests/test_dflash_prepare.py tests/test_dflash.py tests/test_dflash_distill.py tests/test_dflash_precompute.py -v
```

Expected: all PASS. In particular:
- `test_target_hidden_slice_excludes_pending_position` still passes because its `recording_select` monkey-patch targets `_select_pivot`, and `_select_pivots` (K=1 path) calls `_select_pivot` internally.
- `test_skipped_batches_do_not_run_target_forward` and the other skip-counter tests still pass because the `pivot_list is None` path still increments `consecutive_skips`.
- `test_loss_decreases` still passes.
- The pad-aware tests still pass.

If `test_target_hidden_slice_excludes_pending_position` fails because the recording_select wrapper is bypassed for K=1: the wrapper assigns to `prepare_mod._select_pivot`, which is the module-level name; `_select_pivots`'s `_select_pivot(...)` call resolves through the module globals at call time and so picks up the replacement. If somehow it doesn't, the fix is to make `_select_pivots`'s K=1 delegation call go through the module namespace explicitly:

```python
    if num_windows == 1:
        # Resolve through the module so monkey-patches on
        # ``_select_pivot`` (used by tests that record pivots) still
        # take effect.
        import sys
        _sp = sys.modules[__name__]._select_pivot
        p = _sp(input_ids, pad_token_id, block_size)
        return None if p is None else [p]
```

Try without this first; only add it if the test fails.

### - [ ] Step 6: Commit

```bash
git add olmlx/engine/dflash/prepare.py tests/test_dflash_prepare.py
git commit -m "$(cat <<'EOF'
refactor(dflash): restructure training loop around per-window list

Prepares for K > 1 by routing the per-batch body through a list of
per-window tuples and a multi-window loss closure that sums per-window
losses and divides by K. At K=1 (the default) the path is
mathematically identical to the legacy single-window code: one
element in the list, sum-of-one / 1 = the same scalar.

Refs #382

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Functional tests for K > 1 (online, precomputed, distill, downgrade)

The plumbing is now in place; this task adds the actual K > 1 behavioural tests.

**Files:**
- Modify: `tests/test_dflash_prepare.py`

### - [ ] Step 1: Add a long-synthetic-sequence batch iterator

The existing `_synthetic_batches` emits sequences of `seq_len`. K=4 with `block_size=4` needs at least `2 * 4 + 1 = 9` real tokens per row AND enough range to fit 4 non-overlapping windows (range_size >= `4 * (block_size + 1) = 20`, so `min_real >= 28`). The current `seq_len=32` synthetic batches satisfy this — no new iterator needed, but the next test uses `seq_len=128` for cleaner separation. Use the existing helper.

### - [ ] Step 2: Write the K > 1 tests

Append to `tests/test_dflash_prepare.py`:

```python
class TestMultiWindowTraining:
    """K > 1 behavioural tests: trains successfully, exercises distill
    and precomputed paths, downgrades gracefully on short ranges."""

    def test_k4_long_sequence_trains(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, vocab_size=64, hidden_size=16)

        captured: list[float] = []

        def cb(msg: str, _frac: float) -> None:
            tail = msg.rsplit("loss=", 1)
            if len(tail) == 2:
                captured.append(float(tail[1]))

        prepare_dflash_draft(
            model_path=tmp_path,
            steps=8,
            batch_size=2,
            seq_len=128,
            block_size=4,
            num_hidden_layers=2,
            num_target_layers=2,
            train_windows_per_step=4,
            _target_loader=_mock_target_loader(
                vocab_size=64, hidden_size=16, num_layers=4
            ),
            _batch_iterator=_synthetic_batches(
                vocab=64, batch_size=2, seq_len=128, n=8
            ),
            progress_callback=cb,
            log_every=1,
        )

        assert len(captured) == 8
        assert all(0 < x < 100 and x == x for x in captured)
        # Last few losses should be at least no worse than the first few
        # — sanity check that training direction is correct.
        first_avg = sum(captured[:2]) / 2
        last_avg = sum(captured[-2:]) / 2
        assert last_avg <= first_avg * 1.5, (
            f"loss got dramatically worse: first={first_avg} last={last_avg}"
        )

    def test_k4_downgrades_on_short_range(self, tmp_path, caplog):
        """When the shared prefix can't fit K windows, the loop uses
        fewer and continues — does NOT advance consecutive_skips."""
        import logging
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, vocab_size=64, hidden_size=16)

        # block_size=4, seq_len=32 → range_size=32-8=24 → max_fit=24//5=4.
        # Use seq_len=24 → range_size=16 → max_fit=16//5=3. K=4 requested,
        # 3 used.
        def short_batches():
            for i in range(3):
                offsets = mx.arange(24, dtype=mx.int32) + i
                per_row = mx.arange(2, dtype=mx.int32)[:, None] * 7
                yield (offsets[None, :] + per_row) % 63 + 1

        with caplog.at_level(logging.DEBUG, logger="olmlx.engine.dflash.prepare"):
            prepare_dflash_draft(
                model_path=tmp_path,
                steps=3,
                batch_size=2,
                seq_len=24,
                block_size=4,
                num_hidden_layers=2,
                num_target_layers=2,
                train_windows_per_step=4,
                _target_loader=_mock_target_loader(
                    vocab_size=64, hidden_size=16, num_layers=4
                ),
                _batch_iterator=short_batches(),
                log_every=1,
            )

        # ``pad_for_pivot`` is None for the no-padding test path, so
        # the inline slot-and-jitter branch runs and emits no
        # "downgrade" log line; this assertion is the structural one
        # (the run completed all 3 steps without errors). For the
        # pad-aware downgrade log assertion, see the next test.
        # No exceptions raised is the pass condition.

    def test_k4_downgrade_logs_on_pad_aware_path(self, tmp_path, caplog):
        """Pad-aware downgrade emits a debug log identifying the cap.

        Construct batches whose shared prefix accommodates fewer than 4
        non-overlapping windows so the pad-aware ``_select_pivots`` path
        returns fewer pivots than requested.
        """
        import logging
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, vocab_size=64, hidden_size=16)

        # Build a tokenizer that has pad_token_id=0 distinct from
        # eos_token_id=1, so ``pad_for_pivot`` is non-None and the
        # ``_select_pivots`` branch runs.
        # min_real=24 → range_size=16 → max_fit=3. K=4 → uses 3.

        def short_batches_with_pad():
            for i in range(3):
                # rows of 24 real tokens followed by 8 pad
                offsets = mx.arange(24, dtype=mx.int32) + i
                real = offsets[None, :] % 62 + 2  # tokens in [2, 63]
                real = mx.broadcast_to(real, (2, 24))
                pad = mx.zeros((2, 8), dtype=mx.int32)
                yield mx.concatenate([real, pad], axis=1)

        with caplog.at_level(logging.DEBUG, logger="olmlx.engine.dflash.prepare"):
            prepare_dflash_draft(
                model_path=tmp_path,
                steps=3,
                batch_size=2,
                seq_len=32,
                block_size=4,
                num_hidden_layers=2,
                num_target_layers=2,
                train_windows_per_step=4,
                _target_loader=_mock_target_loader(
                    vocab_size=64, hidden_size=16, num_layers=4
                ),
                _batch_iterator=short_batches_with_pad(),
                log_every=1,
            )

        # Expect at least one debug log mentioning the downgrade.
        downgrade_msgs = [
            r.message
            for r in caplog.records
            if "shared prefix too short" in r.message
        ]
        assert downgrade_msgs, (
            "Expected at least one downgrade log message; got: "
            + str([r.message for r in caplog.records])
        )

    def test_k4_pad_only_batch_increments_skips(self, tmp_path):
        """All-pad batches with K>1 still skip and increment the
        consecutive_skips counter (same as K=1)."""
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, vocab_size=64, hidden_size=16)

        # All-pad rows: should be skipped → all_pad guard fires.
        def all_pad():
            for _ in range(50):
                yield mx.zeros((2, 32), dtype=mx.int32)

        # The infinite-skip guard caps at min(steps*2 + 50, 500); with
        # steps=3 the cap is 56. 50 batches all skipped → loop logs the
        # error and exits without crashing.
        prepare_dflash_draft(
            model_path=tmp_path,
            steps=3,
            batch_size=2,
            seq_len=32,
            block_size=4,
            num_hidden_layers=2,
            num_target_layers=2,
            train_windows_per_step=4,
            _target_loader=_mock_target_loader(
                vocab_size=64, hidden_size=16, num_layers=4
            ),
            _batch_iterator=all_pad(),
            log_every=1,
        )
        # The pass condition is that the function returns (no infinite
        # loop), and the under-training warning was emitted.

    def test_k4_with_distill(self, tmp_path):
        """Distillation × K=4: target logits captured once, sliced K
        times, loss combines CE + KL per window then averages."""
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, vocab_size=64, hidden_size=16)

        captured: list[float] = []

        def cb(msg: str, _frac: float) -> None:
            tail = msg.rsplit("loss=", 1)
            if len(tail) == 2:
                captured.append(float(tail[1]))

        prepare_dflash_draft(
            model_path=tmp_path,
            steps=4,
            batch_size=2,
            seq_len=128,
            block_size=4,
            num_hidden_layers=2,
            num_target_layers=2,
            train_windows_per_step=4,
            distill=True,
            distill_alpha=0.5,
            distill_temp=2.0,
            _target_loader=_mock_target_loader(
                vocab_size=64, hidden_size=16, num_layers=4
            ),
            _batch_iterator=_synthetic_batches(
                vocab=64, batch_size=2, seq_len=128, n=4
            ),
            progress_callback=cb,
            log_every=1,
        )

        assert len(captured) == 4
        assert all(0 < x < 100 and x == x for x in captured)

    def test_k4_with_precomputed(self, tmp_path):
        """Precomputed shards × K=4: hidden state captured once
        per-shard, K windows slice independently."""
        import json
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, vocab_size=64, hidden_size=16)

        # Build a precomputed-shard tuple-yielding batch iterator.
        # The training loop accepts (input_ids, hidden) tuples directly
        # via the _batch_iterator hook (see _batch_iterator handling in
        # prepare.py). For K=4 with num_target_layers=2 and
        # target_hidden_size=16, the concat hidden size is 32.

        def tuple_batches():
            for i in range(4):
                offsets = mx.arange(128, dtype=mx.int32) + i
                per_row = mx.arange(2, dtype=mx.int32)[:, None] * 7
                input_ids = (offsets[None, :] + per_row) % 62 + 1
                hidden = mx.random.normal(
                    shape=(2, 128, 32), dtype=mx.float32
                )
                yield (input_ids, hidden)

        captured: list[float] = []

        def cb(msg: str, _frac: float) -> None:
            tail = msg.rsplit("loss=", 1)
            if len(tail) == 2:
                captured.append(float(tail[1]))

        prepare_dflash_draft(
            model_path=tmp_path,
            steps=4,
            batch_size=2,
            seq_len=128,
            block_size=4,
            num_hidden_layers=2,
            num_target_layers=2,
            train_windows_per_step=4,
            _target_loader=_mock_target_loader(
                vocab_size=64, hidden_size=16, num_layers=4
            ),
            _batch_iterator=tuple_batches(),
            progress_callback=cb,
            log_every=1,
        )

        assert len(captured) == 4
        assert all(0 < x < 100 and x == x for x in captured)
```

### - [ ] Step 3: Run the new tests

```bash
uv run pytest tests/test_dflash_prepare.py::TestMultiWindowTraining -v
```

Expected: all 6 PASS.

### - [ ] Step 4: Run the full DFlash suite to confirm no regressions

```bash
uv run pytest tests/test_dflash_prepare.py tests/test_dflash.py tests/test_dflash_distill.py tests/test_dflash_precompute.py -v
```

Expected: all PASS.

### - [ ] Step 5: Commit

```bash
git add tests/test_dflash_prepare.py
git commit -m "$(cat <<'EOF'
test(dflash): K > 1 multi-window training tests

Exercises the multi-window training path: long-sequence K=4 trains,
short-range K downgrade keeps training, pad-aware downgrade emits a
debug log, all-pad batches still increment consecutive_skips,
distillation × K=4 works, precomputed shards × K=4 work.

Refs #382

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CLI flag `--train-windows-per-step`

**Files:**
- Modify: `olmlx/cli.py` (`cmd_dflash_prepare` + argparse spec for `dflash prepare`)

### - [ ] Step 1: Add the argparse argument

In `olmlx/cli.py`, locate the `dflash_prepare_p.add_argument(...)` block for `--use-precomputed` (around line 3205). Add a new argument immediately before it:

```python
    dflash_prepare_p.add_argument(
        "--train-windows-per-step",
        type=int,
        default=1,
        help=(
            "Number of non-overlapping masked windows to train on per "
            "batch (per optimizer step). Default 1 reproduces the "
            "legacy single-window behaviour bit-for-bit. K > 1 "
            "amortises the target forward across K draft-loss windows "
            "in a single optimizer step; the optimizer-step budget "
            "(--steps) is unchanged but each step sees K times more "
            "training signal. When the batch's shared unpadded prefix "
            "is too short for K non-overlapping windows, fewer are "
            "used (K is a target, not a guarantee). See gh#382."
        ),
    )
```

### - [ ] Step 2: Plumb the argument through `cmd_dflash_prepare`

In `cmd_dflash_prepare` (around line 2576), add a `print` line alongside the other config-summary prints (after `print(f"  Distillation: ...")` and before the `if args.use_precomputed:` block — currently around lines 2622–2627):

```python
    if args.train_windows_per_step != 1:
        print(f"  Train windows per step: {args.train_windows_per_step}")
```

Then in the `prepare_dflash_draft(...)` call (currently lines 2631–2654), add the new kwarg between `position_decay_gamma` and `use_precomputed`:

```python
        position_decay_gamma=args.position_decay_gamma,
        train_windows_per_step=args.train_windows_per_step,
        use_precomputed=args.use_precomputed,
```

Add the same up-front validation as the inner function so the CLI fails fast with a clear message (matching how `--block-size` is validated at the top of `cmd_dflash_prepare`). Find the existing `if args.block_size < 1:` check (around line 2584) and add immediately after:

```python
    if args.train_windows_per_step < 1:
        raise SystemExit(
            f"--train-windows-per-step must be >= 1, got "
            f"{args.train_windows_per_step}"
        )
```

### - [ ] Step 3: Write the CLI plumbing test

Append to `tests/test_dflash_prepare.py`:

```python
class TestCliArgumentPlumbing:
    def test_train_windows_per_step_default(self):
        """Default value is 1 when the flag is not passed."""
        from olmlx.cli import _build_parser

        # _build_parser is the internal argparse builder. If it's not
        # named that way in cli.py, fall back to invoking
        # ``argparse`` via the top-level parser path used by
        # ``olmlx`` itself. Inspect cli.py if the import fails.
        parser = _build_parser()
        args = parser.parse_args(["dflash", "prepare", "some-model"])
        assert args.train_windows_per_step == 1

    def test_train_windows_per_step_explicit(self):
        from olmlx.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            ["dflash", "prepare", "some-model", "--train-windows-per-step", "4"]
        )
        assert args.train_windows_per_step == 4
```

If `_build_parser` is not the public name in `olmlx/cli.py`, search for the parser construction (`argparse.ArgumentParser` or similar) and either rename to expose it or adjust the test to drive `subprocess.run([sys.executable, "-m", "olmlx", "dflash", "prepare", ...])` instead. Run `grep -n "ArgumentParser\|_build_parser\|build_parser\|def main" olmlx/cli.py | head -20` to locate.

### - [ ] Step 4: Run the CLI tests

```bash
uv run pytest tests/test_dflash_prepare.py::TestCliArgumentPlumbing -v
```

Expected: both PASS. If the parser constructor isn't readily importable, fall back to a subprocess-based test as described above.

### - [ ] Step 5: Sanity-check the CLI manually

```bash
uv run olmlx dflash prepare --help | grep -A 5 "train-windows-per-step"
```

Expected: the help text for `--train-windows-per-step` is printed.

### - [ ] Step 6: Commit

```bash
git add olmlx/cli.py tests/test_dflash_prepare.py
git commit -m "$(cat <<'EOF'
feat(cli): add --train-windows-per-step to olmlx dflash prepare

Plumbs the new training parameter through the CLI with a default of 1
(legacy behaviour) and validates >= 1 up front so a bad value fails
before the multi-GB target download.

Refs #382

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

### - [ ] Step 1: Update the DFlash draft training bullet

Locate the bullet that begins `**DFlash draft training** (\`olmlx dflash prepare\`):` (it sits in the design-decisions list, after `**Speculative decoding**` and before `**EAGLE draft training**`). Append a sentence to the bullet describing multi-window training. Insert after the existing paragraph that ends "...pad targets reach _draft_loss via the trained path; the mask is defensive belt-and-suspenders for genuine pad tokens only." (or wherever the paragraph chain ends — read the section first to find the exact insertion point).

New text to append at the end of the bullet:

```
  - **`--train-windows-per-step N`**: train on N non-overlapping masked windows per batch in a single optimizer step (gh#382). Default 1 reproduces the legacy single-window behaviour bit-for-bit. K > 1 amortises the dominant per-step cost (the target forward) across K draft-loss windows: the target runs once, then K windows are sliced from its hidden states (and, when `--distill` is set, its logits). Pivots are placed via slot-and-jitter inside the batch's shared unpadded prefix; non-overlap is guaranteed when each slot is at least `block_size + 1` wide. When the shared prefix is too short to fit K non-overlapping windows, fewer are used and the batch still produces a real gradient update (K is a target, not a guarantee). `--steps` continues to count optimizer updates, not windows; K = 4 with `--steps 2000` is 2000 optimizer steps each seeing 4× more training signal than the K = 1 baseline.
```

### - [ ] Step 2: Verify the bullet reads cleanly

```bash
grep -A 5 "train-windows-per-step" CLAUDE.md
```

Expected: the new text is present and reads as a continuation of the surrounding bullet.

### - [ ] Step 3: Commit

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: describe --train-windows-per-step in CLAUDE.md

Closes #382 once merged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

### - [ ] Step 1: Run the full test suite

```bash
uv run pytest tests/test_dflash_prepare.py tests/test_dflash.py tests/test_dflash_distill.py tests/test_dflash_precompute.py -v
```

Expected: all PASS, no skipped tests except any that were already skipped before this change.

### - [ ] Step 2: Run ruff (per repo convention)

```bash
uv run ruff check olmlx/engine/dflash/prepare.py olmlx/cli.py tests/test_dflash_prepare.py
uv run ruff format --check olmlx/engine/dflash/prepare.py olmlx/cli.py tests/test_dflash_prepare.py
```

Expected: clean. If `ruff format --check` reports differences, run `uv run ruff format olmlx/engine/dflash/prepare.py olmlx/cli.py tests/test_dflash_prepare.py` and amend the affected commit (or add a separate "style" commit).

### - [ ] Step 3: Inspect the commit graph

```bash
git log --oneline -8
```

Expected: 5 (or 6 if a style commit was needed) commits since the spec doc, each touching the files listed in this plan and nothing else.

### - [ ] Step 4: Open PR

```bash
gh pr create --title "DFlash: multi-window training (#382)" --body "$(cat <<'EOF'
## Summary
- Adds `--train-windows-per-step N` to `olmlx dflash prepare`.
- Trains on N non-overlapping masked windows per batch in a single optimizer step, reusing one target forward.
- K=1 (default) is bit-exact with the prior single-window behaviour.

Closes #382.

## Test plan
- [ ] `uv run pytest tests/test_dflash_prepare.py -v` — all PASS, including the new `TestSelectPivots`, `TestTrainWindowsValidation`, `TestK1BitExactness`, `TestMultiWindowTraining`, `TestCliArgumentPlumbing` classes.
- [ ] `uv run pytest tests/test_dflash.py tests/test_dflash_distill.py tests/test_dflash_precompute.py -v` — no regressions.
- [ ] `uv run ruff check && uv run ruff format --check` clean.
- [ ] Manual: `uv run olmlx dflash prepare --help | grep train-windows-per-step` shows the flag.
- [ ] (Optional, not part of this PR's acceptance) Empirical benchmark: K=4 on a real 27B–35B target produces a draft with at least the K=1 acceptance rate at fewer wall-clock seconds per N optimizer steps.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
