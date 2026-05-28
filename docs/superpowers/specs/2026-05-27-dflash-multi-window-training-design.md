# DFlash multi-window training

Issue: [#382](https://github.com/motsognirr/olmlx/issues/382)

## Problem

`prepare_dflash_draft` runs the (frozen) target forward once per batch to
capture hidden states for the full sequence, then trains the draft on
**one random `block_size`-length window** sampled from those hiddens. The
remaining ~`L − block_size` positions of the target forward are
discarded. The target forward dominates wall-clock cost (a frozen 27B–35B
model in the production case), so per-step utilisation of the target's
output is ≤ 5% of what's available.

The paper (arxiv:2602.06036) trains with multiple masked blocks per
sequence, anchors resampled each epoch. Adopting that recipe lets one
target forward train K windows of draft gradient — a near-linear speedup
in the rate at which the draft sees independent training signal per unit
of target compute.

## Goal

Train on K windows per batch in a single optimizer step, reusing one
target forward. Keep `K = 1` bit-exact identical to current behaviour so
the change is opt-in and existing runs are unaffected.

## Non-goals

- Per-batch-row pivot variation. The current design picks pivots from
  the batch-wide shared unpadded prefix, and multi-window preserves that
  constraint. Heterogeneous per-row pivots would require gather-based
  hidden-state indexing and are deferred.
- EAGLE multi-window. `eagle/prepare.py` uses autoregressive teacher
  forcing with position subsampling; a different change. Deferred.
- Precompute format changes. Shards already store full-sequence hidden
  states, so the precomputed-shards path picks up multi-window training
  for free (slice K times instead of once). No `precompute.py` change.
- Logit-storage changes for distillation. Distill captures
  `target_logits` for the full sequence in the same forward, and K
  windows are views into that tensor — no extra storage cost.

## Design

### Pivot selection

Replace `_select_pivot(input_ids, pad_token_id, block_size) -> int | None`
with `_select_pivots(input_ids, pad_token_id, block_size, num_windows)
-> list[int] | None`.

- Compute the trailing-pad boundary identically to today (right-reversed
  `argmax`, batch-wide `min().item()` — one CPU sync per batch,
  unchanged).
- Valid pivot range: `[block_size, min_real - block_size - 1]`
  inclusive, same as today.
- Slot-and-jitter placement: divide the valid range into `num_windows`
  equal slots; sample one pivot uniformly per slot.
- Reject any slot shorter than `block_size + 1`. This is what
  guarantees the produced pivots are non-overlapping in the
  `[p, p + block_size]` input-and-target span: two adjacent slots are at
  least `block_size + 1` apart, so adjacent pivots are too.
- Return value semantics:
  - Returns `None` only when no window is possible (i.e. min_real
    < `2 * block_size + 1`). Preserves today's `None` semantics — the
    caller still treats this as "skip the batch, increment
    `consecutive_skips`".
  - Returns a list of length `1..num_windows` otherwise. A list shorter
    than `num_windows` means the valid range was too small for that
    many non-overlapping slots; the caller proceeds with what it got
    (logged at debug, **not** counted as a skip).
- `K = 1` reduces to a single full-range slot. The sample call
  (`random.randint`) sees the same range it does today, so with the
  same RNG seed the produced pivot is identical to the legacy
  single-pivot path.

The no-pad-token regime (`pad_for_pivot is None`, used by test fixtures
with no-padding tokenizers) gets an inline slot-and-jitter variant in
the training loop, mirroring how the existing code special-cases that
path. No CPU sync in that branch — matching current behaviour.

### Loss reduction

`_draft_loss` is **unchanged**. It still returns a per-window scalar
loss (the existing pad-masked, position-decayed mean over the window's
positions).

New closure inside `prepare_dflash_draft`:

```python
def loss_fn_multi(model, windows):
    # windows is a list of (block_input, target_hidden, targets,
    # target_logits_window) tuples — one per pivot.
    total = mx.array(0.0)
    for (block_input, target_hidden, targets, target_logits_window) in windows:
        cache = model.make_cache()  # fresh per window
        total = total + _draft_loss(
            model, block_input, target_hidden, targets, cache,
            target_logits_window=target_logits_window,
            distill_alpha=...,  # same args as today
            distill_temp=...,
            pad_token_id=pad_for_loss,
            position_decay_gamma=position_decay_gamma,
        )
    return total / len(windows)
```

`nn.value_and_grad` differentiates the closure over all K per-window
calls. The optimizer takes one step per batch, applying the combined
gradient.

Mean-of-per-window-means is the chosen reduction (see Trade-offs
below). It is bit-exact at `K = 1`: a one-element sum divided by 1 is
the underlying scalar.

### Training-loop changes

Inside `prepare_dflash_draft`:

1. New parameter `train_windows_per_step: int = 1`. Validate `>= 1`.
2. After resolving `input_ids` / `precomputed_hidden` from the batch,
   call `_select_pivots(...)` (or the inline no-pad variant) to get the
   list of pivots.
3. If pivots is `None` → skip batch, `consecutive_skips += 1` (today's
   behaviour).
4. If `len(pivots) < num_windows` → log at debug. Continue with the
   pivots we have. **Does not increment skip counter** — a real gradient
   update is about to happen.
5. Run the target forward once (or consume `precomputed_hidden` once),
   exactly as today.
6. Build the per-window list: for each pivot `p_k`, slice `pending`,
   `mask_block`, `block_input`, `targets`, `target_hidden_full[:, :p_k,
   :]`, and (if distilling) `target_logits_full[:, p_k:p_k+block_size,
   :]`. Slicing logic is verbatim today's, looped.
7. Call `loss_and_grad_multi(draft, windows)` and apply optimizer.update
   once. `mx.eval(loss, draft.parameters(), optimizer.state)` once per
   batch, as today.

The `real_step` counter advances once per batch (not once per window),
matching the LR schedule, progress callback, and `--steps` budget
semantics. K = 4 with `--steps 2000` is 2000 optimizer steps, each
seeing 4× more windows than the K = 1 baseline — `--steps` continues to
mean "optimizer updates", not "windows".

### CLI surface

Add `--train-windows-per-step N` to `olmlx dflash prepare` in `cli.py`.
Default `1` (legacy behaviour). Argument plumbed verbatim through
`prepare_dflash_draft`.

Existing flags (`--block-size`, `--seq-len`, `--position-decay-gamma`,
`--distill`, `--use-precomputed`, ...) are unchanged.

### Memory considerations

- **Online path, no distill**: K draft forwards through the existing
  `_draft_loss` flow. Each draft forward materialises
  `(B, block_size, vocab)` logits — at `B=4, block_size=16, vocab=250k,
  bf16` this is 32 MB. K = 4 puts ~128 MB of draft-logit memory in
  scope during the backward pass. Manageable.
- **Online path, with distill**: target logits are
  `(B, L, vocab)` and shared across all K windows (sliced views, no
  copy). Same target-logits memory as today.
- **Precomputed path**: `target_hidden` is `(B, L,
  num_target_layers * hidden_size)` and shared across K windows (sliced
  views). No additional precompute storage.

### Backward compatibility

- `K = 1`: bit-exact equivalent of today. Same single pivot, same
  reduction (sum-of-one / 1), same optimizer trajectory under a fixed
  seed.
- Old config-less checkpoints: unaffected (this is a trainer change
  only, not a model-format change).

## Trade-offs considered

**Loss reduction (chose mean of per-window means)**

Alternatives:
- Sum-then-divide-by-total-valid-positions would be slightly less
  biased when one window happens to have more pad-masked positions than
  another. Cost: refactor `_draft_loss` to return `(loss_sum, weight)`,
  changing its public shape. K = 1 would not be bit-exact when
  pad-masking is active. The windows all share a single unpadded prefix
  so the bias is small; the bit-exact-K=1 property is worth more than
  the precision gain.
- Sum without normalisation would effectively scale the learning rate
  K×, requiring users to drop `--lr` to keep the optimiser equivalent.
  Surprising default.

**Pivot placement (chose slot-and-jitter)**

Alternatives:
- Rejection sampling (sample K, reject overlaps) — K becomes fuzzy on
  tight ranges and the code branches more for the same outcome.
- Fixed stride (no jitter) — loses randomness across batches, hurting
  generalisation across positions.

Slot-and-jitter is what the paper's "anchors resampled each epoch"
phrasing describes: K deterministic regions, jittered anchor inside
each.

**Per-window vs single-call draft forward**

Considered batching all K windows into one `(B*K, block_size+1)` draft
forward. Rejected: each window's `target_hidden` slice has a *different*
length (`p_k` varies), so a batched call would require padding to
`max(p_k)`, wasting compute on the smaller windows. K separate calls
keep the per-window context length exact and let MLX kernel-fuse
independently. The dominant cost is still the single target forward,
so K draft forwards add little.

## Test plan

In `tests/test_dflash_prepare.py` (or the existing DFlash-prepare test
module):

1. **K=1 bit-exactness**: with a fixed `random.seed` and a synthetic
   batch iterator, the K=1 multi-window code path produces an identical
   loss curve to the pre-change single-window code. Compares loss
   values exactly.
2. **K=4 on a long-enough synthetic sequence**: trains for N steps
   without crashing; logs report `len(pivots) == 4` per step; loss
   trends down.
3. **K=4 with short sequences**: when `min_real` is just enough for 2
   windows, the training loop runs with `len(pivots) == 2`, logs the
   downgrade at debug, and does **not** advance `consecutive_skips`.
4. **Pad-only batch**: `_select_pivots` returns `None`, batch is
   skipped, `consecutive_skips` increments — identical to today.
5. **`--use-precomputed` × K=4**: precomputed shard reader yields
   `(input_ids, hidden)` tuples; the K windows slice independently from
   the same `hidden` tensor; loss steps complete.
6. **`--distill` × K=4**: target logits captured once, K windows slice
   independently; loss combines CE + KL per window then averages.
7. **CLI argument plumbing**: `--train-windows-per-step 4` arrives at
   `prepare_dflash_draft` with `train_windows_per_step=4`.
8. **Validation**: `train_windows_per_step=0` raises `ValueError`.

No new integration tests against a real target are required — the
existing real-target tests (if any) continue to exercise K=1 and the
synthetic K>1 tests cover the new code path.

## Files touched

- `olmlx/engine/dflash/prepare.py` — replace `_select_pivot` with
  `_select_pivots`, add `train_windows_per_step` parameter, restructure
  the per-step body around the new closure.
- `olmlx/cli.py` — add `--train-windows-per-step` to the `dflash
  prepare` subcommand.
- `tests/test_dflash_prepare.py` (or equivalent) — new tests per the
  plan above.
- `CLAUDE.md` — update the "DFlash draft training" bullet to mention
  multi-window training and the new flag.

No changes to: `engine/dflash/decoder.py`, `engine/dflash/draft_model.py`,
`engine/dflash/precompute.py`, `engine/dflash/training_data.py`,
`engine/eagle/*`.

## Open questions / follow-ups

- Empirical: what value of K is the sweet spot on a 27B–35B target?
  Likely 4–8 based on the issue's estimate; pin this with a benchmark
  before recommending a non-1 default. Not in scope for this change.
- Once K > 1 is validated, the default may move to 4 in a follow-up.
- A natural extension is heterogeneous per-row pivots (each row picks
  its own K pivots from its own unpadded prefix). Worthwhile if dataset
  length variance turns out to be a meaningful pivot-range constraint.
