"""Lookahead (Jacobi) decoding — a draft-free speculative strategy (#502).

PLD (``PromptLookupDecoder``) is the only other draft-free strategy, but it
can only propose n-grams that already appeared in the prompt/history. Lookahead
decoding generates *novel* n-grams in parallel via Jacobi iteration — no draft
model, no training — then verifies them against the target with the same greedy
verify the other strategies use. It is therefore complementary to PLD on fresh
text (prose, novel code) where prompt-lookup finds no match.

Algorithm (a linear-verify variant of Fu et al. 2024, "Break the Sequential
Dependency of LLM Inference Using Lookahead Decoding"):

- Maintain a single Jacobi *guess* trajectory of ``W`` tokens for the next
  ``W`` positions. Each step feeds ``[pending] + guess`` through the target in
  ONE forward, then takes ``argmax`` at every position. Those argmaxes ARE one
  Jacobi iteration: ``refined[i]`` is the target's greedy token at position ``i``
  conditioned on the current guess prefix. The refined tail seeds the next
  step's guess (a parallel fixed-point iteration that converges over steps).
- Simultaneously, the *same* logits drive the standard greedy verify: the longest
  guess prefix that matches the target's argmax is accepted, plus a bonus token.
  Because acceptance is exactly the greedy-verify rule, **output is identical to
  plain greedy decoding regardless of guess quality** — the guess only affects
  speed (tokens-per-step), never correctness.
- An n-gram pool keyed by the preceding token caches refined trajectories across
  steps, warm-starting the guess after a full-acceptance step (when the Jacobi
  tail is exhausted) and on token recurrence.

Unlike the full Fu et al. method this keeps a single linear trajectory (no 2-D
lookahead window / custom attention mask), so it reuses ``verify_draft_greedy``
and the cache-trim arithmetic unchanged — the same machinery PLD uses. Like PLD
it needs no draft model and so composes with Flash-MoE (which blocks the
hidden-state strategies dflash/eagle/mtp). GDN rollback is reused for hybrid
linear-attention targets. Not thread-safe: one instance serves one request.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.gdn_rollback import GDNBuffer, GDNStateCapture, find_gdn_class
from olmlx.engine.spec_decoder_base import SpecDecoderBase
from olmlx.engine.speculative import (
    _logits,
    _prefill_last_logit,
    make_prompt_cache,
    trim_prompt_cache,
)

logger = logging.getLogger(__name__)

#: Cap on distinct keys retained in the n-gram pool, so a long generation with
#: a large vocabulary can't grow it without bound. Insertion-ordered eviction.
_POOL_MAX_KEYS = 8192


class LookaheadDecoder(SpecDecoderBase):
    """Draft-free Jacobi lookahead decoder. See module docstring."""

    def __init__(
        self,
        target_model: nn.Module,
        num_speculative_tokens: int = 5,
    ):
        super().__init__()
        if trim_prompt_cache is None or make_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache imports failed (trim_prompt_cache / "
                "make_prompt_cache unavailable); lookahead requires both — fail "
                "fast at construction rather than crashing on first prefill"
            )
        if num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}"
            )
        self._target = target_model
        #: Jacobi window width = max draft length proposed per step.
        self._window = num_speculative_tokens

        # Persistent per-request state (populated by prefill/step).
        self._target_cache: list | None = None
        self._cache_seq_len: int = 0
        self._pending_token: int | None = None
        #: The Jacobi guess trajectory for the next ``_window`` positions.
        self._guess: list[int] = []
        #: token -> recently-observed length-``_window`` continuation n-gram.
        self._pool: dict[int, list[int]] = {}

        # GDN rollback for hybrid linear-attention targets (Qwen3.5/3.6
        # GatedDeltaNet). Draft-free, so only the target is patched — mirrors
        # PromptLookupDecoder exactly.
        self._gdn_capture: GDNStateCapture | None = None
        self._target_gdn_buffer: GDNBuffer | None = None
        target_gdn_cls = find_gdn_class(target_model)
        if target_gdn_cls is not None:
            self._gdn_capture = GDNStateCapture(target_gdn_cls)
            try:
                self._target_gdn_buffer = self._gdn_capture.create_buffer(target_model)
            except Exception:
                self._gdn_capture.close()
                self._gdn_capture = None
                self._target_gdn_buffer = None
                raise

    def close(self) -> None:
        """Release the decoder-lifetime GDN capture (idempotent), then reset.

        Mirrors ``PromptLookupDecoder.close``: the GDN capture is created in
        ``__init__`` (decoder-lifetime), so the base ``reset()`` never touches
        it — release it here under a ``finally`` so a failing close still drops
        the working cache.
        """
        try:
            if self._gdn_capture is not None:
                self._gdn_capture.close()
                self._gdn_capture = None
                self._target_gdn_buffer = None
        finally:
            self.reset()

    def _reset_state(self) -> None:
        self._target_cache = None
        self._cache_seq_len = 0
        self._pending_token = None
        self._guess = []
        self._pool = {}

    def _stats_extra(self) -> dict[str, Any]:
        return {"window": self._window}

    def _prefill_impl(
        self,
        prompt: mx.array,
        *,
        segmented: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> int:
        """Process the prompt through the target, populating its KV cache.

        Cross-request KV reuse (#421) is not implemented for lookahead v1; the
        ``segmented`` argument is accepted and ignored. A single straight
        prefill keeps the strategy simple while the win is in decode throughput.
        """
        self._target_cache = make_prompt_cache(self._target)

        # Same VLM cache-reset rationale as the other decoders.
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(self._target, attr):
                setattr(self._target, attr, None)

        # No rollback needed for the prompt forward.
        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)

        last_logit = _prefill_last_logit(
            self._target, prompt, self._target_cache, cancel_event=cancel_event
        )
        mx.eval(last_logit)
        self._cache_seq_len = prompt.shape[1]

        first_token = int(mx.argmax(last_logit).item())
        self._pending_token = first_token
        # Seed the Jacobi window with the most recent prompt tokens — a weak
        # but non-empty starting trajectory that the per-step refinement
        # converges away from. (Any seed is correctness-safe; verify guarantees
        # exactness.)
        seq = prompt[0].tolist()
        self._guess = seq[-self._window :] if len(seq) >= self._window else list(seq)
        return first_token

    def _step_impl(self) -> tuple[list[int], int]:
        """One lookahead step: propose the Jacobi guess, verify greedily,
        trim, and refine the trajectory for the next step.

        Returns ``(accepted_tokens, num_draft_proposed)``.
        """
        if self._target_cache is None or self._pending_token is None:
            raise RuntimeError(
                "LookaheadDecoder.step() called before prefill(); "
                "call prefill(prompt) first"
            )

        pending_token = self._pending_token
        draft_tokens = list(self._guess[: self._window])
        num_drafted = len(draft_tokens)

        # Target forward on [pending, guess_1..guess_num_drafted].
        if self._gdn_capture is not None:
            if self._target_gdn_buffer is not None:
                self._target_gdn_buffer.clear()
            self._gdn_capture.use_buffer(self._target_gdn_buffer)
        all_tokens = mx.array([[pending_token] + draft_tokens])
        target_out = _logits(self._target(all_tokens, cache=self._target_cache))
        choices = mx.argmax(target_out[0], axis=-1)  # (num_drafted + 1,)
        mx.eval(choices)
        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)
        refined = choices.tolist()

        # Greedy verification — identical primitive to every other strategy, so
        # the emitted tokens are exactly plain-greedy output. ``refined`` is the
        # argmax per position; ``verify_draft_greedy`` recomputes the same
        # argmax internally, but passing logits keeps the shared contract.
        accepted = self._verify_greedy(draft_tokens, target_out[0])
        num_accepted = len(accepted)

        # Trim the target cache to the accepted prefix (fed num_drafted+1,
        # keep num_accepted). GDN targets roll back recurrent state instead.
        trim_target = max(num_drafted + 1 - num_accepted, 0)
        if trim_target > 0:
            if self._target_gdn_buffer is not None and self._gdn_capture is not None:
                self._gdn_capture.rollback_single(
                    self._target_gdn_buffer,
                    self._target_cache,
                    accepted=num_accepted - 1,
                    trim=trim_target,
                )
            else:
                trim_prompt_cache(self._target_cache, trim_target)

        # Refine the Jacobi trajectory for the next step. ``refined`` is the
        # one-iteration update of [pending] + guess; ``refined[:num_accepted]``
        # equals ``accepted`` (verify accepts only argmax matches), so the
        # not-yet-committed tail ``refined[num_accepted:]`` seeds the next
        # window. Record the refined continuation in the pool keyed by the token
        # we just advanced past, then warm-start any shortfall from it.
        new_pending = accepted[-1]
        self._remember_pool(pending_token, refined[: self._window])
        next_guess = refined[num_accepted:]
        if len(next_guess) < self._window:
            next_guess = next_guess + self._pool.get(new_pending, [])
        self._guess = next_guess[: self._window]

        # Cache now holds prompt + pending + accepted[:-1]; the bonus
        # (accepted[-1]) is the new pending and is not yet in the cache.
        self._cache_seq_len += num_accepted
        self._pending_token = int(new_pending)

        num_accepted_draft = min(num_accepted - 1, num_drafted)
        self._stats_steps += 1
        self._stats_proposed += num_drafted
        self._stats_accepted_draft += num_accepted_draft

        return accepted, num_drafted

    def _remember_pool(self, key: int, ngram: list[int]) -> None:
        """Record ``ngram`` as the continuation following ``key`` (insertion-
        ordered LRU, capped at ``_POOL_MAX_KEYS``)."""
        if not ngram:
            return
        # Refresh recency: re-inserting moves the key to the end.
        self._pool.pop(key, None)
        self._pool[key] = ngram
        if len(self._pool) > _POOL_MAX_KEYS:
            # Drop the oldest key (FIFO ~ LRU given the refresh above).
            oldest = next(iter(self._pool))
            del self._pool[oldest]
