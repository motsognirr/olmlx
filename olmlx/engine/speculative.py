"""Model-agnostic speculative decoding.

Uses a small draft model for fast candidate generation, then verifies
candidates with the target model in a single forward pass.

This base class contains the core algorithm without Flash-specific
dependencies. See engine/flash/speculative.py for the Flash-aware
subclass that adds prefetching and neuron window sizing.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
except ImportError:
    make_prompt_cache = None  # type: ignore[assignment]
    trim_prompt_cache = None  # type: ignore[assignment]

from olmlx.engine.gdn_rollback import (
    GDNBuffer,
    GDNStateCapture,
    find_gdn_class,
)

logger = logging.getLogger(__name__)


def _logits(out: Any) -> mx.array:
    # mlx-vlm's language_model returns LanguageModelOutput(logits=...);
    # mlx-lm models return a raw mx.array.
    return cast(mx.array, getattr(out, "logits", out))


def _eval_cache(cache: list) -> None:
    """Materialise KV cache state without evaluating logits.

    Handles KVCache (.keys/.values as mx.array), quantised caches that wrap
    packed data in a list/tuple, and ArraysCache-style caches (.state
    returning a list of tensors).
    """
    arrs = []
    for c in cache:
        k = getattr(c, "keys", None)
        v = getattr(c, "values", None)
        if isinstance(k, mx.array):
            arrs.append(k)
        elif isinstance(k, (list, tuple)):
            arrs.extend(a for a in k if isinstance(a, mx.array))
        if isinstance(v, mx.array):
            arrs.append(v)
        elif isinstance(v, (list, tuple)):
            arrs.extend(a for a in v if isinstance(a, mx.array))
        if k is None and v is None:
            # ArraysCache and similar: state is a list/tuple of arrays
            state = getattr(c, "state", None)
            if isinstance(state, (list, tuple)):
                arrs.extend(a for a in state if isinstance(a, mx.array))
        # TurboQuantKVCache / SpectralQuantKVCache return values from
        # update_and_fetch are views over a separately maintained dequant
        # side buffer that is not surfaced via .state. Probe explicitly so
        # the dequant graph is forced here instead of fusing into pass 2.
        for attr in ("_key_dequant", "_value_dequant"):
            buf = getattr(c, attr, None)
            if isinstance(buf, mx.array):
                arrs.append(buf)
    if arrs:
        mx.eval(*arrs)
    elif cache:
        # A non-empty cache produced no arrays to evaluate (unrecognised
        # cache type). MLX's lazy graph chains pass-1's cache mutations
        # into pass-2's forward regardless, so tokens stay correct — but
        # the OOM this helper was designed to prevent will resurface
        # (pass-2's eval pulls pass-1's lm_head graph through). Warn so
        # the gap is visible; do not raise, since hard-failing here would
        # break speculative decoding for every new mlx-lm cache type
        # before its probe lands in this function.
        logger.error(
            "_eval_cache: no mx.array entries found in %d cache objects "
            "(types: %s); the OOM-avoidance graph break is a no-op for "
            "this cache type — add an explicit probe here if Metal OOMs "
            "during prefill on long prompts.",
            len(cache),
            sorted({type(c).__name__ for c in cache}),
        )


def _prefill_last_logit(model: Any, prompt: mx.array, cache: list) -> mx.array:
    """Return the logit for the last prompt position without materialising
    the full [batch, seq_len, vocab] tensor.

    For prompts longer than 1 token, runs two passes:
    - Pass 1: prefix[:-1] fills the KV cache; the model output is discarded
      so MLX's lazy evaluation never computes lm_head on seq_len-1 positions.
    - Pass 2: last token produces a [1, 1, vocab] logit; safe to evaluate.

    This avoids the Metal OOM that occurs when seq_len × vocab_size exceeds
    the ~41 GB Metal buffer limit (e.g. Gemma 4 31B, vocab=262 144, long ctx).

    The returned logit is lazy. Cache state is materialised between passes
    (graph-break for OOM avoidance) but the caller is expected to evaluate
    the returned logit, which transitively forces pass-2's cache state.
    """
    if prompt.shape[1] <= 1:
        return _logits(model(prompt, cache=cache))[0, -1, :]
    prefix, last = prompt[:, :-1], prompt[:, -1:]
    model(prefix, cache=cache)  # output discarded; lm_head never materialised
    _eval_cache(cache)  # materialise KV state before the single-token pass
    return _logits(model(last, cache=cache))[0, 0, :]


def verify_draft_greedy(
    draft_tokens: list[int],
    target_logits: mx.array,
) -> list[int]:
    """Verify draft tokens against target model logits (greedy).

    Accept draft tokens that match the target's greedy argmax. On first
    mismatch, return the target's preferred token instead.

    Args:
        draft_tokens: list of draft token IDs, length n.
        target_logits: (n+1, vocab) target model logits.

    Returns:
        List of accepted token IDs (1 to n+1 tokens).
    """
    n = len(draft_tokens)

    target_choices = mx.argmax(target_logits, axis=-1)
    mx.eval(target_choices)

    accepted: list[int] = []
    for i in range(n):
        target_token = int(target_choices[i].item())
        if draft_tokens[i] == target_token:
            accepted.append(draft_tokens[i])
        else:
            accepted.append(target_token)
            return accepted

    # All draft tokens accepted — add bonus token
    accepted.append(int(target_choices[n].item()))
    return accepted


class SpeculativeDecoder:
    """Speculative decoding with a draft model.

    The draft model generates lambda candidate tokens autoregressively.
    The target model verifies all candidates in one forward pass.
    Accepted tokens are returned; on rejection, the target's preferred
    token replaces the first rejected position.

    Supports two modes:
    - Stateless: ``generate_step(prompt)`` — no cross-step caching
    - Cached: ``prefill(prompt)`` then ``step()`` — persistent KV caches

    Not thread-safe: one decoder instance must serve one request at a time.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        num_speculative_tokens: int = 4,
        acceptance_rate_ema: float = 0.9,
    ):
        if trim_prompt_cache is None:
            raise RuntimeError(
                "trim_prompt_cache is unavailable (mlx-lm import missing); "
                "speculative decoding requires it for correct cache trimming"
            )
        self._draft = draft_model
        self._target = target_model
        self._lambda = num_speculative_tokens
        self._alpha = 0.5  # initial acceptance rate estimate
        self._alpha_ema = acceptance_rate_ema

        # Persistent KV cache state (populated by prefill/step)
        self._target_cache: list | None = None
        self._draft_cache: list | None = None
        self._cache_seq_len: int = 0
        self._last_target_logit: mx.array | None = None
        self._pending_token: int | None = None

        # GDN rollback: install a class-level patch on ``GatedDeltaNet``
        # if either model has hybrid linear-attention layers. The single
        # capture instance routes per-call writes to whichever buffer is
        # currently active (target vs draft), so target and draft can
        # share the same ``GDNStateCapture`` when they use the same GDN
        # class (the usual case — e.g. Qwen3.5 target + Qwen3.5 draft).
        # ``find_gdn_class`` returns ``None`` for non-hybrid models.
        # If target and draft use *different* GDN classes (unusual but
        # not impossible — e.g. Qwen3-Coder-Next target + Qwen3.5 draft
        # if they ever ship distinct subclasses), we raise rather than
        # patching two classes silently — one capture per class would
        # require lifting the class-level patch lock.
        self._gdn_capture: GDNStateCapture | None = None
        self._target_gdn_buffer: GDNBuffer | None = None
        self._draft_gdn_buffer: GDNBuffer | None = None
        target_gdn_cls = find_gdn_class(target_model)
        draft_gdn_cls = find_gdn_class(draft_model)
        if target_gdn_cls is not None or draft_gdn_cls is not None:
            if (
                target_gdn_cls is not None
                and draft_gdn_cls is not None
                and target_gdn_cls is not draft_gdn_cls
            ):
                raise NotImplementedError(
                    "Target and draft use different GatedDeltaNet classes "
                    f"({target_gdn_cls.__module__}.{target_gdn_cls.__name__} "
                    f"vs {draft_gdn_cls.__module__}.{draft_gdn_cls.__name__}). "
                    "Classic speculative GDN rollback currently shares one "
                    "class-level patch between target and draft; supporting "
                    "two distinct classes would require lifting the patch "
                    "lock. File an olmlx issue if you hit this."
                )
            gdn_cls = target_gdn_cls if target_gdn_cls is not None else draft_gdn_cls
            # ``assert`` is stripped under ``python -O``; the outer ``if``
            # guarantees at least one of the two is non-None, but a
            # future refactor of that check could silently violate the
            # invariant. Use ``RuntimeError`` so the failure surfaces
            # in production builds too.
            if gdn_cls is None:
                raise RuntimeError(
                    "SpeculativeDecoder GDN-setup invariant violated: "
                    "outer ``if`` branch entered but both target and "
                    "draft GDN classes are None. Please file an olmlx bug."
                )
            self._gdn_capture = GDNStateCapture(gdn_cls)
            # ``create_buffer`` walks the model and can raise (e.g. orphaned
            # GDN modules outside ``get_model_layers``). Close the capture
            # explicitly to release the patch lock — relying on ``__del__``
            # to clean up a partially-constructed decoder leaks the lock
            # until CPython's refcount GC fires, which is fragile under
            # asyncio teardown and blocks any subsequent hybrid load.
            try:
                if target_gdn_cls is not None:
                    self._target_gdn_buffer = self._gdn_capture.create_buffer(
                        target_model
                    )
                if draft_gdn_cls is not None:
                    self._draft_gdn_buffer = self._gdn_capture.create_buffer(
                        draft_model
                    )
            except Exception:
                self._gdn_capture.close()
                self._gdn_capture = None
                self._target_gdn_buffer = None
                self._draft_gdn_buffer = None
                raise

        # Diagnostic counters (reset on prefill)
        self._stats_steps: int = 0
        self._stats_proposed: int = 0
        self._stats_accepted_draft: int = 0

    def close(self) -> None:
        """Release the GDN class-level monkey-patch (idempotent).

        Decoders should be ``close()``-d explicitly when no longer used;
        the ``__del__`` finaliser is best-effort because the patched
        ``__call__`` holds a strong reference to ``GDNStateCapture``
        through its closure, breaking the GC cycle that would otherwise
        run our finaliser.
        """
        if self._gdn_capture is not None:
            self._gdn_capture.close()
            self._gdn_capture = None
            self._target_gdn_buffer = None
            self._draft_gdn_buffer = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Finalisers must never raise.
            pass

    def _update_acceptance_rate(self, num_accepted: int) -> int:
        """Update the rolling acceptance rate via EMA and return accepted-draft count."""
        assert num_accepted >= 1, (
            "_update_acceptance_rate: _verify() must return at least 1 token"
        )
        num_accepted_draft = min(num_accepted - 1, self._lambda)
        acceptance = num_accepted_draft / max(self._lambda, 1)
        self._alpha = self._alpha_ema * self._alpha + (1 - self._alpha_ema) * acceptance
        return num_accepted_draft

    # ------------------------------------------------------------------
    # Cached API: prefill() + step()
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._target_cache = None
        self._draft_cache = None
        self._cache_seq_len = 0
        self._last_target_logit = None
        self._pending_token = None
        self._stats_steps = 0
        self._stats_proposed = 0
        self._stats_accepted_draft = 0

    def stats_summary(self) -> dict[str, Any]:
        steps = self._stats_steps
        proposed = self._stats_proposed
        accepted_draft = self._stats_accepted_draft
        acceptance_rate = accepted_draft / proposed if proposed else 0.0
        # Mean accepted length: accepted drafts plus one guaranteed target
        # token per step, i.e. total new tokens emitted per verification pass.
        avg_tokens_per_step = (accepted_draft + steps) / steps if steps else 0.0
        return {
            "steps": steps,
            "proposed": proposed,
            "accepted_draft": accepted_draft,
            "acceptance_rate": acceptance_rate,
            "avg_tokens_per_step": avg_tokens_per_step,
            "ema_acceptance_rate": self._alpha,
            "lambda": self._lambda,
        }

    def prefill(self, prompt: mx.array) -> int:
        """Process the prompt through both models, populating KV caches.

        Args:
            prompt: (1, seq_len) input token IDs.

        Returns:
            The first generated token (from target model's greedy argmax).
        """
        self.reset()

        if make_prompt_cache is None or trim_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache not available; cannot use cached speculative decoding"
            )

        self._target_cache = make_prompt_cache(self._target)
        self._draft_cache = make_prompt_cache(self._draft)

        # Observed on mlx-vlm 0.4.4 with Qwen3_5: the language model caches
        # `_position_ids` and `_rope_deltas` on the module instance across
        # calls.  Left over from a prior request they produce broadcast
        # mismatches when a new prompt has a different length.  Reset them
        # at the start of each prefill.  If a future VLM caches analogous
        # state under a different attribute name, add it here.
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(self._target, attr):
                setattr(self._target, attr, None)

        # Suppress GDN capture during prefill: no rollback is needed
        # for the prompt forward, and recording it would just bloat the
        # buffers (which are sized for one step's worth of captures).
        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)

        self._last_target_logit = _prefill_last_logit(
            self._target, prompt, self._target_cache
        )
        mx.eval(self._last_target_logit)

        # Populate draft cache; logits not needed.
        self._draft(prompt, cache=self._draft_cache)
        _eval_cache(self._draft_cache)

        self._cache_seq_len = prompt.shape[1]

        first_token = int(mx.argmax(self._last_target_logit).item())
        self._pending_token = first_token
        return first_token

    def step(self) -> tuple[list[int], int]:
        """One speculative decoding step using persistent KV caches.

        Must call ``prefill()`` first to populate caches.

        Returns:
            (accepted_tokens, num_draft_generated).
        """
        assert self._target_cache is not None, "Call prefill() before step()"
        assert self._draft_cache is not None, "Call prefill() before step()"
        assert self._pending_token is not None, "Call prefill() before step()"

        pending_token = self._pending_token

        # 1. Draft: feed pending token, then generate lambda candidates.
        # Route GDN captures to the draft buffer (if draft is hybrid).
        if self._gdn_capture is not None:
            if self._draft_gdn_buffer is not None:
                self._draft_gdn_buffer.clear()
            self._gdn_capture.use_buffer(self._draft_gdn_buffer)
        draft_tokens, draft_ctx = self._draft_generate_cached(
            pending_token, self._lambda
        )

        # Hook for subclasses (e.g. Flash prefetcher submission)
        self._after_draft(draft_ctx)

        # 2. Target: feed [pending, D1, ..., D_lambda] in one pass.
        # Switch GDN capture to the target buffer.
        if self._gdn_capture is not None:
            if self._target_gdn_buffer is not None:
                self._target_gdn_buffer.clear()
            self._gdn_capture.use_buffer(self._target_gdn_buffer)
        all_tokens = mx.array([[pending_token] + draft_tokens])
        target_out = _logits(self._target(all_tokens, cache=self._target_cache))
        mx.eval(target_out)

        # Capture is done: no more model forwards in this step should
        # write to either buffer. Subclass hooks (``_after_verify``) and
        # rollback replays (``rollback_single`` invokes
        # ``gated_delta_update`` directly, not ``GatedDeltaNet.__call__``)
        # would otherwise append spurious captures and fail the next
        # step's buffer-size check.
        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)

        verification_logits = target_out[0]  # (lambda+1, vocab)

        # 3. Verify draft tokens against target logits
        accepted = self._verify(draft_tokens, verification_logits)
        num_accepted = len(accepted)

        # Hook for subclasses (e.g. Flash prefetch cancellation)
        self._after_verify(num_accepted)

        # 4. Roll back caches to keep only the accepted prefix.
        #
        # Target was fed (λ+1) tokens [pending, D_1..D_λ]; keep
        # ``num_accepted`` of them, so trim by (λ+1) - num_accepted.
        # Draft was fed λ tokens autoregressively
        # [pending, D_1..D_{λ-1}]; keep ``num_accepted`` of those if
        # partial acceptance (= a-1 draft tokens between pending and
        # the correction/bonus), so trim by λ - num_accepted.
        trim_target = max(self._lambda + 1 - num_accepted, 0)
        trim_draft = max(self._lambda - num_accepted, 0)

        # Target rollback: GDN replay if hybrid, plain trim otherwise.
        if trim_target > 0:
            if self._target_gdn_buffer is not None and self._gdn_capture is not None:
                # ``rollback_single`` takes ``accepted`` as the number of
                # *additional* tokens beyond the first to keep
                # (n = accepted + 1). We want to keep ``num_accepted``
                # total, so accepted_arg = num_accepted - 1.
                self._gdn_capture.rollback_single(
                    self._target_gdn_buffer,
                    self._target_cache,
                    accepted=num_accepted - 1,
                    trim=trim_target,
                )
            elif trim_prompt_cache is not None:
                trim_prompt_cache(self._target_cache, trim_target)

        # Draft rollback: GDN autoregressive replay if hybrid, plain
        # trim otherwise. On full acceptance (num_accepted == λ+1)
        # trim_draft is 0 and we skip rollback; the align step below
        # then advances the draft cache by feeding D_λ.
        if trim_draft > 0:
            if self._draft_gdn_buffer is not None and self._gdn_capture is not None:
                # ``rollback_autoregressive`` keeps the first
                # ``num_keep_steps`` of ``num_steps`` autoregressive
                # calls. We fed λ tokens and want to keep
                # ``num_accepted`` of them.
                self._gdn_capture.rollback_autoregressive(
                    self._draft_gdn_buffer,
                    self._draft_cache,
                    num_steps=self._lambda,
                    num_keep_steps=num_accepted,
                    trim=trim_draft,
                )
            elif trim_prompt_cache is not None:
                trim_prompt_cache(self._draft_cache, trim_draft)

        # On full acceptance, align draft cache with target cache.
        # ``use_buffer(None)`` was already called after the target
        # forward, so the align step's draft forward — which DOES
        # invoke ``GatedDeltaNet.__call__`` on a hybrid draft — won't
        # write to either buffer.
        if num_accepted > self._lambda:
            last_draft = mx.array([[draft_tokens[-1]]])
            align_logits = _logits(self._draft(last_draft, cache=self._draft_cache))
            mx.eval(align_logits)

        # 5. Update state
        self._cache_seq_len += num_accepted
        assert num_accepted >= 1, "step(): _verify() must return at least 1 token"
        self._last_target_logit = verification_logits[num_accepted - 1]
        mx.eval(self._last_target_logit)
        self._pending_token = int(mx.argmax(self._last_target_logit).item())

        num_accepted_draft = self._update_acceptance_rate(num_accepted)

        self._stats_steps += 1
        self._stats_proposed += self._lambda
        self._stats_accepted_draft += num_accepted_draft

        return accepted, self._lambda

    def _draft_generate_cached(
        self, pending_token: int, n: int
    ) -> tuple[list[int], list[mx.array]]:
        """Generate n candidate tokens using the persistent draft cache.

        Returns:
            (tokens, context) — context is empty in the base class;
            subclasses may return captured hidden states.
        """
        assert self._draft_cache is not None

        next_token = pending_token
        tokens: list[int] = []

        for _ in range(n):
            inp = mx.array([[next_token]])
            logits = _logits(self._draft(inp, cache=self._draft_cache))
            next_logits = logits[:, -1, :]
            mx.eval(next_logits)
            next_token = int(mx.argmax(next_logits, axis=-1).item())
            tokens.append(next_token)

        return tokens, []

    def _after_draft(self, draft_ctx: list[mx.array]) -> None:
        """Hook called after draft generation. Override for prefetch submission."""

    def _after_verify(self, num_accepted: int) -> None:
        """Hook called after verification. Override for prefetch cancellation."""

    # ------------------------------------------------------------------
    # Stateless API
    # ------------------------------------------------------------------

    def _draft_generate(self, prompt: mx.array, n: int) -> list[int]:
        """Generate n candidate tokens with fresh KV cache (stateless)."""
        if make_prompt_cache is None:
            # Caller (generate_step) already raises with the same wording;
            # this guard exists because asserts are stripped under -O.
            raise RuntimeError(
                "mlx_lm.models.cache.make_prompt_cache is not available; "
                "upgrade mlx-lm to a version that exports it."
            )
        try:
            cache = make_prompt_cache(self._draft)
        except (TypeError, AttributeError) as exc:
            raise RuntimeError(
                f"make_prompt_cache failed for draft model "
                f"{type(self._draft).__name__!r}: {exc}. The draft model may "
                "not be compatible with mlx-lm's KV-cache API."
            ) from exc
        tokens: list[int] = []

        logits = _logits(self._draft(prompt, cache=cache))
        next_logits = logits[:, -1, :]
        mx.eval(next_logits)
        next_token = int(mx.argmax(next_logits, axis=-1).item())
        tokens.append(next_token)

        for _ in range(n - 1):
            inp = mx.array([[next_token]])
            logits = _logits(self._draft(inp, cache=cache))
            next_logits = logits[:, -1, :]
            mx.eval(next_logits)
            next_token = int(mx.argmax(next_logits, axis=-1).item())
            tokens.append(next_token)

        return tokens

    def _verify(
        self,
        draft_tokens: list[int],
        target_logits: mx.array,
    ) -> list[int]:
        """Verify draft tokens against target model logits (greedy)."""
        return verify_draft_greedy(draft_tokens, target_logits)

    def generate_step(
        self,
        prompt: mx.array,
    ) -> tuple[list[int], int]:
        """One speculative decoding step (stateless, no cross-step caching).

        Uses a temporary KV cache internally so the target forward over the
        long prompt does not materialise the full [batch, seq_len, vocab]
        logit matrix (same Metal OOM that ``_prefill_last_logit`` avoids).
        """
        if make_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache.make_prompt_cache is not available "
                "(import failed at module load). Upgrade mlx-lm to a "
                "version that exports it (or use the cached prefill+step "
                "API instead, which has the same requirement). The "
                "previous cache-less path was removed because it OOMed "
                "on Metal for large-vocab models on long prompts."
            )

        draft_tokens = self._draft_generate(prompt, self._lambda)

        # See prefill() for why this reset is needed (mlx-vlm 0.4.4 VLMs).
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(self._target, attr):
                setattr(self._target, attr, None)

        # Two-pass split: prefill the prompt into a temporary cache (yields
        # the first verification logit), then feed [pending, D1..D_lambda]
        # for the remaining lambda logits. Total materialised logit shape is
        # [1, lambda+1, vocab] instead of [1, seq_len+lambda, vocab].
        target_cache = make_prompt_cache(self._target)
        first_logit = _prefill_last_logit(self._target, prompt, target_cache)
        draft_ids = mx.array([draft_tokens])
        draft_out = _logits(self._target(draft_ids, cache=target_cache))
        target_logits = mx.concatenate([first_logit[None, :], draft_out[0]], axis=0)
        mx.eval(target_logits)

        accepted = self._verify(draft_tokens, target_logits)

        num_accepted_draft = self._update_acceptance_rate(len(accepted))
        self._stats_steps += 1
        self._stats_proposed += self._lambda
        self._stats_accepted_draft += num_accepted_draft
        return accepted, self._lambda
