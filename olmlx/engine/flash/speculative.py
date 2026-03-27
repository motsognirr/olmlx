"""Speculative decoding with flash inference (Paper §5.2).

Uses a small draft model for fast candidate generation, then verifies
candidates with the big (flash) model in a single forward pass.
Key optimization: neuron retention window is sized to alpha*(lambda+1)
where alpha is the rolling acceptance rate.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
except ImportError:
    make_prompt_cache = None  # type: ignore[assignment]
    trim_prompt_cache = None  # type: ignore[assignment]


class SpeculativeFlashDecoder:
    """Speculative decoding for flash inference.

    The draft model generates lambda candidate tokens autoregressively.
    The target model verifies all candidates in one forward pass.
    Accepted tokens are returned; on rejection, the target's preferred
    token replaces the first rejected position.

    Supports two modes:
    - Stateless: ``generate_step(prompt)`` — no cross-step caching (backward compat)
    - Cached: ``prefill(prompt)`` then ``step()`` — persistent KV caches for O(λ) verification

    Not thread-safe: one decoder instance must serve one request at a time.
    The inference pipeline serializes requests via ``_inference_lock``.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        num_speculative_tokens: int = 4,
        acceptance_rate_ema: float = 0.9,
    ):
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

    def _update_acceptance_rate(self, num_accepted: int) -> None:
        """Update the rolling acceptance rate via EMA."""
        num_accepted_draft = (
            min(num_accepted - 1, self._lambda) if num_accepted > 0 else 0
        )
        acceptance = num_accepted_draft / max(self._lambda, 1)
        self._alpha = self._alpha_ema * self._alpha + (1 - self._alpha_ema) * acceptance

    # ------------------------------------------------------------------
    # Cached API: prefill() + step()
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._target_cache = None
        self._draft_cache = None
        self._cache_seq_len = 0
        self._last_target_logit = None
        self._pending_token = None

    def prefill(self, prompt: mx.array) -> int:
        """Process the prompt through both models, populating KV caches.

        Args:
            prompt: (1, seq_len) input token IDs.

        Returns:
            The first generated token (from target model's greedy argmax).
        """
        self.reset()

        if make_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache not available; cannot use cached speculative decoding"
            )

        self._target_cache = make_prompt_cache(self._target)
        self._draft_cache = make_prompt_cache(self._draft)

        target_out = self._target(prompt, cache=self._target_cache)
        self._last_target_logit = target_out[0, -1, :]
        mx.eval(self._last_target_logit)

        # Populate draft cache (logits discarded — only cache state needed).
        # mx.eval forces cache materialization before step() reads from it.
        draft_logits = self._draft(prompt, cache=self._draft_cache)
        mx.eval(draft_logits)

        self._cache_seq_len = prompt.shape[1]

        first_token = int(mx.argmax(self._last_target_logit).item())
        self._pending_token = first_token
        return first_token

    def step(self) -> tuple[list[int], int]:
        """One speculative decoding step using persistent KV caches.

        Must call ``prefill()`` first to populate caches.

        State invariant: at entry, both caches are at offset ``_cache_seq_len``.
        ``_last_target_logit`` predicts the token at position ``_cache_seq_len``
        (the "pending" token that has been yielded but not yet fed to either cache).

        Returns:
            (accepted_tokens, num_draft_generated).
        """
        assert self._target_cache is not None, "Call prefill() before step()"
        assert self._draft_cache is not None, "Call prefill() before step()"
        assert self._pending_token is not None, "Call prefill() before step()"

        pending_token = self._pending_token

        # 1. Draft: feed pending token, then generate lambda candidates
        #    Draft cache advances from offset to offset + lambda.
        draft_tokens = self._draft_generate_cached(pending_token, self._lambda)

        # 2. Target: feed [pending, D1, ..., D_lambda] in one pass.
        #    Target cache advances from offset to offset + lambda + 1.
        all_tokens = mx.array([[pending_token] + draft_tokens])
        target_out = self._target(all_tokens, cache=self._target_cache)
        mx.eval(target_out)

        # target_out shape: (1, lambda+1, vocab)
        # target_out[0, k] is the logit after processing the token at input position k,
        # predicting what goes at the NEXT position.
        # target_out[0, 0] processed pending → predicts position offset+1 → verifies D1
        # target_out[0, lambda] processed D_lambda → bonus token
        verification_logits = target_out[0]  # (lambda+1, vocab)

        # 3. Verify draft tokens against target logits
        accepted = self._verify(draft_tokens, verification_logits)
        num_accepted = len(accepted)

        # 4. Trim caches to the position after the last accepted token.
        #    Desired offset after step: cache_seq_len + num_accepted
        #    Target was at: cache_seq_len + lambda + 1
        #    Draft was at:  cache_seq_len + lambda
        trim_target = max(self._lambda + 1 - num_accepted, 0)
        trim_draft = max(self._lambda - num_accepted, 0)

        if trim_target > 0:
            trim_prompt_cache(self._target_cache, trim_target)
        if trim_draft > 0:
            trim_prompt_cache(self._draft_cache, trim_draft)

        # On full acceptance (num_accepted == lambda + 1), the draft cache is at
        # offset + lambda while the target is at offset + lambda + 1.  Feed the
        # last draft token through the draft to align both caches at the same offset.
        if num_accepted > self._lambda:
            last_draft = mx.array([[draft_tokens[-1]]])
            align_logits = self._draft(last_draft, cache=self._draft_cache)
            mx.eval(align_logits)

        # 5. Update state
        self._cache_seq_len += num_accepted
        # _verify() always returns at least 1 token (the target's correction).
        assert num_accepted >= 1, "step(): _verify() must return at least 1 token"
        self._last_target_logit = verification_logits[num_accepted - 1]
        mx.eval(self._last_target_logit)
        self._pending_token = int(mx.argmax(self._last_target_logit).item())

        self._update_acceptance_rate(num_accepted)
        return accepted, self._lambda

    def _draft_generate_cached(self, pending_token: int, n: int) -> list[int]:
        """Generate n candidate tokens using the persistent draft cache.

        Feeds ``pending_token`` first (the last accepted token not yet in cache),
        then generates n tokens autoregressively. Draft cache advances by n.

        Args:
            pending_token: Token to feed before generating.
            n: Number of tokens to generate.

        Returns:
            List of n generated token IDs.
        """
        assert self._draft_cache is not None

        next_token = pending_token
        tokens: list[int] = []

        for _ in range(n):
            inp = mx.array([[next_token]])
            logits = self._draft(inp, cache=self._draft_cache)
            next_logits = logits[:, -1, :]
            mx.eval(next_logits)
            next_token = int(mx.argmax(next_logits, axis=-1).item())
            tokens.append(next_token)

        return tokens

    # ------------------------------------------------------------------
    # Stateless API (backward compatibility)
    # ------------------------------------------------------------------

    def _draft_generate(self, prompt: mx.array, n: int) -> list[int]:
        """Generate n candidate tokens with the draft model.

        Creates a fresh KV cache per call so the prompt is processed once
        and each subsequent token is O(1). The cache is discarded after
        the call to avoid accumulating stale state across generate_step
        calls (proper cross-step caching requires offset tracking).

        Args:
            prompt: (1, seq_len) input token IDs.
            n: Number of tokens to generate.

        Returns:
            List of generated token IDs, length n.
        """
        tokens: list[int] = []

        # Try to create a per-call KV cache for O(1) decode steps
        cache = None
        if make_prompt_cache is not None:
            try:
                cache = make_prompt_cache(self._draft)
            except (TypeError, AttributeError):
                pass

        if cache is not None:
            # Prefill: process the full prompt
            logits = self._draft(prompt, cache=cache)
            next_logits = logits[:, -1, :]
            mx.eval(next_logits)
            next_token = int(mx.argmax(next_logits, axis=-1).item())
            tokens.append(next_token)

            # Decode: each step only processes the new token
            for _ in range(n - 1):
                inp = mx.array([[next_token]])
                logits = self._draft(inp, cache=cache)
                next_logits = logits[:, -1, :]
                mx.eval(next_logits)
                next_token = int(mx.argmax(next_logits, axis=-1).item())
                tokens.append(next_token)
        else:
            # Fallback: no KV cache, rebuild full sequence each step
            for _ in range(n):
                if tokens:
                    current = mx.concatenate([prompt, mx.array([tokens])], axis=1)
                else:
                    current = prompt
                logits = self._draft(current)
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
        """Verify draft tokens against target model logits.

        Uses greedy speculative decoding: accept draft token if it matches
        the target's greedy choice. On first mismatch, return the target's
        preferred token instead.

        Args:
            draft_tokens: list of draft token IDs, length lambda.
            target_logits: (lambda+1, vocab) target model logits
                           (positions correspond to verifying each draft
                           token + one extra for the bonus token).

        Returns:
            List of accepted token IDs (1 to lambda+1 tokens).
        """
        n = len(draft_tokens)

        # Vectorized argmax: single GPU kernel for all positions
        target_choices = mx.argmax(target_logits, axis=-1)  # (lambda+1,)
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

    def generate_step(
        self,
        prompt: mx.array,
    ) -> tuple[list[int], int]:
        """One speculative decoding step (stateless, no cross-step caching).

        For cached inference, use ``prefill()`` + ``step()`` instead.

        Args:
            prompt: (1, seq_len) input token IDs.

        Returns:
            (accepted_tokens, num_draft_generated).
        """
        # 1. Draft model generates lambda candidates
        draft_tokens = self._draft_generate(prompt, self._lambda)

        # 2. Target model verifies all candidates + 1 in one pass
        draft_ids = mx.array([draft_tokens])
        combined = mx.concatenate([prompt, draft_ids], axis=1)
        target_out = self._target(combined)  # (1, seq+lambda, vocab)
        mx.eval(target_out)

        # Extract target logits at the verification positions
        seq_len = prompt.shape[1]
        target_logits = target_out[
            0, seq_len - 1 : seq_len + self._lambda, :
        ]  # (lambda+1, vocab)

        # 3. Verify
        accepted = self._verify(draft_tokens, target_logits)

        self._update_acceptance_rate(len(accepted))
        return accepted, self._lambda

    @property
    def effective_window_size(self) -> int:
        """Recommended neuron retention window based on acceptance rate."""
        return max(1, int(self._alpha * (self._lambda + 1)))
