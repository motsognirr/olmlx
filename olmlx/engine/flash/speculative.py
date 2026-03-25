"""Speculative decoding with flash inference (Paper §5.2).

Uses a small draft model for fast candidate generation, then verifies
candidates with the big (flash) model in a single forward pass.
Key optimization: neuron retention window is sized to alpha*(lambda+1)
where alpha is the rolling acceptance rate.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SpeculativeFlashDecoder:
    """Speculative decoding for flash inference.

    The draft model generates lambda candidate tokens autoregressively.
    The target model verifies all candidates in one forward pass.
    Accepted tokens are returned; on rejection, the target's preferred
    token replaces the first rejected position.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        vocab_size: int,
        num_speculative_tokens: int = 4,
        acceptance_rate_ema: float = 0.9,
    ):
        self._draft = draft_model
        self._target = target_model
        self._vocab_size = vocab_size
        self._lambda = num_speculative_tokens
        self._alpha = 0.5  # initial acceptance rate estimate
        self._alpha_ema = acceptance_rate_ema

    def _draft_generate(self, prompt: mx.array, n: int) -> tuple[list[int], mx.array]:
        """Generate n candidate tokens with the draft model.

        Creates a fresh KV cache per call so the prompt is processed once
        and each subsequent token is O(1). The cache is discarded after
        the call to avoid accumulating stale state across generate_step
        calls (proper cross-step caching requires offset tracking).

        Args:
            prompt: (1, seq_len) input token IDs.
            n: Number of tokens to generate.

        Returns:
            (tokens, logits) where tokens is list[int] of length n,
            and logits is (n, vocab_size).
        """
        tokens: list[int] = []
        all_logits: list[mx.array] = []

        # Try to create a per-call KV cache for O(1) decode steps
        cache = None
        try:
            from mlx_lm.models.cache import make_prompt_cache

            cache = make_prompt_cache(self._draft)
        except (ImportError, TypeError, AttributeError):
            pass

        if cache is not None:
            # Prefill: process the full prompt
            logits = self._draft(prompt, cache=cache)
            next_logits = logits[:, -1, :]
            mx.eval(next_logits)
            next_token = int(mx.argmax(next_logits, axis=-1).item())
            tokens.append(next_token)
            all_logits.append(next_logits.squeeze(0))

            # Decode: each step only processes the new token
            for _ in range(n - 1):
                inp = mx.array([[next_token]])
                logits = self._draft(inp, cache=cache)
                next_logits = logits[:, -1, :]
                mx.eval(next_logits)
                next_token = int(mx.argmax(next_logits, axis=-1).item())
                tokens.append(next_token)
                all_logits.append(next_logits.squeeze(0))
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
                all_logits.append(next_logits.squeeze(0))

        return tokens, mx.stack(all_logits)

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
        accepted: list[int] = []
        n = len(draft_tokens)

        for i in range(n):
            target_token = int(mx.argmax(target_logits[i]).item())
            if draft_tokens[i] == target_token:
                accepted.append(draft_tokens[i])
            else:
                # Reject: use target's preferred token instead
                accepted.append(target_token)
                return accepted

        # All draft tokens accepted — add bonus token from target
        bonus = int(mx.argmax(target_logits[n]).item())
        accepted.append(bonus)
        return accepted

    def generate_step(
        self,
        prompt: mx.array,
    ) -> tuple[list[int], int]:
        """One speculative decoding step.

        Args:
            prompt: (1, seq_len) input token IDs.

        Returns:
            (accepted_tokens, num_draft_generated).
        """
        # 1. Draft model generates lambda candidates
        draft_tokens, draft_logits = self._draft_generate(prompt, self._lambda)

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

        # 4. Update acceptance rate
        num_accepted_draft = (
            min(len(accepted) - 1, self._lambda) if len(accepted) > 0 else 0
        )
        acceptance = num_accepted_draft / max(self._lambda, 1)
        self._alpha = self._alpha_ema * self._alpha + (1 - self._alpha_ema) * acceptance

        return accepted, self._lambda

    @property
    def effective_window_size(self) -> int:
        """Recommended neuron retention window based on acceptance rate."""
        return max(1, int(self._alpha * (self._lambda + 1)))
