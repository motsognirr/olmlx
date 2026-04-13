"""Speculative decoding with flash inference (Paper §5.2).

Extends the base SpeculativeDecoder with Flash-specific optimizations:
neuron prefetching (cross-layer and draft-informed) and adaptive
neuron retention window sizing based on acceptance rate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.speculative import SpeculativeDecoder

if TYPE_CHECKING:
    from olmlx.engine.flash.prefetch import Prefetcher

logger = logging.getLogger(__name__)


class SpeculativeFlashDecoder(SpeculativeDecoder):
    """Speculative decoding for flash inference.

    Extends the base decoder with:
    - Draft hidden state capture for prefetch (Path B)
    - Prefetch submission after draft generation
    - Prefetch cancellation on rejection
    - Adaptive neuron retention window based on acceptance rate
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        num_speculative_tokens: int = 4,
        acceptance_rate_ema: float = 0.9,
        prefetcher: Prefetcher | None = None,
    ):
        super().__init__(
            draft_model=draft_model,
            target_model=target_model,
            num_speculative_tokens=num_speculative_tokens,
            acceptance_rate_ema=acceptance_rate_ema,
        )
        self._prefetcher = prefetcher

    def _draft_generate_cached(
        self, pending_token: int, n: int
    ) -> tuple[list[int], list[mx.array]]:
        """Generate n candidates, capturing hidden states for prefetch."""
        assert self._draft_cache is not None

        inner_model = getattr(self._draft, "model", None)
        lm_head = getattr(self._draft, "lm_head", None)
        can_capture = (
            self._prefetcher is not None
            and inner_model is not None
            and lm_head is not None
        )

        captured: list[mx.array] = []
        next_token = pending_token
        tokens: list[int] = []

        for _ in range(n):
            inp = mx.array([[next_token]])
            if can_capture:
                hidden = inner_model(inp, cache=self._draft_cache)
                logits = lm_head(hidden)
                next_logits = logits[:, -1, :]
                mx.eval(next_logits)
                captured.append(hidden[:, -1, :])
            else:
                logits = self._draft(inp, cache=self._draft_cache)
                next_logits = logits[:, -1, :]
                mx.eval(next_logits)

            next_token = int(mx.argmax(next_logits, axis=-1).item())
            tokens.append(next_token)

        return tokens, captured

    def _after_draft(self, draft_ctx: list[mx.array]) -> None:
        """Submit bulk prefetch using captured draft hidden states."""
        if self._prefetcher is not None and draft_ctx:
            self._submit_draft_prefetch(draft_ctx)

    def _after_verify(self, num_accepted: int) -> None:
        """Cancel speculative prefetch for rejected tokens."""
        if self._prefetcher is not None and num_accepted < self._lambda:
            self._prefetcher.cancel()

    def _submit_draft_prefetch(self, draft_hidden_states: list[mx.array]) -> None:
        """Submit bulk prefetch using captured draft hidden states.

        When the draft model's hidden_size differs from the target model's
        (as reported by the prefetcher), skip silently — the cross-layer
        prefetch (Path A) still provides coverage during the target forward pass.
        """
        assert self._prefetcher is not None

        hidden_dim = draft_hidden_states[0].shape[-1]
        expected = self._prefetcher.hidden_size
        if expected is not None and hidden_dim != expected:
            logger.debug(
                "Draft hidden_size %d != target hidden_size %d; skipping Path B",
                hidden_dim,
                expected,
            )
            return

        num_layers = self._prefetcher.num_layers
        num_draft = len(draft_hidden_states)
        draft_to_layers: dict[int, list[int]] = {}
        for i in range(num_layers):
            draft_idx = round(i * (num_draft - 1) / max(num_layers - 1, 1))
            draft_to_layers.setdefault(draft_idx, []).append(i)

        layer_states: dict[int, mx.array] = {}
        for draft_idx, layers in draft_to_layers.items():
            hidden = draft_hidden_states[draft_idx].reshape(1, -1)
            for layer_idx in layers:
                layer_states[layer_idx] = hidden
        mx.eval(*{id(v): v for v in layer_states.values()}.values())

        self._prefetcher.submit_bulk(layer_states)

    @property
    def effective_window_size(self) -> int:
        """Recommended neuron retention window based on acceptance rate."""
        return max(1, int(self._alpha * (self._lambda + 1)))
