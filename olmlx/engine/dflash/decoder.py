"""DFlash block-diffusion speculative decoder.

Uses hidden states from specific target layers to condition a draft model,
which proposes candidate tokens. The target then verifies all candidates
in one forward pass. Compatible with the SpeculativeDecoder interface
(prefill/step/reset) for seamless integration with the streaming pipeline.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.dflash.adapters import TargetAdapter
from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig
from olmlx.engine.speculative import verify_draft_greedy

try:
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
except ImportError:
    make_prompt_cache = None  # type: ignore[assignment]
    trim_prompt_cache = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DFlashDecoder:
    """Block-diffusion speculative decoder.

    Implements the same prefill/step/reset interface as SpeculativeDecoder
    so it can be used interchangeably with the streaming pipeline.

    Uses trim_prompt_cache for cache rollback (like SpeculativeDecoder)
    instead of snapshot/restore, avoiding redundant target forward passes
    and KV cache re-allocation on partial rejection.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: DFlashDraftModel,
        adapter: TargetAdapter,
        draft_config: DraftConfig,
        block_size: int = 4,
    ):
        if trim_prompt_cache is None or make_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache (make_prompt_cache, trim_prompt_cache) "
                "is unavailable; dflash decoding requires it"
            )

        self._target = target_model
        self._draft = draft_model
        self._adapter = adapter
        self._config = draft_config
        self._block_size = block_size

        # State
        self._cache: list | None = None
        self._cache_seq_len: int = 0
        self._last_hidden_states: dict[int, mx.array] = {}
        self._last_target_logit: mx.array | None = None
        self._pending_token: int | None = None

    def reset(self) -> None:
        self._cache = None
        self._cache_seq_len = 0
        self._last_hidden_states = {}
        self._last_target_logit = None
        self._pending_token = None

    def prefill(self, prompt: mx.array) -> int:
        """Process prompt through target, extract hidden states, return first token.

        Args:
            prompt: (1, seq_len) input token IDs.

        Returns:
            First generated token from target's greedy argmax.
        """
        self.reset()
        self._cache = make_prompt_cache(self._target)

        logits, hidden_states, _ = self._adapter.forward_with_hidden(
            self._target,
            prompt,
            cache=self._cache,
            target_layer_ids=self._config.target_layer_ids,
        )

        self._last_target_logit = logits[0, -1, :]
        mx.eval(self._last_target_logit)

        self._last_hidden_states = hidden_states
        self._cache_seq_len = prompt.shape[1]

        first_token = int(mx.argmax(self._last_target_logit).item())
        self._pending_token = first_token
        return first_token

    def step(self) -> tuple[list[int], int]:
        """One block-diffusion speculative decoding step.

        Returns:
            (accepted_tokens, block_size).
        """
        assert self._cache is not None, "Call prefill() before step()"
        assert self._pending_token is not None

        pending = self._pending_token

        # 1. Draft: propose block_size candidate tokens
        draft_tokens = self._draft_block(pending)

        # 2. Target: verify [pending, D1, ..., D_block_size] in one pass
        all_tokens = mx.array([[pending] + draft_tokens])
        logits, hidden_states, _ = self._adapter.forward_with_hidden(
            self._target,
            all_tokens,
            cache=self._cache,
            target_layer_ids=self._config.target_layer_ids,
        )

        verification_logits = logits[0]  # (block_size+1, vocab)

        # 3. Verify: greedy comparison
        accepted = verify_draft_greedy(draft_tokens, verification_logits)
        num_accepted = len(accepted)

        assert num_accepted >= 1, "verify_draft_greedy must return at least 1 token"

        # 4. Trim cache to remove rejected tokens (like SpeculativeDecoder)
        trim_amount = self._block_size + 1 - num_accepted
        if trim_amount > 0:
            self._adapter.trim_cache(self._cache, trim_amount)

        # 5. Slice hidden states to accepted prefix for next draft
        sliced_hidden = {}
        for layer_id, h in hidden_states.items():
            sliced_hidden[layer_id] = h[:, :num_accepted, :]
        self._last_hidden_states = sliced_hidden

        # 6. Update state
        self._last_target_logit = verification_logits[num_accepted - 1]
        mx.eval(self._last_target_logit)
        self._cache_seq_len += num_accepted
        self._pending_token = int(mx.argmax(self._last_target_logit).item())

        return accepted, self._block_size

    def _draft_block(self, pending_token: int) -> list[int]:
        """Use the draft model to propose a block of candidate tokens.

        Each token is fed as input to the next step (autoregressive), but
        unlike SpeculativeDecoder there is no KV cache accumulating across
        draft steps — the draft model's self-attention sees only [context,
        current_token] each time, not the full history of draft tokens.
        This is by design: the block-diffusion approach relies on the
        target's hidden states as the primary conditioning signal.
        """
        inp = mx.array([[pending_token]])

        # Extract last-position hidden states and pre-compute context once
        last_hidden = {}
        for layer_id, h in self._last_hidden_states.items():
            last_hidden[layer_id] = h[:, -1:, :]
        context = self._draft.build_context(last_hidden)

        tokens: list[int] = []
        for _ in range(self._block_size):
            logits = self._draft.forward_with_context(inp, context)
            next_logits = logits[:, -1, :]
            mx.eval(next_logits)
            next_token = int(mx.argmax(next_logits, axis=-1).item())
            tokens.append(next_token)
            inp = mx.array([[next_token]])

        return tokens
