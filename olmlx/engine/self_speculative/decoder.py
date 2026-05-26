"""Self-speculative decoding (LayerSkip-style).

Uses the target model's own early layers as an autoregressive draft.
No external draft model required. Works under Flash and Flash-MoE.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
except ImportError:  # pragma: no cover — mlx-lm is always available at runtime
    make_prompt_cache = None
    trim_prompt_cache = None

from olmlx.engine.gdn_rollback import find_gdn_class, get_model_layers
from olmlx.engine.speculative import verify_draft_greedy

logger = logging.getLogger(__name__)


def _find_module(model: Any, paths: tuple[tuple[str, ...], ...]) -> Any | None:
    for path in paths:
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


_EMBED_PATHS: tuple[tuple[str, ...], ...] = (
    ("embed_tokens",),
    ("model", "embed_tokens"),
    ("language_model", "model", "embed_tokens"),
    ("language_model", "embed_tokens"),
)

_NORM_PATHS: tuple[tuple[str, ...], ...] = (
    ("norm",),
    ("model", "norm"),
    ("language_model", "model", "norm"),
    ("language_model", "norm"),
)

_LM_HEAD_PATHS: tuple[tuple[str, ...], ...] = (
    ("lm_head",),
    ("language_model", "lm_head"),
    ("model", "lm_head"),
    ("language_model", "model", "lm_head"),
)


def _logits(out: Any) -> mx.array:
    return cast(mx.array, getattr(out, "logits", out))


def _eval_cache(cache: list) -> None:
    """Materialise KV cache state without evaluating logits (mirrors
    the version in speculative.py)."""
    arrs: list[mx.array] = []
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
            state = getattr(c, "state", None)
            if isinstance(state, (list, tuple)):
                arrs.extend(a for a in state if isinstance(a, mx.array))
        for attr in ("_key_dequant", "_value_dequant"):
            buf = getattr(c, attr, None)
            if isinstance(buf, mx.array):
                arrs.append(buf)
    if arrs:
        mx.eval(*arrs)


def _prefill_last_logit(model: Any, prompt: mx.array, cache: list) -> mx.array:
    """Return the logit for the last prompt position (mirrors speculative.py).

    Two-pass split: prefix fills cache (logits discarded), last token
    produces a [1, 1, vocab] logit.
    """
    if prompt.shape[1] <= 1:
        return _logits(model(prompt, cache=cache))[0, -1, :]
    prefix, last = prompt[:, :-1], prompt[:, -1:]
    model(prefix, cache=cache)
    _eval_cache(cache)
    return _logits(model(last, cache=cache))[0, 0, :]


class SelfSpeculativeDecoder:
    """Self-speculative decoding using the target's own early layers.

    The first ``num_early_layers`` of the target serve as the draft
    model: they generate λ candidate tokens autoregressively through a
    shared KV cache.  The full target (all layers) then verifies all
    candidates in a single forward pass.

    Works under Flash and Flash-MoE because no hidden-state hooks or
    target patching are involved — just running the model's own layers.

    Limitations
    -----------
    * Hybrid linear-attention targets (Qwen3.5/3.6, Qwen3-Coder-Next)
      with ``GatedDeltaNet`` layers are **not supported** — their
      ``ArraysCache`` RNN-style state cannot be rolled back after the
      autoregressive draft without ``GDNStateCapture`` (planned for a
      follow-up PR).
    * Sliding-window attention targets with ``RotatingKVCache`` /
      ``ChunkedKVCache`` in early layers are **not supported** — the
      per-step trim between draft and verify fails on non-trimmable
      caches (the same models gated off from cross-request prompt caching
      in issue #343).
    """

    def __init__(
        self,
        target_model: nn.Module,
        num_early_layers: int,
        num_speculative_tokens: int = 4,
        acceptance_rate_ema: float = 0.9,
    ):
        if make_prompt_cache is None or trim_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache is unavailable; self-speculative "
                "decoding requires it for KV-cache management."
            )
        if num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1; got {num_speculative_tokens}"
            )

        self._target = target_model

        layers = get_model_layers(target_model)
        self._total_layers = len(layers)
        if num_early_layers < 1 or num_early_layers >= self._total_layers:
            raise ValueError(
                f"num_early_layers ({num_early_layers}) must be in "
                f"[1, {self._total_layers - 1}] for a target with "
                f"{self._total_layers} layers"
            )
        self._N = num_early_layers
        self._layers = layers

        # Discover embed, norm, lm_head via path probing
        self._embed = _find_module(target_model, _EMBED_PATHS)
        if self._embed is None:
            raise ValueError(
                f"Cannot find embed_tokens on model {type(target_model).__name__}"
            )
        self._norm = _find_module(target_model, _NORM_PATHS)
        if self._norm is None:
            raise ValueError(f"Cannot find norm on model {type(target_model).__name__}")
        self._lm_head = _find_module(target_model, _LM_HEAD_PATHS)
        if self._lm_head is None:
            # Tied-embeddings fallback
            as_linear = getattr(self._embed, "as_linear", None)
            if callable(as_linear):
                self._lm_head = as_linear
            else:
                raise ValueError(
                    f"Cannot find lm_head on model {type(target_model).__name__}"
                )

        # --- Guard: hybrid linear-attention targets (GatedDeltaNet) ---
        # The autoregressive draft pass advances ``ArraysCache`` RNN-state
        # for any GDN layers among the first N.  ``trim_prompt_cache`` is
        # a no-op on these, so the state diverges after partial acceptance
        # and corrupts every subsequent step.  Full GDN rollback
        # (``GDNStateCapture`` + ``rollback_autoregressive``) is planned
        # for a follow-up PR.
        gdn_cls = find_gdn_class(target_model)
        if gdn_cls is not None:
            raise NotImplementedError(
                f"self_speculative is not yet supported on hybrid "
                f"linear-attention targets ({gdn_cls.__module__}.{gdn_cls.__name__} "
                "detected). The autoregressive draft mutates GatedDeltaNet "
                "recurrent state which cannot be rolled back with "
                "trim_prompt_cache. Planned for a follow-up PR."
            )

        # --- Guard: non-trimmable caches in early layers ---
        # Probe the first early-layer cache to see if ``trim()`` is
        # supported.  RotatingKVCache / ChunkedKVCache in Gemma 4 /
        # gpt-oss models fail ``trim_prompt_cache`` (issue #343) — the
        # per-step discard of draft entries would either raise or
        # silently corrupt offsets.
        probe_cache = make_prompt_cache(target_model)
        early_non_trimmable = False
        for i in range(min(self._N, len(probe_cache))):
            if not hasattr(probe_cache[i], "trim") or not callable(
                getattr(probe_cache[i], "trim", None)
            ):
                early_non_trimmable = True
                break
        if early_non_trimmable:
            raise NotImplementedError(
                "self_speculative is not supported on models with "
                "non-trimmable KV caches in early layers (e.g. "
                "RotatingKVCache/ChunkedKVCache). The per-step cache "
                "trim between draft and verify cannot be performed."
            )

        self._lambda = num_speculative_tokens
        self._alpha = 0.5
        self._alpha_ema = acceptance_rate_ema

        # Per-request state (populated by prefill)
        self._cache: list | None = None
        self._last_logit: mx.array | None = None
        self._pending_token: int | None = None
        self._prompt_len: int = 0

        # Stats (reset on prefill)
        self._stats_steps: int = 0
        self._stats_proposed: int = 0
        self._stats_accepted_draft: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """No-op — no GDN patches or external resources to release."""

    def reset(self) -> None:
        self._cache = None
        self._last_logit = None
        self._pending_token = None
        self._prompt_len = 0
        self._stats_steps = 0
        self._stats_proposed = 0
        self._stats_accepted_draft = 0

    def stats_summary(self) -> dict[str, Any]:
        steps = self._stats_steps
        proposed = self._stats_proposed
        accepted_draft = self._stats_accepted_draft
        acceptance_rate = accepted_draft / proposed if proposed else 0.0
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

    # ------------------------------------------------------------------
    # Protocol: prefill / step
    # ------------------------------------------------------------------

    def prefill(self, prompt: mx.array) -> int:
        """Process the prompt through the full model, populating KV cache.

        Returns the first generated token.
        """
        self.reset()

        self._cache = make_prompt_cache(self._target)

        # Reset cached RoPE state on VLM targets (mlx-vlm 0.4.4)
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(self._target, attr):
                setattr(self._target, attr, None)

        self._last_logit = _prefill_last_logit(self._target, prompt, self._cache)
        mx.eval(self._last_logit)

        self._prompt_len = prompt.shape[1]
        first_token = int(mx.argmax(self._last_logit).item())
        self._pending_token = first_token
        return first_token

    def step(self) -> tuple[list[int], int]:
        """One self-speculative decoding step.

        Must call ``prefill()`` first.
        Returns (accepted_tokens, num_draft_generated).

        Cache-length invariant
        ----------------------
        After each ``step()``, ``self._cache`` contains exactly
        ``self._prompt_len`` entries across all layers (the original
        prompt plus every accepted token).  The pending token is *not*
        in the cache — it is fed as the first token of the verify batch
        in the next step.  This matches the ``SpeculativeDecoder``
        invariant.
        """
        assert self._cache is not None, "Call prefill() before step()"
        assert self._pending_token is not None, "Call prefill() before step()"

        pending = self._pending_token
        draft_tokens: list[int] = []

        # --- Draft: autoregressive through early layers 0..N-1 ---
        next_token = pending
        for _ in range(self._lambda):
            inp = mx.array([[next_token]])
            h = self._embed(inp)
            for i in range(self._N):
                h = self._layers[i](h, cache=self._cache[i])
            h = self._norm(h)
            logits = self._lm_head(h)
            next_logits = logits[:, -1, :]
            mx.eval(next_logits)
            next_token = int(mx.argmax(next_logits, axis=-1).item())
            draft_tokens.append(next_token)

        # --- Pre-verify: trim early-layer caches back to prompt_len ---
        # The draft advanced cache[0..N-1] by lambda entries; discard
        # them so the verify pass can append uniformly across all layers.
        # Late layers (N..L-1) were never advanced — they're still at
        # prompt_len, so trimming them is a no-op (but we only trim
        # early layers anyway).
        for i in range(self._N):
            trim_prompt_cache([self._cache[i]], self._lambda)

        # --- Verify: full model on [pending, D_1, ..., D_lambda] ---
        # The verify feed is λ+1 tokens.  All layers append uniformly.
        all_tokens = mx.array([[pending] + draft_tokens])
        target_out = _logits(self._target(all_tokens, cache=self._cache))
        mx.eval(target_out)
        verification_logits = target_out[0]  # (lambda+1, vocab)

        # --- Accept/reject ---
        accepted = verify_draft_greedy(draft_tokens, verification_logits)
        num_accepted = len(accepted)

        # --- Trim caches to accepted prefix ---
        # After verify the cache grew by λ+1 entries (for all layers).
        # We keep ``num_accepted`` of those entries; trim the rest.
        # ``num_accepted`` is always >= 1 (greedy accept guarantees at
        # least one token).
        trim_count = max(self._lambda + 1 - num_accepted, 0)
        if trim_count > 0:
            trim_prompt_cache(self._cache, trim_count)
        # Post-trim cache length = old_prompt_len + num_accepted.
        # ``_prompt_len`` is updated to the same value below so the
        # invariant holds.

        # --- Update state ---
        self._prompt_len += num_accepted
        self._last_logit = verification_logits[num_accepted - 1]
        mx.eval(self._last_logit)
        self._pending_token = int(mx.argmax(self._last_logit).item())

        # --- Update acceptance-rate EMA and stats ---
        num_accepted_draft = min(num_accepted - 1, self._lambda)
        acceptance = num_accepted_draft / max(self._lambda, 1)
        self._alpha = self._alpha_ema * self._alpha + (1 - self._alpha_ema) * acceptance
        self._stats_steps += 1
        self._stats_proposed += self._lambda
        self._stats_accepted_draft += num_accepted_draft

        return accepted, self._lambda
