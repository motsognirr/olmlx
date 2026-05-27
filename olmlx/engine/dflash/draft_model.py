"""DFlash block-diffusion draft model.

Mirrors the upstream z-lab/dflash MLX architecture
(https://github.com/z-lab/dflash, MIT-licensed) so any pre-trained DFlash
draft (Qwen3, Qwen3.5, Qwen3-Coder, Gemma 4, MiniMax, Kimi, gpt-oss,
Llama 3.1, …) loads without per-target adapter code.

Key shape: the draft is a Qwen-style transformer whose attention takes
two streams — current/proposal tokens and a per-step "context" stream
that carries target hidden states (concatenated across the configured
``target_layer_ids``, projected through ``fc + hidden_norm``). Only the
context stream's K/V is appended to the per-layer KV cache; the
proposal stream's K/V is concatenated transiently for SDPA.

The draft borrows ``embed_tokens`` and ``lm_head`` from the target via
``bind(target_model)`` so vocab alignment is automatic and weights are
not duplicated in memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope

logger = logging.getLogger(__name__)


@dataclass
class DraftConfig:
    """Configuration for the dflash draft model.

    Schema matches upstream's ``DFlashConfig`` (z-lab/dflash). Required
    fields raise on missing keys at load time. ``target_layer_ids`` and
    ``mask_token_id`` come from the nested ``dflash_config`` block in
    the draft's ``config.json``; the loader flattens them before
    constructing this dataclass.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    num_target_layers: int
    target_layer_ids: list[int]
    mask_token_id: int
    rope_scaling: dict[str, Any] | None = None
    layer_types: tuple[str, ...] = field(default_factory=tuple)
    sliding_window: int | None = None
    final_logit_softcapping: float | None = None
    attention_causal: bool = False

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = tuple(["full_attention"] * self.num_hidden_layers)
        else:
            self.layer_types = tuple(self.layer_types)
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has length {len(self.layer_types)} but "
                f"num_hidden_layers is {self.num_hidden_layers}; they must match"
            )
        unsupported = {
            t
            for t in self.layer_types
            if t
            not in {
                "full_attention",
                "sliding_attention",
            }
        }
        if unsupported:
            raise ValueError(
                f"Unsupported layer_types: {sorted(unsupported)}. "
                "Only 'full_attention' and 'sliding_attention' are supported."
            )
        if any(t == "sliding_attention" for t in self.layer_types) and (
            self.sliding_window is None or self.sliding_window < 2
        ):
            # ``sliding_window <= 0`` makes ``DFlashAttention``'s
            # ``keep = sliding_window - 1`` go negative; ``sliding_window
            # == 1`` makes ``keep = 0`` so every context token is
            # discarded the moment it's produced and the draft attends
            # to nothing. Both are silent failure modes — reject early.
            raise ValueError(
                "sliding_window must be >= 2 when any layer uses "
                f"'sliding_attention'; got {self.sliding_window!r}"
            )
        if len(self.target_layer_ids) != self.num_target_layers:
            raise ValueError(
                f"target_layer_ids has length {len(self.target_layer_ids)} but "
                f"num_target_layers is {self.num_target_layers}"
            )


def _build_rope(config: DraftConfig) -> nn.Module:
    return initialize_rope(
        config.head_dim,
        config.rope_theta,
        traditional=False,
        scaling_config=config.rope_scaling,
        max_position_embeddings=config.max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    """Cross/self attention over (proposal, context) streams.

    The context stream contributes K/V that go into the persistent cache;
    the proposal stream contributes K/V that are concatenated transiently
    for SDPA but never written to cache. Both streams share Q/K/V/O
    projections and QK-RMSNorm. RoPE is applied to Q (offset by the
    context length), proposal-K (offset by the context length), and
    context-K (offset by the prior cache offset).

    When ``config.attention_causal`` is ``False`` (bidirectional), the
    mask is ``None`` for full-attention layers only. Sliding-attention
    layers always apply a sliding-causal mask (via
    ``create_causal_mask``) when the combined context + proposal length
    exceeds the sliding window — matching upstream z-lab/dflash
    behaviour regardless of the ``causal`` flag. For sequences shorter
    than the window the mask remains ``None`` so in-window attention is
    fully bidirectional. The ``RotatingKVCache`` eviction (enforced by
    ``make_cache()`` via ``layer_types``) works alongside the explicit
    spatial mask — the mask bounds the attention dot-product to the
    window while the cache discards keys outside it.

    .. note::

       ``create_causal_mask`` enforces *both* the spatial window and
       causal ordering within the proposal stream.  When the window is
       exceeded, proposal token ``i`` cannot attend to proposal token
       ``j > i`` even when ``attention_causal`` is ``False``.  This
       matches upstream z-lab/dflash and avoids a train/inference
       mismatch, but a custom draft architecture that was trained with
       full bidirectionality for proposal tokens would silently degrade
       under this mask.

    .. note::

       With a full ``RotatingKVCache`` (``max_size = window - 1``,
       the ``make_cache()`` convention for sliding layers),
       ``ctx_len`` is ``window - 1``, so ``ctx_len + L > window``
       holds for ``L >= 2`` — the mask kicks in at two or more
       proposal tokens.  In practice DFlash always processes
       ``block_size >= 2`` tokens per step, so the mask is
       effectively unconditional in steady state.
    """

    def __init__(self, config: DraftConfig, layer_idx: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.causal = config.attention_causal

        if self.is_sliding and self.sliding_window is None:
            raise ValueError(
                "DFlashAttention layer has is_sliding=True but no "
                "sliding_window set — a sliding-attention layer "
                "requires a window size.  This occurs when "
                "DFlashAttention is constructed directly without "
                "DraftConfig validation."
            )

        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        x_ctx: mx.array,
        rope: nn.Module,
        cache: Any,
    ) -> mx.array:
        B, L, _ = x.shape
        S = x_ctx.shape[1]

        # When sliding, drop ctx tokens that would fall outside the window.
        # Advance ``cache.offset`` by the skip amount so subsequent steps
        # pick up the correct RoPE base — the skipped tokens leave a
        # phantom positional gap that must be tracked across calls.
        # ``update_and_fetch`` then advances from this corrected base.
        #
        # This relies on the current mlx-lm invariant that
        # ``RotatingKVCache.update_and_fetch`` uses ``self.offset`` as
        # both the ring-buffer write pointer and the RoPE position
        # accumulator, and increments it by exactly ``S`` (the number of
        # tokens written, *not* ``S + skip``). If mlx-lm ever changes
        # this, the rollback trim in ``_trim_recent_cache`` would also
        # break — the invariant is observable at the protocol level.
        if self.is_sliding and S > (self.sliding_window or 0) - 1:
            keep = (self.sliding_window or 0) - 1
            skip = S - keep
            x_ctx = x_ctx[:, skip:]
            S = x_ctx.shape[1]
            cache.offset += skip

        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)

        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)

        ctx_len = keys.shape[2] - L
        mask = "causal" if self.causal else None
        # The ``or 0`` fallback is dead code after the __init__ guard
        # (raising ValueError when is_sliding=True without a window),
        # but kept as belt-and-suspenders in case the guard is ever
        # bypassed (e.g. direct construction in tests).
        if self.is_sliding and ctx_len + L > (self.sliding_window or 0):
            # Upstream z-lab/dflash always applies a sliding-causal mask on
            # sliding layers regardless of causal mode (see
            # dflash/model_mlx.py:110-114). The spatial window constrains
            # which positions a token can interact with independently of the
            # causal directionality; matching this avoids a train/inference
            # mismatch for sliding-attention draft architectures.
            mask = create_causal_mask(
                L, offset=ctx_len, window_size=self.sliding_window
            )

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    """Qwen-style SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    """One draft transformer block: attention + SwiGLU MLP with RMSNorm."""

    def __init__(self, config: DraftConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        x_ctx: mx.array,
        rope: nn.Module,
        cache: Any,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    """Block-diffusion draft model conditioned on target hidden states.

    Embedding and LM head are populated by ``bind(target_model)``; until
    then the draft cannot run. The ``fc`` projection takes the
    concatenation of target hidden states across the configured
    ``target_layer_ids`` and projects them down to the draft hidden size.
    ``hidden_norm`` normalizes the projected context once per call; the
    same ``h_ctx`` tensor is fed to every layer's attention.
    """

    def __init__(self, config: DraftConfig):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(
            config.num_target_layers * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(config)

        # The warning fires for every non-causal sliding draft —
        # including correctly-trained ones that were prepared with the
        # patched pipeline.  The checkpoint format does not record which
        # code version trained it, so we cannot distinguish legacy
        # (mask-free) training from post-#317 (mask-applied) training.
        # Newly-trained drafts correctly expect the mask at inference
        # time and can safely ignore this warning; the heuristic "if
        # acceptance is unexpectedly low, re-train" still holds.
        if any(
            t == "sliding_attention" and not config.attention_causal
            for t in config.layer_types
        ):
            logger.warning(
                "DFlash draft (%d layers) contains sliding_attention "
                "layers with attention_causal=False.  As of gh#317 the "
                "inference path applies a sliding-causal mask whenever "
                "ctx_len + L > sliding_window regardless of the causal "
                "flag.  Correctly-trained post-#317 drafts expect this "
                "mask and can safely ignore this warning.  Older drafts "
                "trained without the mask may show degraded acceptance "
                "rates — re-train if acceptance is unexpectedly low.",
                config.num_hidden_layers,
            )

        # Populated by ``bind()``. We use ``object.__setattr__`` there to
        # bypass mlx's ``Module.__setattr__`` — assigning an
        # ``nn.Module`` (which is also a ``dict``) registers it as a
        # tracked child, so the borrowed ``embed_tokens`` / ``lm_head``
        # would otherwise appear in ``self.parameters()`` and serialise
        # via ``save_weights``. Initial ``None`` assignments are safe
        # (``None`` isn't tracked) but we mirror the bypass for symmetry.
        self.embed_tokens: nn.Embedding | None = None
        self.embed_scale: float = 1.0
        self.lm_head: nn.Module | None = None
        self._bound: bool = False

    def bind(self, target_model: nn.Module) -> None:
        """Borrow ``embed_tokens``, ``embed_scale``, and ``lm_head`` from
        the target model.

        Walks a small list of known model shapes:
        ``target.embed_tokens``, ``target.model.embed_tokens``,
        ``target.language_model.model.embed_tokens``. The lm_head chain
        is the same with ``embed_tokens.as_linear`` as a tied-embeddings
        fallback for models that share input/output projections.

        Borrowed modules are stored via ``object.__setattr__`` so they
        do not enter the draft's parameter tree.
        """
        embed = self._find_embed(target_model)
        if embed is None:
            raise ValueError(
                f"Cannot find embed_tokens on target model "
                f"{type(target_model).__name__}; tried .embed_tokens, "
                ".model.embed_tokens, .language_model.model.embed_tokens"
            )
        scale = self._find_embed_scale(target_model, embed)
        lm_head = self._find_lm_head(target_model, embed)
        if lm_head is None:
            raise ValueError(
                f"Cannot find lm_head on target model "
                f"{type(target_model).__name__}; tried .lm_head, "
                ".language_model.lm_head, embed.as_linear (tied embeddings)"
            )
        object.__setattr__(self, "embed_tokens", embed)
        object.__setattr__(self, "embed_scale", scale)
        object.__setattr__(self, "lm_head", lm_head)
        self._bound = True

    def unbind(self) -> None:
        object.__setattr__(self, "embed_tokens", None)
        object.__setattr__(self, "embed_scale", 1.0)
        object.__setattr__(self, "lm_head", None)
        self._bound = False

    @staticmethod
    def _find_embed(target: nn.Module) -> nn.Embedding | None:
        for path in (
            ("embed_tokens",),
            ("model", "embed_tokens"),
            ("language_model", "model", "embed_tokens"),
        ):
            obj: Any = target
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        return None

    @staticmethod
    def _find_embed_scale(target: nn.Module, embed: nn.Module) -> float:
        for obj in (
            embed,
            target,
            getattr(target, "model", None),
            getattr(target, "language_model", None),
        ):
            if obj is None:
                continue
            scale = getattr(obj, "embed_scale", None)
            if scale is not None:
                try:
                    return float(scale)
                except (TypeError, ValueError):
                    continue
        return 1.0

    @staticmethod
    def _find_lm_head(target: nn.Module, embed: nn.Module) -> nn.Module | None:
        # Mirror ``_find_embed``'s walk so VLMs that bury the head under
        # ``language_model.model`` (next to ``embed_tokens``) still
        # resolve. Order matters: top-level wins over the same attribute
        # under a wrapper, matching mlx-lm's lookup convention.
        for path in (
            ("lm_head",),
            ("language_model", "lm_head"),
            ("language_model", "model", "lm_head"),
        ):
            obj: Any = target
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        # Tied embeddings: many mlx-lm models expose ``as_linear`` on the
        # input embedding to project the final hidden state back to vocab.
        as_linear = getattr(embed, "as_linear", None)
        if callable(as_linear):
            return as_linear  # type: ignore[return-value]
        return None

    def make_cache(self) -> list[Any]:
        """Construct per-layer draft KV caches: rotating for sliding
        layers, plain for full-attention layers."""
        caches: list[Any] = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                window = self.config.sliding_window or 1
                caches.append(RotatingKVCache(max_size=max(window - 1, 1), keep=0))
            else:
                caches.append(KVCache())
        return caches

    def __call__(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: list[Any],
        logits_start: int = 0,
    ) -> mx.array:
        """Run the draft.

        Args:
            inputs: ``(B, L)`` token ids.
            target_hidden: ``(B, L, num_target_layers * hidden_size)`` —
                the concatenation of target hidden states across all
                configured ``target_layer_ids`` (the caller produces this
                from the layer-hook output).
            cache: per-layer KV caches from ``make_cache()``.
            logits_start: slice the hidden tensor at this position
                before applying the LM head. Block-diffusion drafting
                uses ``logits_start=1`` to drop the position-0 logit
                (which corresponds to the already-known last committed
                token).

        Returns:
            ``(B, L - logits_start, vocab_size)`` logits.
        """
        if not self._bound:
            raise RuntimeError(
                "DFlashDraftModel.bind(target_model) must be called before "
                "running the draft"
            )
        # Hard guards rather than ``assert`` — ``assert`` is stripped
        # under ``python -O``, and the closely-related ``_bound`` check
        # above is a ``raise`` for the same reason. ``_bound`` should
        # imply both attributes are set, but we surface a loud error
        # anyway in case of programmatic misuse.
        if self.embed_tokens is None or self.lm_head is None:
            raise RuntimeError(
                "DFlashDraftModel ``_bound`` is True but ``embed_tokens`` / "
                "``lm_head`` are unset; bind() invariant violated."
            )
        h = self.embed_tokens(inputs) * self.embed_scale
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            h = layer(h, h_ctx, self.rope, layer_cache)
        h = self.norm(h)
        if logits_start:
            h = h[:, logits_start:, :]
        logits = self.lm_head(h)
        cap = self.config.final_logit_softcapping
        # ``cap is not None and cap > 0.0`` rather than ``if cap`` so a
        # config with ``final_logit_softcapping: 0.0`` is rejected
        # (would otherwise silently skip softcapping AND become a
        # divide-by-zero if someone later flips this to ``is not None``).
        if cap is not None and cap > 0.0:
            logits = mx.tanh(logits / cap) * cap
        return logits
