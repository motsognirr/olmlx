"""EAGLE-style autoregressive draft model.

The draft is a small transformer (typically 1-2 decoder layers) that
predicts the *next hidden state* in feature space, not the next token
directly. Inputs at each step are
``concat(h_prev, embed(token_prev))`` projected to ``hidden_size``. The
output hidden goes through the target's ``lm_head`` (shared via
``bind()``) for the next-token distribution.

Architecture (per EAGLE-1, arxiv 2401.15077):

1. **Input projection**: Linear(2 * hidden_size, hidden_size). Takes
   ``concat([h_target, embed(token)])`` and produces a ``hidden_size``
   feature.
2. **Decoder layer(s)**: standard Qwen-style block (RMSNorm + GQA
   self-attention with RoPE + MLP). One layer is the EAGLE-1 default;
   two is the EAGLE-2 variant.
3. **Output norm**: RMSNorm on the final hidden.
4. **LM head (shared)**: borrowed from target via ``bind()``. Mapping
   final hidden to logits.

The draft does not own ``embed_tokens`` or ``lm_head`` — those are
borrowed from the target. This avoids duplicating ~vocab*hidden weights
(~250M parameters for Qwen3.5's 248k vocab × 5120 hidden) in the draft
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class EagleConfig:
    """Configuration for the EAGLE draft model.

    Most fields are inherited from the target's config so weights are
    dimensionally compatible at inference. ``block_size`` is the
    EAGLE-specific draft-token count per verify (analog of DFlash's
    ``block_size``).
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
    rope_scaling: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(
                f"block_size must be >= 1 for EAGLE drafts; got {self.block_size}"
            )
        if self.num_hidden_layers < 1:
            raise ValueError(
                f"num_hidden_layers must be >= 1; got {self.num_hidden_layers}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a "
                f"multiple of num_key_value_heads ({self.num_key_value_heads}) "
                "for GQA — same constraint mlx-lm enforces."
            )


class _Attention(nn.Module):
    """Standard GQA self-attention with RoPE.

    Mirrors mlx-lm's Qwen3-style attention block. Projects q/k/v with
    GQA shapes (``num_heads`` for q, ``num_kv_heads`` for k/v), applies
    RoPE, runs SDPA (with causal mask in prefill, none in generation
    since the cache enforces order), and projects back to
    ``hidden_size``.
    """

    def __init__(self, args: EagleConfig):
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.rope = initialize_rope(
            args.head_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, L, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, L, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, L, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class _MLP(nn.Module):
    def __init__(self, args: EagleConfig):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _DecoderLayer(nn.Module):
    def __init__(self, args: EagleConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = _Attention(args)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = _MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class EagleDraftModel(nn.Module):
    """Autoregressive draft head conditioned on target hidden states.

    Forward signature: ``(token_ids, h_prev, cache=None) -> (logits,
    h_new)``.

    - ``token_ids``: ``(B, L)`` int32 tensor of "previous tokens" — for
      the first draft step after a target forward, this is the target's
      most recently sampled token; for subsequent steps it's the
      draft's own previously-sampled token.
    - ``h_prev``: ``(B, L, hidden_size)`` — previous-position hidden
      states. For the first draft step this is the target's last-layer
      hidden at the corresponding positions; for subsequent steps it's
      the draft's own ``h_new`` from the prior call.
    - ``cache``: optional list of ``KVCache`` (one per draft layer) so
      autoregressive generation does not redo the prefix on every
      step.

    Returns ``(logits, h_new)``:

    - ``logits``: ``(B, L, vocab_size)``.
    - ``h_new``: ``(B, L, hidden_size)`` post-norm draft hidden, used
      both for the next autoregressive step and (during training) as a
      regression target if you want auxiliary supervision against the
      target's true hidden.

    ``embed_tokens`` and ``lm_head`` are not owned by this module — they
    must be borrowed from the target via ``bind()`` before any forward
    pass.
    """

    def __init__(self, args: EagleConfig):
        super().__init__()
        self.args = args
        # 2 * hidden because we concatenate h_prev (one hidden_size) and
        # embed(token) (one hidden_size) before projecting back down.
        self.input_proj = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.layers = [_DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Borrowed at bind() time from the target. Set via
        # ``object.__setattr__`` (in ``bind``/``unbind``) to bypass
        # ``nn.Module.__setattr__``, which would register them as
        # tracked children — that would (a) duplicate the target's huge
        # embed/lm_head tensors in the draft's saved checkpoint, and
        # (b) include them in ``draft.parameters()`` so
        # ``nn.value_and_grad(draft, ...)`` would compute gradients
        # against the target's frozen weights. Both are silent
        # correctness/scale failures we want to avoid.
        object.__setattr__(self, "embed_tokens", None)
        object.__setattr__(self, "lm_head", None)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def bind(self, target_model: Any) -> None:
        """Borrow ``embed_tokens`` and ``lm_head`` from the target.

        Walks several attribute chains so VLM and other wrapped
        targets are supported (mirrors DFlash's lookup pattern):

        - embed: ``.embed_tokens``, ``.model.embed_tokens``,
          ``.language_model.model.embed_tokens``,
          ``.language_model.embed_tokens``
        - lm_head: ``.lm_head``, ``.language_model.lm_head``,
          ``.model.lm_head``, ``.language_model.model.lm_head``,
          and finally ``embed.as_linear`` for tied-embeddings models
        """
        embed = self._find_embed(target_model)
        if embed is None:
            raise AttributeError(
                f"Cannot find embed_tokens on target model "
                f"{type(target_model).__name__}; tried .embed_tokens, "
                ".model.embed_tokens, .language_model.model.embed_tokens, "
                ".language_model.embed_tokens"
            )
        lm_head = self._find_lm_head(target_model, embed)
        if lm_head is None:
            raise AttributeError(
                f"Cannot find lm_head on target model "
                f"{type(target_model).__name__}; tried .lm_head, "
                ".language_model.lm_head, .model.lm_head, "
                ".language_model.model.lm_head, embed.as_linear"
            )
        # ``object.__setattr__`` keeps these out of the parameter tree
        # (see ``__init__`` comment) — DO NOT switch to plain ``self.x =``.
        object.__setattr__(self, "embed_tokens", embed)
        object.__setattr__(self, "lm_head", lm_head)

    @staticmethod
    def _find_embed(target: Any) -> nn.Module | None:
        for path in (
            ("embed_tokens",),
            ("model", "embed_tokens"),
            ("language_model", "model", "embed_tokens"),
            ("language_model", "embed_tokens"),
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
    def _find_lm_head(target: Any, embed: nn.Module) -> nn.Module | None:
        for path in (
            ("lm_head",),
            ("language_model", "lm_head"),
            ("model", "lm_head"),
            ("language_model", "model", "lm_head"),
        ):
            obj: Any = target
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        # Tied-embeddings fallback: many small Qwen variants share the
        # input embedding with the lm_head and expose
        # ``embed_tokens.as_linear``.
        as_linear = getattr(embed, "as_linear", None)
        if callable(as_linear):
            return as_linear  # type: ignore[no-any-return]
        return None

    def bind_via_modules(self, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        """Test-only entry point: bind to externally-provided modules
        without going through a target wrapper. Useful in unit tests
        that want to drive forward without instantiating a fake target.
        """
        object.__setattr__(self, "embed_tokens", embed_tokens)
        object.__setattr__(self, "lm_head", lm_head)

    def unbind(self) -> None:
        object.__setattr__(self, "embed_tokens", None)
        object.__setattr__(self, "lm_head", None)

    def __call__(
        self,
        token_ids: mx.array,
        h_prev: mx.array,
        cache: list[KVCache] | None = None,
        compute_logits: bool = True,
    ) -> tuple[mx.array | None, mx.array]:
        """Forward pass.

        ``compute_logits=False`` skips the final ``lm_head`` projection
        and returns ``(None, h_new)``. Training uses this to apply
        ``lm_head`` only at a small subset of sampled positions —
        materialising the full ``(B, L, vocab_size)`` logits tensor on
        a 250k-vocab × 2048-token sequence is the dominant cost (~4 GB
        per forward) and dwarfs the rest of the draft.
        """
        if self.embed_tokens is None:
            raise RuntimeError(
                "EagleDraftModel.__call__ requires bind() to attach the "
                "target's embed_tokens before forward."
            )
        if compute_logits and self.lm_head is None:
            raise RuntimeError(
                "EagleDraftModel.__call__(compute_logits=True) requires "
                "bind() to attach the target's lm_head."
            )
        # Embed tokens, concat with h_prev along feature axis, project
        # back down to hidden_size.
        emb = self.embed_tokens(token_ids)
        x = self.input_proj(mx.concatenate([h_prev, emb], axis=-1))

        # Causal mask for prefill (L > 1); for single-token decode the
        # cache enforces ordering and no mask is needed.
        L = x.shape[1]
        mask: mx.array | None = None
        if L > 1:
            mask = create_causal_mask(L, offset=cache[0].offset if cache else 0)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, mask=mask, cache=layer_cache)

        h_new = self.norm(x)
        if compute_logits:
            assert self.lm_head is not None  # narrow for type checker
            return self.lm_head(h_new), h_new
        return None, h_new
