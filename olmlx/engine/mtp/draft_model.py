"""MTP draft head: one full-attention Qwen3.6 layer + MTP front-end.

The head consumes ``(token_{i+1}, h_i)`` where ``h_i`` is the target's
last-layer (pre-``model.norm``) hidden, and produces ``(logits, h_new)``.
``h_new`` is the layer output BEFORE the head's own ``norm`` — it is fed
back as ``h_prev`` for the next autoregressive draft step (DeepSeek/Qwen
MTP convention). ``norm`` is applied only to compute logits via the
target's borrowed ``lm_head``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer as _Qwen35DecoderLayer
from mlx_lm.models.qwen3_5 import TextModelArgs as _Qwen35TextArgs


@dataclass
class MTPConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    full_attention_interval: int
    block_size: int
    rope_parameters: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    # MoE (0 => dense)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    # Quantization (None => not quantized)
    quant_group_size: int | None = None
    quant_bits: int | None = None

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MTPConfig":
        text = config.get("text_config", config)
        quant = config.get("quantization") or config.get("quantization_config") or {}
        return cls(
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text.get(
                "head_dim", text["hidden_size"] // text["num_attention_heads"]
            ),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            vocab_size=text["vocab_size"],
            max_position_embeddings=text.get("max_position_embeddings", 262144),
            full_attention_interval=text.get("full_attention_interval", 4),
            block_size=config.get("block_size", 1),
            rope_parameters=text.get("rope_parameters"),
            tie_word_embeddings=text.get("tie_word_embeddings", False),
            num_experts=text.get("num_experts", 0),
            num_experts_per_tok=text.get("num_experts_per_tok", 0),
            moe_intermediate_size=text.get("moe_intermediate_size", 0),
            shared_expert_intermediate_size=text.get(
                "shared_expert_intermediate_size", 0
            ),
            norm_topk_prob=text.get("norm_topk_prob", True),
            decoder_sparse_step=text.get("decoder_sparse_step", 1),
            quant_group_size=quant.get("group_size"),
            quant_bits=quant.get("bits"),
        )

    def to_qwen35_text_args(self) -> _Qwen35TextArgs:
        """Build the mlx-lm ``TextModelArgs`` the reused ``DecoderLayer``
        expects. ``num_hidden_layers`` is forced to ``full_attention_interval``
        so layer_idx ``full_attention_interval - 1`` is a FULL-attention
        layer (``(idx+1) % interval == 0``)."""
        return _Qwen35TextArgs(
            model_type="qwen3_5_text",
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.full_attention_interval,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            full_attention_interval=self.full_attention_interval,
            tie_word_embeddings=self.tie_word_embeddings,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size,
            shared_expert_intermediate_size=self.shared_expert_intermediate_size,
            norm_topk_prob=self.norm_topk_prob,
            decoder_sparse_step=self.decoder_sparse_step,
            rope_parameters=self.rope_parameters
            or {
                "type": "default",
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        )


class MTPDraftModel(nn.Module):
    """Single-layer MTP draft head conditioned on target hidden states.

    Forward: ``(token_ids, h_prev, cache=None, compute_logits=True) ->
    (logits|None, h_new)``. ``embed_tokens``/``lm_head`` are borrowed from
    the target via ``bind()`` (kept out of the parameter tree via
    ``object.__setattr__``, same as EAGLE).

    Two deliberate differences from EAGLE:

    1. Front-end: two separate RMSNorms (``pre_fc_norm_hidden`` and
       ``pre_fc_norm_embedding``) applied to ``h_prev`` and ``emb``
       respectively, then concatenated and projected via ``fc``.
    2. Chained hidden: returns ``h_new = x`` (layer output BEFORE the
       head's own ``norm``); ``logits = lm_head(norm(x))``. EAGLE
       returns ``norm(x)`` for both — we must not do that here.
    """

    def __init__(self, args: MTPConfig):
        super().__init__()
        self.args = args
        self.concat_hidden_first = True  # [hidden, embed]; a later task may flip
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        # One full-attention Qwen3.6 layer. layer_idx = interval-1 forces
        # the non-linear (full attention) branch:
        # is_linear = (layer_idx + 1) % full_attention_interval != 0
        #           = (interval) % interval != 0  =>  False  =>  full attn.
        text_args = args.to_qwen35_text_args()
        self.layers = [
            _Qwen35DecoderLayer(text_args, layer_idx=args.full_attention_interval - 1)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Borrowed at bind() time from the target. Set via
        # ``object.__setattr__`` to bypass ``nn.Module.__setattr__``,
        # which would register them as tracked children — that would
        # duplicate the target's embed/lm_head tensors in the draft
        # checkpoint and include them in ``draft.parameters()`` so
        # gradients would flow into the frozen target weights.
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
    def _find_lm_head(
        target: Any, embed: nn.Module
    ) -> nn.Module | Callable[..., Any] | None:
        """Resolve the target's ``lm_head``.

        Returns either an ``nn.Module`` (the standard case — a dedicated
        ``Linear`` layer) or a callable (the tied-embeddings fallback for
        models that expose ``embed_tokens.as_linear``). Both forms are
        consumed identically by ``self.lm_head(x)`` at the call site.
        """
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
        # Tied-embeddings fallback.
        as_linear = getattr(embed, "as_linear", None)
        if callable(as_linear):
            return as_linear
        return None

    def bind_via_modules(self, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        """Test-only entry point: bind to externally-provided modules
        without going through a target wrapper."""
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
        if self.embed_tokens is None:
            raise RuntimeError("MTPDraftModel.__call__ requires bind() first.")
        if compute_logits and self.lm_head is None:
            raise RuntimeError(
                "MTPDraftModel.__call__(compute_logits=True) requires bind()."
            )
        emb = self.embed_tokens(token_ids)
        h = self.pre_fc_norm_hidden(h_prev)
        e = self.pre_fc_norm_embedding(emb)
        parts = [h, e] if self.concat_hidden_first else [e, h]
        x = self.fc(mx.concatenate(parts, axis=-1))

        L = x.shape[1]
        mask = None
        if L > 1:
            mask = create_causal_mask(L, offset=cache[0].offset if cache else 0)
        layer_cache = cache[0] if cache is not None else None
        x = self.layers[0](x, mask=mask, cache=layer_cache)

        h_new = x  # pre-norm; chained for the next draft step (MTP convention)
        if compute_logits:
            lm_head = self.lm_head
            if lm_head is None:
                raise RuntimeError("MTPDraftModel internal invariant: lm_head None.")
            return lm_head(self.norm(x)), h_new
        return None, h_new
