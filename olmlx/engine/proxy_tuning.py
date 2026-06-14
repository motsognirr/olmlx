"""Proxy-Tuning decode mode (Liu et al. 2024, *Tuning Language Models by Proxy*).

Steers a large base model ``M`` at decode time — without touching its weights
— using a small tuned expert ``M⁺`` and small untuned anti-expert ``M⁻``
(``M⁺`` is ``M⁻`` fine-tuned). Each decode step combines per-token logits as::

    logits = base + alpha * (expert - antiexpert)

then samples greedily. The learned "tuning delta" from the cheap small models
is transplanted onto the big model.

Unlike speculative decoding (which is *exactness-preserving*), proxy-tuning
*deliberately alters* the output distribution, so it cannot reuse the
draft -> verify -> accept logic. It is registered as a sibling
``speculative_strategy`` ("proxy_tuning") only to reuse the mechanical
plumbing (lifecycle, stream bridge, dispatch); the algorithm is its own.

Hard constraint: all three models must share one exact tokenizer / vocabulary.

v1 targets **dense** model families only (Qwen2.5/Qwen3 dense, Llama 3.x).
Hybrid linear-attention / GDN families are out of scope — this decoder installs
no ``GDNStateCapture`` and runs every forward on the default stream (the same
stream the speculative path decodes on).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import mlx.core as mx

from mlx_lm.models.cache import make_prompt_cache

from olmlx.engine.spec_decoder_base import SpecDecoderBase
from olmlx.engine.speculative import _eval_cache, _prefill_last_logit

logger = logging.getLogger(__name__)


def combine_proxy_logits(
    base: mx.array,
    expert: mx.array,
    antiexpert: mx.array,
    alpha: float,
) -> mx.array:
    """Proxy-tuning logit arithmetic: ``base + alpha * (expert - antiexpert)``.

    All three arrays are ``(vocab,)`` last-position logits from the three
    models forwarded over the same token. ``alpha`` scales the tuning delta;
    ``alpha == 0`` reduces to the unsteered base distribution.
    """
    return base + alpha * (expert - antiexpert)


def _safe_get_vocab(tokenizer: Any) -> dict[str, int] | None:
    """Return ``tokenizer.get_vocab()`` as a dict, or ``None`` if unavailable.

    mlx-lm's ``TokenizerWrapper`` forwards attribute access to the underlying
    HuggingFace tokenizer, so ``get_vocab()`` works on the wrapper too. Any
    failure (no method, raises) yields ``None`` so the caller can fall back to
    the loader's ``vocab_size`` check rather than hard-failing on an exotic
    tokenizer.
    """
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        return None
    try:
        vocab = get_vocab()
    except Exception:  # noqa: BLE001 — any failure -> fall back to size check
        return None
    return vocab if isinstance(vocab, dict) else None


def check_vocab_identity(
    reference_tokenizer: Any,
    other_tokenizer: Any,
    *,
    reference_label: str,
    other_label: str,
) -> None:
    """Raise ``ValueError`` if two tokenizers map tokens differently.

    Proxy-tuning adds logits across models position-by-position; a token id
    that means different things in two models silently corrupts the output.
    This is stricter than a ``vocab_size`` comparison (two vocabularies can
    match on size yet differ in mapping). When either tokenizer does not expose
    ``get_vocab()``, this warns and returns — the loader's ``vocab_size`` guard
    is the hard floor in that case.
    """
    ref_vocab = _safe_get_vocab(reference_tokenizer)
    other_vocab = _safe_get_vocab(other_tokenizer)
    if ref_vocab is None or other_vocab is None:
        logger.warning(
            "Could not verify tokenizer/vocab identity for proxy-tuning "
            "(%s or %s tokenizer has no usable get_vocab()); relying on the "
            "vocab_size check only. A token-mapping mismatch would corrupt "
            "the combined logits.",
            reference_label,
            other_label,
        )
        return
    if ref_vocab != other_vocab:
        raise ValueError(
            f"Proxy-tuning requires identical vocabularies: the {other_label} "
            f"tokenizer's token->id mapping differs from the {reference_label} "
            f"tokenizer's (sizes {len(other_vocab)} vs {len(ref_vocab)}). All "
            f"three models (base, expert, anti-expert) must share one exact "
            f"tokenizer — use models from the same family and tokenizer revision."
        )


class ProxyTuningDecoder(SpecDecoderBase):
    """Decode-time logit-arithmetic steering with three models.

    Holds three independent ``(model, KV-cache)`` pairs — base, expert,
    anti-expert. Each decode step forwards the single pending token through all
    three, combines their last-position logits via :func:`combine_proxy_logits`,
    and greedily argmaxes the result. Every model advances over the same
    committed token sequence, so the caches stay aligned with no trimming.

    Installs **no** layer hooks, **no** draft bind, and **no** GDN capture
    (dense-only v1), so the base-class ``reset()`` teardown is effectively a
    no-op beyond clearing the caches in :meth:`_reset_state`.

    Implements the speculative decoder protocol (``prefill() -> int``,
    ``step() -> (list[int], int)``) so it slots into ``speculative_stream``
    unchanged. ``num_draft`` is always 0 (proxy-tuning does not speculate; one
    token per step at ``base + 2*small`` forward cost).

    Not thread-safe: one decoder instance serves one request at a time.
    """

    def __init__(
        self,
        base_model: Any,
        expert_model: Any,
        antiexpert_model: Any,
        *,
        alpha: float = 1.0,
    ):
        super().__init__()
        self._base = base_model
        self._expert = expert_model
        self._antiexpert = antiexpert_model
        # Base teardown reference for the inherited reset() path. We never call
        # _install_layer_hooks/_bind_draft, so _patched/_bound stay False and
        # reset() never actually touches _target — but set it for correctness.
        self._target = base_model
        self._alpha = float(alpha)

        # Per-request state (populated by prefill, cleared by _reset_state).
        self._base_cache: list | None = None
        self._expert_cache: list | None = None
        self._antiexpert_cache: list | None = None
        self._pending_token: int | None = None

    def _reset_state(self) -> None:
        self._base_cache = None
        self._expert_cache = None
        self._antiexpert_cache = None
        self._pending_token = None

    def _stats_extra(self) -> dict[str, Any]:
        return {"alpha": self._alpha}

    def _prefill_impl(
        self,
        prompt: mx.array,
        *,
        segmented: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> int:
        """Prefill all three models over ``prompt`` and return the first token.

        Each model gets its own fresh KV cache. ``_prefill_last_logit`` (reused
        from the speculative module) sub-chunks the prefix so a long prompt
        cannot OOM Metal, and returns the model's final-position ``(vocab,)``
        logit. The three logits are combined and argmaxed for the first token.
        ``segmented`` is accepted and ignored — proxy-tuning has no cross-request
        snapshot store in v1.

        Runs on the default stream (no ``mx.stream`` wrapper), the same stream
        ``step()`` decodes on — consistent with the speculative path's
        single-stream invariant.
        """
        self._base_cache = make_prompt_cache(self._base)
        self._expert_cache = make_prompt_cache(self._expert)
        self._antiexpert_cache = make_prompt_cache(self._antiexpert)

        base_logit = _prefill_last_logit(
            self._base, prompt, self._base_cache, cancel_event=cancel_event
        )
        expert_logit = _prefill_last_logit(
            self._expert, prompt, self._expert_cache, cancel_event=cancel_event
        )
        antiexpert_logit = _prefill_last_logit(
            self._antiexpert, prompt, self._antiexpert_cache, cancel_event=cancel_event
        )
        combined = combine_proxy_logits(
            base_logit, expert_logit, antiexpert_logit, self._alpha
        )
        # Materialize the combined logit and every cache's final-token write
        # before decode begins — keeps the per-model lazy graphs from chaining
        # across step() boundaries.
        first_token = int(mx.argmax(combined).item())
        _eval_cache(self._base_cache)
        _eval_cache(self._expert_cache)
        _eval_cache(self._antiexpert_cache)

        self._pending_token = first_token
        return first_token

    def _step_impl(self) -> tuple[list[int], int]:
        raise NotImplementedError  # implemented in Task 5
