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

from olmlx.engine.spec_decoder_base import SpecDecoderBase

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
