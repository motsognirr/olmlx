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
