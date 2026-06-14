"""Tests for proxy-tuning decode mode (engine/proxy_tuning.py)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.proxy_tuning import combine_proxy_logits


def test_combine_proxy_logits_basic():
    base = mx.array([1.0, 2.0, 3.0])
    expert = mx.array([0.0, 5.0, 0.0])
    antiexpert = mx.array([0.0, 1.0, 0.0])
    # base + alpha*(expert - antiexpert) with alpha=1.0
    out = combine_proxy_logits(base, expert, antiexpert, 1.0)
    assert out.tolist() == [1.0, 6.0, 3.0]


def test_combine_proxy_logits_alpha_scales_delta():
    base = mx.array([0.0, 0.0])
    expert = mx.array([0.0, 4.0])
    antiexpert = mx.array([0.0, 0.0])
    out = combine_proxy_logits(base, expert, antiexpert, 0.5)
    assert out.tolist() == [0.0, 2.0]


def test_combine_proxy_logits_alpha_zero_is_base():
    base = mx.array([7.0, -3.0])
    expert = mx.array([100.0, 100.0])
    antiexpert = mx.array([-100.0, -100.0])
    out = combine_proxy_logits(base, expert, antiexpert, 0.0)
    assert out.tolist() == [7.0, -3.0]
