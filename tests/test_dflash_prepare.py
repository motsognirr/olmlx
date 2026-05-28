"""Tests for ``olmlx dflash prepare`` — DFlash draft training.

These tests exercise the full pipeline against a synthetic target so
they run without network access and without downloading any real
models. Coverage:

- Target-config → DraftConfig derivation (vocab, head dims, GQA shape)
- Target layer-id selection (explicit + evenly-spaced default)
- One end-to-end training run that asserts loss decreases over a few
  steps
- Round-trip: a freshly-saved draft loads back through the same
  ``DFlashDraftModel(DraftConfig(...))`` constructor used by
  ``_load_dflash_decoder``
- Saved ``config.json`` matches the upstream-compatible schema (nested
  ``dflash_config`` block)
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Synthetic target + tokenizer
# ---------------------------------------------------------------------------


class _MockSelfAttn(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.n_heads = 1
        self.n_kv_heads = 1
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        if cache is not None:
            k = v = x.reshape(x.shape[0], 1, -1, x.shape[-1])
            cache.update_and_fetch(k, v)
        return self.proj(x)


class _MockLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = _MockSelfAttn(hidden_size)

    def __call__(self, x, mask=None, cache=None):
        return x + self.self_attn(x, mask=mask, cache=cache)


class _Inner(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [_MockLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)


class _Target(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=16, num_layers=4):
        super().__init__()
        self.model = _Inner(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, input_ids, cache=None):
        h = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, cache=cache[i] if cache is not None else None)
        h = self.model.norm(h)
        return self.lm_head(h)


class _MockTokenizer:
    """Tokenizer just rich enough for ``stream_training_batches``."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab_size: int = 64, seq_len: int = 64):
        self.vocab_size = vocab_size
        self._seq_len = seq_len

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Deterministic hash-based fake tokenization.
        tokens = []
        for i, ch in enumerate(text):
            tokens.append((ord(ch) + i) % self.vocab_size)
            if len(tokens) >= self._seq_len * 2:
                break
        # Ensure we exceed min_seq_len in the loader so the example isn't dropped.
        while len(tokens) < self._seq_len:
            tokens.append((len(tokens) + 7) % self.vocab_size)
        return tokens


def _mock_target_loader(vocab_size: int, hidden_size: int, num_layers: int):
    target = _Target(vocab_size, hidden_size, num_layers)
    tokenizer = _MockTokenizer(vocab_size=vocab_size, seq_len=64)
    return lambda _path: (target, tokenizer)


def _synthetic_batches(vocab: int, batch_size: int, seq_len: int, n: int):
    """Deterministic batch iterator so tests don't hit the network.

    Emits sequences whose next-token IS the previous-token + 1 (mod
    ``vocab - 1``, then shifted into ``[1, vocab)``) so there's *actual*
    learnable structure for the loss-decrease sanity check. Previous
    versions yielded pure ``mx.random.randint`` noise, which worked
    only because the pre-#317 training pipeline let the draft cheat
    via a target-hidden copy shortcut at the pending position (the
    very bug gh#317 Gap 1 fixes). With the shortcut removed the draft
    can't reduce CE on noise — so the synthetic data must carry a
    pattern.

    Tokens stay in ``[1, vocab)`` because ``_MockTokenizer.pad_token_id
    == 0`` and the trainer's pad-aware pivot helper would otherwise
    treat any 0 as padding and shrink the valid pivot range. Tests
    that need to exercise the pad-aware path build batches explicitly
    with pad tokens (see ``TestPivotSelection``).
    """
    # Build a deterministic per-batch starting offset, then walk
    # ``seq_len`` consecutive tokens in 1..vocab-1.
    period = vocab - 1
    for i in range(n):
        offsets = mx.arange(seq_len, dtype=mx.int32) + i
        # Each row starts at a slightly different offset so batches
        # don't all carry the same window.
        per_row = mx.arange(batch_size, dtype=mx.int32)[:, None] * 7
        tokens = (offsets[None, :] + per_row) % period + 1
        yield tokens


def _write_target_config(tmp_path: Path, vocab_size: int, hidden_size: int) -> Path:
    cfg = {
        "model_type": "qwen3",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": hidden_size // 2,
        "intermediate_size": hidden_size * 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestEvenlySpaced:
    def test_picks_inner_indices(self):
        from olmlx.engine.dflash.prepare import _evenly_spaced

        # 32-layer target, 4 hooks
        ids = _evenly_spaced(32, 4)
        assert len(ids) == 4
        assert all(0 < i < 31 for i in ids)
        # Strictly increasing
        assert ids == sorted(ids)
        assert len(set(ids)) == 4

    def test_handles_small_targets(self):
        from olmlx.engine.dflash.prepare import _evenly_spaced

        ids = _evenly_spaced(2, 4)
        # When k >= num_layers, returns the full layer range.
        assert ids == [0, 1]

    def test_matches_upstream_recipe_for_n32_k5(self):
        """Upstream ``build_target_layer_ids(32, 5) == [1, 8, 15, 22,
        29]``: indices evenly spread over ``[1, N-3]`` (early-biased,
        skipping the embedding-adjacent layer 0 and the lm_head-adjacent
        last two layers). See gh#317 (Gap 4).
        """
        from olmlx.engine.dflash.prepare import _evenly_spaced

        assert _evenly_spaced(32, 5) == [1, 8, 15, 22, 29]

    def test_non_degenerate_skips_layer_zero_and_last_two_layers(self):
        """The upstream recipe explicitly avoids layer 0 (its signal
        duplicates ``embed_tokens``) and the last 2 layers (their
        signal duplicates the bound ``lm_head``). Regression guard
        for the *non-degenerate* path only: any selection where the
        ``[1, N-3]`` range can fit ``k`` unique indices must respect
        the boundary. The degenerate fallback (``N-3 < k``) is
        covered by ``test_degenerate_fallback_returns_k_unique_indices``.
        """
        from olmlx.engine.dflash.prepare import _evenly_spaced

        for n in (12, 24, 32, 48, 80):
            for k in range(2, min(n - 3, 8) + 1):
                ids = _evenly_spaced(n, k)
                assert min(ids) >= 1, f"_evenly_spaced({n},{k}) included layer 0"
                assert max(ids) <= n - 3, (
                    f"_evenly_spaced({n},{k}) reached layer {max(ids)} > {n - 3}"
                )

    def test_degenerate_fallback_returns_k_unique_indices(self):
        """When the upstream ``[1, N-3]`` range is too small to fit
        ``k`` unique indices (``N - 3 < k`` but ``k < N``), the
        fallback spreads across the full ``0..N-1`` range. The
        contract here is **k unique indices, no rounding collisions**
        — and in this regime layer 0 *can* appear, in contradiction
        with the non-degenerate path's invariant. The trade is
        intentional so small synthetic targets in unit tests still
        produce ``k`` distinct hidden-state hooks; documenting it as
        an explicit test prevents a future refactor from silently
        re-introducing the rounding-collision bug.
        """
        from olmlx.engine.dflash.prepare import _evenly_spaced

        # ``(num_layers, k)`` pairs that exercise the fallback:
        # ``num_layers - 3 < k`` but ``k < num_layers``.
        cases = [
            (6, 4),  # end=3 < 4
            (5, 3),  # end=2 < 3
            (7, 5),  # end=4 < 5
            (4, 2),  # end=1 < 2 — the test fixture in TestPrepareDflashDraft
        ]
        for n, k in cases:
            ids = _evenly_spaced(n, k)
            assert len(ids) == k, (
                f"_evenly_spaced({n},{k}) returned {ids} ({len(ids)} indices, "
                f"expected {k}); rounding collision in the fallback"
            )
            assert len(set(ids)) == k, (
                f"_evenly_spaced({n},{k}) returned duplicates: {ids}"
            )
            assert all(0 <= i < n for i in ids), (
                f"_evenly_spaced({n},{k}) returned out-of-range index: {ids}"
            )


class TestResolveTargetLayerIds:
    def test_explicit_list(self):
        from olmlx.engine.dflash.prepare import _resolve_target_layer_ids

        out = _resolve_target_layer_ids([1, 5, 11], None, 16)
        assert out == [1, 5, 11]

    def test_out_of_range_raises(self):
        from olmlx.engine.dflash.prepare import _resolve_target_layer_ids

        with pytest.raises(ValueError, match="out of range"):
            _resolve_target_layer_ids([3, 99], None, 16)

    def test_default_when_none(self):
        from olmlx.engine.dflash.prepare import _resolve_target_layer_ids

        out = _resolve_target_layer_ids(None, 4, 16)
        assert len(out) == 4


class TestDraftConfigDerivation:
    def test_inherits_target_dims(self):
        from olmlx.engine.dflash.prepare import _build_draft_config

        target_cfg = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 32768,
        }
        cfg = _build_draft_config(
            target_cfg,
            target_layer_ids=[5, 11, 17, 23],
            num_hidden_layers=4,
            block_size=4,
            mask_token_id=0,
        )
        assert cfg.hidden_size == 4096
        assert cfg.head_dim == 128
        assert cfg.num_key_value_heads == 8
        assert cfg.vocab_size == 32000
        assert cfg.target_layer_ids == [5, 11, 17, 23]
        # ``block_size`` on disk is the draft-token count directly,
        # matching #287's ``_load_dflash_decoder`` consumer.
        assert cfg.block_size == 4
        assert cfg.mask_token_id == 0

    def test_requires_num_attention_heads(self):
        # Missing ``num_attention_heads`` previously fell back to
        # ``hidden_size // 64``, silently producing 2× too many heads
        # for the dominant 128-dim head convention. The field is now
        # required so a malformed/minimal config fails loudly.
        from olmlx.engine.dflash.prepare import _build_draft_config

        target_cfg = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            # num_attention_heads intentionally omitted
            "head_dim": 128,
            "intermediate_size": 11008,
        }
        with pytest.raises(ValueError, match="num_attention_heads"):
            _build_draft_config(
                target_cfg,
                target_layer_ids=[0],
                num_hidden_layers=1,
                block_size=4,
                mask_token_id=0,
            )

    def test_descends_into_text_config_for_vlm_targets(self):
        # Multimodal targets (e.g. Qwen3.6-35B-A3B's
        # ``Qwen3_5MoeForConditionalGeneration``) nest text-tower fields
        # under a ``text_config`` block. A flat ``target_cfg["hidden_size"]``
        # read KeyErrors. The builder must descend into ``text_config`` when
        # present, otherwise the only Qwen3.6 target on disk can't have a
        # DFlash draft trained for it at all.
        from olmlx.engine.dflash.prepare import _build_draft_config

        target_cfg = {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 262144,
            },
            "vision_config": {"hidden_size": 1024},
        }
        cfg = _build_draft_config(
            target_cfg,
            target_layer_ids=[8, 16, 23, 31],
            num_hidden_layers=4,
            block_size=4,
            mask_token_id=151643,
        )
        assert cfg.hidden_size == 2048
        assert cfg.head_dim == 256
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 2
        assert cfg.vocab_size == 248320
        assert cfg.max_position_embeddings == 262144

    def test_vlm_config_with_rope_parameters_nested_under_text_config(self):
        """The actual Qwen3.6-35B-A3B config shape: ``rope_parameters``
        lives *inside* ``text_config``, not at the top level. Both
        descent paths (``_text_config()`` to find the text-tower
        block, then the nested ``rope_parameters`` lookup) must
        compose. The two are individually covered by
        ``test_descends_into_text_config_for_vlm_targets`` and
        ``test_picks_up_rope_theta_from_rope_parameters_block``, but
        only the end-to-end training run actually exercises both
        together. With the fallback wrong by 3 orders of magnitude
        (10_000 vs 10_000_000), a regression that broke composition
        would produce a silently misconfigured draft and only show up
        as poor acceptance at inference time.
        """
        from olmlx.engine.dflash.prepare import _build_draft_config

        target_cfg = {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 262144,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000000,
                },
            },
            "vision_config": {"hidden_size": 1024},
        }
        cfg = _build_draft_config(
            target_cfg,
            target_layer_ids=[8, 16, 23, 31],
            num_hidden_layers=4,
            block_size=4,
            mask_token_id=151643,
        )
        assert cfg.rope_theta == 10000000.0
        # Sanity: the other text_config fields also resolved.
        assert cfg.hidden_size == 2048
        assert cfg.vocab_size == 248320

    def test_rope_parameters_falls_back_to_top_level_for_vlm(self):
        """Symmetric to ``rope_scaling``: a VLM that carries
        ``rope_parameters`` at the *top level* of ``config.json``
        (without mirroring it inside ``text_config``) must still be
        discoverable. Otherwise the lookup falls through to the
        legacy 10000.0 default — three orders of magnitude wrong for
        long-context targets — silently producing a misconfigured
        draft.
        """
        from olmlx.engine.dflash.prepare import _build_draft_config

        target_cfg = {
            "model_type": "future_vlm",
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
            "text_config": {
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
            },
        }
        cfg = _build_draft_config(
            target_cfg,
            target_layer_ids=[0],
            num_hidden_layers=1,
            block_size=4,
            mask_token_id=0,
        )
        assert cfg.rope_theta == 10000000.0

    def test_rope_scaling_falls_back_to_top_level_for_vlm(self):
        """``rope_scaling`` regression guard. After ``_build_draft_config``
        was switched to read everything from ``text_config`` for VLMs,
        a config that carries ``rope_scaling`` at the *top level*
        (without mirroring it inside ``text_config``) would silently
        drop scaling, producing a draft whose RoPE doesn't match the
        target's effective frequencies.
        """
        from olmlx.engine.dflash.prepare import _build_draft_config

        target_cfg = {
            "model_type": "qwen3_5_moe",
            "rope_scaling": {"type": "yarn", "factor": 4.0},
            "text_config": {
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rope_theta": 10000.0,
            },
        }
        cfg = _build_draft_config(
            target_cfg,
            target_layer_ids=[0],
            num_hidden_layers=1,
            block_size=4,
            mask_token_id=0,
        )
        assert cfg.rope_scaling == {"type": "yarn", "factor": 4.0}

    def test_picks_up_rope_theta_from_rope_parameters_block(self):
        # Newer config schemas (Qwen3.5+, Qwen3.6) replace the flat
        # ``rope_theta`` field with a nested ``rope_parameters`` block
        # carrying ``rope_theta``, ``rope_type``, and friends. The
        # default 10000.0 is wildly off from the 10_000_000 base these
        # long-context targets actually use, so silently falling back
        # to it produces a draft whose RoPE is incompatible with the
        # context positions the target was trained on.
        from olmlx.engine.dflash.prepare import _build_draft_config

        target_cfg = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 11008,
            # No top-level rope_theta — only the nested block.
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
            },
        }
        cfg = _build_draft_config(
            target_cfg,
            target_layer_ids=[0],
            num_hidden_layers=1,
            block_size=4,
            mask_token_id=0,
        )
        assert cfg.rope_theta == 10000000.0


class _FakeDraft:
    """Stand-in for ``DFlashDraftModel`` that returns pre-baked logits.

    Lets us drive ``_draft_loss`` without instantiating the real draft
    (which would require building a target, capturing hidden states, and
    walking RoPE). Mirrors the call signature used in ``prepare.py``:
    ``draft(block_input, target_hidden, cache, logits_start=1)``.
    """

    def __init__(self, logits: mx.array):
        self._logits = logits

    def __call__(self, block_input, target_hidden, cache, logits_start=1):
        return self._logits


def _logits_with_target_at_index(
    batch_size: int, block_size: int, vocab: int, target_id: int, peak: float
) -> mx.array:
    """Return logits whose softmax peaks sharply at ``target_id``.

    Filling the row with 0 except a positive value at ``target_id`` gives
    a known closed-form CE: ``-log_softmax(logits)[target_id] = -peak +
    log(exp(peak) + (vocab - 1))``.
    """
    arr = mx.zeros((batch_size, block_size, vocab))
    one_hot = mx.full((batch_size, block_size), peak)
    indices = mx.full((batch_size, block_size), target_id, dtype=mx.int32)
    arr = mx.put_along_axis(arr, indices[..., None], one_hot[..., None], axis=-1)
    return arr


class TestDraftLossPadMasking:
    """Loss-mask behavior for padding-collision targets.

    When the data loader right-pads sequences with ``pad_token_id`` AND
    the trainer aliases ``mask_token_id == pad_token_id`` (the upstream
    convention for tokenizers without a dedicated MASK token, which is
    the common case for Qwen3.x), pivots that land in the padding region
    construct a window of ``[PAD, MASK*N]`` whose targets are also all
    PAD. The ``bind()``-tied lm_head then trivially predicts PAD with
    near-1.0 probability, producing exact-zero CE that contaminates the
    running average without giving any gradient signal.

    The fix passes ``pad_token_id`` into ``_draft_loss`` and zero-weights
    pad-target positions in the CE reduction. With no pad targets in the
    batch the behavior is unchanged; with all-pad targets the loss is
    exactly zero (genuine no-op step rather than misleading sub-epsilon
    value).
    """

    def test_no_pad_id_preserves_legacy_behavior(self):
        # Backward compat: callers that don't supply pad_token_id still
        # see the old uniform-mean reduction.
        from olmlx.engine.dflash.prepare import _draft_loss

        vocab = 8
        # Logits sharply peak at index 1 in every batch position.
        logits = _logits_with_target_at_index(1, 3, vocab, target_id=1, peak=20.0)
        targets = mx.array([[1, 1, 1]])
        loss = _draft_loss(
            _FakeDraft(logits),
            block_input=mx.zeros((1, 4), dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            targets=targets,
            cache=[],
        )
        # Sharp logit at the right index gives loss ≈ 0; no masking
        # kwarg means we exercise the legacy path.
        assert float(loss) < 1e-3

    def test_all_pad_targets_yield_zero_loss(self):
        # Every target is the pad id → no real positions → loss is
        # exactly 0 (vs. the unmasked path, which would also approach 0
        # via lm_head's identity-bias and look misleadingly identical).
        from olmlx.engine.dflash.prepare import _draft_loss

        vocab = 8
        # Pick a logit shape that would *not* trivially zero-out without
        # masking: huge nll at every position. If the mask isn't applied
        # the mean would be large.
        bad_logits = _logits_with_target_at_index(1, 3, vocab, target_id=1, peak=20.0)
        targets = mx.array([[5, 5, 5]])  # all pad
        loss = _draft_loss(
            _FakeDraft(bad_logits),
            block_input=mx.zeros((1, 4), dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            targets=targets,
            cache=[],
            pad_token_id=5,
        )
        assert float(loss) == 0.0

    def test_mixed_targets_average_only_real_positions(self):
        # Real target at position 0 (loss ~0), pad at positions 1-2.
        # Without masking, mean is dragged toward (real_loss + 2*pad_loss)/3.
        # With masking, mean equals real_loss alone.
        from olmlx.engine.dflash.prepare import _draft_loss

        vocab = 8
        # Logits peak at id=1 sharply; real target uses id=1 → near-zero
        # nll there. Pad positions use id=5 → very high nll (~20).
        logits = _logits_with_target_at_index(1, 3, vocab, target_id=1, peak=20.0)
        targets = mx.array([[1, 5, 5]])
        loss = _draft_loss(
            _FakeDraft(logits),
            block_input=mx.zeros((1, 4), dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            targets=targets,
            cache=[],
            pad_token_id=5,
        )
        # If masking works, only position 0 contributes; loss ≈ 0.
        # Without masking the unweighted mean would be ~13.3 (one near-zero
        # plus two ~20s averaged).
        assert float(loss) < 1e-3

    def test_kl_distillation_with_all_pad_targets_yields_zero_loss(self):
        # Distillation path with every target marked as pad. Both the CE
        # and KL terms must zero-weight all positions, so the combined
        # loss is exactly 0.0 — even though a non-zero KL would
        # otherwise dominate via the (1 - alpha) * CE + alpha * T^2 *
        # KL mix.
        from olmlx.engine.dflash.prepare import _draft_loss

        vocab = 8
        # Logits sharply favor id=1 in both draft and target — CE is
        # tiny but non-zero in absolute terms; without masking, the KL
        # term contributes the dominant share of any apparent loss.
        draft_logits = _logits_with_target_at_index(1, 3, vocab, target_id=1, peak=20.0)
        # Target softmax peaks at id=2 — disagrees with draft, so an
        # unmasked KL would be very large.
        target_logits = _logits_with_target_at_index(
            1, 3, vocab, target_id=2, peak=20.0
        )
        targets = mx.array([[5, 5, 5]])  # all pad
        loss = _draft_loss(
            _FakeDraft(draft_logits),
            block_input=mx.zeros((1, 4), dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            targets=targets,
            cache=[],
            target_logits_window=target_logits,
            distill_alpha=0.5,
            distill_temp=2.0,
            pad_token_id=5,
        )
        assert float(loss) == 0.0

    def test_pad_for_loss_disabled_when_pad_aliases_eos(self, tmp_path):
        """Qwen3.x tokenizers set ``pad_token_id == eos_token_id ==
        151643``. Until now ``pad_for_loss = int(_tok_pad)`` flowed
        that value into ``_draft_loss``, where ``valid = (targets !=
        pad_token_id)`` zero-weighted every mid-stream EOS marker —
        not just trailing pad. In multi-turn instruction-tuning data
        each turn ends with EOS, so the draft would receive zero
        gradient at turn boundaries and never learn to predict them.

        The fix: if the tokenizer's pad_token_id is the same value as
        eos_token_id, set ``pad_for_loss = None`` so ``_draft_loss``
        skips the mask entirely. ``_select_pivot`` (which uses
        ``pad_for_pivot``) already keeps the targets inside the
        unpadded prefix, so trailing-pad windows can't reach the loss
        and the mask is redundant.
        """
        from olmlx.engine.dflash import prepare as prepare_mod
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        # Tokenizer with pad == eos (the Qwen3.x default).
        class _Qwen3xTokenizer(_MockTokenizer):
            pad_token_id = 7
            eos_token_id = 7

        # Capture every ``pad_token_id`` argument fed into
        # ``_draft_loss``.
        original_loss = prepare_mod._draft_loss
        seen_pad_token_ids: list[int | None] = []

        def recording_loss(*args, **kwargs):
            seen_pad_token_ids.append(kwargs.get("pad_token_id"))
            return original_loss(*args, **kwargs)

        target = _Target(vocab, hidden, num_layers)
        tokenizer = _Qwen3xTokenizer(vocab_size=vocab, seq_len=64)

        prepare_dflash_draft(
            tmp_path,
            steps=1,
            batch_size=2,
            seq_len=32,
            block_size=2,
            num_hidden_layers=1,
            num_target_layers=2,
            lr=1e-2,
            output_dir=tmp_path / "dflash_out",
            _target_loader=lambda _path: (target, tokenizer),
            _batch_iterator=_synthetic_batches(vocab, batch_size=2, seq_len=32, n=3),
        )
        # Patch happens after the loader, so recording_loss may not be
        # in scope. Verify via the resolved variable in the prepare
        # module instead — re-import and inspect the closure that
        # built the step. Simpler: re-run with monkeypatching the loss.
        # Since the above run already completed, do a second invocation
        # with the monkeypatch in place to capture the kwarg.
        seen_pad_token_ids.clear()
        prepare_mod._draft_loss = recording_loss  # type: ignore[assignment]
        try:
            prepare_dflash_draft(
                tmp_path,
                steps=1,
                batch_size=2,
                seq_len=32,
                block_size=2,
                num_hidden_layers=1,
                num_target_layers=2,
                lr=1e-2,
                output_dir=tmp_path / "dflash_out2",
                _target_loader=lambda _path: (
                    _Target(vocab, hidden, num_layers),
                    tokenizer,
                ),
                _batch_iterator=_synthetic_batches(
                    vocab, batch_size=2, seq_len=32, n=3
                ),
            )
        finally:
            prepare_mod._draft_loss = original_loss  # type: ignore[assignment]
        # When pad == eos, ``pad_for_loss`` must be None so the loss
        # falls through to the unmasked mean reduction.
        assert seen_pad_token_ids, "expected at least one _draft_loss call"
        for pid in seen_pad_token_ids:
            assert pid is None, (
                f"pad_for_loss = {pid} when pad_token_id == eos_token_id; "
                "must be None to avoid zero-weighting mid-stream EOS markers"
            )

    def test_kl_distillation_with_mixed_targets_masks_pad_positions(self):
        # Distillation path with one real position and two pad
        # positions. The KL contribution from the pad positions must be
        # zero-weighted; only the real position contributes to either
        # term. With ``alpha = 0.5`` and ``T = 1.0``, ``T^2 == 1`` so
        # the loss is ``0.5 * CE_real + 0.5 * KL_real``. Both pieces
        # individually approach zero when draft and target both peak at
        # the real target id.
        from olmlx.engine.dflash.prepare import _draft_loss

        vocab = 8
        # Both draft and target sharply favor id=1 at every position.
        # At the real position (target=1) the predictions agree → both
        # CE and KL ≈ 0.  At the pad positions (target=5) draft+target
        # still agree; an unmasked CE would be huge (~20) while KL
        # would be ~0. Masking must suppress the CE explosion.
        draft_logits = _logits_with_target_at_index(1, 3, vocab, target_id=1, peak=20.0)
        target_logits = _logits_with_target_at_index(
            1, 3, vocab, target_id=1, peak=20.0
        )
        targets = mx.array([[1, 5, 5]])
        loss = _draft_loss(
            _FakeDraft(draft_logits),
            block_input=mx.zeros((1, 4), dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            targets=targets,
            cache=[],
            target_logits_window=target_logits,
            distill_alpha=0.5,
            distill_temp=1.0,
            pad_token_id=5,
        )
        # If masking works, only the agreeing position 0 contributes;
        # both CE and KL → 0. Without masking, CE would be ~13.3
        # (one near-zero + two ~20s averaged).
        assert float(loss) < 1e-3


class TestPerPositionLossWeighting:
    """Gap 2 (gh#317): the paper trains with ``w_k = exp(-(k-1)/gamma)``
    to emphasise early positions because acceptance length compounds —
    a wrong token at position 1 wastes positions 2..N regardless of
    how correct those would have been. Without the weighting the
    optimizer spends equal gradient budget on positions whose impact
    is only conditional on every earlier position being accepted.
    """

    def test_legacy_uniform_mean_when_gamma_none(self):
        """When ``position_decay_gamma`` is omitted (or ``None``), the
        reduction collapses to the legacy uniform mean — required so
        the existing tests pass unchanged and existing checkpoints
        remain reproducible.
        """
        from olmlx.engine.dflash.prepare import _draft_loss

        vocab = 8
        # Construct a CE that depends on position so weighted and
        # uniform mean give visibly different numbers.
        # Logits peak sharply at id=1 for position 0 only; positions 1..2
        # peak at id=2. With targets [1, 1, 1] only position 0 is near-zero
        # NLL; positions 1-2 have very high NLL (~20 each).
        per_pos_peaks = [1, 2, 2]
        target_id = 1
        peak = 20.0
        # Build logits manually so each position uses a different peak id.
        arr = mx.zeros((1, 3, vocab))
        for k, pid in enumerate(per_pos_peaks):
            arr = arr.at[:, k, pid].add(peak)
        targets = mx.array([[target_id, target_id, target_id]])
        loss = _draft_loss(
            _FakeDraft(arr),
            block_input=mx.zeros((1, 4), dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            targets=targets,
            cache=[],
        )
        # Legacy uniform mean over 3 positions: (~0 + ~20 + ~20) / 3 ≈ 13.3.
        assert 12.0 < float(loss) < 15.0

    def test_weighted_mean_emphasises_early_positions(self):
        """With ``position_decay_gamma`` set, the weighted mean must
        upweight the first (easy) position and downweight the later
        (hard) ones, producing a lower aggregate loss than the
        uniform-mean baseline on the same logits.
        """
        from olmlx.engine.dflash.prepare import _draft_loss

        vocab = 8
        per_pos_peaks = [1, 2, 2]
        target_id = 1
        peak = 20.0
        arr = mx.zeros((1, 3, vocab))
        for k, pid in enumerate(per_pos_peaks):
            arr = arr.at[:, k, pid].add(peak)
        targets = mx.array([[target_id, target_id, target_id]])
        # gamma=1 strongly downweights positions 1..2 (weights ~ [1.0,
        # 0.37, 0.14]). Weighted mean: (0*1 + 20*0.37 + 20*0.14) /
        # (1 + 0.37 + 0.14) ≈ 6.7 — well below the uniform-mean ~13.3.
        loss = _draft_loss(
            _FakeDraft(arr),
            block_input=mx.zeros((1, 4), dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            targets=targets,
            cache=[],
            position_decay_gamma=1.0,
        )
        assert 5.0 < float(loss) < 9.0

    def test_disabled_when_gamma_non_positive(self):
        """``position_decay_gamma`` of 0 (or negative) disables the
        weighting and recovers the uniform-mean reduction. Important
        for the CLI's escape hatch (operators sweeping the
        hyperparameter who want a no-weighting baseline).
        """
        from olmlx.engine.dflash.prepare import _draft_loss

        vocab = 8
        per_pos_peaks = [1, 2, 2]
        arr = mx.zeros((1, 3, vocab))
        for k, pid in enumerate(per_pos_peaks):
            arr = arr.at[:, k, pid].add(20.0)
        targets = mx.array([[1, 1, 1]])
        loss_uniform = _draft_loss(
            _FakeDraft(arr),
            block_input=mx.zeros((1, 4), dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            targets=targets,
            cache=[],
        )
        loss_disabled = _draft_loss(
            _FakeDraft(arr),
            block_input=mx.zeros((1, 4), dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            targets=targets,
            cache=[],
            position_decay_gamma=0.0,
        )
        assert float(loss_disabled) == pytest.approx(float(loss_uniform), abs=1e-6)

    def test_weighting_combines_with_pad_mask(self):
        """The pad mask and per-position weighting must compose: pad
        positions stay zero-weighted, and the remaining real positions
        are reduced via the weighted mean. The all-pad case must still
        deliver exact 0.0 so the optimizer takes an honest no-op step.
        """
        from olmlx.engine.dflash.prepare import _draft_loss

        vocab = 8
        # All targets are pad → loss must be exactly 0 regardless of
        # the weighting.
        bad_logits = _logits_with_target_at_index(1, 3, vocab, target_id=1, peak=20.0)
        targets = mx.array([[5, 5, 5]])
        loss = _draft_loss(
            _FakeDraft(bad_logits),
            block_input=mx.zeros((1, 4), dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            targets=targets,
            cache=[],
            pad_token_id=5,
            position_decay_gamma=1.0,
        )
        assert float(loss) == 0.0


class TestSliceMatchesInferenceConvention:
    """Gap 1 (gh#317): training must slice ``target_hidden_full[:, :p,
    :]`` — ctx covers positions 0..p-1 and pending sits at position p.
    The pre-fix slice ``[:, :p+1, :]`` included the target's hidden
    for the pending position itself, giving the draft a copy-shortcut
    that didn't exist at inference and silently mis-aligning the
    proposal RoPE positions by 1.
    """

    def test_target_hidden_slice_excludes_pending_position(self, tmp_path, monkeypatch):
        """Inspect the ``target_hidden`` shape fed into ``_draft_loss``:
        its sequence dim must equal the pivot ``p``, not ``p + 1``.
        """
        from olmlx.engine.dflash import prepare as prepare_mod
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        # ``_select_pivot`` returns a host int, so we can record the
        # pivot directly. Capture (pivot, target_hidden.shape[1]) pairs
        # to verify equality.
        observed_pivots: list[int] = []
        original_select = prepare_mod._select_pivot

        def recording_select(input_ids, pad_token_id, block_size):
            p = original_select(input_ids, pad_token_id, block_size)
            if p is not None:
                observed_pivots.append(p)
            return p

        original_loss = prepare_mod._draft_loss
        observed_shapes: list[int] = []

        def recording_loss(draft, block_input, target_hidden, *args, **kwargs):
            observed_shapes.append(target_hidden.shape[1])
            return original_loss(draft, block_input, target_hidden, *args, **kwargs)

        monkeypatch.setattr(prepare_mod, "_select_pivot", recording_select)
        monkeypatch.setattr(prepare_mod, "_draft_loss", recording_loss)

        prepare_dflash_draft(
            tmp_path,
            steps=3,
            batch_size=2,
            seq_len=32,
            block_size=2,
            num_hidden_layers=1,
            num_target_layers=2,
            lr=1e-2,
            output_dir=tmp_path / "dflash_out",
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_batches(vocab, batch_size=2, seq_len=32, n=3),
        )

        assert observed_pivots, "expected at least one pivot to be recorded"
        assert len(observed_shapes) == len(observed_pivots)
        for p, ctx_len in zip(observed_pivots, observed_shapes):
            assert ctx_len == p, (
                f"target_hidden ctx len = {ctx_len}, expected {p} so that "
                "the draft conditions on positions 0..p-1 and pending sits "
                "at RoPE position p (matching inference's convention "
                "where ctx covers 0..L-1 and pending is at L)"
            )


class TestPivotSelection:
    """The pivot must land inside the unpadded prefix shared by all batch
    rows; otherwise the trainer wastes a forward+backward pass on a
    window where every target is pad. The selector returns ``None`` when
    no row has enough real content to fit a ``2*block_size + 1``
    window."""

    def test_returns_pivot_inside_real_content(self):
        from olmlx.engine.dflash.prepare import _select_pivot

        # 2 rows, both with 20 real tokens then padding to 32.
        pad = 0
        real_len = 20
        seq_len = 32
        rows = []
        for _ in range(2):
            row = list(range(1, real_len + 1)) + [pad] * (seq_len - real_len)
            rows.append(row)
        ids = mx.array(rows)
        block_size = 4
        # Run several times; every pivot must satisfy the constraint.
        import random as _r

        _r.seed(0)
        for _ in range(20):
            p = _select_pivot(ids, pad_token_id=pad, block_size=block_size)
            assert p is not None
            assert block_size <= p <= real_len - block_size - 1

    def test_returns_none_when_all_rows_too_short(self):
        # Every row has fewer real tokens than ``2 * block_size + 1`` so
        # no pivot fits. The selector signals "skip" rather than picking
        # a degenerate position.
        from olmlx.engine.dflash.prepare import _select_pivot

        pad = 0
        seq_len = 32
        # 4 real tokens, rest padding.
        rows = [[1, 2, 3, 4] + [pad] * (seq_len - 4) for _ in range(2)]
        ids = mx.array(rows)
        assert _select_pivot(ids, pad_token_id=pad, block_size=4) is None

    def test_handles_pad_token_appearing_mid_content(self):
        """The pivot range must reflect the right-padded *prefix* length,
        not the count of non-pad tokens. With ``mask_token_id ==
        pad_token_id == eos_token_id`` (the Qwen3.x default), multi-turn
        sequences contain EOS markers mid-stream that are content, not
        padding. Counting non-pad tokens globally undercounts the prefix
        and narrows the pivot range below what's actually safe.

        Construct a row of 30 real tokens (some equal to ``pad_token_id``
        because they're EOS markers ending intermediate turns), with the
        last 2 positions being trailing pad. The right-padded prefix
        boundary is at index 30; the global non-pad count is < 30. The
        helper must use the prefix boundary.
        """
        from olmlx.engine.dflash.prepare import _select_pivot

        pad = 0
        block_size = 2
        # 30-token "real" prefix that contains 3 EOS=pad markers
        # mid-stream (positions 5, 11, 17), then trailing pad to len 32.
        real_prefix = [
            1,
            2,
            3,
            4,
            5,
            pad,
            6,
            7,
            8,
            9,
            10,
            pad,
            11,
            12,
            13,
            14,
            15,
            pad,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ]
        assert len(real_prefix) == 30
        row = real_prefix + [pad, pad]  # seq_len = 32
        ids = mx.array([row, row])

        # The correct unpadded-prefix length is 30 (first trailing pad
        # at index 30). Pivot must be allowed up to 30 - block_size - 1
        # = 27. The buggy implementation that counts non-pad tokens
        # globally would see 27 non-pad tokens (30 - 3 EOS markers) and
        # cap the pivot at 27 - 3 - 1 = 23.
        import random as _r

        _r.seed(0)
        max_pivot_seen = 0
        for _ in range(200):
            p = _select_pivot(ids, pad_token_id=pad, block_size=block_size)
            assert p is not None
            max_pivot_seen = max(max_pivot_seen, p)
        # If the helper uses the prefix-boundary semantics, ``p`` can
        # reach 27 with high probability over 200 draws. If it uses the
        # buggy non-pad count, it caps at 24.
        assert max_pivot_seen >= 25, (
            f"max pivot seen = {max_pivot_seen}; helper appears to be "
            "counting non-pad tokens globally instead of using the "
            "right-padded prefix boundary"
        )

    def test_uses_min_real_length_across_batch(self):
        # One row has 20 real tokens, the other only 10. The shared
        # pivot must respect the shorter row so that *every* row's
        # targets are real content.
        from olmlx.engine.dflash.prepare import _select_pivot

        pad = 0
        seq_len = 32
        block_size = 2
        long_row = list(range(1, 21)) + [pad] * (seq_len - 20)
        short_row = list(range(1, 11)) + [pad] * (seq_len - 10)
        ids = mx.array([long_row, short_row])
        import random as _r

        _r.seed(0)
        for _ in range(20):
            p = _select_pivot(ids, pad_token_id=pad, block_size=block_size)
            assert p is not None
            # Must fit inside the SHORT row's real content.
            assert block_size <= p <= 10 - block_size - 1


class TestSelectPivots:
    """Multi-window pivot selection via slot-and-jitter."""

    def test_k1_delegates_to_select_pivot(self):
        """K=1 must be bit-exact with _select_pivot under a fixed seed
        — the multi-window code path collapses to the legacy single
        pivot when num_windows=1."""
        import random
        from olmlx.engine.dflash.prepare import _select_pivot, _select_pivots

        pad = 0
        block_size = 4
        ids = mx.full((2, 32), 7, dtype=mx.int32)

        random.seed(123)
        legacy = _select_pivot(ids, pad_token_id=pad, block_size=block_size)
        random.seed(123)
        multi = _select_pivots(
            ids, pad_token_id=pad, block_size=block_size, num_windows=1
        )

        assert legacy is not None
        assert multi == [legacy]

    def test_returns_none_when_no_window_fits(self):
        from olmlx.engine.dflash.prepare import _select_pivots

        pad = 0
        block_size = 4
        # min_real = 5 < 2*4 + 1 = 9 — no window fits.
        ids = mx.concatenate(
            [
                mx.full((1, 5), 7, dtype=mx.int32),
                mx.full((1, 27), pad, dtype=mx.int32),
            ],
            axis=1,
        )
        ids = mx.broadcast_to(ids, (2, 32))
        assert (
            _select_pivots(ids, pad_token_id=pad, block_size=block_size, num_windows=4)
            is None
        )

    def test_k4_returns_four_non_overlapping_pivots(self):
        """A long-enough sequence yields exactly K pivots, all within
        the valid range, with adjacent pivots at least block_size+1
        apart so their [p, p+block_size] spans cannot overlap.

        Tested across many seeds — the non-overlap invariant is a hard
        guarantee, not a probabilistic one. A single seed would let a
        broken implementation pass by luck (and previously did — see
        gh#382 review).
        """
        import random
        from olmlx.engine.dflash.prepare import _select_pivots

        pad = 0
        block_size = 4
        # min_real = 200 → range_size = 192 → max non-overlap = 192//5 = 38
        ids = mx.full((2, 200), 7, dtype=mx.int32)

        for seed in range(200):
            random.seed(seed)
            pivots = _select_pivots(
                ids, pad_token_id=pad, block_size=block_size, num_windows=4
            )
            assert pivots is not None, f"seed={seed}: got None"
            assert len(pivots) == 4, f"seed={seed}: got {pivots}"
            for p in pivots:
                assert block_size <= p <= 200 - block_size - 1, (
                    f"seed={seed}: pivot {p} outside valid range"
                )
            assert pivots == sorted(pivots), f"seed={seed}: not sorted {pivots}"
            for i in range(len(pivots) - 1):
                assert pivots[i + 1] - pivots[i] >= block_size + 1, (
                    f"seed={seed}: pivots {pivots[i]} and {pivots[i + 1]} are "
                    f"within block_size+1={block_size + 1} of each other"
                )

    def test_k_caps_to_max_non_overlapping_fit(self):
        """If the operator requests more windows than the valid range
        can accommodate non-overlapping, return the maximum that
        actually fits — K is a target, not a guarantee. Non-overlap
        must hold across all seeds."""
        import random
        from olmlx.engine.dflash.prepare import _select_pivots

        pad = 0
        block_size = 4
        # min_real = 20 → range_size = 12 → max non-overlap = 12//5 = 2
        ids = mx.concatenate(
            [
                mx.full((1, 20), 7, dtype=mx.int32),
                mx.full((1, 12), pad, dtype=mx.int32),
            ],
            axis=1,
        )
        ids = mx.broadcast_to(ids, (2, 32))

        for seed in range(200):
            random.seed(seed)
            pivots = _select_pivots(
                ids, pad_token_id=pad, block_size=block_size, num_windows=8
            )
            assert pivots is not None, f"seed={seed}: got None"
            assert 1 <= len(pivots) <= 2, f"seed={seed}: got {pivots}"
            for i in range(len(pivots) - 1):
                assert pivots[i + 1] - pivots[i] >= block_size + 1, (
                    f"seed={seed}: gap too small in {pivots}"
                )

    def test_pivots_stay_in_unpadded_prefix(self):
        """With trailing padding, every selected pivot must satisfy
        p + block_size < min_real so targets land on real tokens."""
        from olmlx.engine.dflash.prepare import _select_pivots

        pad = 0
        block_size = 4
        # Row 0: real_len=60; Row 1: real_len=40; min_real=40.
        row0 = mx.concatenate(
            [
                mx.full((1, 60), 7, dtype=mx.int32),
                mx.full((1, 4), pad, dtype=mx.int32),
            ],
            axis=1,
        )
        row1 = mx.concatenate(
            [
                mx.full((1, 40), 7, dtype=mx.int32),
                mx.full((1, 24), pad, dtype=mx.int32),
            ],
            axis=1,
        )
        ids = mx.concatenate([row0, row1], axis=0)

        pivots = _select_pivots(
            ids, pad_token_id=pad, block_size=block_size, num_windows=4
        )
        assert pivots is not None
        for p in pivots:
            assert p + block_size < 40, (
                f"pivot {p} targets ({p + 1}..{p + block_size}) extend past min_real=40"
            )


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------


class TestSkippedBatchesPreserveStepBudget:
    """Skipped (all-padding) batches must NOT reduce the gradient-step
    count. ``enumerate(batches)`` advances ``step`` on every iteration,
    including the ones the pivot-selector rejects via ``continue``;
    without a separate counter the loop exits with fewer real gradient
    updates than the operator asked for, producing a quietly
    under-trained checkpoint.

    Test strategy: feed an iterator that alternates pad-only batches
    (which trigger ``_select_pivot is None``) with real-content
    batches. Assert that the number of progress-callback ``loss=``
    messages equals ``steps`` — i.e. every requested step ran a real
    optimizer update.
    """

    def test_skipped_batches_do_not_run_target_forward(self, tmp_path, monkeypatch):
        """Performance: when ``_select_pivot`` returns ``None`` the
        batch is discarded, so the (dominant) target forward pass for
        that batch must be skipped too. Otherwise every pad-only batch
        burns the same compute as a real training step on a frozen
        target — a 35B-A3B forward thrown away for nothing.

        Count invocations of ``_capture_target_outputs`` (the wrapper
        that runs the target forward and harvests captured hiddens).
        With one real-content batch followed by one pad-only batch and
        ``steps=1``, it must run exactly once: the pad-only batch
        should be rejected before any target compute is spent.
        """
        from olmlx.engine.dflash import prepare as prepare_mod
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        original_capture = prepare_mod._capture_target_outputs
        call_count = {"n": 0}

        def counting_capture(*args, **kwargs):
            call_count["n"] += 1
            return original_capture(*args, **kwargs)

        monkeypatch.setattr(prepare_mod, "_capture_target_outputs", counting_capture)

        seq_len = 32
        batch_size = 2

        # First batch is real content (avoids id 0 = pad), second is
        # all-pad and must be skipped without a target forward.
        def first_real_then_pad():
            real = mx.full((batch_size, seq_len), 5, dtype=mx.int32)
            pad_only = mx.zeros((batch_size, seq_len), dtype=mx.int32)
            yield real
            yield pad_only

        prepare_dflash_draft(
            tmp_path,
            steps=1,
            batch_size=batch_size,
            seq_len=seq_len,
            block_size=2,
            num_hidden_layers=1,
            num_target_layers=2,
            lr=1e-2,
            output_dir=tmp_path / "dflash_out",
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=first_real_then_pad(),
        )
        # Exactly one target forward — for the real-content batch.
        # The pad-only batch must have been skipped before
        # ``_capture_target_outputs`` ran.
        assert call_count["n"] == 1, (
            f"target forward ran {call_count['n']} times, expected 1; "
            "pad-only batches are wasting target forward compute"
        )

    def test_aborts_when_infinite_all_padding_iterator(self, tmp_path, caplog):
        """Critical: HuggingFace streaming datasets are typically
        infinite. If every batch in a window is all-padding (e.g.
        misconfigured ``block_size`` for a short-sequence dataset),
        the training loop must terminate rather than spin forever
        waiting for ``real_step`` to reach ``steps``. A
        max-consecutive-skip guard turns the silent hang into a
        clear, actionable error.
        """
        import logging

        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        seq_len = 32
        batch_size = 2

        def infinite_pad_batches():
            pad_only = mx.zeros((batch_size, seq_len), dtype=mx.int32)
            while True:  # Truly infinite — mirrors HF streaming.
                yield pad_only

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.dflash.prepare"):
            # Should terminate (via error log + break), not hang.
            # pad_token_id=0 is wired through _MockTokenizer so pad_for_pivot
            # is active — the consecutive_skips guard fires with an ERROR log
            # and breaks out, letting the post-loop warning + checkpoint save
            # run (preserving any partial progress from real steps).
            prepare_dflash_draft(
                tmp_path,
                steps=20,
                batch_size=batch_size,
                seq_len=seq_len,
                block_size=2,
                num_hidden_layers=1,
                num_target_layers=2,
                lr=1e-2,
                output_dir=tmp_path / "dflash_out",
                _target_loader=_mock_target_loader(vocab, hidden, num_layers),
                _batch_iterator=infinite_pad_batches(),
            )
        # Verify the abort error was logged.
        assert any(
            "aborted" in rec.message
            for rec in caplog.records
            if rec.levelno >= logging.ERROR
        )

    def test_warns_when_batch_stream_exhausts_before_steps(self, tmp_path, caplog):
        """If the batch iterator ends before ``steps`` real updates ran
        (every batch was a pad-only skip, or finite dataset short of the
        budget), the operator gets a saved checkpoint that's quietly
        under-trained or completely untrained. Surface that as a warning
        on the way out so it isn't silent.
        """
        import logging

        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        # All-pad batches → every step is skipped → real_step never
        # advances → loop exits when the iterator drains.
        seq_len = 32
        batch_size = 2

        def all_pad_batches():
            pad_only = mx.zeros((batch_size, seq_len), dtype=mx.int32)
            for _ in range(10):
                yield pad_only

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.dflash.prepare"):
            prepare_dflash_draft(
                tmp_path,
                steps=20,
                batch_size=batch_size,
                seq_len=seq_len,
                block_size=2,
                num_hidden_layers=1,
                num_target_layers=2,
                lr=1e-2,
                output_dir=tmp_path / "dflash_out",
                _target_loader=_mock_target_loader(vocab, hidden, num_layers),
                _batch_iterator=all_pad_batches(),
            )
        # Operator must see a clear signal that no real updates ran.
        # Use ``caplog.messages`` (the formatted-message property) rather
        # than ``rec.message`` directly: ``LogRecord.message`` is only set
        # after ``Formatter.format()`` runs, which pytest's
        # ``LogCaptureHandler`` does not guarantee to call.
        assert any(
            "no real gradient steps" in msg.lower() or "0/20" in msg or "0 of 20" in msg
            for msg in caplog.messages
        ), f"expected an under-training warning, got: {caplog.messages}"

    def test_skipped_batches_do_not_eat_step_budget(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        # ``_MockTokenizer.pad_token_id == 0``; build alternating
        # all-pad-only and real-content batches. The training loop
        # should ``continue`` past pad-only batches without consuming
        # one of its ``steps`` real gradient updates.
        seq_len = 32
        batch_size = 2

        def alternating_batches():
            real = mx.full((batch_size, seq_len), 5, dtype=mx.int32)  # all id=5
            pad_only = mx.zeros((batch_size, seq_len), dtype=mx.int32)
            i = 0
            while True:
                yield pad_only if (i % 2 == 0) else real
                i += 1

        steps = 5
        log_every = 1
        loss_lines: list[str] = []

        def cb(msg: str, _frac: float) -> None:
            if "loss=" in msg:
                loss_lines.append(msg)

        prepare_dflash_draft(
            tmp_path,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            block_size=2,
            num_hidden_layers=1,
            num_target_layers=2,
            lr=1e-2,
            output_dir=tmp_path / "dflash_out",
            progress_callback=cb,
            log_every=log_every,
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=alternating_batches(),
        )
        # Every requested step must have run a real gradient update.
        # Bug behavior: only ``ceil(steps/2)`` lines because half the
        # iterations were skipped pad-only batches that still consumed
        # a slot.
        assert len(loss_lines) == steps, (
            f"got {len(loss_lines)} real gradient steps, expected {steps}; "
            "skipped batches are eating into the step budget"
        )


class TestPrepareDflashDraft:
    def test_loss_decreases(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        # Seed for determinism; the synthetic-data + tiny-target setup
        # is noisy enough that an unseeded run occasionally fails to
        # show monotone descent in 20 steps. Both ``mx.random`` (data
        # init) and Python ``random`` (pivot sampling — see
        # ``prepare.py`` for why we don't use ``mx.random.randint``)
        # need seeding.
        import random as _stdlib_random

        mx.random.seed(0)
        _stdlib_random.seed(0)
        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        # Capture losses by patching the optimizer step indirectly: we
        # wrap stream_training_batches to deterministic batches AND read
        # the training log via a custom callback.
        losses: list[float] = []

        def cb(msg: str, _frac: float) -> None:
            # The progress callback message is "Training step N/M loss=X.XXXX"
            if "loss=" in msg:
                losses.append(float(msg.split("loss=")[-1].strip()))

        out_dir = prepare_dflash_draft(
            tmp_path,
            steps=20,
            batch_size=2,
            seq_len=32,
            block_size=2,
            num_hidden_layers=1,
            num_target_layers=2,
            lr=1e-2,
            output_dir=tmp_path / "dflash_out",
            progress_callback=cb,
            log_every=1,
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_batches(vocab, batch_size=2, seq_len=32, n=20),
        )
        assert (out_dir / "config.json").exists()
        assert (out_dir / "model-00001-of-00001.safetensors").exists()
        assert len(losses) >= 5
        # Losses are noisy on synthetic data; require the *trailing*
        # window to be lower than the *leading* window rather than a
        # strict per-step monotone descent.
        head = sum(losses[:3]) / 3
        tail = sum(losses[-3:]) / 3
        assert tail < head, f"loss did not decrease: head={head}, tail={tail}"

    def test_saved_config_round_trips_through_loader(self, tmp_path):
        from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        out_dir = prepare_dflash_draft(
            tmp_path,
            steps=2,
            batch_size=2,
            seq_len=32,
            block_size=2,
            num_hidden_layers=1,
            num_target_layers=2,
            lr=1e-2,
            output_dir=tmp_path / "dflash_out",
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_batches(vocab, batch_size=2, seq_len=32, n=20),
        )

        # Re-parse via the same logic _load_dflash_decoder uses on disk.
        cfg_dict = json.loads((out_dir / "config.json").read_text())
        assert "dflash_config" in cfg_dict
        assert "target_layer_ids" in cfg_dict["dflash_config"]
        assert "mask_token_id" in cfg_dict["dflash_config"]

        cfg = DraftConfig(
            hidden_size=cfg_dict["hidden_size"],
            num_hidden_layers=cfg_dict["num_hidden_layers"],
            num_attention_heads=cfg_dict["num_attention_heads"],
            num_key_value_heads=cfg_dict["num_key_value_heads"],
            head_dim=cfg_dict["head_dim"],
            intermediate_size=cfg_dict["intermediate_size"],
            vocab_size=cfg_dict["vocab_size"],
            rms_norm_eps=cfg_dict["rms_norm_eps"],
            rope_theta=cfg_dict["rope_theta"],
            max_position_embeddings=cfg_dict["max_position_embeddings"],
            block_size=cfg_dict["block_size"],
            num_target_layers=cfg_dict["num_target_layers"],
            target_layer_ids=cfg_dict["dflash_config"]["target_layer_ids"],
            mask_token_id=cfg_dict["dflash_config"]["mask_token_id"],
            layer_types=tuple(cfg_dict.get("layer_types", ())),
        )
        draft = DFlashDraftModel(cfg)

        # Load the saved weights — should match shapes exactly.
        weights = mx.load(str(out_dir / "model-00001-of-00001.safetensors"))
        draft.load_weights(list(weights.items()), strict=True)

    def test_unpatches_on_failure(self, tmp_path, monkeypatch):
        """If training raises mid-loop, the target hooks must be uninstalled."""
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        target_holder = {}
        original_loader = _mock_target_loader(vocab, hidden, num_layers)

        def trapping_loader(path):
            target, tok = original_loader(path)
            target_holder["target"] = target
            return target, tok

        # Force the draft step to raise.
        from olmlx.engine.dflash import prepare as prepare_mod

        def boom(*_a, **_kw):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(prepare_mod, "_draft_loss", boom)

        with pytest.raises(RuntimeError, match="simulated failure"):
            prepare_dflash_draft(
                tmp_path,
                steps=5,
                batch_size=2,
                seq_len=32,
                block_size=2,
                num_hidden_layers=1,
                num_target_layers=2,
                _target_loader=trapping_loader,
                _batch_iterator=_synthetic_batches(vocab, 2, 32, n=5),
            )

        # The post-merge ``_patch_model`` API never sets
        # ``_hidden_states`` on the model (storage is caller-owned), so
        # the meaningful invariant is that no layer remains wrapped.
        from olmlx.engine.dflash.decoder import _LayerHook, _get_layers

        target = target_holder["target"]
        layers = _get_layers(target)
        for i, layer in enumerate(layers):
            assert not isinstance(layer, _LayerHook), (
                f"Target layer {i} still wrapped in _LayerHook after a "
                "training run that raised — _unpatch_model must run in the "
                "exception path"
            )

    def test_min_seq_len_runs(self, tmp_path):
        """``seq_len == 2*block_size + 1`` must not trip the pivot guard.

        Regression test for the off-by-one fixed in this PR: with
        ``hi = seq - block_size`` the guard ``hi <= lo`` triggers only
        when ``seq <= 2*block_size``, so the minimum runnable sequence
        length drops to ``2*block_size + 1``. If a future change
        restores the old ``hi = seq - block_size - 1``, this test fails
        with ``ValueError: seq_len=... too small``.
        """
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        block_size = 2
        seq_len = 2 * block_size + 1  # 5
        _write_target_config(tmp_path, vocab, hidden)

        prepare_dflash_draft(
            tmp_path,
            steps=3,
            batch_size=2,
            seq_len=seq_len,
            block_size=block_size,
            num_hidden_layers=1,
            num_target_layers=2,
            output_dir=tmp_path / "dflash_out",
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_batches(
                vocab, batch_size=2, seq_len=seq_len, n=3
            ),
        )

    def test_pivot_upper_bound_reaches_last_window(self, tmp_path, monkeypatch):
        """Pivot sampling must reach the last valid window position.

        The original regression was ``hi = seq - block_size - 1`` with
        ``mx.random.randint`` (exclusive upper bound), which made the
        last valid window position ``seq - block_size - 1`` unreachable.
        The pivot sampler now uses Python's ``random.randint`` (which
        is *inclusive* on both ends) with ``hi_inclusive = seq -
        block_size - 1``, so the last window position is reachable. We
        assert the call site passes the corrected upper bound — catches
        a future regression even if no run happens to sample the
        boundary value.
        """
        from olmlx.engine.dflash import prepare as prepare_mod
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        block_size = 2
        seq_len = 16
        _write_target_config(tmp_path, vocab, hidden)

        original_randint = prepare_mod.random.randint
        pivot_calls: list[tuple[int, int]] = []

        def recording_randint(lo, hi):
            # ``random.randint`` is only used by the pivot sampler in
            # this module today — record every call.
            pivot_calls.append((lo, hi))
            return original_randint(lo, hi)

        monkeypatch.setattr(prepare_mod.random, "randint", recording_randint)

        prepare_dflash_draft(
            tmp_path,
            steps=3,
            batch_size=2,
            seq_len=seq_len,
            block_size=block_size,
            num_hidden_layers=1,
            num_target_layers=2,
            output_dir=tmp_path / "dflash_out",
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_batches(
                vocab, batch_size=2, seq_len=seq_len, n=3
            ),
        )

        assert pivot_calls, "expected pivot randint call to be recorded"
        for lo, hi in pivot_calls:
            assert lo == block_size, f"pivot lo={lo}, expected {block_size}"
            # ``random.randint`` is inclusive on the upper bound, so we
            # pass ``seq_len - block_size - 1`` (the last valid pivot).
            assert hi == seq_len - block_size - 1, (
                f"pivot hi={hi}, expected {seq_len - block_size - 1} "
                "(last valid window position; random.randint is "
                "inclusive on the upper bound)"
            )
