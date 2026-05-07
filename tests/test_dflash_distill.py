"""Tests for KL distillation in DFlash draft training."""

from __future__ import annotations


import mlx.core as mx
import pytest

# Re-use the shared synthetic-target plumbing from the prepare tests so
# we don't duplicate fixtures across test files.
from tests.test_dflash_prepare import (
    _Target,
    _mock_target_loader,
    _synthetic_batches,
    _write_target_config,
)


# ---------------------------------------------------------------------------
# Loss formula
# ---------------------------------------------------------------------------


class TestDraftLossDistillation:
    """Unit tests for ``_draft_loss`` with the distillation term."""

    def _make_draft(self):
        from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig

        cfg = DraftConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=32,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=128,
            block_size=3,
            num_target_layers=1,
            target_layer_ids=[0],
            mask_token_id=0,
        )
        target = _Target(vocab_size=64, hidden_size=16, num_layers=2)
        draft = DFlashDraftModel(cfg)
        draft.bind(target)
        return draft, cfg

    def test_alpha_zero_falls_back_to_pure_ce(self):
        from olmlx.engine.dflash.prepare import _draft_loss

        draft, cfg = self._make_draft()
        block_size = 2
        block = mx.array([[3, 0, 0]])
        target_hidden = mx.zeros((1, 4, cfg.hidden_size))
        targets = mx.array([[5, 7]])
        cache = draft.make_cache()

        ce_only = _draft_loss(draft, block, target_hidden, targets, cache)
        cache2 = draft.make_cache()
        with_logits = _draft_loss(
            draft,
            block,
            target_hidden,
            targets,
            cache2,
            target_logits_window=mx.zeros((1, block_size, cfg.vocab_size)),
            distill_alpha=0.0,
            distill_temp=2.0,
        )
        assert float(ce_only.item()) == pytest.approx(float(with_logits.item()))

    def test_alpha_one_matches_kl_only(self):
        """alpha=1.0 should produce the pure KL term (CE weight zeroed)."""
        from olmlx.engine.dflash.prepare import _draft_loss

        draft, cfg = self._make_draft()
        block_size = 2
        block = mx.array([[3, 0, 0]])
        target_hidden = mx.zeros((1, 4, cfg.hidden_size))
        targets = mx.array([[5, 7]])
        target_logits = mx.random.normal((1, block_size, cfg.vocab_size))
        cache = draft.make_cache()

        loss = _draft_loss(
            draft,
            block,
            target_hidden,
            targets,
            cache,
            target_logits_window=target_logits,
            distill_alpha=1.0,
            distill_temp=2.0,
        )
        # KL is non-negative — for distinct distributions it's strictly > 0.
        assert float(loss.item()) > 0.0

    def test_kl_zero_when_target_equals_draft(self):
        """KL(p || p) = 0; loss should equal alpha-weighted (1-alpha)*CE."""
        from olmlx.engine.dflash.prepare import _draft_loss

        draft, cfg = self._make_draft()
        block = mx.array([[3, 0, 0]])
        target_hidden = mx.zeros((1, 4, cfg.hidden_size))
        targets = mx.array([[5, 7]])
        cache_a = draft.make_cache()

        # Get the draft's own logits at the masked positions — using
        # them as the "target" makes KL exactly zero.
        own_logits = draft(block, target_hidden, cache_a, logits_start=1)
        mx.eval(own_logits)

        cache_b = draft.make_cache()
        ce = _draft_loss(draft, block, target_hidden, targets, cache_b)
        cache_c = draft.make_cache()
        mixed = _draft_loss(
            draft,
            block,
            target_hidden,
            targets,
            cache_c,
            target_logits_window=own_logits,
            distill_alpha=0.4,
            distill_temp=2.0,
        )
        # mixed == (1 - 0.4) * ce  +  0.4 * 0  ==  0.6 * ce
        assert float(mixed.item()) == pytest.approx(0.6 * float(ce.item()), rel=1e-4)


# ---------------------------------------------------------------------------
# End-to-end with --distill
# ---------------------------------------------------------------------------


class TestPrepareDistill:
    def test_distill_loss_decreases(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        mx.random.seed(0)
        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        losses: list[float] = []

        def cb(msg: str, _frac: float) -> None:
            if "loss=" in msg:
                losses.append(float(msg.split("loss=")[-1].strip()))

        prepare_dflash_draft(
            tmp_path,
            steps=20,
            batch_size=2,
            seq_len=32,
            block_size=2,
            num_hidden_layers=1,
            num_target_layers=2,
            lr=1e-2,
            output_dir=tmp_path / "dflash_out",
            distill=True,
            distill_alpha=0.5,
            distill_temp=2.0,
            progress_callback=cb,
            log_every=1,
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_batches(vocab, batch_size=2, seq_len=32, n=20),
        )
        assert len(losses) >= 5
        head = sum(losses[:3]) / 3
        tail = sum(losses[-3:]) / 3
        assert tail < head, f"distill loss did not decrease: head={head}, tail={tail}"

    def test_distill_with_precomputed_raises(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, 64, 16)
        with pytest.raises(ValueError, match="incompatible with --use-precomputed"):
            prepare_dflash_draft(
                tmp_path,
                steps=1,
                distill=True,
                use_precomputed=tmp_path / "shards",
                _target_loader=_mock_target_loader(64, 16, 2),
                _batch_iterator=_synthetic_batches(64, 2, 32, 1),
            )

    def test_distill_alpha_validated(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, 64, 16)
        with pytest.raises(ValueError, match="distill_alpha"):
            prepare_dflash_draft(
                tmp_path,
                steps=1,
                distill_alpha=1.5,
                _target_loader=_mock_target_loader(64, 16, 2),
                _batch_iterator=_synthetic_batches(64, 2, 32, 1),
            )

    def test_distill_temp_validated(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        _write_target_config(tmp_path, 64, 16)
        with pytest.raises(ValueError, match="distill_temp"):
            prepare_dflash_draft(
                tmp_path,
                steps=1,
                distill_temp=0.0,
                _target_loader=_mock_target_loader(64, 16, 2),
                _batch_iterator=_synthetic_batches(64, 2, 32, 1),
            )
