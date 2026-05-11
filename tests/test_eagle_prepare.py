"""Tests for ``olmlx eagle prepare`` — EAGLE draft training.

These tests run the training loop end-to-end against a synthetic
target so they don't need network access or a real model. Coverage:

- Target-config → ``EagleConfig`` derivation (vocab, head dims, GQA
  shape) including the multimodal ``text_config`` descent path
- ``rope_theta`` resolution: flat field, nested ``rope_parameters``,
  fallback warning
- The training loop: loss decreases over a handful of steps when fed
  deterministic batches against a synthetic target
- Saved-checkpoint round-trip: a freshly-saved draft loads back through
  ``EagleDraftModel(EagleConfig(...))`` with matching parameter shapes
- Saved ``config.json`` carries the EAGLE-specific marker so the
  inference loader (Phase E) can distinguish from DFlash drafts
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Synthetic target + tokenizer
# ---------------------------------------------------------------------------


class _MockSelfAttn(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        if cache is not None:
            B, L, D = x.shape
            kv = x.reshape(B, 1, L, D)
            cache.update_and_fetch(kv, kv)
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
            layer_cache = cache[i] if cache is not None else None
            h = layer(h, cache=layer_cache)
        h = self.model.norm(h)
        return self.lm_head(h)


class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1


def _mock_target_loader(vocab_size: int, hidden_size: int, num_layers: int):
    target = _Target(vocab_size, hidden_size, num_layers)
    tokenizer = _MockTokenizer()
    return lambda _path: (target, tokenizer)


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


def _synthetic_precomputed_batches(
    vocab: int, batch_size: int, seq_len: int, hidden: int, n: int
):
    """Yield (input_ids, target_hidden) tuples mirroring the format
    ``iter_precomputed_shards`` produces for production runs."""
    rng = mx.random.key(0)
    for _ in range(n):
        rng, sub_ids, sub_h = mx.random.split(rng, num=3)
        ids = mx.random.randint(1, vocab, (batch_size, seq_len), key=sub_ids)
        # Hidden has shape (batch, seq, hidden) — single-layer (EAGLE
        # only consumes the last layer; tests don't need multi-layer
        # concatenation).
        h = mx.random.normal((batch_size, seq_len, hidden), key=sub_h)
        yield ids, h


# ---------------------------------------------------------------------------
# Config derivation
# ---------------------------------------------------------------------------


class TestBuildEagleConfig:
    def test_inherits_target_dims(self):
        from olmlx.engine.eagle.prepare import _build_eagle_config

        target_cfg = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 32768,
        }
        cfg = _build_eagle_config(
            target_cfg,
            num_hidden_layers=1,
            block_size=4,
        )
        assert cfg.hidden_size == 4096
        assert cfg.head_dim == 128
        assert cfg.num_key_value_heads == 8
        assert cfg.vocab_size == 32000
        assert cfg.block_size == 4
        assert cfg.rope_theta == 1_000_000.0

    def test_descends_into_text_config(self):
        # Multimodal target (Qwen3.5-style nested config).
        from olmlx.engine.eagle.prepare import _build_eagle_config

        target_cfg = {
            "model_type": "qwen3_5",
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 5120,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "intermediate_size": 17408,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_theta": 10_000_000},
                "max_position_embeddings": 262144,
            },
            "vision_config": {"hidden_size": 1024},
        }
        cfg = _build_eagle_config(target_cfg, num_hidden_layers=1, block_size=4)
        assert cfg.hidden_size == 5120
        assert cfg.vocab_size == 248320
        assert cfg.rope_theta == 10_000_000.0

    def test_rope_theta_fallback_warns(self, caplog):
        import logging

        from olmlx.engine.eagle.prepare import _build_eagle_config

        target_cfg = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 11008,
            # No rope_theta and no rope_parameters.
        }
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.eagle.prepare"):
            cfg = _build_eagle_config(target_cfg, num_hidden_layers=1, block_size=4)
        assert cfg.rope_theta == 10000.0
        assert any("rope_theta" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Training loop end-to-end
# ---------------------------------------------------------------------------


class TestPrepareEagleDraft:
    def test_loss_decreases(self, tmp_path):
        from olmlx.engine.eagle.prepare import prepare_eagle_draft

        mx.random.seed(0)
        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        losses: list[float] = []

        def cb(msg: str, _frac: float) -> None:
            if "loss=" in msg:
                losses.append(float(msg.split("loss=")[-1].strip()))

        out_dir = prepare_eagle_draft(
            tmp_path,
            steps=20,
            batch_size=2,
            seq_len=32,
            block_size=2,
            num_hidden_layers=1,
            lr=1e-2,
            output_dir=tmp_path / "eagle_out",
            progress_callback=cb,
            log_every=1,
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_precomputed_batches(
                vocab, batch_size=2, seq_len=32, hidden=hidden, n=20
            ),
        )
        assert (out_dir / "config.json").exists()
        assert (out_dir / "model-00001-of-00001.safetensors").exists()
        assert len(losses) >= 5
        # Loss is noisy on synthetic data; require trailing window
        # below the leading window rather than strict monotone descent.
        head = sum(losses[:3]) / 3
        tail = sum(losses[-3:]) / 3
        assert tail < head, f"loss did not decrease: head={head}, tail={tail}"

    def test_saved_config_carries_eagle_marker(self, tmp_path):
        # Phase E will dispatch DFlash vs EAGLE drafts by the
        # ``eagle_config`` block in the saved config.json. Lock the
        # field name in.
        from olmlx.engine.eagle.prepare import prepare_eagle_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)
        out_dir = prepare_eagle_draft(
            tmp_path,
            steps=2,
            batch_size=2,
            seq_len=32,
            block_size=2,
            num_hidden_layers=1,
            lr=1e-2,
            output_dir=tmp_path / "eagle_out",
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_precomputed_batches(
                vocab, batch_size=2, seq_len=32, hidden=hidden, n=5
            ),
        )
        cfg_dict = json.loads((out_dir / "config.json").read_text())
        assert "eagle_config" in cfg_dict
        # The block_size lives under the eagle_config marker so the
        # loader can distinguish multi-architecture drafts.
        assert cfg_dict["eagle_config"]["block_size"] == 2

    def test_saved_weights_round_trip(self, tmp_path):
        # A freshly-saved draft loads back through
        # ``EagleDraftModel(EagleConfig(...))`` with matching parameter
        # shapes. This catches a save/load schema drift before Phase E
        # writes the inference loader.
        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel
        from olmlx.engine.eagle.prepare import prepare_eagle_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)
        out_dir = prepare_eagle_draft(
            tmp_path,
            steps=2,
            batch_size=2,
            seq_len=32,
            block_size=2,
            num_hidden_layers=1,
            lr=1e-2,
            output_dir=tmp_path / "eagle_out",
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_precomputed_batches(
                vocab, batch_size=2, seq_len=32, hidden=hidden, n=5
            ),
        )
        cfg_dict = json.loads((out_dir / "config.json").read_text())
        cfg = EagleConfig(
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
            block_size=cfg_dict["eagle_config"]["block_size"],
            rope_scaling=cfg_dict.get("rope_scaling"),
        )
        m = EagleDraftModel(cfg)
        # Load weights; should not raise on shape mismatch.
        m.load_weights(str(out_dir / "model-00001-of-00001.safetensors"))


class TestEagleLossSignature:
    """The loss function takes (draft, h_target, input_ids) and returns
    a scalar mx.array. Cheap unit test against a hand-built draft."""

    def test_loss_is_scalar_and_finite(self):
        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel
        from olmlx.engine.eagle.prepare import _eagle_loss

        cfg = EagleConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=32,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=512,
            block_size=4,
        )
        draft = EagleDraftModel(cfg)
        embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        lm = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        draft.bind_via_modules(embed, lm)

        # Fake batch: (B=2, L=8) tokens + (B=2, L=8, H=16) hiddens.
        ids = mx.random.randint(0, cfg.vocab_size, (2, 8))
        h = mx.random.normal((2, 8, cfg.hidden_size))
        loss = _eagle_loss(draft, h, ids)
        assert loss.ndim == 0  # scalar
        assert mx.isfinite(loss).item()

    def test_subsampled_loss_is_scalar_and_finite(self):
        """``sample_positions`` skips lm_head over un-sampled positions
        but still produces a finite scalar loss. We don't assert
        equality with the full path because position sampling is random
        and per-position CE varies — instead we verify the path runs
        end-to-end and produces a sensible value within ~3 sigma of the
        full-sequence loss."""
        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel
        from olmlx.engine.eagle.prepare import _eagle_loss

        cfg = EagleConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=32,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=512,
            block_size=4,
        )
        draft = EagleDraftModel(cfg)
        embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        lm = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        draft.bind_via_modules(embed, lm)

        ids = mx.random.randint(0, cfg.vocab_size, (2, 32))
        h = mx.random.normal((2, 32, cfg.hidden_size))
        loss_full = _eagle_loss(draft, h, ids)
        loss_sub = _eagle_loss(draft, h, ids, sample_positions=8)
        assert loss_sub.ndim == 0
        assert mx.isfinite(loss_sub).item()
        # An untrained tied-embedding draft on a 64-vocab batch sits
        # near ln(64) ≈ 4.16; both paths should land in the same
        # ballpark.
        assert abs(float(loss_sub.item()) - float(loss_full.item())) < 2.0

    def test_subsampled_loss_clamps_to_seq_len(self):
        """Asking for more sampled positions than the sequence has
        should not crash — it just scores every available position."""
        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel
        from olmlx.engine.eagle.prepare import _eagle_loss

        cfg = EagleConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=32,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=512,
            block_size=4,
        )
        draft = EagleDraftModel(cfg)
        embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        lm = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        draft.bind_via_modules(embed, lm)

        ids = mx.random.randint(0, cfg.vocab_size, (1, 4))
        h = mx.random.normal((1, 4, cfg.hidden_size))
        loss = _eagle_loss(draft, h, ids, sample_positions=1024)
        assert mx.isfinite(loss).item()
