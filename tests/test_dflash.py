"""Tests for the DFlash block-diffusion speculative decoder.

Covers the universal layer-hooking infrastructure (`_LayerHook`,
`_get_layers`, `_patch_model`), the rewritten `DraftConfig` schema,
`DFlashDraftModel.bind()` for various target shapes, the
`_trim_recent_cache` helper across `KVCache` and `RotatingKVCache`,
end-to-end `DFlashDecoder.prefill`/`step`/`reset`, and the
`speculative_strategy` migration path.
"""

from __future__ import annotations

import threading

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache

from olmlx.engine.dflash.decoder import (
    DFlashDecoder,
    _find_gdn_class,
    _LayerHook,
    _get_layers,
    _patch_model,
    _trim_recent_cache,
    _unpatch_model,
)
from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig
from olmlx.engine.speculative_stream import speculative_stream_generate


# ---------------------------------------------------------------------------
# Shared synthetic models
# ---------------------------------------------------------------------------


class _MockSelfAttn(nn.Module):
    """Minimal attention module that exercises the KV cache protocol."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.n_heads = 1
        self.n_kv_heads = 1
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        if cache is not None:
            k = v = x.reshape(x.shape[0], 1, -1, x.shape[-1])
            cache.update_and_fetch(k, v)
        return self.proj(x)


class _MockLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = _MockSelfAttn(hidden_size)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        return x + self.self_attn(x, mask=mask, cache=cache)


class _Inner(nn.Module):
    """Inner transformer with .layers and .embed_tokens (mimics mlx-lm)."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [_MockLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)


class _Target(nn.Module):
    """mlx-lm style target with .model and .lm_head."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 4):
        super().__init__()
        self.model = _Inner(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def layers(self):
        # Mirrors mlx_lm.models.qwen3.Model so make_prompt_cache works.
        return self.model.layers

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        h = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, cache=cache[i] if cache is not None else None)
        h = self.model.norm(h)
        return self.lm_head(h)


def _make_draft_config(
    vocab_size: int,
    hidden_size: int,
    target_layer_ids: list[int],
    *,
    num_hidden_layers: int = 1,
    sliding_window: int | None = None,
    layer_types: tuple[str, ...] | None = None,
    block_size: int = 4,
) -> DraftConfig:
    return DraftConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hidden_size // 2,
        intermediate_size=hidden_size * 2,
        vocab_size=vocab_size,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=2048,
        block_size=block_size,
        num_target_layers=len(target_layer_ids),
        target_layer_ids=target_layer_ids,
        mask_token_id=0,
        layer_types=layer_types or (("full_attention",) * num_hidden_layers),
        sliding_window=sliding_window,
    )


# ---------------------------------------------------------------------------
# Layer hooks
# ---------------------------------------------------------------------------


class TestGetLayers:
    def test_resolves_via_model_layers(self):
        target = _Target(vocab_size=8, hidden_size=8, num_layers=2)
        assert _get_layers(target) is target.model.layers

    def test_resolves_via_language_model_layers(self):
        class VLMLanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [_MockLayer(4), _MockLayer(4)]

        class VLMTarget(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = VLMLanguageModel()

        target = VLMTarget()
        assert _get_layers(target) is target.language_model.layers

    def test_resolves_via_flat_layers(self):
        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [_MockLayer(4)]

        target = Flat()
        assert _get_layers(target) is target.layers

    def test_raises_when_missing(self):
        class Empty(nn.Module):
            pass

        with pytest.raises(AttributeError, match="Cannot find layers"):
            _get_layers(Empty())


class TestLayerHook:
    def test_proxies_attribute_access(self):
        layer = _MockLayer(8)
        storage: list = [None]
        hook = _LayerHook(layer, 0, storage)
        # Attribute lookups should pass through to the wrapped layer.
        assert hook.self_attn is layer.self_attn

    def test_captures_call_output(self):
        layer = _MockLayer(8)
        storage: list = [None]
        hook = _LayerHook(layer, 0, storage)
        x = mx.zeros((1, 3, 8))
        out = hook(x)
        mx.eval(out)
        assert storage[0] is not None
        assert storage[0].shape == (1, 3, 8)


class TestPatchModel:
    def test_idempotent(self):
        target = _Target(vocab_size=8, hidden_size=8, num_layers=4)
        original_layers = list(target.model.layers)
        storage: list = [None, None]
        _patch_model(target, [1, 3], storage)
        first_hook = target.model.layers[1]
        # Second call detects existing hooks and is a no-op.
        _patch_model(target, [1, 3], storage)
        assert target.model.layers[1] is first_hook
        _unpatch_model(target)
        assert target.model.layers == original_layers
        # Storage stays caller-owned; the target ``nn.Module`` never
        # receives a ``_hidden_states`` attribute (would otherwise leak
        # into ``model.parameters()`` and corrupt distributed eval).
        assert not hasattr(target, "_hidden_states")

    def test_captures_at_configured_indices(self):
        target = _Target(vocab_size=16, hidden_size=8, num_layers=4)
        target_layer_ids = [1, 3]
        storage: list = [None, None]
        _patch_model(target, target_layer_ids, storage)
        out = target(mx.array([[1, 2, 3, 4]]))
        mx.eval(out)
        assert storage[0] is not None
        assert storage[1] is not None
        assert storage[0].shape == (1, 4, 8)
        _unpatch_model(target)
        # The target should remain free of decoder-owned state.
        assert not hasattr(target, "_hidden_states")


# ---------------------------------------------------------------------------
# Cache trim
# ---------------------------------------------------------------------------


class TestTrimRecentCache:
    def test_kv_cache(self):
        target = _Target(vocab_size=16, hidden_size=8, num_layers=2)
        cache = [KVCache() for _ in target.model.layers]
        target(mx.array([[1, 2, 3, 4, 5]]), cache=cache)
        assert cache[0].offset == 5
        _trim_recent_cache(cache, 2)
        assert cache[0].offset == 3

    def test_rotating_cache(self):
        c = RotatingKVCache(max_size=8, keep=0)
        # Populate with 5 K/V tokens (head_dim=4, n_kv_heads=1).
        for _ in range(5):
            k = v = mx.zeros((1, 1, 1, 4))
            c.update_and_fetch(k, v)
        assert c.offset == 5
        _trim_recent_cache([c], 2)
        assert c.offset == 3

    def test_zero_or_negative_is_noop(self):
        target = _Target(vocab_size=8, hidden_size=8, num_layers=1)
        cache = [KVCache() for _ in target.model.layers]
        target(mx.array([[1, 2, 3]]), cache=cache)
        original = cache[0].offset
        _trim_recent_cache(cache, 0)
        _trim_recent_cache(cache, -5)
        assert cache[0].offset == original


# ---------------------------------------------------------------------------
# Dynamic GatedDeltaNet discovery
# ---------------------------------------------------------------------------


class TestFindGDNClass:
    """Verify ``_find_gdn_class`` discovers the GDN class from the model
    rather than relying on a hardcoded import path. Targets that define
    ``GatedDeltaNet`` in different modules (e.g. ``qwen3_5`` vs.
    ``qwen3_5_moe`` vs. some future hybrid) must all be supported."""

    def test_returns_none_when_no_gdn_layer(self):
        target = _Target(vocab_size=8, hidden_size=8, num_layers=2)
        assert _find_gdn_class(target) is None

    def test_finds_gdn_class_by_name(self):
        # Class name plus the structural attributes ``_capturing_gdn_call``
        # actually reads — same-named classes without those attributes are
        # skipped to avoid silently patching unrelated modules.
        class GatedDeltaNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj_qkv = nn.Linear(4, 4, bias=False)
                self.in_proj_z = nn.Linear(4, 4, bias=False)
                self.in_proj_b = nn.Linear(4, 4, bias=False)
                self.in_proj_a = nn.Linear(4, 4, bias=False)
                self.out_proj = nn.Linear(4, 4, bias=False)
                self.conv1d = nn.Conv1d(4, 4, kernel_size=3)
                self.conv_kernel_size = 3
                self.conv_dim = 4
                self.A_log = mx.zeros((4,))
                self.dt_bias = mx.zeros((4,))
                self.norm = nn.RMSNorm(4)
                self.num_k_heads = 1
                self.num_v_heads = 1
                self.head_k_dim = 4
                self.head_v_dim = 4
                self.key_dim = 4

        class _GDNLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_attn = GatedDeltaNet()

        class _GDNTarget(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [_GDNLayer()]

        target = _GDNTarget()
        found = _find_gdn_class(target)
        assert found is GatedDeltaNet

    def test_skips_namesake_without_required_attrs(self):
        # A class that happens to be named ``GatedDeltaNet`` but does not
        # expose the GDN interface must not be patched — patching its
        # ``__call__`` would silently corrupt inference.
        class GatedDeltaNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4, bias=False)

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.fake_gdn = GatedDeltaNet()

        class _Target(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [_Layer()]

        assert _find_gdn_class(_Target()) is None


# ---------------------------------------------------------------------------
# DraftConfig validation
# ---------------------------------------------------------------------------


class TestDraftConfig:
    def test_default_layer_types(self):
        cfg = _make_draft_config(
            vocab_size=64,
            hidden_size=32,
            target_layer_ids=[0],
            num_hidden_layers=3,
            layer_types=(),  # exercise the default-fill path
        )
        assert cfg.layer_types == (
            "full_attention",
            "full_attention",
            "full_attention",
        )

    def test_layer_types_length_mismatch(self):
        with pytest.raises(ValueError, match="layer_types has length"):
            _make_draft_config(
                vocab_size=64,
                hidden_size=32,
                target_layer_ids=[0],
                num_hidden_layers=3,
                layer_types=("full_attention", "full_attention"),
            )

    def test_unsupported_layer_type(self):
        with pytest.raises(ValueError, match="Unsupported layer_types"):
            _make_draft_config(
                vocab_size=64,
                hidden_size=32,
                target_layer_ids=[0],
                num_hidden_layers=2,
                layer_types=("full_attention", "linear_attention"),
            )

    def test_sliding_requires_window(self):
        with pytest.raises(ValueError, match="sliding_window must be"):
            _make_draft_config(
                vocab_size=64,
                hidden_size=32,
                target_layer_ids=[0],
                num_hidden_layers=2,
                layer_types=("full_attention", "sliding_attention"),
                sliding_window=None,
            )

    def test_sliding_window_must_be_at_least_two(self):
        # ``sliding_window <= 0`` pushes ``DFlashAttention.keep`` to -1
        # and silently empties x_ctx; ``sliding_window == 1`` pushes
        # ``keep`` to 0 so context is evicted as soon as it's produced.
        # Both reject at config build.
        for bad in (-1, 0, 1):
            with pytest.raises(ValueError, match="sliding_window must be"):
                _make_draft_config(
                    vocab_size=64,
                    hidden_size=32,
                    target_layer_ids=[0],
                    num_hidden_layers=2,
                    layer_types=("full_attention", "sliding_attention"),
                    sliding_window=bad,
                )

    def test_target_layer_ids_length_mismatch(self):
        with pytest.raises(ValueError, match="target_layer_ids has length"):
            DraftConfig(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                intermediate_size=64,
                vocab_size=64,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                max_position_embeddings=2048,
                block_size=4,
                num_target_layers=3,
                target_layer_ids=[0, 1],  # length 2 != num_target_layers 3
                mask_token_id=0,
            )


# ---------------------------------------------------------------------------
# DFlashDraftModel.bind()
# ---------------------------------------------------------------------------


class TestDraftModelBind:
    def test_bind_via_model_chain(self):
        target = _Target(vocab_size=32, hidden_size=8, num_layers=2)
        cfg = _make_draft_config(32, 8, [0])
        draft = DFlashDraftModel(cfg)
        draft.bind(target)
        assert draft.embed_tokens is target.model.embed_tokens
        assert draft.lm_head is target.lm_head

    def test_bind_does_not_register_borrowed_weights_as_children(self):
        """Borrowed ``embed_tokens`` and ``lm_head`` must stay invisible
        to ``draft.parameters()`` / ``draft.named_modules()``. mlx's
        ``Module.__setattr__`` registers any ``dict``-typed value (which
        ``nn.Module`` is, via inheritance) as a tracked child by default
        — ``bind()`` uses ``object.__setattr__`` to bypass that so
        ``mx.eval(draft.parameters())`` and ``draft.save_weights()``
        don't pull in the target's tensors."""
        target = _Target(vocab_size=32, hidden_size=8, num_layers=2)
        cfg = _make_draft_config(32, 8, [0])
        draft = DFlashDraftModel(cfg)

        params_before = set(draft.parameters().keys())
        modules_before = {n for n, _ in draft.named_modules()}

        draft.bind(target)
        params_after = set(draft.parameters().keys())
        modules_after = {n for n, _ in draft.named_modules()}

        # No new parameter-tree entries appeared.
        assert params_after == params_before
        # No new sub-modules either.
        assert modules_after == modules_before
        # Attribute access still works.
        assert draft.embed_tokens is target.model.embed_tokens
        assert draft.lm_head is target.lm_head

        draft.unbind()
        # State cleared; tree still pristine.
        assert draft.embed_tokens is None
        assert draft.lm_head is None
        assert set(draft.parameters().keys()) == params_before

    def test_bind_via_flat_chain(self):
        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(32, 8)
                self.lm_head = nn.Linear(8, 32, bias=False)
                self.layers = [_MockLayer(8)]

        target = Flat()
        cfg = _make_draft_config(32, 8, [0])
        draft = DFlashDraftModel(cfg)
        draft.bind(target)
        assert draft.embed_tokens is target.embed_tokens
        assert draft.lm_head is target.lm_head

    def test_bind_via_language_model_chain(self):
        class _LMInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(32, 8)
                self.layers = [_MockLayer(8)]

        class _LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _LMInner()
                self.lm_head = nn.Linear(8, 32, bias=False)

        class VLMTarget(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = _LM()

        target = VLMTarget()
        cfg = _make_draft_config(32, 8, [0])
        draft = DFlashDraftModel(cfg)
        draft.bind(target)
        assert draft.embed_tokens is target.language_model.model.embed_tokens
        assert draft.lm_head is target.language_model.lm_head

    def test_bind_required_before_forward(self):
        cfg = _make_draft_config(32, 8, [0])
        draft = DFlashDraftModel(cfg)
        with pytest.raises(RuntimeError, match="bind"):
            draft(
                mx.array([[1, 2, 3]]),
                mx.zeros((1, 3, 8)),
                draft.make_cache(),
            )


# ---------------------------------------------------------------------------
# DFlashDecoder end-to-end
# ---------------------------------------------------------------------------


class TestDFlashDecoder:
    @pytest.fixture()
    def components(self):
        vocab_size, hidden_size = 32, 16
        target = _Target(vocab_size, hidden_size, num_layers=4)
        cfg = _make_draft_config(vocab_size, hidden_size, [1, 3])
        draft = DFlashDraftModel(cfg)
        return target, draft, cfg

    def test_prefill_returns_token(self, components):
        target, draft, cfg = components
        decoder = DFlashDecoder(target, draft, cfg, block_size=2)
        first = decoder.prefill(mx.array([[1, 2, 3]]))
        assert isinstance(first, int)
        assert 0 <= first < cfg.vocab_size

    def test_step_returns_at_least_one_token(self, components):
        target, draft, cfg = components
        decoder = DFlashDecoder(target, draft, cfg, block_size=2)
        decoder.prefill(mx.array([[1, 2, 3]]))
        accepted, num_draft = decoder.step()
        assert 1 <= len(accepted) <= num_draft + 1
        assert num_draft == 2

    def test_reset_clears_state(self, components):
        target, draft, cfg = components
        decoder = DFlashDecoder(target, draft, cfg, block_size=2)
        decoder.prefill(mx.array([[1, 2, 3]]))
        decoder.step()
        decoder.reset()
        assert decoder._target_cache is None
        assert decoder._draft_cache is None
        assert decoder._pending_token is None
        # Hook should be removed; a re-prefill should re-install it.
        assert not hasattr(target, "_hidden_states")

    def test_multi_step(self, components):
        target, draft, cfg = components
        decoder = DFlashDecoder(target, draft, cfg, block_size=3)
        decoder.prefill(mx.array([[1, 2, 3]]))
        for _ in range(3):
            accepted, _ = decoder.step()
            assert len(accepted) >= 1

    def test_compatible_with_speculative_stream(self, components):
        target, draft, cfg = components
        decoder = DFlashDecoder(target, draft, cfg, block_size=2)
        cancel = threading.Event()
        responses = list(
            speculative_stream_generate(
                decoder, [1, 2, 3], max_tokens=5, cancel_event=cancel
            )
        )
        assert len(responses) >= 1
        for resp in responses:
            assert hasattr(resp, "token")
            assert 0 <= resp.token < cfg.vocab_size

    def test_stats_summary(self, components):
        target, draft, cfg = components
        decoder = DFlashDecoder(target, draft, cfg, block_size=2)
        decoder.prefill(mx.array([[1, 2, 3]]))
        decoder.step()
        stats = decoder.stats_summary()
        assert stats["steps"] == 1
        assert stats["proposed"] == 2
        assert stats["lambda"] == 2


# ---------------------------------------------------------------------------
# Migration / config routing
# ---------------------------------------------------------------------------


class TestDflashLegacyMigrationError:
    def test_experimental_dflash_raises(self):
        from olmlx.engine.registry import ModelConfig

        with pytest.raises(ValueError, match="folded into the unified speculative"):
            ModelConfig.from_entry(
                {
                    "hf_path": "Qwen/Qwen3-8B",
                    "experimental": {"dflash": True},
                }
            )

    def test_experimental_dflash_draft_model_raises(self):
        from olmlx.engine.registry import ModelConfig

        with pytest.raises(ValueError, match="folded into the unified speculative"):
            ModelConfig.from_entry(
                {
                    "hf_path": "Qwen/Qwen3-8B",
                    "experimental": {"dflash_draft_model": "z-lab/Qwen3-8B-DFlash"},
                }
            )

    def test_experimental_dflash_block_size_raises(self):
        from olmlx.engine.registry import ModelConfig

        with pytest.raises(ValueError, match="folded into the unified speculative"):
            ModelConfig.from_entry(
                {
                    "hf_path": "Qwen/Qwen3-8B",
                    "experimental": {"dflash_block_size": 8},
                }
            )


class TestSpeculativeStrategyConfig:
    def test_strategy_default_classic(self):
        from olmlx.engine.registry import ModelConfig

        cfg = ModelConfig(hf_path="Qwen/Qwen3-8B", speculative=True)
        resolved = cfg.resolved_speculative()
        assert resolved.strategy == "classic"

    def test_strategy_per_model_dflash(self):
        from olmlx.engine.registry import ModelConfig

        cfg = ModelConfig(
            hf_path="Qwen/Qwen3-8B",
            speculative=True,
            speculative_strategy="dflash",
            speculative_draft_model="z-lab/Qwen3-8B-DFlash",
        )
        resolved = cfg.resolved_speculative()
        assert resolved.strategy == "dflash"
        assert resolved.draft_model == "z-lab/Qwen3-8B-DFlash"

    def test_strategy_invalid_value_rejected(self):
        from olmlx.engine.registry import ModelConfig

        with pytest.raises(ValueError, match="speculative_strategy"):
            ModelConfig(
                hf_path="Qwen/Qwen3-8B",
                speculative_strategy="bogus",  # type: ignore[arg-type]
            )

    def test_strategy_round_trip_via_from_entry(self):
        from olmlx.engine.registry import ModelConfig

        cfg = ModelConfig.from_entry(
            {
                "hf_path": "Qwen/Qwen3-8B",
                "speculative": True,
                "speculative_strategy": "dflash",
                "speculative_draft_model": "z-lab/Qwen3-8B-DFlash",
                "speculative_tokens": 5,
            }
        )
        assert cfg.speculative_strategy == "dflash"
        round_trip = cfg.to_entry()
        assert isinstance(round_trip, dict)
        assert round_trip["speculative_strategy"] == "dflash"
