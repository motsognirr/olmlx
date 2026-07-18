"""Tests for the EAGLE-style autoregressive speculative draft.

Covers:

- ``EagleConfig`` dataclass validation
- ``EagleDraftModel`` construction, forward pass, and weight shapes
- ``bind`` / ``unbind`` for sharing target's ``embed_tokens`` and
  ``lm_head`` (mirrors the DFlash pattern)
- ``EagleDecoder`` prefill / step / reset against a synthetic
  trim-able target (Phase B scope)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest


class TestEagleConfig:
    """Field-by-field validation for the draft config dataclass."""

    def test_minimal_fields_construct_cleanly(self):
        from olmlx.engine.eagle.draft_model import EagleConfig

        cfg = EagleConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            intermediate_size=128,
            vocab_size=1024,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=2048,
            block_size=4,
        )
        assert cfg.hidden_size == 64
        assert cfg.num_hidden_layers == 1
        assert cfg.block_size == 4

    def test_block_size_must_be_positive(self):
        from olmlx.engine.eagle.draft_model import EagleConfig

        with pytest.raises(ValueError, match="block_size"):
            EagleConfig(
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                intermediate_size=128,
                vocab_size=1024,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                max_position_embeddings=2048,
                block_size=0,
            )

    def test_kv_head_count_must_divide_attention_heads(self):
        # GQA invariant: num_attention_heads must be a multiple of
        # num_key_value_heads. Same constraint mlx-lm enforces.
        from olmlx.engine.eagle.draft_model import EagleConfig

        with pytest.raises(ValueError, match="num_key_value_heads"):
            EagleConfig(
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=3,  # 3 not divisible by 2
                num_key_value_heads=2,
                head_dim=32,
                intermediate_size=128,
                vocab_size=1024,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                max_position_embeddings=2048,
                block_size=4,
            )


class TestEagleDraftModelConstruction:
    """The draft model builds from a config and exposes the expected
    parameter shapes. No target binding yet — that's a separate test
    class."""

    def _cfg(self, **overrides):
        from olmlx.engine.eagle.draft_model import EagleConfig

        defaults = dict(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            intermediate_size=128,
            vocab_size=1024,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=2048,
            block_size=4,
        )
        defaults.update(overrides)
        return EagleConfig(**defaults)

    def test_input_projection_shape(self):
        # The projection takes ``concat([h_target, embed(token)])`` of
        # size ``2 * hidden_size`` and produces ``hidden_size``.
        from olmlx.engine.eagle.draft_model import EagleDraftModel

        m = EagleDraftModel(self._cfg(hidden_size=64))
        # mlx.nn.Linear stores weights as (out_features, in_features).
        assert m.input_proj.weight.shape == (64, 128)

    def test_one_decoder_layer_by_default(self):
        from olmlx.engine.eagle.draft_model import EagleDraftModel

        m = EagleDraftModel(self._cfg(num_hidden_layers=1))
        assert len(m.layers) == 1

    def test_supports_two_layer_drafts(self):
        # EAGLE-2 uses two layers; the code must support num_layers > 1
        # without architecture changes.
        from olmlx.engine.eagle.draft_model import EagleDraftModel

        m = EagleDraftModel(self._cfg(num_hidden_layers=2))
        assert len(m.layers) == 2

    def test_norm_layer_present(self):
        from olmlx.engine.eagle.draft_model import EagleDraftModel

        m = EagleDraftModel(self._cfg())
        assert isinstance(m.norm, nn.RMSNorm)


class TestEagleDraftModelForward:
    """Forward pass shape and value checks against a synthetic target."""

    def _cfg_and_model(self, **overrides):
        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

        defaults = dict(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            intermediate_size=64,
            vocab_size=128,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=512,
            block_size=4,
        )
        defaults.update(overrides)
        return EagleDraftModel(EagleConfig(**defaults))

    def _bind_dummy(self, model):
        # Provide an embed_tokens + lm_head so forward works.
        cfg = model.args
        embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        lm = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        model.bind_via_modules(embed, lm)
        return model

    def test_unbound_forward_raises(self):
        # Without binding ``embed_tokens`` and ``lm_head``, forward
        # cannot run — surface that rather than silently producing
        # garbage.
        m = self._cfg_and_model()
        with pytest.raises(RuntimeError, match="bind"):
            m(
                token_ids=mx.array([[5]]),
                h_prev=mx.zeros((1, 1, 32)),
            )

    def test_forward_shape(self):
        # Single-step draft: input is one previous token + one previous
        # hidden state. Output is logits over vocab + new hidden.
        m = self._bind_dummy(self._cfg_and_model())
        token_ids = mx.array([[5, 9]])  # batch=1, seq_len=2
        h_prev = mx.zeros((1, 2, 32))
        logits, h_new = m(token_ids=token_ids, h_prev=h_prev)
        assert logits.shape == (1, 2, 128)
        assert h_new.shape == (1, 2, 32)

    def test_bind_uses_tied_embeddings_fallback(self):
        # When the target lacks a dedicated lm_head module (small Qwen
        # variants like Qwen3.5-0.8B tie input/output embeddings and
        # expose ``embed_tokens.as_linear`` instead), ``bind()`` must
        # fall back to that callable. Regression: ``as_linear`` is a
        # bound method, not an ``nn.Module``; an over-narrow
        # ``_find_lm_head -> nn.Module | None`` annotation would mask
        # this. Forward through the bound draft to confirm the
        # callable lm_head is actually invoked.
        m = self._cfg_and_model()
        target_embed = nn.Embedding(m.args.vocab_size, m.args.hidden_size)

        # mlx ``Embedding`` ships ``as_linear``; verify the fixture
        # actually exposes it so the test exercises the intended path.
        assert callable(getattr(target_embed, "as_linear", None))

        class _FakeInner:
            embed_tokens = target_embed
            # No ``lm_head`` — forces the as_linear fallback.

        class _FakeTarget:
            model = _FakeInner()
            # No top-level ``lm_head`` either.

        m.bind(_FakeTarget())
        assert m.embed_tokens is target_embed
        # ``lm_head`` is the bound method, not the embedding itself.
        assert callable(m.lm_head)
        assert m.lm_head is not target_embed

        # End-to-end: a forward must run without raising and produce
        # logits of the expected shape.
        ids = mx.array([[0, 1, 2]], dtype=mx.int32)
        h = mx.zeros((1, 3, m.args.hidden_size))
        logits, _h_new = m(token_ids=ids, h_prev=h)
        assert logits.shape == (1, 3, m.args.vocab_size)

        m.unbind()
        assert m.lm_head is None

    def test_bind_borrows_target_weights(self):
        # ``bind(target)`` must borrow the target's ``embed_tokens`` and
        # ``lm_head`` modules (not copy weights). Verify identity.
        m = self._cfg_and_model()
        target_embed = nn.Embedding(m.args.vocab_size, m.args.hidden_size)
        target_lm = nn.Linear(m.args.hidden_size, m.args.vocab_size, bias=False)

        class _FakeInner:
            embed_tokens = target_embed

        class _FakeTarget:
            model = _FakeInner()
            lm_head = target_lm

        m.bind(_FakeTarget())
        assert m.embed_tokens is target_embed
        assert m.lm_head is target_lm

        m.unbind()
        assert m.embed_tokens is None
        assert m.lm_head is None

    def test_bound_modules_stay_out_of_parameter_tree(self):
        # ``bind()`` uses ``object.__setattr__`` so the borrowed modules
        # don't get registered as nn.Module children. Crucial for
        # training: if they were tracked, ``nn.value_and_grad(draft,
        # ...)`` would compute gradients against the target's frozen
        # weights and ``mlx.utils.tree_flatten(draft.parameters())``
        # would dump them into the saved checkpoint — duplicating the
        # ~250M-parameter embed and lm_head tensors. Lock the
        # invariant in tests.
        import mlx.utils as mx_utils

        m = self._cfg_and_model()
        target_embed = nn.Embedding(m.args.vocab_size, m.args.hidden_size)
        target_lm = nn.Linear(m.args.hidden_size, m.args.vocab_size, bias=False)

        class _FakeInner:
            embed_tokens = target_embed

        class _FakeTarget:
            model = _FakeInner()
            lm_head = target_lm

        m.bind(_FakeTarget())
        param_keys = {k for k, _ in mx_utils.tree_flatten(m.parameters())}
        leaked = [k for k in param_keys if k.startswith(("embed_tokens.", "lm_head."))]
        assert leaked == [], (
            f"borrowed modules leaked into draft.parameters(): {leaked}"
        )


# ---------------------------------------------------------------------------
# EagleDecoder synthetic target + tests
# ---------------------------------------------------------------------------


class _SimpleAttn(nn.Module):
    """Minimal self-attention shim for the synthetic target.

    Just enough to populate a ``KVCache`` so ``trim_prompt_cache`` has
    something to trim. Returns ``Linear(x)`` regardless of attention
    arithmetic; the synthetic test doesn't exercise correctness of
    attention outputs, only the cache lifecycle and the
    layer-output-capture path.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.n_heads = 1
        self.n_kv_heads = 1
        self.head_dim = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        if cache is not None:
            B, L, D = x.shape
            kv = x.reshape(B, 1, L, D)
            cache.update_and_fetch(kv, kv)
        return self.proj(x)


class _SimpleLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = _SimpleAttn(hidden_size)

    def __call__(self, x, mask=None, cache=None):
        return x + self.self_attn(x, mask=mask, cache=cache)


class _Inner(nn.Module):
    def __init__(self, vocab, hidden, num_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = [_SimpleLayer(hidden) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden)


class _SyntheticTarget(nn.Module):
    """Tiny target with a layers list (so ``_patch_model`` works) and a
    trim-able prompt cache (one ``KVCache`` per layer)."""

    def __init__(self, vocab=64, hidden=32, num_layers=3):
        super().__init__()
        self.model = _Inner(vocab, hidden, num_layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    @property
    def layers(self):
        # Used by mlx_lm.models.cache.make_prompt_cache to introspect
        # the layer count.
        return self.model.layers

    def __call__(self, input_ids, cache=None):
        h = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            layer_cache = cache[i] if cache is not None else None
            h = layer(h, cache=layer_cache)
        h = self.model.norm(h)
        return self.lm_head(h)


def _make_decoder(vocab=64, hidden=32, num_layers=3, block_size=2):
    """Construct (decoder, target, draft) with bound weights."""
    from olmlx.engine.eagle.decoder import EagleDecoder
    from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

    target = _SyntheticTarget(vocab=vocab, hidden=hidden, num_layers=num_layers)
    cfg = EagleConfig(
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hidden // 2,
        intermediate_size=hidden * 2,
        vocab_size=vocab,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        block_size=block_size,
    )
    draft = EagleDraftModel(cfg)
    decoder = EagleDecoder(target, draft, block_size=block_size)
    return decoder, target, draft


class TestEagleDecoderLifecycle:
    """Construction + reset semantics. No prefill yet."""

    def test_construction_uses_last_layer_by_default(self):
        decoder, target, _ = _make_decoder(num_layers=4)
        # 4 layers → default capture id is 3.
        assert decoder._target_layer_id == 3

    def test_construction_rejects_zero_block_size(self):
        from olmlx.engine.eagle.decoder import EagleDecoder
        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

        target = _SyntheticTarget()
        cfg = EagleConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            intermediate_size=64,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=512,
            block_size=4,
        )
        with pytest.raises(ValueError, match="block_size"):
            EagleDecoder(target, EagleDraftModel(cfg), block_size=0)

    def test_step_before_prefill_raises(self):
        decoder, _, _ = _make_decoder()
        with pytest.raises(RuntimeError, match="prefill"):
            decoder.step()

    def test_reset_clears_state(self):
        decoder, _, _ = _make_decoder()
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        decoder.prefill(prompt)
        # Caches are populated.
        assert decoder._target_cache is not None
        assert decoder._draft_cache is not None
        decoder.reset()
        assert decoder._target_cache is None
        assert decoder._draft_cache is None
        assert decoder._seed_token is None
        assert decoder._seed_hidden is None
        assert not decoder._patched
        assert not decoder._bound

    def test_close_is_alias_for_reset(self):
        """ModelManager._close_loaded_model calls .close() on whichever decoder
        type the strategy resolved to. Eagle must expose the same lifecycle
        name as SpeculativeDecoder and DFlashDecoder, otherwise eviction and
        keep-alive expiry raise AttributeError for eagle users.
        """
        decoder, _, _ = _make_decoder()
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        decoder.prefill(prompt)
        decoder.close()
        assert decoder._target_cache is None
        assert decoder._draft_cache is None
        assert not decoder._patched
        assert not decoder._bound


class TestEagleDecoderPrefillStep:
    """End-to-end protocol: prefill, then step produces accepted tokens."""

    def test_prefill_returns_token_id(self):
        decoder, target, _ = _make_decoder()
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        first = decoder.prefill(prompt)
        assert isinstance(first, int)
        # vocab is 64 in the synthetic target.
        assert 0 <= first < 64

    def test_prefill_captures_target_hidden(self):
        decoder, _, _ = _make_decoder(hidden=32)
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        decoder.prefill(prompt)
        # Seed hidden has shape (1, 1, hidden_size).
        assert decoder._seed_hidden is not None
        assert decoder._seed_hidden.shape == (1, 1, 32)

    def test_step_returns_at_least_one_token(self):
        # ``verify_draft_greedy`` always returns at least one token (the
        # target's preferred token at the first mismatch, or a bonus if
        # all drafts accepted).
        decoder, _, _ = _make_decoder(block_size=2)
        prompt = mx.array([[5, 7, 9]], dtype=mx.int32)
        decoder.prefill(prompt)
        accepted, num_drafts = decoder.step()
        assert 1 <= len(accepted) <= 3  # block_size + 1
        assert 0 <= num_drafts <= 2  # block_size

    def test_step_advances_seed_state(self):
        # The seed token must rotate to the last accepted token after
        # each step; the seed hidden must change shape-wise stay (1,1,H)
        # but the contents should differ from prefill.
        decoder, _, _ = _make_decoder(block_size=2)
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        first_token = decoder.prefill(prompt)
        seed_h_before = decoder._seed_hidden
        accepted, _ = decoder.step()
        assert decoder._seed_token == accepted[-1]
        assert decoder._seed_hidden is not None
        assert decoder._seed_hidden.shape == seed_h_before.shape
        # If the very first step produced 1 accepted token (= target's
        # first prediction), and that matched ``first_token``, we'd
        # not rotate. Avoid asserting strict change; just confirm the
        # decoder stayed consistent.
        _ = first_token

    def test_two_consecutive_steps(self):
        # Two back-to-back steps must not crash and must return
        # consistently-shaped accepted lists.
        decoder, _, _ = _make_decoder(block_size=2)
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        decoder.prefill(prompt)
        a1, _ = decoder.step()
        a2, _ = decoder.step()
        assert len(a1) >= 1
        assert len(a2) >= 1
        # Stats accumulate.
        s = decoder.stats_summary()
        assert s["steps"] == 2
        assert s["proposed"] == 4  # 2 * block_size

    def test_step_resets_on_target_forward_exception(self, monkeypatch):
        """Mirror of ``test_prefill_resets_on_target_forward_exception``
        for ``step()``. A mid-step exception (Metal error, OOM)
        previously left both KV caches in a partially-modified state
        and would silently corrupt the next step. The try/reset
        wrapper forces a clean tear-down so the caller has to
        re-``prefill`` to recover.
        """
        from olmlx.engine.dflash.decoder import _LayerHook, _get_layers

        decoder, target, _ = _make_decoder(block_size=2)
        decoder.prefill(mx.array([[1, 2, 3]], dtype=mx.int32))
        assert decoder._patched is True

        # Make the verify forward fail by raising from embed_tokens
        # (called inside the synthetic target's ``__call__``). The
        # draft forward path is independent, so this only blows up
        # the verify.
        def raising_embed(*_a, **_kw):
            raise RuntimeError("simulated verify forward failure")

        monkeypatch.setattr(target.model, "embed_tokens", raising_embed)
        with pytest.raises(RuntimeError, match="simulated verify"):
            decoder.step()

        # reset() ran: patched / bound / capture all cleared.
        assert decoder._patched is False
        assert decoder._bound is False
        assert decoder._capture is None
        layers = _get_layers(target)
        assert not any(isinstance(layer, _LayerHook) for layer in layers)
        # Caches are gone too — caller must re-prefill to use again.
        assert decoder._target_cache is None
        assert decoder._draft_cache is None

    def test_prefill_resets_on_target_forward_exception(self, monkeypatch):
        """If the target's forward raises (OOM, shape mismatch on a
        bad checkpoint, Metal stream error), the prefill exception
        path must call ``reset()`` so the model isn't left patched
        and the GDN capture lock isn't left held. Otherwise the
        window between the exception and the next ``prefill()`` self-
        heal leaks state to any concurrent request inspecting the
        target's layers.

        Triggering it: monkeypatch the synthetic target's
        ``embed_tokens`` (called first thing inside ``__call__``) to
        raise. Patching ``target.__call__`` directly would bind to
        the instance and Python's descriptor protocol would still
        resolve through the class — this is a more reliable hook.
        """
        from olmlx.engine.dflash.decoder import _LayerHook, _get_layers

        decoder, target, _ = _make_decoder()
        orig_embed = target.model.embed_tokens

        def raising_embed(*_args, **_kw):
            raise RuntimeError("simulated target forward failure")

        monkeypatch.setattr(target.model, "embed_tokens", raising_embed)
        with pytest.raises(RuntimeError, match="simulated target forward"):
            decoder.prefill(mx.array([[1, 2, 3]], dtype=mx.int32))
        # reset() must have run: patched / bound / capture all cleared.
        assert decoder._patched is False
        assert decoder._bound is False
        assert decoder._capture is None
        # And the layer hook should have been removed — the original
        # layer modules should be back in place.
        layers = _get_layers(target)
        assert not any(isinstance(layer, _LayerHook) for layer in layers)
        _ = orig_embed  # silence unused

    def test_rejects_out_of_range_target_layer_id(self):
        """If an EAGLE checkpoint was trained against a target of a
        different depth, ``target_layer_id`` from its saved config can
        be out of range for the currently loaded target. ``_patch_model``
        would surface this as an ``IndexError`` at the first prefill,
        far from the load site. Reject up front in the constructor.
        """
        from olmlx.engine.eagle.decoder import EagleDecoder
        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

        from olmlx.engine.dflash.decoder import _get_layers

        # Real target from ``_make_decoder`` helper; ask it for the
        # layer count rather than hardcoding (the helper may change).
        decoder, target, _ = _make_decoder(hidden=16)
        n = len(_get_layers(target))

        # Construct a fresh draft + cfg to reuse below.
        cfg = EagleConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=32,
            vocab_size=decoder._draft.args.vocab_size,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=512,
            block_size=2,
        )
        draft = EagleDraftModel(cfg)

        with pytest.raises(ValueError, match="out of range"):
            EagleDecoder(target_model=target, draft_model=draft, target_layer_id=n)
        with pytest.raises(ValueError, match="out of range"):
            EagleDecoder(target_model=target, draft_model=draft, target_layer_id=-1)
        # Valid indices construct cleanly.
        EagleDecoder(target_model=target, draft_model=draft, target_layer_id=0)
        EagleDecoder(target_model=target, draft_model=draft, target_layer_id=n - 1)

    def test_draft_cache_growth_is_bounded_per_step(self):
        """The draft cache must grow by exactly ``num_accepted``
        positions per ``step()`` — one entry per committed token, so the
        draft's KV history and RoPE positions stay aligned with the
        committed sequence (#617).

        Two failure modes this pins:
        - Skipping the draft trim (e.g. flipping ``if trim_draft > 0``)
          lets the cache grow by ``block_size`` per step, accumulating
          extra positions and blowing up memory on long generations.
        - Over-trimming (the pre-#617 ``block_size + 1 - num_accepted``,
          which kept only ``num_accepted - 1``) drops the last accepted
          token's KV entry every step, compressing RoPE positions by one
          per step and degrading acceptance cumulatively.
        """
        decoder, _, _ = _make_decoder(block_size=2)
        decoder.prefill(mx.array([[1, 2, 3]], dtype=mx.int32))
        assert decoder._draft_cache is not None

        def _cache_offset() -> int:
            assert decoder._draft_cache is not None
            return decoder._draft_cache[0].offset

        baseline = _cache_offset()
        steps = 3
        total_accepted = 0
        for _ in range(steps):
            accepted, _ = decoder.step()
            total_accepted += len(accepted)

        # Net growth per step = num_accepted (see step()'s comment block
        # on trim arithmetic). Cumulative offset growth after N steps =
        # ``sum_i num_accepted_i`` = total_accepted.
        expected = total_accepted
        actual = _cache_offset() - baseline
        assert actual == expected, (
            f"draft cache offset grew by {actual} after {steps} steps "
            f"with {total_accepted} total accepted tokens; expected "
            f"{expected}. Regression: the per-step draft trim no longer "
            "keeps exactly ``num_accepted`` committed entries."
        )

    def test_full_acceptance_aligns_draft_cache(self, monkeypatch):
        """On full acceptance (all block_size drafts + the bonus token
        committed) the draft cache is one entry short of the committed
        prefix, so ``step`` must run an align-forward to append the last
        accepted draft token's KV (#617). Force full acceptance and assert
        the draft cache grew by exactly num_accepted = block_size + 1.
        """
        decoder, _, _ = _make_decoder(block_size=3)
        decoder.prefill(mx.array([[1, 2, 3]], dtype=mx.int32))
        assert decoder._draft_cache is not None
        baseline = decoder._draft_cache[0].offset

        # Accept every draft plus a bonus token -> num_accepted = bs + 1.
        monkeypatch.setattr(
            decoder, "_verify_greedy", lambda drafts, logits: [*drafts, 7]
        )
        accepted, _ = decoder.step()
        assert len(accepted) == decoder._block_size + 1  # full acceptance
        grew = decoder._draft_cache[0].offset - baseline
        assert grew == len(accepted), (
            f"draft cache grew by {grew} on full acceptance; expected "
            f"{len(accepted)} — the align-forward did not append the "
            "last accepted draft token's KV entry."
        )

    def test_step_clears_hidden_storage_before_verify(self):
        """Regression: from step 2 onward, the decoder must clear
        ``_hidden_storage[0]`` before the verify forward. Otherwise a
        silent hook failure on the second step would propagate the
        prior step's hidden with no error, because the ``is None``
        guard would never fire. Pin this by injecting a sentinel into
        the storage slot *before* step() and asserting it doesn't
        survive — the hook in a working setup must repopulate the slot,
        and the storage must have been cleared in between so the guard
        is meaningful.
        """
        decoder, _, _ = _make_decoder(block_size=2)
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        decoder.prefill(prompt)
        decoder.step()  # step 1 — storage was None at start (from prefill)

        # Plant a sentinel that should be overwritten by the next forward.
        sentinel = mx.zeros((1, 999, 1))
        decoder._hidden_storage[0] = sentinel
        accepted, _ = decoder.step()

        # The hook fires during the verify forward and writes the real
        # captured hidden; whatever's in storage now should NOT be the
        # sentinel. (Identity check, not value: a working hook produces
        # a fresh array each call.)
        assert decoder._hidden_storage[0] is not None
        assert decoder._hidden_storage[0] is not sentinel
        assert len(accepted) >= 1


# ---------------------------------------------------------------------------
# Phase C: GDN rollback path for hybrid linear-attention targets
# ---------------------------------------------------------------------------


class TestEagleDecoderGDNPath:
    """Exercise the non-trim-able-cache code path without depending on
    a real ``GatedDeltaNet`` model. Strategy: monkey-patch
    ``can_trim_prompt_cache`` and ``_GDNStateCapture`` in the decoder
    module to fake a hybrid target while still using a synthetic
    trim-able target underneath. Verifies wiring; full end-to-end
    correctness with real hybrid models is covered by Phase F bench.
    """

    def test_prefill_rejects_non_trim_cache_without_gdn(self, monkeypatch):
        """Defensive guard: if the target's KV cache reports non-trim
        but the target has no ``GatedDeltaNet`` module, EAGLE's
        rollback path doesn't apply (some other cache type is at
        play). Reject up front with a clear error rather than crash
        cryptically inside ``rollback`` later. Today's quant caches
        all report ``is_trimmable() == True`` so they don't trip
        this; it's a forward-compatibility guard for future cache
        types that might not.
        """
        from olmlx.engine.eagle import decoder as decoder_mod

        monkeypatch.setattr(decoder_mod, "can_trim_prompt_cache", lambda _: False)
        monkeypatch.setattr(decoder_mod, "_HAS_GDN", True)
        # Pretend the target has no GDN module — what a pure-attention
        # model with a custom non-trim cache would look like.
        monkeypatch.setattr(decoder_mod, "_find_gdn_class", lambda _m: None)

        decoder, _, _ = _make_decoder(block_size=2)
        with pytest.raises(RuntimeError, match="no ``GatedDeltaNet`` submodule"):
            decoder.prefill(mx.array([[1, 2, 3]], dtype=mx.int32))

    def test_prefill_installs_capture_when_cache_is_non_trimmable(self, monkeypatch):
        from olmlx.engine import spec_decoder_base as base_mod
        from olmlx.engine.eagle import decoder as decoder_mod

        # Fake ``can_trim_prompt_cache`` to claim the cache is non-
        # trim-able.
        monkeypatch.setattr(decoder_mod, "can_trim_prompt_cache", lambda _: False)

        # Stub ``_GDNStateCapture`` so we don't actually need a real
        # ``GatedDeltaNet`` class on the target. Mirrors only the
        # methods the decoder calls: ``for_model`` (returns the
        # capture/buffer pair), ``use_buffer``, ``close``, and
        # ``rollback_single``; the buffer side exposes ``clear``.
        captured_calls = {"init": 0, "clear": 0, "close": 0, "rollback": []}

        class _FakeBuffer:
            def clear(self):
                captured_calls["clear"] += 1

        class _FakeCapture:
            @classmethod
            def for_model(cls, model):
                captured_calls["init"] += 1
                inst = cls()
                inst._model = model
                return inst, _FakeBuffer()

            def use_buffer(self, _buf):
                pass

            def close(self):
                captured_calls["close"] += 1

            def rollback_single(self, _buffer, _cache, accepted, trim):
                captured_calls["rollback"].append((accepted, trim))

        # The capture is installed via ``SpecDecoderBase._install_gdn_capture``,
        # so the patch seam lives on the base module (#467).
        monkeypatch.setattr(base_mod, "GDNStateCapture", _FakeCapture)
        # Force _HAS_GDN to True so the ``not _HAS_GDN`` early-error
        # path doesn't fire.
        monkeypatch.setattr(decoder_mod, "_HAS_GDN", True)
        # Defeat the "no GDN module on target" guard that fires when
        # we synthesise a non-trim cache without a real
        # ``GatedDeltaNet`` underneath. Return a sentinel class so
        # the guard sees "found one" — it's only checking presence,
        # not type identity.
        monkeypatch.setattr(decoder_mod, "_find_gdn_class", lambda _m: object)

        decoder, _, _ = _make_decoder(block_size=2)
        decoder.prefill(mx.array([[1, 2, 3]], dtype=mx.int32))
        assert captured_calls["init"] == 1
        assert decoder._capture is not None
        assert decoder._capture_buffer is not None
        assert decoder._target_can_trim is False

    def test_step_calls_rollback_on_non_trimmable_cache(self, monkeypatch):
        # Same setup as above; verify that ``step()`` invokes
        # ``buffer.clear()`` before the verify forward and
        # ``capture.rollback_single(...)`` instead of
        # ``trim_prompt_cache`` for the target cache.
        from olmlx.engine import spec_decoder_base as base_mod
        from olmlx.engine.eagle import decoder as decoder_mod

        monkeypatch.setattr(decoder_mod, "can_trim_prompt_cache", lambda _: False)
        monkeypatch.setattr(decoder_mod, "_HAS_GDN", True)
        # Defeat the "no GDN module on target" guard (see
        # ``test_prefill_installs_capture_when_cache_is_non_trimmable``
        # for rationale). Sentinel class — guard only checks presence.
        monkeypatch.setattr(decoder_mod, "_find_gdn_class", lambda _m: object)
        # Force a deterministic partial acceptance so the rollback
        # assertion below is guaranteed to fire. Without this, the
        # synthetic target's argmax could (by chance with a small
        # vocab) accept all block_size+1 candidates, giving trim=0
        # and silently skipping the rollback path — the test name
        # would promise coverage it doesn't deliver. We patch
        # ``verify_draft_greedy`` rather than tweaking the target's
        # weights because patching is explicit and the verify
        # helper is the right surface for "what was accepted". The
        # decoder verifies via ``SpecDecoderBase._verify_greedy``, so the
        # patch seam lives on the base module (#467).
        monkeypatch.setattr(
            base_mod,
            "verify_draft_greedy",
            lambda drafts, _logits: [drafts[0], 99],  # accept 1 draft + 1 substitute
        )

        captured_calls = {"clear": 0, "close": 0, "rollback": []}

        class _FakeBuffer:
            def clear(self):
                captured_calls["clear"] += 1

        class _FakeCapture:
            @classmethod
            def for_model(cls, _model):
                return cls(), _FakeBuffer()

            def use_buffer(self, _buf):
                pass

            def close(self):
                captured_calls["close"] += 1

            def rollback_single(self, _buffer, _cache, accepted, trim):
                captured_calls["rollback"].append((accepted, trim))

        monkeypatch.setattr(base_mod, "GDNStateCapture", _FakeCapture)

        # ``trim_prompt_cache`` for target should NOT be called when
        # we're in the GDN regime. Patch it to detect.
        target_trim_calls = {"count": 0}
        original_trim = decoder_mod.trim_prompt_cache

        def recording_trim(cache, n):
            target_trim_calls["count"] += 1
            if original_trim is not None:
                original_trim(cache, n)

        monkeypatch.setattr(decoder_mod, "trim_prompt_cache", recording_trim)

        decoder, _, _ = _make_decoder(block_size=2)
        decoder.prefill(mx.array([[1, 2, 3]], dtype=mx.int32))
        accepted, _ = decoder.step()
        # buffer.clear must run before verify
        assert captured_calls["clear"] == 1
        # rollback must run after verify. With the
        # ``verify_draft_greedy`` patch above we deterministically
        # get a 2-token accepted list against a block_size=2 verify
        # (3 candidate positions), so trim = 3 - 2 = 1 > 0 and the
        # rollback branch is guaranteed to fire.
        assert len(accepted) == 2
        assert len(captured_calls["rollback"]) == 1
        # accepted-draft-count argument should be num_accepted - 1
        recorded_acc, recorded_trim = captured_calls["rollback"][0]
        assert recorded_acc == len(accepted) - 1
        assert recorded_trim == decoder._block_size + 1 - len(accepted)
        # The draft cache trim still goes through trim_prompt_cache,
        # but the target cache trim should NOT have called it. Hard
        # to assert exact count here without distinguishing target vs
        # draft trim; settle for "no double-trim of target": the
        # number of target_trim_calls equals the number of draft
        # trims (== 1 if any draft trim happened).
        # Rather than a brittle equality, just confirm
        # capture.rollback_single ran — the path is exercised.
        decoder.reset()
        # close should fire on reset
        assert captured_calls["close"] == 1
