"""Tests for classic SpeculativeDecoder on hybrid linear-attention models.

Covers the GDN-rollback wiring added in ``engine/speculative.py``:

- Detection of hybrid target/draft at construction.
- Buffer routing during ``step()`` (draft buffer for the autoregressive
  loop, target buffer for the verify forward).
- Rollback dispatch (single vs autoregressive) based on which side is
  hybrid.
- Lifecycle: ``close()`` releases the class-level patch lock; ``__del__``
  is best-effort.

The actual GDN math (``gated_delta_update`` replay) is exercised by the
DFlash test suite via ``rollback_single``; the unit tests here use a
fake GDN class with the required structural attrs so the patch installs
without exercising real mlx-lm GDN forward passes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.gdn_rollback import _GDN_PATCH_LOCK, get_active_capture
from olmlx.engine.speculative import SpeculativeDecoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_gdn_cls(name: str = "GatedDeltaNet") -> type:
    """Build a class named *name* that carries every attribute the
    capturing forward reads.

    A unique subclass per call so tests that exercise
    ``find_gdn_class`` against multiple models with different classes
    can do so without the second class shadowing the first.
    """

    class _FakeGDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkv = nn.Linear(4, 12, bias=False)
            self.in_proj_z = nn.Linear(4, 4, bias=False)
            self.in_proj_b = nn.Linear(4, 4, bias=False)
            self.in_proj_a = nn.Linear(4, 4, bias=False)
            self.out_proj = nn.Linear(4, 4, bias=False)
            self.conv1d = nn.Conv1d(12, 12, kernel_size=3)
            self.conv_kernel_size = 3
            self.conv_dim = 12
            self.A_log = mx.zeros((4,))
            self.dt_bias = mx.zeros((4,))
            self.norm = nn.RMSNorm(4)
            self.num_k_heads = 1
            self.num_v_heads = 1
            self.head_k_dim = 4
            self.head_v_dim = 4
            self.key_dim = 4

    _FakeGDN.__name__ = name
    return _FakeGDN


def _make_hybrid_model(gdn_cls: type | None) -> nn.Module:
    """Build a tiny model with one layer; the layer holds a GDN
    submodule if *gdn_cls* is not None, otherwise just a linear stub.

    Exposes ``model.layers`` (the path ``get_model_layers`` finds first
    after ``.model.layers``). We use the bare ``.layers`` path here so
    each call to this helper produces a fully-distinct module tree.
    """

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            if gdn_cls is not None:
                self.linear_attn = gdn_cls()
            else:
                self.attn = nn.Linear(4, 4, bias=False)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [_Layer()]

        def __call__(self, x, cache=None):
            return mx.zeros((x.shape[0], x.shape[1], 4))

    return _Model()


@pytest.fixture(autouse=True)
def _ensure_gdn_lock_released():
    """Defensively close any leaked ``GDNStateCapture`` between tests.

    A test that fails with the patch installed would leave the
    class-level ``GatedDeltaNet.__call__`` pointing at a stale closure
    AND the lock held. Just releasing the lock would let the next test
    acquire it, but its ``__init__`` would save the stale patch as
    ``_orig_call`` — and on close, restore the stale patch instead of
    the original ``__call__``.

    ``get_active_capture()`` returns the singleton active capture (the
    patch lock guarantees at most one), so we can close it cleanly
    rather than scanning ``gc.get_objects()``. Falls back to a bare
    lock release as a last resort.
    """
    yield
    active = get_active_capture()
    if active is not None:
        try:
            active.close()
        except Exception:
            pass
    try:
        _GDN_PATCH_LOCK.release()
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Construction-time GDN detection
# ---------------------------------------------------------------------------


class TestGDNDetection:
    def test_no_capture_for_non_hybrid_models(self):
        """Plain models with no GatedDeltaNet — no capture installed."""
        target = _make_hybrid_model(gdn_cls=None)
        draft = _make_hybrid_model(gdn_cls=None)
        dec = SpeculativeDecoder(draft, target)
        try:
            assert dec._gdn_capture is None
            assert dec._target_gdn_buffer is None
            assert dec._draft_gdn_buffer is None
        finally:
            dec.close()

    def test_capture_when_only_target_hybrid(self):
        """Hybrid target + dense draft: capture installed, only target buffer."""
        gdn = _make_fake_gdn_cls()
        target = _make_hybrid_model(gdn_cls=gdn)
        draft = _make_hybrid_model(gdn_cls=None)
        dec = SpeculativeDecoder(draft, target)
        try:
            assert dec._gdn_capture is not None
            assert dec._gdn_capture.gdn_cls is gdn
            assert dec._target_gdn_buffer is not None
            assert dec._draft_gdn_buffer is None
            # The target buffer's expected modules are the GDN
            # instances inside *target*'s layer 0.
            assert len(dec._target_gdn_buffer.expected_modules) == 1
            assert dec._target_gdn_buffer.expected_modules[0] is (
                target.layers[0].linear_attn
            )
        finally:
            dec.close()

    def test_capture_when_only_draft_hybrid(self):
        """Dense target + hybrid draft (unusual but supported)."""
        gdn = _make_fake_gdn_cls()
        target = _make_hybrid_model(gdn_cls=None)
        draft = _make_hybrid_model(gdn_cls=gdn)
        dec = SpeculativeDecoder(draft, target)
        try:
            assert dec._gdn_capture is not None
            assert dec._target_gdn_buffer is None
            assert dec._draft_gdn_buffer is not None
            assert dec._draft_gdn_buffer.expected_modules[0] is (
                draft.layers[0].linear_attn
            )
        finally:
            dec.close()

    def test_capture_when_both_hybrid_same_class(self):
        """The typical Qwen3.5 target + Qwen3.5 draft case."""
        gdn = _make_fake_gdn_cls()
        target = _make_hybrid_model(gdn_cls=gdn)
        draft = _make_hybrid_model(gdn_cls=gdn)
        dec = SpeculativeDecoder(draft, target)
        try:
            assert dec._gdn_capture is not None
            assert dec._gdn_capture.gdn_cls is gdn
            assert dec._target_gdn_buffer is not None
            assert dec._draft_gdn_buffer is not None
            # Each buffer collects modules from its OWN model — they
            # must be distinct instance lists.
            assert (
                dec._target_gdn_buffer.expected_modules[0]
                is not dec._draft_gdn_buffer.expected_modules[0]
            )
            assert dec._target_gdn_buffer.expected_modules[0] is (
                target.layers[0].linear_attn
            )
            assert dec._draft_gdn_buffer.expected_modules[0] is (
                draft.layers[0].linear_attn
            )
        finally:
            dec.close()

    def test_different_gdn_classes_raise_not_implemented(self):
        """Two distinct GDN classes can't share one class-level patch."""
        # Two classes with the same __name__ "GatedDeltaNet" but
        # different identities — exactly the situation a future
        # qwen3_next vs qwen3_5 split would create.
        gdn_a = _make_fake_gdn_cls("GatedDeltaNet")
        gdn_b = _make_fake_gdn_cls("GatedDeltaNet")
        assert gdn_a is not gdn_b
        target = _make_hybrid_model(gdn_cls=gdn_a)
        draft = _make_hybrid_model(gdn_cls=gdn_b)
        with pytest.raises(
            NotImplementedError, match="different GatedDeltaNet classes"
        ):
            SpeculativeDecoder(draft, target)


# ---------------------------------------------------------------------------
# Lifecycle: close() and __del__
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_releases_lock(self):
        gdn = _make_fake_gdn_cls()
        target = _make_hybrid_model(gdn_cls=gdn)
        draft = _make_hybrid_model(gdn_cls=gdn)
        dec = SpeculativeDecoder(draft, target)
        # Lock should be held by the capture.
        assert _GDN_PATCH_LOCK.acquire(blocking=False) is False
        dec.close()
        # Now the lock should be free.
        assert _GDN_PATCH_LOCK.acquire(blocking=False) is True
        _GDN_PATCH_LOCK.release()  # restore for autouse fixture's no-op release

    def test_close_idempotent(self):
        gdn = _make_fake_gdn_cls()
        target = _make_hybrid_model(gdn_cls=gdn)
        draft = _make_hybrid_model(gdn_cls=gdn)
        dec = SpeculativeDecoder(draft, target)
        dec.close()
        dec.close()  # must not raise
        assert dec._gdn_capture is None

    def test_close_on_non_hybrid_is_noop(self):
        target = _make_hybrid_model(gdn_cls=None)
        draft = _make_hybrid_model(gdn_cls=None)
        dec = SpeculativeDecoder(draft, target)
        dec.close()
        assert dec._gdn_capture is None

    def test_del_swallows_exceptions(self):
        """__del__ runs during GC and must never raise."""
        gdn = _make_fake_gdn_cls()
        target = _make_hybrid_model(gdn_cls=gdn)
        draft = _make_hybrid_model(gdn_cls=gdn)
        dec = SpeculativeDecoder(draft, target)
        # Force close to raise on next call by corrupting internal
        # state — __del__ should still not raise.
        dec._gdn_capture._gdn_cls = None  # type: ignore[union-attr]
        try:
            dec.__del__()
        except Exception as e:
            pytest.fail(f"__del__ raised: {e}")
        # Manually release if needed.
        try:
            _GDN_PATCH_LOCK.release()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# step() rollback dispatch
# ---------------------------------------------------------------------------


class TestStepRollbackDispatch:
    """``step()`` must call the right rollback method based on which
    side is hybrid. We patch out ``rollback_single`` /
    ``rollback_autoregressive`` and validate the call arguments rather
    than running real GDN math."""

    def _make_decoder(
        self, target_hybrid: bool, draft_hybrid: bool, same_class: bool = True
    ):
        if target_hybrid and draft_hybrid and same_class:
            gdn = _make_fake_gdn_cls()
            target_cls = draft_cls = gdn
        else:
            target_cls = _make_fake_gdn_cls() if target_hybrid else None
            draft_cls = _make_fake_gdn_cls() if draft_hybrid else None
        target = _make_hybrid_model(target_cls)
        draft = _make_hybrid_model(draft_cls)
        return SpeculativeDecoder(draft, target, num_speculative_tokens=4)

    def test_target_hybrid_calls_rollback_single(self):
        """Hybrid target + dense draft: target rollback uses
        rollback_single with accepted=num_accepted-1; draft falls
        through to plain trim_prompt_cache."""
        dec = self._make_decoder(target_hybrid=True, draft_hybrid=False)
        try:
            # Simulate post-prefill state.
            dec._pending_token = 1
            dec._target_cache = [MagicMock(is_trimmable=MagicMock(return_value=True))]
            dec._draft_cache = [MagicMock(is_trimmable=MagicMock(return_value=True))]
            # Fake draft generation: returns 4 token ids, no ctx.
            dec._draft_generate_cached = MagicMock(return_value=([10, 20, 30, 40], []))
            # Fake target forward: returns logits with shape (1, 5, vocab).
            target_logits = mx.array(
                [[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
            )
            dec._target = MagicMock(return_value=target_logits)
            # Verify returns 2 tokens (1 accepted draft + correction).
            dec._verify = MagicMock(return_value=[10, 1])
            # Patch rollback methods.
            dec._gdn_capture.rollback_single = MagicMock()
            dec._gdn_capture.rollback_autoregressive = MagicMock()
            with patch("olmlx.engine.speculative.trim_prompt_cache") as trim_fn:
                dec.step()
            # Target rollback: single-forward, accepted=num_accepted-1=1, trim=λ+1-na=3
            dec._gdn_capture.rollback_single.assert_called_once_with(
                dec._target_gdn_buffer,
                dec._target_cache,
                accepted=1,
                trim=3,
            )
            # Draft rollback: no GDN — falls through to trim_prompt_cache
            dec._gdn_capture.rollback_autoregressive.assert_not_called()
            # trim_prompt_cache called on draft only (target uses rollback_single)
            trim_fn.assert_called_once_with(dec._draft_cache, 2)
        finally:
            dec.close()

    def test_draft_hybrid_calls_rollback_autoregressive(self):
        """Dense target + hybrid draft: draft uses
        rollback_autoregressive with num_steps=λ, num_keep_steps=
        num_accepted; target falls through to plain trim_prompt_cache."""
        dec = self._make_decoder(target_hybrid=False, draft_hybrid=True)
        try:
            dec._pending_token = 1
            dec._target_cache = [MagicMock(is_trimmable=MagicMock(return_value=True))]
            dec._draft_cache = [MagicMock(is_trimmable=MagicMock(return_value=True))]
            dec._draft_generate_cached = MagicMock(return_value=([10, 20, 30, 40], []))
            target_logits = mx.array(
                [[[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
            )
            dec._target = MagicMock(return_value=target_logits)
            dec._verify = MagicMock(return_value=[1])  # num_accepted=1
            dec._gdn_capture.rollback_single = MagicMock()
            dec._gdn_capture.rollback_autoregressive = MagicMock()
            with patch("olmlx.engine.speculative.trim_prompt_cache") as trim_fn:
                dec.step()
            # Draft rollback: autoregressive, num_keep_steps=1, trim=λ-na=3
            dec._gdn_capture.rollback_autoregressive.assert_called_once_with(
                dec._draft_gdn_buffer,
                dec._draft_cache,
                num_steps=4,
                num_keep_steps=1,
                trim=3,
            )
            dec._gdn_capture.rollback_single.assert_not_called()
            # Target trim via plain trim_prompt_cache (trim_target=4).
            trim_fn.assert_called_once_with(dec._target_cache, 4)
        finally:
            dec.close()

    def test_both_hybrid_full_acceptance_no_rollback(self):
        """Qwen3.5+Qwen3.5, full acceptance (num_accepted == λ+1):
        neither rollback fires because trim_target and trim_draft are 0;
        the align step advances the draft cache by feeding D_λ."""
        dec = self._make_decoder(target_hybrid=True, draft_hybrid=True)
        try:
            dec._pending_token = 1
            dec._target_cache = [MagicMock(is_trimmable=MagicMock(return_value=True))]
            dec._draft_cache = [MagicMock(is_trimmable=MagicMock(return_value=True))]
            dec._draft_generate_cached = MagicMock(return_value=([10, 20, 30, 40], []))
            # All drafts match: full acceptance + bonus → num_accepted=5.
            # verify_draft_greedy needs the argmax at positions 0..3 to
            # equal D_1..D_4 in turn.
            target_logits = mx.zeros((1, 5, 50))
            target_logits[0, 0, 10] = 1.0  # position 0: argmax 10 == D_1
            target_logits[0, 1, 20] = 1.0  # position 1: argmax 20 == D_2
            target_logits[0, 2, 30] = 1.0  # position 2: argmax 30 == D_3
            target_logits[0, 3, 40] = 1.0  # position 3: argmax 40 == D_4
            target_logits[0, 4, 7] = 1.0  # position 4 (bonus): argmax 7
            dec._target = MagicMock(return_value=target_logits)
            dec._gdn_capture.rollback_single = MagicMock()
            dec._gdn_capture.rollback_autoregressive = MagicMock()
            # Avoid the align step's real forward by stubbing it.
            dec._draft = MagicMock(return_value=mx.zeros((1, 1, 50)))
            with patch("olmlx.engine.speculative.trim_prompt_cache") as trim_fn:
                dec.step()
            # Full acceptance: trim_target=0, trim_draft<0 (clamped to 0).
            # Neither rollback nor trim_prompt_cache should fire.
            dec._gdn_capture.rollback_single.assert_not_called()
            dec._gdn_capture.rollback_autoregressive.assert_not_called()
            trim_fn.assert_not_called()
            # Align step ran (full acceptance) — draft was called with
            # the last draft token to bring its cache to position λ+1.
            dec._draft.assert_called_once()
        finally:
            dec.close()

    def test_both_hybrid_partial_acceptance(self):
        """The primary new scenario this PR enables — Qwen3.5+Qwen3.5
        with partial acceptance. BOTH rollback paths must fire with the
        right args, no plain ``trim_prompt_cache`` call escapes to the
        underlying hybrid caches."""
        dec = self._make_decoder(target_hybrid=True, draft_hybrid=True)
        try:
            dec._pending_token = 1
            dec._target_cache = [MagicMock(is_trimmable=MagicMock(return_value=True))]
            dec._draft_cache = [MagicMock(is_trimmable=MagicMock(return_value=True))]
            dec._draft_generate_cached = MagicMock(return_value=([10, 20, 30, 40], []))
            # First two drafts match, third doesn't → num_accepted=3
            # (= D_1 + D_2 + correction). verify_draft_greedy needs
            # argmax at positions 0,1 to match D_1, D_2 and at position 2
            # to be the correction token (not D_3).
            target_logits = mx.zeros((1, 5, 50))
            target_logits[0, 0, 10] = 1.0  # match D_1=10
            target_logits[0, 1, 20] = 1.0  # match D_2=20
            target_logits[0, 2, 99] = 1.0  # mismatch D_3=30 → correction=99
            target_logits[0, 3, 0] = 1.0  # unused after mismatch
            target_logits[0, 4, 0] = 1.0  # unused
            dec._target = MagicMock(return_value=target_logits)
            dec._gdn_capture.rollback_single = MagicMock()
            dec._gdn_capture.rollback_autoregressive = MagicMock()
            with patch("olmlx.engine.speculative.trim_prompt_cache") as trim_fn:
                dec.step()
            # Partial acceptance: num_accepted=3, λ=4 → trim_target=2,
            # trim_draft=1.
            dec._gdn_capture.rollback_single.assert_called_once_with(
                dec._target_gdn_buffer,
                dec._target_cache,
                accepted=2,  # num_accepted - 1
                trim=2,  # λ+1 - num_accepted
            )
            dec._gdn_capture.rollback_autoregressive.assert_called_once_with(
                dec._draft_gdn_buffer,
                dec._draft_cache,
                num_steps=4,  # λ
                num_keep_steps=3,  # num_accepted
                trim=1,  # λ - num_accepted
            )
            # Crucially: plain ``trim_prompt_cache`` must NOT be called.
            # If it were, the hybrid caches' ArraysCache layers would
            # silently desync — exactly the bug this PR fixes.
            trim_fn.assert_not_called()
        finally:
            dec.close()
