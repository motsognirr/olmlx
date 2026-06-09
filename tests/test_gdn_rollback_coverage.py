"""Regression coverage for ``olmlx.engine.gdn_rollback``.

The sibling ``test_gdn_rollback.py`` exercises the rollback *math*
(``rollback_single`` / ``rollback_autoregressive``). This file targets
the surrounding machinery that was previously uncovered:

- model/module discovery (``get_model_layers``, ``find_gdn_class``,
  ``collect_gdn_modules``),
- the mlx-lm signature probe (``validate_gated_delta_update_signature``),
- the identity-based ordering helper (``_order_matches``),
- the class-level monkey-patch lifecycle (``__init__`` / ``_patch`` /
  ``close`` / ``__del__``, ``for_model``, ``create_buffer``,
  ``use_buffer``, ``get_active_capture``),
- ``GDNBuffer`` bookkeeping, and a few rollback branch outcomes not
  reached by the sibling test (mask concat, mixed-mask error,
  single-step ordering error, per-step ordering error).

Everything uses a hand-built fake ``GatedDeltaNet`` ``nn.Module``
subclass plus fake cache layers — no real model load, no GPU, no
network. Each test finishes in well under a second.

Lock hygiene: ``GDNStateCapture`` acquires a *module-level* lock in
``__init__`` and releases it in ``close()``. A test that constructs a
capture and forgets to close it (or fails before closing) would leave
the lock held and deadlock every subsequent capture-constructing test.
The autouse ``_drain_gdn_lock`` fixture force-releases the lock after
each test so one failure can never cascade into a hang.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

import olmlx.engine.gdn_rollback as gr
from olmlx.engine.gdn_rollback import (
    GDNBuffer,
    GDNStateCapture,
    _GDN_REQUIRED_ATTRS,
    _order_matches,
    collect_gdn_modules,
    find_gdn_class,
    get_active_capture,
    get_model_layers,
    validate_gated_delta_update_signature,
)


@pytest.fixture(autouse=True)
def _drain_gdn_lock():
    """Guarantee the module patch lock is free after every test.

    If a test leaks a held lock (construct-without-close, or a failure
    mid-construction), force-release it so the next test that builds a
    ``GDNStateCapture`` does not block forever. ``Lock.release`` on an
    unlocked lock raises ``RuntimeError`` — swallow it.
    """
    yield
    gr._active_capture = None
    try:
        gr._GDN_PATCH_LOCK.release()
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Fake model building blocks
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    """Stand-in for mlx-lm's ``GatedDeltaNet``.

    The class *name* is what ``find_gdn_class`` matches on; every
    attribute in ``_GDN_REQUIRED_ATTRS`` is attached so the structural
    check passes. ``__call__`` is never invoked here (we never run a
    forward), so placeholder scalar values are fine.
    """

    def __init__(self) -> None:
        super().__init__()
        for attr in _GDN_REQUIRED_ATTRS:
            setattr(self, attr, 1)


class _ImpostorGDN(nn.Module):
    """Named ``GatedDeltaNet`` but lacking the required attributes."""


_ImpostorGDN.__name__ = "GatedDeltaNet"


class _Inner(nn.Module):
    def __init__(self, layers: list) -> None:
        super().__init__()
        self.layers = layers


class _ModelDotModel(nn.Module):
    """Exposes ``.model.layers`` (first lookup path)."""

    def __init__(self, layers: list) -> None:
        super().__init__()
        self.model = _Inner(layers)


class _ModelLanguageModel(nn.Module):
    """Exposes ``.language_model.layers`` (second lookup path)."""

    def __init__(self, layers: list) -> None:
        super().__init__()
        self.language_model = _Inner(layers)


class _ModelFlatLayers(nn.Module):
    """Exposes ``.layers`` directly (third lookup path)."""

    def __init__(self, layers: list) -> None:
        super().__init__()
        self.layers = layers


class _ModelNoLayers(nn.Module):
    """No recognizable layers attribute."""


def _gdn_layer() -> nn.Module:
    """A transformer-style layer wrapping one GDN submodule."""

    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = GatedDeltaNet()

    return _Layer()


def _plain_layer() -> nn.Module:
    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Linear(2, 2)

    return _Layer()


# ---------------------------------------------------------------------------
# Cache layer stubs (mirror the sibling test's interface)
# ---------------------------------------------------------------------------


class _StubArrays:
    """Non-trimmable (ArraysCache-like) cache slot pair."""

    def __init__(self) -> None:
        self._slots: list = [None, None]

    def is_trimmable(self) -> bool:
        return False

    def __getitem__(self, idx: int):
        return self._slots[idx]

    def __setitem__(self, idx: int, value) -> None:
        self._slots[idx] = value


class _StubTrimmable:
    def __init__(self) -> None:
        self.trim_calls: list[int] = []

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        self.trim_calls.append(n)
        return n


def _capture_tuple(q=1.0, k=10.0, v=100.0, state=0.0, mask=None):
    """One capture matching ``_capturing_gdn_call``'s 10-tuple layout:
    (q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)."""
    return (
        mx.full((1, 1, 1, 2), q),
        mx.full((1, 1, 1, 2), k),
        mx.full((1, 1, 1, 2), v),
        mx.full((1, 1, 1, 2), q + 0.1),
        mx.full((1, 1, 1, 2), q + 0.2),
        mx.array([0.5]),
        mx.array([0.1]),
        mx.full((1,), state),
        mask,
        True,
    )


def _conv_data(value=1.0, K=3, S_total=1):
    return (mx.full((1, K - 1 + S_total, 4), value), K)


def _detached_capture() -> GDNStateCapture:
    """A capture whose patch-install path is bypassed (no lock held).

    Mirrors the sibling test's fixture: ``_closed`` starts ``True`` so
    ``close()`` is a no-op and the module lock is never touched. Use for
    pure rollback-math tests that never need a live patch.
    """
    cap = GDNStateCapture.__new__(GDNStateCapture)
    cap._gdn_cls = object
    cap._active_buffer = None
    cap._orig_call = None
    cap._patched_call = None
    cap._closed = True
    return cap


# ---------------------------------------------------------------------------
# get_model_layers
# ---------------------------------------------------------------------------


class TestGetModelLayers:
    def test_model_dot_model_layers(self):
        layers = [_plain_layer()]
        assert get_model_layers(_ModelDotModel(layers)) is layers

    def test_language_model_layers(self):
        layers = [_plain_layer()]
        assert get_model_layers(_ModelLanguageModel(layers)) is layers

    def test_flat_layers(self):
        layers = [_plain_layer()]
        assert get_model_layers(_ModelFlatLayers(layers)) is layers

    def test_no_layers_raises(self):
        with pytest.raises(AttributeError, match="Cannot find layers"):
            get_model_layers(_ModelNoLayers())


# ---------------------------------------------------------------------------
# find_gdn_class
# ---------------------------------------------------------------------------


class TestFindGDNClass:
    def test_finds_gdn_class_in_model(self):
        model = _ModelDotModel([_plain_layer(), _gdn_layer()])
        assert find_gdn_class(model) is GatedDeltaNet

    def test_returns_none_when_no_gdn(self):
        assert find_gdn_class(_ModelDotModel([_plain_layer()])) is None

    def test_skips_same_named_class_missing_attrs(self):
        """A class literally named ``GatedDeltaNet`` but lacking the
        required attributes must be skipped, not matched."""

        class _Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = _ImpostorGDN()

        assert find_gdn_class(_ModelDotModel([_Layer()])) is None


# ---------------------------------------------------------------------------
# collect_gdn_modules
# ---------------------------------------------------------------------------


class TestCollectGDNModules:
    def test_collects_in_layer_order(self):
        l0 = _plain_layer()
        l1 = _gdn_layer()
        l2 = _gdn_layer()
        model = _ModelDotModel([l0, l1, l2])
        mods = collect_gdn_modules(model, GatedDeltaNet)
        assert len(mods) == 2
        assert mods[0] is l1.self_attn
        assert mods[1] is l2.self_attn

    def test_multiple_gdn_per_layer_raises(self):
        class _DoubleLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = GatedDeltaNet()
                self.b = GatedDeltaNet()

        model = _ModelDotModel([_DoubleLayer()])
        with pytest.raises(RuntimeError, match="GatedDeltaNet submodules"):
            collect_gdn_modules(model, GatedDeltaNet)

    def test_empty_when_no_gdn_and_none_orphaned(self):
        model = _ModelDotModel([_plain_layer()])
        assert collect_gdn_modules(model, GatedDeltaNet) == []

    def test_orphaned_gdn_outside_layers_raises(self):
        """GDN reachable via ``named_modules`` but not via the layers
        list (attached at top level) must raise — rollback can't
        address it."""

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Inner([_plain_layer()])
                self.stray = GatedDeltaNet()

        with pytest.raises(RuntimeError, match="none are reachable"):
            collect_gdn_modules(_Model(), GatedDeltaNet)


# ---------------------------------------------------------------------------
# _order_matches
# ---------------------------------------------------------------------------


class TestOrderMatches:
    def test_identity_match(self):
        a, b, c = object(), object(), object()
        assert _order_matches([a, b, c], [a, b, c]) is True

    def test_length_mismatch(self):
        a, b = object(), object()
        assert _order_matches([a, b], [a]) is False

    def test_reordered_is_false(self):
        a, b = object(), object()
        assert _order_matches([a, b], [b, a]) is False

    def test_equal_but_distinct_arrays_is_false(self):
        """Identity, not equality: two arrays that compare equal but are
        distinct instances are NOT a match — and crucially this does not
        raise on ``bool()`` of a multi-element array (the bug
        ``_order_matches`` exists to avoid)."""
        x = mx.array([1.0, 2.0])
        y = mx.array([1.0, 2.0])
        assert _order_matches([x], [y]) is False


# ---------------------------------------------------------------------------
# validate_gated_delta_update_signature
# ---------------------------------------------------------------------------


class TestValidateSignature:
    def test_passes_with_real_mlx_lm(self):
        # Real mlx-lm is installed here; the probe must accept it.
        validate_gated_delta_update_signature()

    def test_raises_when_gd_mod_is_none(self):
        with patch.object(gr, "_gd_mod", None):
            with pytest.raises(RuntimeError, match="unavailable"):
                validate_gated_delta_update_signature()

    def test_raises_on_missing_param(self):
        fake_mod = MagicMock()

        def _bad_update(q, k, v):  # missing most params
            return None

        fake_mod.gated_delta_update = _bad_update
        with patch.object(gr, "_gd_mod", fake_mod):
            with pytest.raises(RuntimeError, match="missing parameters"):
                validate_gated_delta_update_signature()

    def test_raises_when_signature_uninspectable(self):
        fake_mod = MagicMock()
        fake_mod.gated_delta_update = object()  # any non-callable target
        # Force ``inspect.signature`` to fail the way an un-introspectable
        # callable would (some builtins raise ValueError/TypeError).
        with (
            patch.object(gr, "_gd_mod", fake_mod),
            patch("inspect.signature", side_effect=ValueError("boom")),
        ):
            with pytest.raises(RuntimeError, match="Cannot introspect"):
                validate_gated_delta_update_signature()

    def test_raises_on_reordered_params(self):
        """#468: a presence-only check accepts a reordering that breaks
        the positional call in ``_capturing_gdn_call``/rollback. The
        probe must compare order, not just the set of names.
        """
        fake_mod = MagicMock()

        # Same names as upstream, ``q`` and ``k`` swapped.
        def _reordered(k, q, v, a, b, A_log, dt_bias, state, mask, use_kernel):
            return None

        fake_mod.gated_delta_update = _reordered
        with patch.object(gr, "_gd_mod", fake_mod):
            with pytest.raises(RuntimeError, match="order"):
                validate_gated_delta_update_signature()


# ---------------------------------------------------------------------------
# validate_gdn_upstream_sources (#468 follow-up)
# ---------------------------------------------------------------------------


class TestValidateUpstreamSources:
    """Source-hash guard: an out-of-sync vendored ``GatedDeltaNet.__call__``
    copy (or a changed ``gated_delta_update`` kernel) must fail at patch
    time instead of silently degrading acceptance / corrupting state.
    """

    @staticmethod
    def _real_gdn_cls():
        from mlx_lm.models.qwen3_5 import GatedDeltaNet as RealGDN

        return RealGDN

    def test_passes_with_real_mlx_lm(self):
        # The verified-hash table must match the installed (pinned) mlx-lm.
        gr.validate_gdn_upstream_sources(self._real_gdn_cls())

    def test_raises_on_drifted_gdn_call_source(self):
        stale = dict(gr._VERIFIED_UPSTREAM_SHA256)
        stale["GatedDeltaNet.__call__"] = frozenset({"0" * 64})
        with patch.object(gr, "_VERIFIED_UPSTREAM_SHA256", stale):
            with pytest.raises(RuntimeError, match="GatedDeltaNet.__call__"):
                gr.validate_gdn_upstream_sources(self._real_gdn_cls())

    def test_raises_on_drifted_gated_delta_update_source(self):
        stale = dict(gr._VERIFIED_UPSTREAM_SHA256)
        stale["gated_delta_update"] = frozenset({"0" * 64})
        with patch.object(gr, "_VERIFIED_UPSTREAM_SHA256", stale):
            with pytest.raises(RuntimeError, match="gated_delta_update"):
                gr.validate_gdn_upstream_sources(self._real_gdn_cls())

    def test_skips_non_mlx_lm_classes(self):
        # The contract is with upstream mlx-lm; hand-built stand-ins
        # (this file's fake ``GatedDeltaNet``) are exempt from the class
        # hash even with a poisoned table.
        stale = dict(gr._VERIFIED_UPSTREAM_SHA256)
        stale["GatedDeltaNet.__call__"] = frozenset({"0" * 64})
        with patch.object(gr, "_VERIFIED_UPSTREAM_SHA256", stale):
            gr.validate_gdn_upstream_sources(GatedDeltaNet)

    def test_skips_uninspectable_gated_delta_update(self):
        # A MagicMock module (as other tests install) has no source —
        # skip with a warning rather than blocking inference.
        fake_mod = MagicMock()
        with patch.object(gr, "_gd_mod", fake_mod):
            gr.validate_gdn_upstream_sources(GatedDeltaNet)

    def test_capture_init_validates_sources(self):
        # The guard must run before the patch is installed, and a
        # failed construction must leave the lock free.
        stale = dict(gr._VERIFIED_UPSTREAM_SHA256)
        stale["GatedDeltaNet.__call__"] = frozenset({"0" * 64})
        real_cls = self._real_gdn_cls()
        orig_call = real_cls.__call__
        with patch.object(gr, "_VERIFIED_UPSTREAM_SHA256", stale):
            with pytest.raises(RuntimeError, match="GatedDeltaNet.__call__"):
                GDNStateCapture(real_cls)
        assert real_cls.__call__ is orig_call  # never patched
        # Lock free → a capture on the fake class still installs.
        cap = GDNStateCapture(GatedDeltaNet)
        cap.close()


# ---------------------------------------------------------------------------
# Patch lifecycle: __init__ / _patch / close / __del__ / for_model
# ---------------------------------------------------------------------------


class TestPatchLifecycle:
    def test_patch_install_swaps_call_and_registers(self):
        cap = GDNStateCapture(GatedDeltaNet)
        try:
            # The patched closure is now the class ``__call__`` and the
            # active-capture pointer references this instance.
            assert GatedDeltaNet.__call__ is cap._patched_call
            assert cap.gdn_cls is GatedDeltaNet
            assert get_active_capture() is cap
            assert cap._orig_call is not None
        finally:
            cap.close()

    def test_close_restores_original_and_clears_active(self):
        cap = GDNStateCapture(GatedDeltaNet)
        patched = cap._patched_call
        orig = cap._orig_call
        cap.close()
        # The patched closure is gone; the stored original is reinstated.
        assert GatedDeltaNet.__call__ is not patched
        assert GatedDeltaNet.__call__ is orig
        assert get_active_capture() is None

    def test_close_is_idempotent(self):
        cap = GDNStateCapture(GatedDeltaNet)
        cap.close()
        # A second close must not raise (else it would release an
        # already-released lock).
        cap.close()
        assert get_active_capture() is None

    def test_lock_freed_after_close_allows_reinstall(self):
        cap = GDNStateCapture(GatedDeltaNet)
        cap.close()
        # If the lock were leaked, this construction would deadlock.
        cap2 = GDNStateCapture(GatedDeltaNet)
        cap2.close()
        assert get_active_capture() is None

    def test_use_buffer_routes_active_buffer(self):
        cap = GDNStateCapture(GatedDeltaNet)
        try:
            buf = GDNBuffer(expected_modules=[])
            cap.use_buffer(buf)
            assert cap._active_buffer is buf
            cap.use_buffer(None)
            assert cap._active_buffer is None
        finally:
            cap.close()

    def test_del_invokes_close_and_frees_lock(self):
        """``__del__`` is the belt-and-braces finaliser: if the owner is
        dropped without an explicit close, it must still restore the
        class ``__call__`` and release the module lock.

        Garbage-collection timing is not deterministic across CPython
        builds (a traceback frame can pin the object), so invoke the
        finaliser directly rather than relying on ``gc.collect``.
        """
        cap = GDNStateCapture(GatedDeltaNet)
        patched = cap._patched_call
        assert get_active_capture() is cap
        assert GatedDeltaNet.__call__ is patched
        cap.__del__()  # idempotent close()
        assert get_active_capture() is None
        assert GatedDeltaNet.__call__ is not patched
        # Lock freed → another capture can install.
        cap2 = GDNStateCapture(GatedDeltaNet)
        cap2.close()

    def test_init_raises_when_no_gdn_support(self):
        with patch.object(gr, "_HAS_GDN", False):
            with pytest.raises(RuntimeError, match="unavailable"):
                GDNStateCapture(GatedDeltaNet)
        # The failed construction must not have acquired the lock.
        cap = GDNStateCapture(GatedDeltaNet)
        cap.close()

    def test_for_model_builds_capture_and_buffer(self):
        model = _ModelDotModel([_plain_layer(), _gdn_layer()])
        cap, buf = GDNStateCapture.for_model(model)
        try:
            assert isinstance(buf, GDNBuffer)
            assert buf.num_gdn_layers == 1
            assert cap.gdn_cls is GatedDeltaNet
        finally:
            cap.close()

    def test_for_model_raises_when_no_gdn(self):
        model = _ModelDotModel([_plain_layer()])
        with pytest.raises(RuntimeError, match="no .*GatedDeltaNet.* submodule"):
            GDNStateCapture.for_model(model)
        # for_model never installed a patch, so the lock is free.
        cap = GDNStateCapture(GatedDeltaNet)
        cap.close()

    def test_for_model_closes_capture_on_buffer_failure(self):
        """If buffer creation fails (orphaned GDN), ``for_model`` must
        close the capture so the lock is released."""

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # find_gdn_class sees the stray via named_modules, but the
                # layers list is empty so collect_gdn_modules raises.
                self.model = _Inner([])
                self.stray = GatedDeltaNet()

        with pytest.raises(RuntimeError, match="none are reachable"):
            GDNStateCapture.for_model(_Model())
        # Lock released → a fresh capture can install.
        cap = GDNStateCapture(GatedDeltaNet)
        cap.close()

    def test_create_buffer_independent_per_model(self):
        cap = GDNStateCapture(GatedDeltaNet)
        try:
            b1 = cap.create_buffer(_ModelDotModel([_gdn_layer()]))
            b2 = cap.create_buffer(_ModelDotModel([_gdn_layer(), _gdn_layer()]))
            assert b1.num_gdn_layers == 1
            assert b2.num_gdn_layers == 2
            assert b1.expected_modules != b2.expected_modules
        finally:
            cap.close()


# ---------------------------------------------------------------------------
# GDNBuffer
# ---------------------------------------------------------------------------


class TestGDNBuffer:
    def test_num_gdn_layers_and_clear(self):
        mods = [object(), object()]
        buf = GDNBuffer(expected_modules=mods)
        assert buf.num_gdn_layers == 2
        buf.gdn_inputs = [_capture_tuple()]
        buf.conv_data = [_conv_data()]
        buf.captured_modules = [object()]
        buf.clear()
        assert buf.gdn_inputs == []
        assert buf.conv_data == []
        assert buf.captured_modules == []
        # expected_modules survives clear().
        assert buf.num_gdn_layers == 2


# ---------------------------------------------------------------------------
# rollback_single branch coverage
# ---------------------------------------------------------------------------


class TestRollbackSingleBranches:
    def test_replays_accepted_prefix_and_trims(self):
        layer = MagicMock(name="layer")
        buf = GDNBuffer(expected_modules=[layer])
        # One GDN capture with S=5 tokens so a prefix slice is meaningful.
        q = mx.broadcast_to(mx.arange(5).reshape(1, 5, 1, 1), (1, 5, 1, 2))
        cap = list(_capture_tuple())
        cap[0] = q  # distinct per-position values
        buf.gdn_inputs = [tuple(cap)]
        buf.conv_data = [_conv_data(value=7.0, K=3, S_total=5)]
        buf.captured_modules = [layer]

        trimmable = _StubTrimmable()
        gdn = _StubArrays()
        cache = [trimmable, gdn]

        replayed = mx.array([55.0])
        with patch.object(gr, "_gd_mod") as mock_gd:
            mock_gd.gated_delta_update.return_value = (None, replayed)
            _detached_capture().rollback_single(buf, cache=cache, accepted=2, trim=3)

        # Trimmable layer trimmed with the provided amount.
        assert trimmable.trim_calls == [3]
        # GDN state slot written with replay output.
        assert gdn._slots[1] is replayed
        # gated_delta_update received the first accepted+1 = 3 q positions.
        q_arg = mock_gd.gated_delta_update.call_args.args[0]
        assert q_arg.shape == (1, 3, 1, 2)
        # And the conv state slot was written too (index 0).
        assert gdn._slots[0] is not None

    def test_single_ordering_mismatch_raises(self):
        """Single-mode alignment check: captured modules in a different
        order than expected_modules raises before any replay."""
        a = MagicMock(name="a")
        b = MagicMock(name="b")
        buf = GDNBuffer(expected_modules=[a, b])
        buf.gdn_inputs = [_capture_tuple(), _capture_tuple()]
        buf.conv_data = [_conv_data(), _conv_data()]
        buf.captured_modules = [b, a]  # swapped

        with pytest.raises(RuntimeError, match="ordering invariant violated"):
            _detached_capture().rollback_single(
                buf, cache=[_StubArrays(), _StubArrays()], accepted=0, trim=0
            )


# ---------------------------------------------------------------------------
# rollback_autoregressive branch coverage (new vs sibling)
# ---------------------------------------------------------------------------


class TestRollbackAutoregressiveBranches:
    def test_full_keep_with_nonzero_trim_raises(self):
        buf = GDNBuffer(expected_modules=[])
        with pytest.raises(ValueError, match="implies no rollback"):
            _detached_capture().rollback_autoregressive(
                buf, cache=[], num_steps=3, num_keep_steps=3, trim=2
            )

    def test_mask_present_concatenated(self):
        """When captures carry a mask, the no-mask fast path is skipped
        and masks are concatenated along the sequence dim."""
        layer = MagicMock(name="layer")
        buf = GDNBuffer(expected_modules=[layer])
        buf.gdn_inputs = [
            _capture_tuple(q=1.0, mask=mx.array([[True]])),
            _capture_tuple(q=2.0, mask=mx.array([[False]])),
            _capture_tuple(q=3.0, mask=mx.array([[True]])),
        ]
        buf.conv_data = [_conv_data(1.0), _conv_data(2.0), _conv_data(3.0)]
        buf.captured_modules = [layer, layer, layer]

        gdn = _StubArrays()
        with patch.object(gr, "_gd_mod") as mock_gd:
            mock_gd.gated_delta_update.return_value = (None, mx.array([0.0]))
            # Keep 2 of 3 steps so the rollback body (not the no-op) runs.
            _detached_capture().rollback_autoregressive(
                buf, cache=[gdn], num_steps=3, num_keep_steps=2, trim=0
            )
        mask_arg = mock_gd.gated_delta_update.call_args.args[8]
        assert mask_arg is not None
        assert mask_arg.shape == (1, 2)

    def test_mixed_mask_raises(self):
        """Some captures with mask, others None → ambiguous concat → raise."""
        layer = MagicMock(name="layer")
        buf = GDNBuffer(expected_modules=[layer])
        buf.gdn_inputs = [
            _capture_tuple(q=1.0, mask=mx.array([[True]])),
            _capture_tuple(q=2.0, mask=None),
            _capture_tuple(q=3.0, mask=mx.array([[True]])),
        ]
        buf.conv_data = [_conv_data(1.0), _conv_data(2.0), _conv_data(3.0)]
        buf.captured_modules = [layer, layer, layer]

        with patch.object(gr, "_gd_mod") as mock_gd:
            mock_gd.gated_delta_update.return_value = (None, mx.array([0.0]))
            with pytest.raises(RuntimeError, match="Mixed-mask"):
                _detached_capture().rollback_autoregressive(
                    buf, cache=[_StubArrays()], num_steps=3, num_keep_steps=2, trim=0
                )

    def test_first_step_ordering_mismatch_raises(self):
        """Step 0 visiting modules in the wrong order vs expected_modules
        must raise. ``num_steps`` must exceed ``num_keep_steps`` or the
        method short-circuits on the no-op branch before the alignment
        check runs."""
        a = MagicMock(name="a")
        b = MagicMock(name="b")
        buf = GDNBuffer(expected_modules=[a, b])
        # 2 layers x 2 steps = 4 captures (alignment count must match).
        buf.gdn_inputs = [_capture_tuple()] * 4
        buf.conv_data = [_conv_data()] * 4
        # Step 0 captured (b, a) instead of (a, b); step 1 mirrors it.
        buf.captured_modules = [b, a, b, a]

        with pytest.raises(RuntimeError, match="ordering invariant violated"):
            _detached_capture().rollback_autoregressive(
                buf,
                cache=[_StubArrays(), _StubArrays()],
                num_steps=2,
                num_keep_steps=1,
                trim=0,
            )

    def test_per_step_ordering_mismatch_raises(self):
        """Step 1 visiting GDN layers in a different order than step 0
        is caught with a per-step diagnostic."""
        a = MagicMock(name="a")
        b = MagicMock(name="b")
        buf = GDNBuffer(expected_modules=[a, b])
        buf.gdn_inputs = [_capture_tuple()] * 4
        buf.conv_data = [_conv_data()] * 4
        # Step 0: (a, b) correct. Step 1: (b, a) swapped.
        buf.captured_modules = [a, b, b, a]

        with pytest.raises(RuntimeError, match="different order than step 0"):
            _detached_capture().rollback_autoregressive(
                buf,
                cache=[_StubArrays(), _StubArrays()],
                num_steps=2,
                num_keep_steps=1,
                trim=0,
            )
