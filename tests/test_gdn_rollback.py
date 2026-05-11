"""Tests for ``rollback_autoregressive``, the new GDN rollback path
used by classic speculative on a hybrid draft model.

``rollback_single`` is already covered end-to-end by DFlash's test
suite; what's new here is the multi-capture-per-layer math:
concatenating per-step (q, k, v, a, b, mask) tensors along the
sequence dim and replaying ``gated_delta_update`` once with the FIRST
capture's pre-state as the init state.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from olmlx.engine.gdn_rollback import GDNBuffer, GDNStateCapture


@pytest.fixture
def fake_capture():
    """A ``GDNStateCapture`` with the patch-install path stubbed so
    tests don't need a real ``GatedDeltaNet`` class."""
    cap = GDNStateCapture.__new__(GDNStateCapture)
    cap._gdn_cls = object  # placeholder; never invoked
    cap._active_buffer = None
    cap._orig_call = None
    cap._patched_call = None
    cap._closed = True  # close() is a no-op
    return cap


def _make_capture_tuple(
    q_val: float, k_val: float, v_val: float, state_val: float, mask=None
):
    """Build one capture tuple matching ``_capturing_gdn_call``'s layout:
    (q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel).

    Per-token tensors (q, k, v, a, b) have shape (B=1, S=1, H=1, D=2);
    constants (A_log, dt_bias, state) are scalars."""
    q = mx.full((1, 1, 1, 2), q_val)
    k = mx.full((1, 1, 1, 2), k_val)
    v = mx.full((1, 1, 1, 2), v_val)
    a = mx.full((1, 1, 1, 2), q_val + 0.1)
    b = mx.full((1, 1, 1, 2), q_val + 0.2)
    A_log = mx.array([0.5])
    dt_bias = mx.array([0.1])
    state = mx.full((1,), state_val)
    return (q, k, v, a, b, A_log, dt_bias, state, mask, True)


def _make_conv_data(value: float, K: int = 3, S_total: int = 1):
    """Build a conv_data entry (conv_input, K). conv_input is built so
    that its last K-1 elements are deterministic markers for the
    post-step conv state."""
    # conv_input shape: (B, prev_state_len + S, conv_dim). For
    # tests we just need a recognizable tail.
    conv_input = mx.full((1, K - 1 + S_total, 4), value)
    return (conv_input, K)


class _StubLayer:
    """A stand-in for an mlx-lm cache layer that's NOT trimmable
    (mimics ``ArraysCache``). Exposes the cache[0]/cache[1] write
    interface rollback uses."""

    def __init__(self, name: str):
        self.name = name
        self._slots: list = [None, None]

    def is_trimmable(self) -> bool:
        return False

    def __getitem__(self, idx: int):
        return self._slots[idx]

    def __setitem__(self, idx: int, value) -> None:
        self._slots[idx] = value


class _TrimmableLayer:
    """A trimmable cache layer (mimics ``KVCache``). Records calls."""

    def __init__(self, name: str):
        self.name = name
        self.trim_calls: list[int] = []

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        self.trim_calls.append(n)
        return n


class TestRollbackAutoregressive:
    def test_rejects_num_keep_steps_out_of_range(self, fake_capture):
        buf = GDNBuffer(expected_modules=[])
        with pytest.raises(ValueError, match="num_keep_steps"):
            fake_capture.rollback_autoregressive(
                buf, cache=[], num_steps=4, num_keep_steps=0, trim=0
            )
        with pytest.raises(ValueError, match="num_keep_steps"):
            fake_capture.rollback_autoregressive(
                buf, cache=[], num_steps=4, num_keep_steps=5, trim=0
            )

    def test_full_keep_is_noop(self, fake_capture):
        """num_keep_steps == num_steps: no rollback needed, no
        gated_delta_update call."""
        buf = GDNBuffer(expected_modules=[])
        cache = [_TrimmableLayer("L0")]
        with patch("olmlx.engine.gdn_rollback._gd_mod") as mock_gd:
            fake_capture.rollback_autoregressive(
                buf, cache=cache, num_steps=4, num_keep_steps=4, trim=0
            )
        mock_gd.gated_delta_update.assert_not_called()
        # Trimmable layer left alone (trim=0 doesn't matter; we return early).
        assert cache[0].trim_calls == []

    def test_concatenates_first_n_captures_per_layer(self, fake_capture):
        """rollback_autoregressive with one GDN layer + 3 captures,
        keep first 2: should concatenate captures 0 and 1, pass the
        FIRST capture's state as init_state, and write back."""
        layer_a = MagicMock(name="layer_a")
        buf = GDNBuffer(expected_modules=[layer_a])
        # 3 captures, one per autoregressive step. Distinguishable
        # values per step so we can verify which ones got concatenated.
        buf.gdn_inputs = [
            _make_capture_tuple(q_val=1.0, k_val=10.0, v_val=100.0, state_val=999.0),
            _make_capture_tuple(q_val=2.0, k_val=20.0, v_val=200.0, state_val=888.0),
            _make_capture_tuple(q_val=3.0, k_val=30.0, v_val=300.0, state_val=777.0),
        ]
        buf.conv_data = [
            _make_conv_data(value=1.0),
            _make_conv_data(value=2.0),
            _make_conv_data(value=3.0),
        ]
        # Captured modules must match expected for the ordering check.
        buf.captured_modules = [layer_a, layer_a, layer_a]

        gdn_cache = _StubLayer("gdn0")
        cache = [gdn_cache]

        # Patch gated_delta_update so we can inspect its call and return
        # a known state to write into the cache.
        replayed_state = mx.array([42.0])
        with patch("olmlx.engine.gdn_rollback._gd_mod") as mock_gd:
            mock_gd.gated_delta_update.return_value = (None, replayed_state)
            fake_capture.rollback_autoregressive(
                buf,
                cache=cache,
                num_steps=3,
                num_keep_steps=2,
                trim=0,
            )

        # Exactly one replay call (one GDN layer, one batched call).
        assert mock_gd.gated_delta_update.call_count == 1
        call_args = mock_gd.gated_delta_update.call_args
        # Positional args: q, k, v, a, b, A_log, dt_bias, state, mask
        q_arg, k_arg, v_arg, a_arg, b_arg, _, _, init_state_arg, mask_arg = (
            call_args.args
        )
        # Each per-token tensor should have S=2 (the first two captures
        # concatenated along axis=1).
        assert q_arg.shape == (1, 2, 1, 2), f"q shape {q_arg.shape}"
        # Values: step 0 had q=1.0 across the slot, step 1 had q=2.0.
        # After concatenation, q_arg[:, 0] should be 1.0 and q_arg[:, 1] 2.0.
        assert q_arg[0, 0, 0, 0].item() == pytest.approx(1.0)
        assert q_arg[0, 1, 0, 0].item() == pytest.approx(2.0)
        # k: step 0 was 10.0, step 1 was 20.0.
        assert k_arg[0, 0, 0, 0].item() == pytest.approx(10.0)
        assert k_arg[0, 1, 0, 0].item() == pytest.approx(20.0)
        # v: step 0 was 100.0, step 1 was 200.0.
        assert v_arg[0, 0, 0, 0].item() == pytest.approx(100.0)
        assert v_arg[0, 1, 0, 0].item() == pytest.approx(200.0)
        # a, b similar.
        assert a_arg[0, 0, 0, 0].item() == pytest.approx(1.1)
        assert b_arg[0, 1, 0, 0].item() == pytest.approx(2.2)
        # init_state comes from the FIRST capture, not later ones.
        assert init_state_arg[0].item() == pytest.approx(999.0)
        # mask was None in all captures → None passed through.
        assert mask_arg is None
        # The cache's GDN state slot (index 1) was updated with the
        # replayed state.
        assert gdn_cache._slots[1] is replayed_state
        # The conv state slot (index 0) was written with the last K-1
        # elements of the step ``num_keep_steps-1`` = step 1's
        # conv_input (value=2.0).
        conv_state_written = gdn_cache._slots[0]
        # Last K-1 = 2 positions, all filled with value=2.0.
        assert conv_state_written.shape == (1, 2, 4)
        assert conv_state_written[0, 0, 0].item() == pytest.approx(2.0)

    def test_trims_trimmable_layers_in_mixed_cache(self, fake_capture):
        """Hybrid cache: GDN layers do replay, trimmable layers get
        c.trim(trim)."""
        layer_a = MagicMock(name="layer_a")
        buf = GDNBuffer(expected_modules=[layer_a])
        buf.gdn_inputs = [
            _make_capture_tuple(1.0, 10.0, 100.0, 999.0),
            _make_capture_tuple(2.0, 20.0, 200.0, 888.0),
        ]
        buf.conv_data = [_make_conv_data(1.0), _make_conv_data(2.0)]
        buf.captured_modules = [layer_a, layer_a]

        kv_layer_pre = _TrimmableLayer("kv-pre")
        gdn_layer = _StubLayer("gdn")
        kv_layer_post = _TrimmableLayer("kv-post")
        cache = [kv_layer_pre, gdn_layer, kv_layer_post]

        with patch("olmlx.engine.gdn_rollback._gd_mod") as mock_gd:
            mock_gd.gated_delta_update.return_value = (None, mx.array([0.0]))
            fake_capture.rollback_autoregressive(
                buf, cache=cache, num_steps=2, num_keep_steps=1, trim=1
            )

        assert kv_layer_pre.trim_calls == [1]
        assert kv_layer_post.trim_calls == [1]
        # GDN layer got a state write, not a trim.
        assert gdn_layer._slots[1] is not None

    def test_buffer_size_mismatch_raises(self, fake_capture):
        """If captured count != expected_modules * num_steps,
        rollback raises with a clear diagnostic."""
        layer_a = MagicMock(name="layer_a")
        buf = GDNBuffer(expected_modules=[layer_a])
        # Buffer has 5 captures but caller claims num_steps=2 → expected 2.
        buf.gdn_inputs = [_make_capture_tuple(1.0, 1.0, 1.0, 1.0)] * 5
        buf.conv_data = [_make_conv_data(1.0)] * 5
        buf.captured_modules = [layer_a] * 5

        with pytest.raises(RuntimeError, match="buffer size mismatch"):
            fake_capture.rollback_autoregressive(
                buf, cache=[_StubLayer("g")], num_steps=2, num_keep_steps=1, trim=0
            )

    def test_per_step_ordering_mismatch_raises(self, fake_capture):
        """If steps visit GDN layers in inconsistent order, fail loudly."""
        layer_a = MagicMock(name="layer_a")
        layer_b = MagicMock(name="layer_b")
        buf = GDNBuffer(expected_modules=[layer_a, layer_b])
        buf.gdn_inputs = [_make_capture_tuple(1.0, 1.0, 1.0, 1.0)] * 4
        buf.conv_data = [_make_conv_data(1.0)] * 4
        # Step 0: a, b (correct). Step 1: b, a (swapped).
        buf.captured_modules = [layer_a, layer_b, layer_b, layer_a]

        with pytest.raises(RuntimeError, match="different order than step 0"):
            fake_capture.rollback_autoregressive(
                buf,
                cache=[_StubLayer("g1"), _StubLayer("g2")],
                num_steps=2,
                num_keep_steps=1,
                trim=0,
            )
