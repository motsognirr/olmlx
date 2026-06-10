"""Tests for the shared speculative-decoder base class (#467).

``SpecDecoderBase`` owns the mechanical parts every decoder had hand-rolled
(and drifted on): the ``step()`` try/reset exception boundary (#460), the
canonical ``prefill(prompt, *, segmented=None, cancel_event=None)``
signature, the hook/GDN-capture teardown ordering in ``reset()``, the
shared stats counters, and the ``spec.prefill``/``spec.step``/``spec.verify``
tracing seams. The verify/draft algorithms stay in the subclasses.
"""

from __future__ import annotations

import threading

import mlx.core as mx
import pytest

from olmlx.engine.spec_decoder_base import SpecDecoderBase

from tests.test_flash_speculative import MockModel


class _Boom(Exception):
    pass


class _DummyDecoder(SpecDecoderBase):
    """Minimal concrete decoder exercising only the base scaffolding."""

    def __init__(self):
        super().__init__()
        self.reset_state_calls = 0
        self.prefill_seen: dict | None = None
        self.fail_prefill = False
        self.fail_step = False

    def _prefill_impl(self, prompt, *, segmented, cancel_event):
        self.prefill_seen = {"segmented": segmented, "cancel_event": cancel_event}
        if self.fail_prefill:
            raise _Boom("prefill failed")
        return 7

    def _step_impl(self):
        if self.fail_step:
            raise _Boom("step failed")
        self._stats_steps += 1
        self._stats_proposed += 4
        self._stats_accepted_draft += 2
        return [1, 2, 3], 4

    def _reset_state(self):
        self.reset_state_calls += 1


class TestStepExceptionSafety:
    """step() must reset() on any _step_impl exception (the #460 guarantee)."""

    def test_step_resets_on_exception(self):
        dec = _DummyDecoder()
        dec.fail_step = True
        before = dec.reset_state_calls
        with pytest.raises(_Boom):
            dec.step()
        assert dec.reset_state_calls == before + 1

    def test_step_success_does_not_reset(self):
        dec = _DummyDecoder()
        before = dec.reset_state_calls
        accepted, proposed = dec.step()
        assert accepted == [1, 2, 3]
        assert proposed == 4
        assert dec.reset_state_calls == before


class TestPrefillCanonicalSignature:
    def test_prefill_defaults(self):
        dec = _DummyDecoder()
        assert dec.prefill(mx.array([[1, 2, 3]])) == 7
        assert dec.prefill_seen == {"segmented": None, "cancel_event": None}

    def test_prefill_forwards_kwargs(self):
        dec = _DummyDecoder()
        ev = threading.Event()
        seg = object()
        dec.prefill(mx.array([[1, 2, 3]]), segmented=seg, cancel_event=ev)
        assert dec.prefill_seen == {"segmented": seg, "cancel_event": ev}

    def test_prefill_resets_before_impl_and_again_on_failure(self):
        dec = _DummyDecoder()
        dec.fail_prefill = True
        with pytest.raises(_Boom):
            dec.prefill(mx.array([[1, 2, 3]]))
        # One reset at the top of prefill, one from the exception path.
        assert dec.reset_state_calls == 2

    def test_prefill_resets_once_on_success(self):
        dec = _DummyDecoder()
        dec.prefill(mx.array([[1, 2, 3]]))
        assert dec.reset_state_calls == 1


class TestResetTeardownOrdering:
    """reset() tears down: GDN capture close → unpatch hooks → unbind draft,
    then clears stats and calls _reset_state()."""

    class _FakeCapture:
        def __init__(self, log):
            self.log = log

        def close(self):
            self.log.append("capture")

    class _FakeDraft:
        def __init__(self, log):
            self.log = log

        def unbind(self):
            self.log.append("unbind")

    def test_teardown_order_and_flags(self):
        from olmlx.engine.spec_decoder_base import _LayerHook

        log: list[str] = []
        dec = _DummyDecoder()
        target = MockModel(8, 4)
        dec._target = target
        dec._draft = self._FakeDraft(log)
        dec._capture = self._FakeCapture(log)
        dec._capture_buffer = object()
        dec._install_layer_hooks([0], [None])
        assert isinstance(target.layers[0], _LayerHook)
        dec._bound = True

        dec.reset()

        assert log == ["capture", "unbind"]
        assert not isinstance(target.layers[0], _LayerHook)
        assert dec._capture is None
        assert dec._capture_buffer is None
        assert dec._patched is False
        assert dec._bound is False
        assert dec.reset_state_calls == 1

    def test_reset_swallows_teardown_errors_and_completes(self, caplog):
        """A raising teardown step must not abort the rest of reset() —
        reset() never raises (close()/__del__ and the step()/prefill()
        exception paths rely on it)."""
        from olmlx.engine.spec_decoder_base import _LayerHook

        class _RaisingCapture:
            def close(self):
                raise RuntimeError("capture close failed")

        log: list[str] = []
        dec = _DummyDecoder()
        target = MockModel(8, 4)
        dec._target = target
        dec._draft = TestResetTeardownOrdering._FakeDraft(log)
        dec._capture = _RaisingCapture()
        dec._capture_buffer = object()
        dec._install_layer_hooks([0], [None])
        dec._bound = True

        dec.reset()  # must not raise

        assert dec._capture is None
        assert not isinstance(target.layers[0], _LayerHook)
        assert log == ["unbind"]
        assert dec._patched is False
        assert dec._bound is False
        assert dec.reset_state_calls == 1

    def test_step_original_error_survives_teardown_failure(self):
        """The exception out of step() must be _step_impl's error, not a
        secondary teardown error from the reset."""

        class _RaisingCapture:
            def close(self):
                raise RuntimeError("teardown error")

        dec = _DummyDecoder()
        dec.fail_step = True
        dec._capture = _RaisingCapture()
        with pytest.raises(_Boom):
            dec.step()

    def test_reset_clears_stats(self):
        dec = _DummyDecoder()
        dec.step()
        assert dec._stats_steps == 1
        dec.reset()
        assert dec._stats_steps == 0
        assert dec._stats_proposed == 0
        assert dec._stats_accepted_draft == 0

    def test_reset_idempotent(self):
        dec = _DummyDecoder()
        dec.reset()
        dec.reset()


class TestInstallHelpers:
    def test_install_layer_hooks_sets_patched(self):
        dec = _DummyDecoder()
        dec._target = MockModel(8, 4)
        storage = [None]
        dec._install_layer_hooks([1], storage)
        assert dec._patched is True

    def test_install_layer_hooks_flags_before_patching(self, monkeypatch):
        """If _patch_model raises mid-loop with some layers already
        wrapped, _patched must already be True so reset() unpatches the
        partial wrap instead of leaking hooks onto the shared target."""
        import olmlx.engine.spec_decoder_base as base_mod

        dec = _DummyDecoder()
        target = MockModel(8, 4)
        dec._target = target
        real_patch = base_mod._patch_model

        def patch_then_raise(model, layer_ids, storage):
            real_patch(model, layer_ids[:1], storage)  # wrap one layer
            raise IndexError("corrupt layer id")

        monkeypatch.setattr(base_mod, "_patch_model", patch_then_raise)
        with pytest.raises(IndexError):
            dec._install_layer_hooks([0, 99], [None, None])
        assert dec._patched is True
        dec.reset()
        assert not isinstance(target.layers[0], base_mod._LayerHook)

    def test_bind_draft_sets_bound(self):
        class _Draft:
            def __init__(self):
                self.bound_to = None

            def bind(self, target):
                self.bound_to = target

            def unbind(self):
                self.bound_to = None

        dec = _DummyDecoder()
        dec._target = object()
        dec._draft = _Draft()
        dec._bind_draft()
        assert dec._bound is True
        assert dec._draft.bound_to is dec._target


class TestStats:
    def test_stats_summary_common_keys(self):
        dec = _DummyDecoder()
        dec.step()
        s = dec.stats_summary()
        assert s["steps"] == 1
        assert s["proposed"] == 4
        assert s["accepted_draft"] == 2
        assert s["acceptance_rate"] == 0.5
        assert s["avg_tokens_per_step"] == 3.0

    def test_stats_summary_zero_safe(self):
        s = _DummyDecoder().stats_summary()
        assert s["acceptance_rate"] == 0.0
        assert s["avg_tokens_per_step"] == 0.0

    def test_stats_extra_merged(self):
        class _Extra(_DummyDecoder):
            def _stats_extra(self):
                return {"block_size": 3}

        s = _Extra().stats_summary()
        assert s["block_size"] == 3


class TestLifecycle:
    def test_close_defaults_to_reset(self):
        dec = _DummyDecoder()
        dec.close()
        assert dec.reset_state_calls == 1

    def test_del_never_raises(self):
        class _Bad(_DummyDecoder):
            def _reset_state(self):
                raise RuntimeError("teardown failure")

        dec = _Bad()
        dec.__del__()  # must swallow


class TestCloseReleasesState:
    """close() must release the last request's KV state eagerly at model
    unload for every decoder — classic/PLD additionally drop their
    decoder-lifetime capture + snapshot store."""

    def test_classic_close_releases_per_request_state(self):
        from olmlx.engine.speculative import SpeculativeDecoder

        dec = SpeculativeDecoder(
            draft_model=MockModel(32, 16),
            target_model=MockModel(32, 16),
            num_speculative_tokens=2,
        )
        dec.prefill(mx.array([[1, 2, 3]]))
        assert dec._target_cache is not None
        dec.close()
        assert dec._target_cache is None
        assert dec._draft_cache is None

    def test_pld_close_releases_per_request_state(self):
        from olmlx.engine.speculative import PromptLookupDecoder

        dec = PromptLookupDecoder(
            target_model=MockModel(32, 16), num_speculative_tokens=2
        )
        dec.prefill(mx.array([[1, 2, 3]]))
        assert dec._target_cache is not None
        dec.close()
        assert dec._target_cache is None
        assert dec._pending_token is None


class TestTracingSeams:
    def test_prefill_step_verify_spans(self, memory_exporter):
        dec = _DummyDecoder()
        dec.prefill(mx.array([[1, 2, 3]]))
        dec.step()
        dec._verify_greedy([1], mx.zeros((2, 8)))
        names = [s.name for s in memory_exporter.get_finished_spans()]
        assert "spec.prefill" in names
        assert "spec.step" in names
        assert "spec.verify" in names

    def test_step_span_records_proposed_accepted(self, memory_exporter):
        dec = _DummyDecoder()
        dec.step()
        step = next(
            s for s in memory_exporter.get_finished_spans() if s.name == "spec.step"
        )
        attrs = dict(step.attributes)
        assert attrs["proposed"] == 4
        assert attrs["accepted"] == 3


class TestAllDecodersSubclassBase:
    def test_every_decoder_inherits_base(self):
        from olmlx.engine.dflash.decoder import DFlashDecoder
        from olmlx.engine.eagle.decoder import EagleDecoder
        from olmlx.engine.flash.speculative import SpeculativeFlashDecoder
        from olmlx.engine.mtp.decoder import MTPDecoder
        from olmlx.engine.self_speculative.decoder import SelfSpeculativeDecoder
        from olmlx.engine.speculative import (
            PromptLookupDecoder,
            SpeculativeDecoder,
        )

        for cls in (
            SpeculativeDecoder,
            SpeculativeFlashDecoder,
            PromptLookupDecoder,
            DFlashDecoder,
            EagleDecoder,
            MTPDecoder,
            SelfSpeculativeDecoder,
        ):
            assert issubclass(cls, SpecDecoderBase), cls.__name__


class TestConcreteDecoderExceptionSafety:
    """The try/reset boundary applies to the decoders that previously
    lacked it: classic, PLD, and self-speculative (#460 drift)."""

    def test_classic_step_resets_on_exception(self, monkeypatch):
        from olmlx.engine.speculative import SpeculativeDecoder

        dec = SpeculativeDecoder(
            draft_model=MockModel(32, 16),
            target_model=MockModel(32, 16),
            num_speculative_tokens=2,
        )
        dec.prefill(mx.array([[1, 2, 3]]))
        monkeypatch.setattr(
            dec, "_step_linear", lambda *_: (_ for _ in ()).throw(_Boom())
        )
        with pytest.raises(_Boom):
            dec.step()
        assert dec._target_cache is None
        assert dec._draft_cache is None

    def test_pld_step_resets_on_exception(self, monkeypatch):
        from olmlx.engine.speculative import PromptLookupDecoder

        dec = PromptLookupDecoder(
            target_model=MockModel(32, 16), num_speculative_tokens=2
        )
        dec.prefill(mx.array([[1, 2, 3, 1, 2]]))
        monkeypatch.setattr(
            dec, "_lookup_draft", lambda: (_ for _ in ()).throw(_Boom())
        )
        with pytest.raises(_Boom):
            dec.step()
        assert dec._target_cache is None
        assert dec._pending_token is None

    def test_self_speculative_step_resets_on_exception(self, monkeypatch):
        from olmlx.engine.self_speculative.decoder import SelfSpeculativeDecoder

        from tests.test_self_speculative_coverage import _SelfSpecModel

        dec = SelfSpeculativeDecoder(
            target_model=_SelfSpecModel(vocab_size=16, hidden_size=8, num_layers=2),
            num_early_layers=1,
            num_speculative_tokens=2,
        )
        dec.prefill(mx.array([[1, 2, 3]]))
        monkeypatch.setattr(
            "olmlx.engine.spec_decoder_base.verify_draft_greedy",
            lambda *_: (_ for _ in ()).throw(_Boom()),
        )
        with pytest.raises(_Boom):
            dec.step()
        assert dec._cache is None
        assert dec._pending_token is None


class TestBridgeCanonicalSignature:
    """speculative_stream_generate always passes both kwargs — no more
    per-decoder signature special-casing."""

    def test_bridge_passes_segmented_and_cancel_event(self):
        from olmlx.engine.speculative_stream import speculative_stream_generate

        class _Stub:
            def __init__(self):
                self.seen = None

            # Required keyword-only params: omitting either kwarg at the
            # call site is a TypeError, so this stub proves the bridge
            # passes both unconditionally.
            def prefill(self, prompt, *, segmented, cancel_event):
                self.seen = {"segmented": segmented, "cancel_event": cancel_event}
                return 5

            def step(self):
                return [6], 1

            def reset(self):
                pass

        stub = _Stub()
        ev = threading.Event()
        tokens = list(
            speculative_stream_generate(stub, [1, 2, 3], max_tokens=1, cancel_event=ev)
        )
        assert tokens[0].token == 5
        assert stub.seen == {"segmented": None, "cancel_event": ev}

    def test_protocol_declares_segmented(self):
        import inspect

        from olmlx.engine.speculative_stream import SpeculativeDecoderProtocol

        params = inspect.signature(SpeculativeDecoderProtocol.prefill).parameters
        assert "segmented" in params
        assert "cancel_event" in params
