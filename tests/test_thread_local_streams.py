"""Semantics gates for mlx >= 0.31.2 thread-local streams (#499).

These tests pin the exact behaviors the sync-in-worker migration relies on:

1. ``mx.new_thread_local_stream`` exists and its proxy is usable from any
   thread (resolving to that thread's own underlying stream).
2. mlx-lm's and mlx-vlm's module-level ``generation_stream`` are such
   proxies (mlx-lm >= 0.31.3 / mlx-vlm >= 0.5.0).
3. A *plain* stream created on one thread still cannot be adopted by
   another thread. This constraint is why the checkpoint-prefill drive had
   to move into the generation worker thread — if this test ever FAILS
   (adoption becomes legal), that design constraint may have been lifted;
   revisit before deleting the test.
4. Materialized arrays may cross threads and feed ops built on the
   consuming thread.

Metal-gated: stream-thread affinity is enforced by the Metal backend; the
CPU backend used under OLMLX_TESTS_CPU_DEVICE=1 does not exercise it.
"""

import threading

import mlx.core as mx
import pytest


def _run_in_thread(fn):
    """Run fn() on a fresh thread; return {'value': ...} or {'error': exc}."""
    result: dict = {}

    def _target():
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001 — the exception IS the result
            result["error"] = exc

    t = threading.Thread(target=_target)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "worker thread hung"
    return result


@pytest.mark.usefixtures("metal_default_device")
class TestThreadLocalStreamSemantics:
    def test_new_thread_local_stream_api_exists(self):
        assert hasattr(mx, "new_thread_local_stream"), (
            "mx.new_thread_local_stream missing — mlx < 0.31.2?"
        )

    def test_proxy_usable_from_worker_thread(self):
        proxy = mx.new_thread_local_stream(mx.default_device())

        def work():
            with mx.stream(proxy):
                x = mx.ones((32, 32))
                y = x @ x
                mx.eval(y)
            mx.synchronize(proxy)
            return float(y[0, 0].item())

        res = _run_in_thread(work)
        assert res.get("error") is None, f"worker failed: {res.get('error')!r}"
        assert res["value"] == 32.0

    def test_mlx_lm_generation_stream_usable_from_worker(self):
        from mlx_lm.generate import generation_stream

        def work():
            with mx.stream(generation_stream):
                x = mx.ones((8, 8))
                mx.eval(x @ x)
            mx.synchronize(generation_stream)
            return True

        res = _run_in_thread(work)
        assert res.get("error") is None, (
            f"mlx-lm generation_stream not worker-usable: {res.get('error')!r} "
            "— mlx-lm < 0.31.3 (module-level plain stream)?"
        )

    def test_mlx_vlm_generation_stream_usable_from_worker(self):
        from mlx_vlm.generate import generation_stream

        def work():
            with mx.stream(generation_stream):
                x = mx.ones((8, 8))
                mx.eval(x @ x)
            mx.synchronize(generation_stream)
            return True

        res = _run_in_thread(work)
        assert res.get("error") is None, (
            f"mlx-vlm generation_stream not worker-usable: {res.get('error')!r} "
            "— mlx-vlm < 0.5.0 (module-level plain stream)?"
        )

    def test_foreign_plain_stream_rejected_in_worker(self):
        s = mx.new_stream(mx.default_device())  # plain, owned by this thread

        def work():
            with mx.stream(s):
                mx.eval(mx.ones((4, 4)) * 2)
            return True

        res = _run_in_thread(work)
        assert res.get("error") is not None, (
            "a plain stream was adopted by a foreign thread — mlx may have "
            "made streams thread-portable; the deferred-drive design "
            "constraint (#499) may be liftable. Investigate before deleting."
        )

    def test_materialized_arrays_cross_threads(self):
        x = mx.random.normal((64, 64))
        mx.eval(x)  # materialize on this thread

        def work():
            y = x @ x  # new op on the worker, consuming materialized input
            mx.eval(y)
            return float(y.sum().item())

        res = _run_in_thread(work)
        assert res.get("error") is None, f"worker failed: {res.get('error')!r}"


def _yarn_rope_model():
    """A minimal module tree carrying a scaled-RoPE ``_freqs`` buffer.

    Mirrors mlx-lm's ``Qwen3_5``/``Qwen3Next`` attention layout: each attention
    submodule owns a ``YarnRoPE`` whose ``self._freqs`` is a *lazy* array built
    in ``__init__`` (no ``with mx.stream``), bound to the constructing thread's
    default stream. Because the key is underscore-prefixed, it is NOT part of
    ``nn.Module.parameters()`` — so mlx-lm's load-time ``mx.eval(parameters())``
    never materializes it.
    """
    import mlx.nn as nn
    from mlx_lm.models.rope_utils import YarnRoPE

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.rope = YarnRoPE(
                dims=32,
                max_position_embeddings=1048576,
                scaling_factor=4.0,
                original_max_position_embeddings=262144,
            )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Attn(), Attn()]

    return Model()


def _rope_forward_on_worker(model):
    """Run ``model.layers[0].rope`` on a fresh thread; return _run_in_thread result."""

    def work():
        x = mx.random.normal((1, 4, 8, 32))
        mx.eval(model.layers[0].rope(x))
        return True

    return _run_in_thread(work)


@pytest.mark.usefixtures("metal_default_device")
class TestScaledRopeBufferMaterialization:
    """Regression gate for the Yarn/longrope ``_freqs`` cross-thread crash.

    A 1M-context model (e.g. Qwen3.5 hybrid ``empero-ai/Qwythos-9B``) uses a
    scaled RoPE whose ``_freqs`` buffer is not reached by
    ``mx.eval(model.parameters())``. Loaded on one worker thread and generated
    on another, the first forward eval raised "There is no Stream(gpu, 0) in
    current thread". ``_materialize_module_buffers`` closes that gap by eager-
    evaluating the non-parameter buffers on the load thread.
    """

    def test_lazy_rope_freqs_crashes_cross_thread_without_fix(self):
        # Negative control: reproduces the reported crash. Locks in the fact
        # that mx.eval(parameters()) is insufficient, so a future refactor that
        # drops _materialize_module_buffers can't silently pass.
        model = _yarn_rope_model()
        mx.eval(model.parameters())  # mimic mlx-lm load (misses _freqs)

        res = _rope_forward_on_worker(model)
        assert res.get("error") is not None, (
            "expected the load-thread-bound lazy _freqs to crash on the worker "
            "thread — if this passes, scaled-RoPE buffers no longer stay lazy "
            "and the fix may be unnecessary"
        )
        assert "Stream" in str(res["error"])

    def test_materialize_module_buffers_fixes_cross_thread(self):
        from olmlx.engine.model_manager import _materialize_module_buffers

        model = _yarn_rope_model()
        mx.eval(model.parameters())
        _materialize_module_buffers(model)  # the fix, on the load thread

        res = _rope_forward_on_worker(model)
        assert res.get("error") is None, (
            f"scaled-RoPE _freqs not materialized cross-thread: {res.get('error')!r}"
        )
        assert res["value"] is True


@pytest.mark.usefixtures("metal_default_device")
class TestScaledRopeBufferMaterializationWiring:
    """The three loader chokepoints materialize buffers on the load thread.

    Text goes through ``_load_with_model_type_fallback``; every mlx-vlm load
    (main VLM, lm-then-vlm fallback, flash / flash-MoE VLM fallback) through
    ``load_vlm``; the flash dense path through
    ``load_model_with_strict_fallback``. Each is monkeypatched to return a
    scaled-RoPE model so the wiring is exercised without a real model load.
    """

    def test_load_vlm_materializes_buffers(self, monkeypatch):
        import mlx_vlm

        from olmlx.engine.vlm_load import load_vlm

        model = _yarn_rope_model()
        mx.eval(model.parameters())  # load-time param eval (misses _freqs)
        monkeypatch.setattr(mlx_vlm, "load", lambda *a, **k: (model, object()))

        out_model, _ = load_vlm("fake/path", lazy=True)
        assert out_model is model
        res = _rope_forward_on_worker(out_model)
        assert res.get("error") is None, (
            f"load_vlm left scaled-RoPE _freqs lazy: {res.get('error')!r}"
        )

    def test_strict_fallback_materializes_buffers(self, monkeypatch):
        import mlx_lm

        from olmlx.engine.flash.prepare import load_model_with_strict_fallback

        model = _yarn_rope_model()
        mx.eval(model.parameters())
        monkeypatch.setattr(mlx_lm, "load", lambda *a, **k: (model, object()))

        out_model, _ = load_model_with_strict_fallback("fake/path", lazy=False)
        assert out_model is model
        res = _rope_forward_on_worker(out_model)
        assert res.get("error") is None, (
            f"strict-fallback load left scaled-RoPE _freqs lazy: {res.get('error')!r}"
        )


def _whisper_buffer_model():
    """A minimal module tree carrying mlx_whisper's two lazy underscore buffers.

    Mirrors ``mlx_whisper.whisper``: ``AudioEncoder.__init__`` sets
    ``self._positional_embedding = sinusoids(...).astype(dtype)`` and
    ``TextDecoder.__init__`` sets ``self._mask =
    create_additive_causal_mask(...).astype(dtype)`` — both *lazy* arrays built
    in ``__init__`` and stored under underscore keys, so they are NOT part of
    ``nn.Module.parameters()``. ``mlx_whisper.load_models.load_model``'s
    ``mx.eval(model.parameters())`` therefore never materializes them, leaving
    them bound to the load thread's stream.
    """
    import mlx.nn as nn
    from mlx_whisper.whisper import sinusoids

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Linear(8, 8, bias=False)  # a real parameter
            self._positional_embedding = sinusoids(16, 8).astype(mx.float16)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self._mask = nn.MultiHeadAttention.create_additive_causal_mask(16).astype(
                mx.float16
            )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

    return Model()


def _whisper_buffers_on_worker(model):
    """Consume both underscore buffers via new ops on a fresh thread."""

    def work():
        mx.eval(model.encoder._positional_embedding + 0.0)
        mx.eval(model.decoder._mask + 0.0)
        return True

    return _run_in_thread(work)


@pytest.mark.usefixtures("metal_default_device")
class TestWhisperBufferMaterialization:
    """Regression gate for the whisper cross-thread crash (issue #651).

    ``mlx_whisper.load_models.load_model`` runs ``mx.eval(model.parameters())``,
    which skips the underscore-keyed ``_positional_embedding`` / ``_mask``
    buffers. Loaded on one ``asyncio.to_thread`` pool thread and run on another
    (whisper *load* vs. *transcribe* land on different pool threads once prior
    traffic has rotated the executor), the first forward that evaluates them
    raised the uncatchable ``libc++abi ... There is no Stream(gpu, N) in current
    thread`` process abort. Same class as the scaled-RoPE ``_freqs`` gap.
    """

    def test_whisper_lazy_buffers_crash_cross_thread_without_fix(self):
        # Negative control: reproduces the reported crash so a future refactor
        # that drops _materialize_module_buffers on the whisper path can't
        # silently pass. Mirrors the scaled-RoPE negative control.
        model = _whisper_buffer_model()
        mx.eval(model.parameters())  # mimic mlx_whisper.load_model (misses them)

        res = _whisper_buffers_on_worker(model)
        assert res.get("error") is not None, (
            "expected the load-thread-bound lazy whisper buffers to crash on the "
            "worker thread — if this passes, mlx_whisper's _positional_embedding/"
            "_mask no longer stay lazy and the fix may be unnecessary"
        )
        assert "Stream" in str(res["error"])

    def test_materialize_module_buffers_fixes_whisper(self):
        from olmlx.engine.model_manager import _materialize_module_buffers

        model = _whisper_buffer_model()
        mx.eval(model.parameters())
        _materialize_module_buffers(model)  # the fix, on the load thread

        res = _whisper_buffers_on_worker(model)
        assert res.get("error") is None, (
            f"whisper buffers not materialized cross-thread: {res.get('error')!r}"
        )
        assert res["value"] is True


@pytest.mark.usefixtures("metal_default_device")
class TestWhisperTtsLoaderWiring:
    """The whisper and TTS loader chokepoints materialize buffers at load.

    ``ModelManager._load_model_whisper`` (whisper STT) and ``_load_model_tts``
    (Kokoro etc.) both return a model whose non-parameter buffers must be
    materialized on the load thread — otherwise a cross-thread forward aborts
    the process. Each underlying loader is monkeypatched to return a model with
    lazy underscore buffers so the wiring is exercised without a real load.
    """

    def test_load_model_whisper_materializes_buffers(self, monkeypatch):
        import mlx_whisper.load_models as whisper_loader

        from olmlx.engine.model_manager import ModelManager

        model = _whisper_buffer_model()
        mx.eval(model.parameters())  # load-time param eval (misses the buffers)
        monkeypatch.setattr(whisper_loader, "load_model", lambda *a, **k: model)

        out_model, tokenizer, is_vlm, _caps, decoder = ModelManager._load_model_whisper(
            "fake/path"
        )
        assert out_model is model
        assert tokenizer is None and is_vlm is False and decoder is None
        res = _whisper_buffers_on_worker(out_model)
        assert res.get("error") is None, (
            f"_load_model_whisper left buffers lazy: {res.get('error')!r}"
        )

    def test_load_model_tts_materializes_buffers(self, monkeypatch):
        tts_utils = pytest.importorskip("mlx_audio.tts.utils")

        from olmlx.engine.model_manager import ModelManager

        model = _whisper_buffer_model()  # stands in for a TTS model w/ lazy bufs
        mx.eval(model.parameters())
        monkeypatch.setattr(tts_utils, "load_model", lambda *a, **k: model)

        # _load_model_tts ignores self; a bare instance avoids full construction.
        mgr = ModelManager.__new__(ModelManager)
        out_model, tokenizer, is_vlm, _caps, decoder = mgr._load_model_tts(
            "fake/hf", "fake/path"
        )
        assert out_model is model
        assert tokenizer is None and is_vlm is False and decoder is None
        res = _whisper_buffers_on_worker(out_model)
        assert res.get("error") is None, (
            f"_load_model_tts left buffers lazy: {res.get('error')!r}"
        )
