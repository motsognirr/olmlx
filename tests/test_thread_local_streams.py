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
