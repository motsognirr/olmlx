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
