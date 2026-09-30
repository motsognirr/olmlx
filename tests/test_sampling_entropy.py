"""Per-request sampling entropy on generation worker threads (#708).

MLX's random state is thread-local, and olmlx samples on worker threads
(``CancellableStream`` spawns a fresh thread per streaming request; the
non-streaming path runs on an ``asyncio.to_thread`` pool thread). Two
things broke real sampling there:

1. mlx-lm's / mlx-vlm's ``categorical_sampling`` is
   ``mx.compile(inputs=mx.random.state, outputs=mx.random.state)``. The
   ``mx.random.state`` list captured at import belongs to the importing
   thread, so on any other thread the compiled draw neither reads nor
   advances that thread's state: every token gets identical noise and
   ``mx.random.seed`` is ignored.
2. Every fresh thread starts from the same default random state, so an
   unseeded request never drew real per-request entropy.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

from olmlx.engine.generation_options import _apply_seed, _build_generate_kwargs

VOCAB = 1000
N_DRAWS = 12


def _draws(sampler, n: int = N_DRAWS) -> list[int]:
    # Logits built on the calling thread: a lazy array made on another
    # thread would be bound to that thread's stream.
    logprobs = mx.zeros((1, VOCAB))
    out = []
    for _ in range(n):
        tok = sampler(logprobs)
        mx.eval(tok)
        out.append(tok.item())
    return out


def _on_fresh_thread(fn):
    result: dict = {}

    def _target():
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — surface to the test
            result["error"] = exc

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def _request(options: dict, *, is_vlm: bool = False) -> list[int]:
    """Mimic one generation request: build kwargs, seed, sample."""
    kwargs = _build_generate_kwargs(options, is_vlm=is_vlm)
    _apply_seed(kwargs, consume=not is_vlm)
    return _draws(kwargs["sampler"])


class TestWorkerThreadSampling:
    def test_sampler_advances_rng_on_worker_thread(self):
        """Successive draws on a worker thread must not repeat the same noise."""
        draws = _on_fresh_thread(lambda: _request({"temperature": 1.0}))
        assert len(set(draws)) > 1, draws

    def test_explicit_seed_is_honored_on_worker_thread(self):
        """Same seed → same tokens; different seed → different tokens."""
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            a = pool.submit(_request, {"temperature": 1.0, "seed": 1}).result()
            b = pool.submit(_request, {"temperature": 1.0, "seed": 2}).result()
            a2 = pool.submit(_request, {"temperature": 1.0, "seed": 1}).result()
        finally:
            pool.shutdown()
        assert a == a2
        assert a != b

    def test_unseeded_requests_on_fresh_threads_differ(self):
        """Each unseeded request must draw fresh entropy, not the thread default."""
        first = _on_fresh_thread(lambda: _request({"temperature": 1.0}))
        second = _on_fresh_thread(lambda: _request({"temperature": 1.0}))
        assert first != second

    def test_filters_still_applied(self):
        """top_k=1 on a peaked distribution must always pick the argmax."""

        def run():
            kwargs = _build_generate_kwargs({"temperature": 1.0, "top_k": 1})
            _apply_seed(kwargs)
            logprobs = mx.zeros((1, VOCAB)).at[:, 7].add(1.0)
            toks = [kwargs["sampler"](logprobs) for _ in range(N_DRAWS)]
            mx.eval(toks)
            return {t.item() for t in toks}

        assert _on_fresh_thread(run) == {7}

    def test_zero_temperature_is_greedy(self):
        def run():
            kwargs = _build_generate_kwargs({"temperature": 0.0})
            logprobs = mx.zeros((1, VOCAB)).at[:, 3].add(1.0)
            return kwargs["sampler"](logprobs).item()

        assert _on_fresh_thread(run) == 3

    def test_vlm_gets_thread_safe_sampler(self):
        """mlx-vlm's own sampler has the same compiled-state bug; we pass ours."""
        draws = _on_fresh_thread(lambda: _request({"temperature": 1.0}, is_vlm=True))
        assert len(set(draws)) > 1, draws
