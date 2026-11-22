"""top_k-first sampling filter: same token support, without a full-vocab sort.

olmlx's default sampling (Ollama parity, #646) sets ``top_k=40`` and
``top_p=0.9``. The filters used to run in mlx-lm's order (top_p, min_p,
top_k), so every sampled token paid an ``argsort`` over the whole vocabulary
(150k-260k entries): ~0.3 ms/token, -11% decode on Qwen3-0.6B and -5% on
Qwen3.6-35B-A3B. When ``top_k`` is set, the sampler now selects the k
candidates first and runs top_p / min_p over those k only. top_p accounts for
the probability mass outside the k candidates, so the kept set is exactly the
old one: top_p and top_k each keep a prefix of the descending order, and the
old chain kept the shorter prefix.
"""

import threading

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

from olmlx.engine import generation_options
from olmlx.engine.generation_options import _make_thread_safe_sampler


def _on_fresh_thread(fn):
    """Run on a worker thread like real generation (MLX RNG is thread-local, #708)."""
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


def _logprobs(
    probs: list[float], vocab: int, seed: int = 0
) -> tuple[mx.array, np.ndarray]:
    """Normalized logprobs with *probs* placed at shuffled positions; the rest
    of the mass is spread over the remaining tokens."""
    rng = np.random.default_rng(seed)
    p = np.full(vocab, (1.0 - sum(probs)) / (vocab - len(probs)), dtype=np.float64)
    pos = rng.permutation(vocab)[: len(probs)]
    p[pos] = probs
    lp = mx.array(np.log(p).astype(np.float32)[None])
    # Materialize: a lazy op built here is bound to this thread's stream and
    # would abort the process if a worker thread evaluated it (#499).
    mx.eval(lp)
    return lp, pos


def _reference_support(logprobs, top_k, top_p, min_p) -> set[int]:
    """The pre-change chain: top_p -> min_p -> top_k over the full vocabulary."""
    x = logprobs
    if 0 < top_p < 1.0:
        x = apply_top_p(x, top_p)
    if min_p != 0.0:
        x = apply_min_p(x, min_p, 1)
    x = apply_top_k(x, top_k)
    return set(np.nonzero(np.array(x[0]) > -np.inf)[0].tolist())


class TestFilterTopKFirst:
    @pytest.mark.parametrize(
        "top_k,top_p,min_p",
        [
            (40, 0.9, 0.0),  # olmlx defaults
            (4, 0.8, 0.0),  # top_p cut lands inside the k candidates
            (3, 0.99, 0.0),  # top_k is the binding filter
            (40, 0.0, 0.0),  # top_p disabled
            (40, 1.0, 0.0),  # top_p disabled (== 1)
            (40, 0.9, 0.05),  # min_p binds
            (10, 0.5, 0.2),  # all three
        ],
    )
    def test_support_matches_full_vocab_chain(self, top_k, top_p, min_p):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            logits = rng.normal(size=1000).astype(np.float32) * 3
            lp = mx.array(logits - np.log(np.exp(logits.astype(np.float64)).sum()))[
                None
            ]
            lp = lp.astype(mx.float32)

            idx, vals = generation_options._filter_top_k_first(lp, top_k, top_p, min_p)
            kept = {
                int(i)
                for i, v in zip(np.array(idx[0]), np.array(vals[0]))
                if v > -np.inf
            }
            assert kept == _reference_support(lp, top_k, top_p, min_p), seed

    def test_top_p_counts_mass_outside_the_candidates(self):
        """p = [.5, .25, .12, .06, ...rest]; top_k=4, top_p=0.8.

        Full-vocab top_p keeps {.5, .25, .12} (ascending cumulative mass at the
        .12 token is 0.25 > 0.2). A naive top_p over only the 4 candidates
        would see 0.18 there and wrongly drop it.

        The random-logits parity test rarely puts the top_p cut inside the k
        candidates, so this case is also checked against mlx-lm's own chain:
        a release that moves ``apply_top_p``'s boundary fails here.
        """
        lp, pos = _logprobs([0.5, 0.25, 0.12, 0.06], vocab=200)
        idx, vals = generation_options._filter_top_k_first(lp, 4, 0.8, 0.0)
        kept = {
            int(i) for i, v in zip(np.array(idx[0]), np.array(vals[0])) if v > -np.inf
        }
        assert kept == _reference_support(lp, 4, 0.8, 0.0)
        assert kept == {int(pos[0]), int(pos[1]), int(pos[2])}

    @pytest.mark.parametrize("shift", [20.0, -20.0, 100.0])
    def test_raw_logits_keep_the_same_set_as_logprobs(self, shift):
        """mlx-vlm's speculative loop (Edge0 drafting at temperature > 0) hands
        the sampler raw logits, not logprobs. top_p is a fraction of the total
        mass, so the filter has to normalize them itself. Unnormalized, +20 kept
        only the argmax (sampling silently became greedy), -20 kept every
        candidate, and +100 overflowed ``exp`` and dropped them all.
        """
        lp, pos = _logprobs([0.5, 0.25, 0.12, 0.06], vocab=200)
        idx, vals = generation_options._filter_top_k_first(lp + shift, 4, 0.8, 0.0)
        kept = {
            int(i) for i, v in zip(np.array(idx[0]), np.array(vals[0])) if v > -np.inf
        }
        assert kept == {int(pos[0]), int(pos[1]), int(pos[2])}

    def test_kept_values_are_the_original_logprobs(self):
        lp, _ = _logprobs([0.4, 0.3, 0.2], vocab=100)
        idx, vals = generation_options._filter_top_k_first(lp, 5, 0.0, 0.0)
        np.testing.assert_array_equal(
            np.array(vals[0]), np.array(mx.take_along_axis(lp, idx, axis=-1)[0])
        )


class TestTopKFirstSampler:
    def test_never_sorts_the_full_vocabulary(self, monkeypatch):
        """The default sampler must not run mlx-lm's full-vocab top_p/top_k."""
        seen: list[int] = []

        def spy(name):
            real = getattr(generation_options, name)

            def wrapper(x, *a, **kw):
                seen.append(x.shape[-1])
                return real(x, *a, **kw)

            return wrapper

        monkeypatch.setattr(generation_options, "apply_top_p", spy("apply_top_p"))
        sampler = _make_thread_safe_sampler(temp=0.8, top_p=0.9, top_k=40)
        lp, _ = _logprobs([0.5, 0.3], vocab=5000)
        mx.eval(sampler(lp))
        assert all(n <= 40 for n in seen), seen

    def test_draws_match_renormalized_distribution(self):
        """Batched draws land on the right token ids with the right frequencies."""
        lp, pos = _logprobs([0.5, 0.25, 0.12, 0.06], vocab=300, seed=3)
        n = 4000

        def run():
            mx.random.seed(1234)
            sampler = _make_thread_safe_sampler(temp=1.0, top_p=0.8, top_k=4)
            toks = sampler(mx.broadcast_to(lp, (n, lp.shape[-1])))
            assert toks.shape == (n,)
            return np.array(toks)

        toks = _on_fresh_thread(run)
        support = [int(pos[0]), int(pos[1]), int(pos[2])]
        assert set(toks.tolist()) <= set(support)
        expected = np.array([0.5, 0.25, 0.12]) / 0.87
        observed = np.array([(toks == t).mean() for t in support])
        np.testing.assert_allclose(observed, expected, atol=0.03)

    def test_top_k_at_least_vocab_is_a_no_op(self):
        """top_k >= vocab used to raise mid-generation; it now just keeps all."""
        lp, _ = _logprobs([0.9], vocab=20)

        def run():
            mx.random.seed(0)
            sampler = _make_thread_safe_sampler(temp=1.0, top_k=50)
            return int(sampler(lp).item())

        assert 0 <= _on_fresh_thread(run) < 20

    def test_temperature_zero_stays_greedy(self):
        lp, pos = _logprobs([0.6, 0.3], vocab=50)
        sampler = _make_thread_safe_sampler(temp=0.0, top_p=0.9, top_k=40)
        assert int(sampler(lp).item()) == int(pos[0])
