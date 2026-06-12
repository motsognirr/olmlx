"""Phase 0 tests for continuous batching (docs/batching-plan.md).

Covers the shared rope-bug workaround (`engine/ropefix.py`), the
batch-convertible cache probe (`engine/batching.py`), and the batching
settings. The scheduler itself is Phase 1.
"""

import asyncio
import threading

import mlx.core as mx
import pytest
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)

from olmlx.config import Settings


# ---------------------------------------------------------------------------
# safe_rope_patch — promoted to engine/ropefix.py
# ---------------------------------------------------------------------------


class TestRopefix:
    def _rope_kwargs(self):
        return dict(traditional=False, base=10000.0, scale=1.0, offset=7)

    def test_importable_from_ropefix(self):
        from olmlx.engine.ropefix import safe_rope_patch  # noqa: F401

    def test_selfgen_reexport_is_same_object(self):
        """Old importers (`dflash.selfgen`) must keep working."""
        from olmlx.engine.dflash.selfgen import safe_rope_patch as old
        from olmlx.engine.ropefix import safe_rope_patch as new

        assert old is new

    def test_batched_decode_rope_matches_per_row_reference(self):
        """The patched call must equal per-row (B=1) reference output for
        the buggy B>1, L==1 shape."""
        from olmlx.engine.ropefix import safe_rope_patch

        B, H, L, D = 3, 2, 1, 8
        mx.random.seed(0)
        x = mx.random.normal((B, H, L, D))
        rows = [x[i : i + 1] for i in range(B)]
        kwargs = self._rope_kwargs()
        with safe_rope_patch():
            batched = mx.fast.rope(x, D, **kwargs)
            refs = [mx.fast.rope(r, D, **kwargs) for r in rows]
        for i in range(B):
            assert mx.allclose(batched[i : i + 1], refs[i], atol=1e-5).item()

    def test_vector_offset_passes_through(self):
        """Per-row offset vectors (BatchKVCache's decode shape) cannot be
        folded and are *correct* on the unpatched kernel — the patch must
        leave them alone."""
        from olmlx.engine.ropefix import safe_rope_patch

        B, H, L, D = 3, 2, 1, 8
        x = mx.random.normal((B, H, L, D))
        offsets = mx.array([5, 12, 60])
        kwargs = dict(traditional=False, base=10000.0, scale=1.0)
        refs = mx.concatenate(
            [
                mx.fast.rope(x[i : i + 1], D, offset=int(offsets[i].item()), **kwargs)
                for i in range(B)
            ]
        )
        with safe_rope_patch():
            patched = mx.fast.rope(x, D, offset=offsets, **kwargs)
        assert mx.allclose(patched, refs, atol=1e-5).item()

    def test_prefill_shape_passes_through(self):
        """L > 1 (prefill) must hit the original kernel path unchanged."""
        from olmlx.engine.ropefix import safe_rope_patch

        x = mx.random.normal((2, 2, 5, 8))
        kwargs = self._rope_kwargs()
        ref = mx.fast.rope(x, 8, **kwargs)
        with safe_rope_patch():
            patched = mx.fast.rope(x, 8, **kwargs)
        assert mx.allclose(ref, patched, atol=1e-6).item()

    def test_patch_restored_on_exit(self):
        from olmlx.engine.ropefix import safe_rope_patch

        orig = mx.fast.rope
        with safe_rope_patch():
            assert mx.fast.rope is not orig
        assert mx.fast.rope is orig

    def test_patch_restored_on_exception(self):
        from olmlx.engine.ropefix import safe_rope_patch

        orig = mx.fast.rope
        with pytest.raises(RuntimeError):
            with safe_rope_patch():
                raise RuntimeError("boom")
        assert mx.fast.rope is orig

    @pytest.mark.skipif(
        not mx.metal.is_available(), reason="Metal-kernel bug; CPU path is fine"
    )
    def test_rope_bug_still_present_remove_patch_when_this_fails(self):
        """Removal gate for safe_rope_patch (#499).

        Asserts the mlx B>1/L==1 ``mx.fast.rope`` corruption still
        reproduces on the unpatched kernel. When an mlx upgrade fixes it,
        this test FAILS — that is the signal to delete engine/ropefix.py,
        drop the patch from its holders, and delete this test.
        """
        B, H, L, D = 4, 2, 1, 64
        mx.random.seed(1)
        x = mx.random.normal((B, H, L, D))
        kwargs = dict(traditional=False, base=10000.0, scale=1.0, offset=33)
        direct = mx.fast.rope(x, D, **kwargs)
        refs = mx.concatenate(
            [mx.fast.rope(x[i : i + 1], D, **kwargs) for i in range(B)]
        )
        assert not mx.allclose(direct, refs, atol=1e-4).item(), (
            "mx.fast.rope now matches per-row reference at B>1/L==1 — the "
            "mlx bug appears fixed; remove safe_rope_patch and this test"
        )


# ---------------------------------------------------------------------------
# Cache probe — mirrors mlx-lm's _make_cache convertibility rules
# ---------------------------------------------------------------------------


class TestCacheProbe:
    def test_plain_kvcache_convertible(self):
        from olmlx.engine.batching import caches_batch_convertible

        assert caches_batch_convertible([KVCache(), KVCache()])

    def test_arrays_cache_convertible(self):
        from olmlx.engine.batching import caches_batch_convertible

        assert caches_batch_convertible([ArraysCache(size=2)])

    def test_rotating_without_keep_convertible(self):
        from olmlx.engine.batching import caches_batch_convertible

        assert caches_batch_convertible([RotatingKVCache(max_size=512)])

    def test_rotating_with_keep_not_convertible(self):
        from olmlx.engine.batching import caches_batch_convertible

        assert not caches_batch_convertible([RotatingKVCache(max_size=512, keep=4)])

    def test_kvcache_subclass_not_convertible(self):
        """mlx-lm uses an exact ``type(c) is KVCache`` check — subclasses
        (olmlx's quantized caches) are rejected, so the probe must be too."""
        from olmlx.engine.batching import caches_batch_convertible

        class FancyKVCache(KVCache):
            pass

        assert not caches_batch_convertible([FancyKVCache()])

    def test_unknown_cache_not_convertible(self):
        from olmlx.engine.batching import caches_batch_convertible

        class WeirdCache:
            pass

        assert not caches_batch_convertible([KVCache(), WeirdCache()])

    def test_cache_list_recurses(self):
        from olmlx.engine.batching import caches_batch_convertible

        ok = CacheList(KVCache(), RotatingKVCache(max_size=64))
        bad = CacheList(KVCache(), RotatingKVCache(max_size=64, keep=2))
        assert caches_batch_convertible([ok])
        assert not caches_batch_convertible([bad])

    def test_empty_is_not_convertible(self):
        """No layers → nothing to batch; treat as not eligible."""
        from olmlx.engine.batching import caches_batch_convertible

        assert not caches_batch_convertible([])

    def test_mock_caches_not_convertible(self):
        """A MagicMock is truthy but iterates empty — it must not pass the
        probes vacuously (it routed mocked-settings inference tests onto
        the batched path and hung the suite)."""
        from unittest.mock import MagicMock

        from olmlx.engine.batching import (
            caches_batch_convertible,
            caches_plain_kv,
        )

        assert not caches_batch_convertible(MagicMock())
        assert not caches_plain_kv(MagicMock())


# ---------------------------------------------------------------------------
# BatchScheduler (fake generator; no MLX work)
# ---------------------------------------------------------------------------


class _Resp:
    """Duck-typed stand-in for mlx-lm's batch Response objects."""

    def __init__(
        self,
        uid,
        *,
        token=None,
        finish_reason=None,
        progress=None,
        prompt_cache=None,
        all_tokens=None,
    ):
        self.uid = uid
        self.token = token
        self.finish_reason = finish_reason
        self.progress = progress
        self.prompt_cache = prompt_cache
        self.all_tokens = all_tokens


class FakeCache:
    """Per-layer cache stand-in; ``state`` keeps mx.eval happy (empty tree)."""

    def __init__(self, tag=None):
        self.tag = tag

    @property
    def state(self):
        return ()


class FakeGen:
    """Scripted BatchGenerator: each insert consumes the next output script.

    A script is a list of (token, finish_reason) steps. The first next()
    tick after insert emits a prompt-progress response; subsequent ticks
    emit one generation response per active sequence. Mirrors mlx-lm's
    token bookkeeping: per-uid ``all_tokens`` = history + prompt + every
    generated step token (including the finish-step token), and finished
    responses carry the extracted per-sequence cache.
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.inserted = 0
        self.next_calls = 0
        self.active = {}  # uid -> {"steps": [...], "progressed": bool, ...}
        self.removed = []
        self.closed = False
        self.on_next = None  # optional hook called at the top of next()
        self.insert_calls = []  # kwargs of every insert, for assertions

    def insert(
        self,
        prompts,
        max_tokens,
        caches=None,
        all_tokens=None,
        samplers=None,
        logits_processors=None,
    ):
        self.insert_calls.append(
            {
                "prompts": [list(p) for p in prompts],
                "caches": caches,
                "all_tokens": all_tokens,
            }
        )
        uids = []
        for i, p in enumerate(prompts):
            uid = self.inserted
            history = list(all_tokens[i]) if all_tokens else []
            self.active[uid] = {
                "steps": list(self.scripts[self.inserted]),
                "progressed": False,
                "tokens": history + list(p),
            }
            self.inserted += 1
            uids.append(uid)
        return uids

    def _extract(self, uid):
        st = self.active[uid]
        return [FakeCache(f"extracted-{uid}")], list(st["tokens"])

    def next(self):
        self.next_calls += 1
        if self.on_next is not None:
            self.on_next(self)
        prompt_rs, gen_rs = [], []
        for uid, st in list(self.active.items()):
            if not st["progressed"]:
                st["progressed"] = True
                prompt_rs.append(_Resp(uid, progress=(7, 7)))
                continue
            token, reason = st["steps"].pop(0)
            st["tokens"].append(token)
            if reason is not None:
                cache, tokens = self._extract(uid)
                gen_rs.append(
                    _Resp(
                        uid,
                        token=token,
                        finish_reason=reason,
                        prompt_cache=cache,
                        all_tokens=tokens,
                    )
                )
                del self.active[uid]
            else:
                gen_rs.append(_Resp(uid, token=token, finish_reason=None))
        return prompt_rs, gen_rs

    def remove(self, uids, return_prompt_caches=False):
        caches = {}
        for uid in uids:
            if return_prompt_caches and uid in self.active:
                caches[uid] = self._extract(uid)
            self.active.pop(uid, None)
            self.removed.append(uid)
        return caches

    def close(self):
        self.closed = True


class _GpuLog:
    """Records acquire/release ordering for assertions."""

    def __init__(self):
        self.events = []

    async def acquire(self):
        self.events.append("acquire")

    def release(self):
        self.events.append("release")


def _make_scheduler(scripts, *, exclusive_pending=None, gens=None, on_next=None):
    from olmlx.engine.batching import BatchScheduler

    gpu = _GpuLog()
    made = gens if gens is not None else []

    def factory():
        gen = FakeGen(scripts)
        gen.on_next = on_next
        made.append(gen)
        return gen

    sched = BatchScheduler(
        generator_factory=factory,
        acquire_gpu=gpu.acquire,
        release_gpu=gpu.release,
        exclusive_pending=exclusive_pending,
        name="test",
    )
    return sched, gpu, made


async def _collect(seq):
    events = []
    while True:
        ev = await asyncio.wait_for(seq.out.get(), timeout=5.0)
        events.append(ev)
        if ev["type"] in ("done", "error"):
            return events


def _tokens(events):
    return [ev["token"] for ev in events if ev["type"] == "token"]


class TestBatchScheduler:
    async def test_single_request_round_trip(self):
        from olmlx.engine.batching import BatchRequest

        sched, gpu, gens = _make_scheduler([[(10, None), (11, None), (0, "stop")]])
        seq = await sched.submit(BatchRequest(tokens=[1, 2, 3], max_tokens=8))
        events = await _collect(seq)
        # EOS-step token is not emitted; progress precedes tokens.
        assert _tokens(events) == [10, 11]
        assert events[0]["type"] == "progress"
        assert events[-1] == {"type": "done", "reason": "stop"}
        assert gpu.events == ["acquire", "release"]
        assert gens[0].closed

    async def test_length_finish_emits_final_token(self):
        from olmlx.engine.batching import BatchRequest

        sched, _, _ = _make_scheduler([[(5, None), (6, "length")]])
        seq = await sched.submit(BatchRequest(tokens=[1], max_tokens=2))
        events = await _collect(seq)
        assert _tokens(events) == [5, 6]
        assert events[-1]["reason"] == "length"

    async def test_concurrent_requests_share_one_busy_period(self):
        from olmlx.engine.batching import BatchRequest

        gate = threading.Event()

        # Block the first tick until both requests are in.
        def hold(gen):
            if gen.next_calls == 1:
                gate.wait(timeout=5.0)

        sched, gpu, gens = _make_scheduler(
            [
                [(10, None), (11, None), (0, "stop")],
                [(20, "length")],
            ],
            on_next=hold,
        )
        seq1 = await sched.submit(BatchRequest(tokens=[1], max_tokens=8))
        seq2 = await sched.submit(BatchRequest(tokens=[2], max_tokens=8))
        gate.set()
        ev1, ev2 = await asyncio.gather(_collect(seq1), _collect(seq2))
        assert _tokens(ev1) == [10, 11]
        assert _tokens(ev2) == [20]
        # One busy period: single acquire/release pair.
        assert gpu.events == ["acquire", "release"]
        assert len(gens) == 1

    async def test_new_request_after_drain_starts_new_period(self):
        from olmlx.engine.batching import BatchRequest

        sched, gpu, gens = _make_scheduler(
            [[(1, "length")], [(2, "length")]],
        )
        await _collect(await sched.submit(BatchRequest(tokens=[1], max_tokens=1)))
        # Fresh generator per busy period: second submit consumes script
        # index 0 of a NEW FakeGen, so give it the same script shape.
        await _collect(await sched.submit(BatchRequest(tokens=[2], max_tokens=1)))
        assert gpu.events == ["acquire", "release", "acquire", "release"]
        assert len(gens) == 2
        assert all(g.closed for g in gens)

    async def test_cancel_frees_slot_and_emits_cancelled(self):
        from olmlx.engine.batching import BatchRequest, BatchScheduler

        gate = threading.Event()
        long_script = [(i, None) for i in range(100)] + [(0, "stop")]

        # Hold the generator at tick 3 until the test has cancelled, so the
        # instant fake can't race through the whole script first.
        def hold_for_cancel(gen):
            if gen.next_calls == 3:
                gate.wait(timeout=5.0)

        sched, _, gens = _make_scheduler([long_script], on_next=hold_for_cancel)
        seq = await sched.submit(BatchRequest(tokens=[1], max_tokens=200))
        # Wait for generation to demonstrably start (first token).
        while True:
            ev = await asyncio.wait_for(seq.out.get(), timeout=5.0)
            if ev["type"] == "token":
                break
        BatchScheduler.cancel(seq)
        gate.set()
        events = await _collect(seq)
        assert events[-1] == {"type": "done", "reason": "cancelled"}
        assert gens[0].removed == [0]

    async def test_worker_error_fails_active_and_queued(self):
        from olmlx.engine.batching import BatchRequest

        gens = []

        def boom_first_gen_only(gen):
            if gen is gens[0]:
                raise RuntimeError("metal exploded")

        sched, gpu, _ = _make_scheduler(
            [[(1, None), (2, "stop")]], gens=gens, on_next=boom_first_gen_only
        )
        seq = await sched.submit(BatchRequest(tokens=[1], max_tokens=8))
        events = await _collect(seq)
        assert events[-1]["type"] == "error"
        assert "metal exploded" in str(events[-1]["exc"])
        # Lock released despite the failure (the error event can reach the
        # consumer a beat before the manager's finally runs — wait for it);
        # generator closed.
        for _ in range(200):
            if "release" in gpu.events:
                break
            await asyncio.sleep(0.01)
        assert gpu.events == ["acquire", "release"]
        assert gens[0].closed
        # Scheduler survives: a new request gets a fresh generator and
        # completes normally.
        events2 = await _collect(
            await sched.submit(BatchRequest(tokens=[2], max_tokens=8))
        )
        assert events2[-1] == {"type": "done", "reason": "stop"}
        assert _tokens(events2) == [1]
        assert len(gens) == 2

    async def test_pause_latches_mid_period(self):
        """A request arriving after the exclusive flag rises stays in the
        inbox for the whole busy period (admission latched shut)."""
        from olmlx.engine.batching import BatchRequest, BatchSequence

        flag = {"up": False}
        loop = asyncio.get_running_loop()
        seq2 = BatchSequence(BatchRequest(tokens=[2], max_tokens=8), loop)
        holder = {}

        def hook(gen):
            if gen.next_calls == 2 and not holder.get("injected"):
                # Exclusive waiter appears, then a second batched request
                # lands mid-period — straight into the inbox, bypassing
                # submit() so the timing is deterministic.
                holder["injected"] = True
                flag["up"] = True
                holder["sched"]._inbox.put(seq2)

        sched, _, gens = _make_scheduler(
            [[(10, None), (11, None), (0, "stop")], [(99, "length")]],
            exclusive_pending=lambda: flag["up"],
            on_next=hook,
        )
        holder["sched"] = sched

        seq1 = await sched.submit(BatchRequest(tokens=[1], max_tokens=8))
        ev1 = await _collect(seq1)
        assert _tokens(ev1) == [10, 11]
        # seq2 was never admitted into period 1...
        assert gens[0].inserted == 1
        # ...and is served in period 2 once the exclusive flag clears
        # (manager re-arms because the inbox is non-empty). Each period
        # gets a fresh FakeGen, so seq2 consumes script index 0 again.
        flag["up"] = False
        ev2 = await _collect(seq2)
        assert _tokens(ev2) == [10, 11]
        assert len(gens) == 2
        assert gens[1].inserted == 1

    def test_rebinds_after_event_loop_close(self):
        """One event loop per test (pytest-asyncio) means a scheduler can
        outlive its loop on the LoadedModel. A submit from a fresh loop
        must rebind — the stale-manager variant hung the suite forever."""
        from olmlx.engine.batching import BatchRequest

        sched, gpu, gens = _make_scheduler([[(1, "length")]])

        async def round_trip():
            seq = await sched.submit(BatchRequest(tokens=[1], max_tokens=1))
            return await _collect(seq)

        loop_a = asyncio.new_event_loop()
        try:
            ev1 = loop_a.run_until_complete(round_trip())
        finally:
            loop_a.close()
        loop_b = asyncio.new_event_loop()
        try:
            ev2 = loop_b.run_until_complete(round_trip())
        finally:
            loop_b.close()
        assert ev1[-1] == {"type": "done", "reason": "length"}
        assert ev2[-1] == {"type": "done", "reason": "length"}
        assert gpu.events == ["acquire", "release", "acquire", "release"]

    def test_rejects_use_from_other_loop_while_bound_loop_alive(self):
        """Cross-loop use while the bound loop is still alive (merely not
        running) is a programming error, not a rebind."""
        from olmlx.engine.batching import BatchRequest

        sched, _, _ = _make_scheduler([[(1, "length")]])

        async def round_trip():
            seq = await sched.submit(BatchRequest(tokens=[1], max_tokens=1))
            return await _collect(seq)

        async def submit_other():
            await sched.submit(BatchRequest(tokens=[1], max_tokens=1))

        loop_a = asyncio.new_event_loop()
        try:
            loop_a.run_until_complete(round_trip())  # binds to loop_a
            loop_b = asyncio.new_event_loop()
            try:
                with pytest.raises(RuntimeError, match="different running event loop"):
                    loop_b.run_until_complete(submit_other())
            finally:
                loop_b.close()
        finally:
            loop_a.close()

    async def test_manager_cancellation_holds_lock_until_worker_exits(self):
        """Cancelling the manager task (shutdown) must NOT release the GPU
        while the worker thread is still running — the lock is held until
        the worker drains, then released and CancelledError propagates."""
        from olmlx.engine.batching import BatchRequest

        running = threading.Event()
        gate = threading.Event()
        long_script = [(i, None) for i in range(1000)] + [(0, "stop")]

        def hold(gen):
            running.set()
            if gen.next_calls >= 2:
                gate.wait(timeout=5.0)
                gate.clear()

        sched, gpu, gens = _make_scheduler([long_script], on_next=hold)
        seq = await sched.submit(BatchRequest(tokens=[1], max_tokens=2000))
        await asyncio.to_thread(running.wait, 5.0)

        assert sched._manager_task is not None
        sched._manager_task.cancel()
        # Give the cancellation a moment to land; the worker is blocked at
        # the gate, so the lock must still be held.
        await asyncio.sleep(0.05)
        assert "release" not in gpu.events
        # Unblock the worker: it observes _closing, drains, and exits; the
        # manager then releases and finishes cancelled.
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(sched._manager_task, timeout=5.0)
        assert gpu.events == ["acquire", "release"]
        events = await _collect(seq)
        assert events[-1] == {"type": "done", "reason": "cancelled"}

    async def test_close_cancels_queued_and_stops_manager(self):
        from olmlx.engine.batching import BatchRequest

        sched, gpu, _ = _make_scheduler([[(1, "stop")]])
        # Close with a queued item before any manager run is forced.
        seq = await sched.submit(BatchRequest(tokens=[1], max_tokens=1))
        sched.close()
        events = await _collect(seq)
        assert events[-1]["reason"] == "cancelled" or events[-1]["type"] == "done"
        with pytest.raises(RuntimeError):
            await sched.submit(BatchRequest(tokens=[1], max_tokens=1))
        # Manager exits (possibly after finishing an in-flight period).
        if sched._manager_task is not None:
            await asyncio.wait_for(sched._manager_task, timeout=5.0)


# ---------------------------------------------------------------------------
# Scheduler cache round trip (Phase 2)
# ---------------------------------------------------------------------------


class TestSchedulerCacheRoundTrip:
    async def test_done_carries_cache_and_tokens_when_requested(self):
        from olmlx.engine.batching import BatchRequest

        sched, _, _ = _make_scheduler([[(10, None), (0, "stop")]])
        seq = await sched.submit(
            BatchRequest(tokens=[1, 2, 3], max_tokens=8, return_cache=True)
        )
        events = await _collect(seq)
        done = events[-1]
        assert done["type"] == "done" and done["reason"] == "stop"
        assert isinstance(done["cache"][0], FakeCache)
        # mlx-lm semantics: all_tokens = prompt + every step token,
        # including the finish-step (EOS) token.
        assert done["tokens"] == [1, 2, 3, 10, 0]

    async def test_done_omits_cache_by_default(self):
        from olmlx.engine.batching import BatchRequest

        sched, _, _ = _make_scheduler([[(10, None), (0, "stop")]])
        seq = await sched.submit(BatchRequest(tokens=[1], max_tokens=8))
        events = await _collect(seq)
        assert "cache" not in events[-1]
        assert "tokens" not in events[-1]

    async def test_pre_seeded_cache_and_history_reach_insert(self):
        from olmlx.engine.batching import BatchRequest

        sched, _, gens = _make_scheduler([[(9, "length")]])
        cache = [FakeCache("seed")]
        seq = await sched.submit(
            BatchRequest(
                tokens=[4, 5],
                max_tokens=8,
                cache=cache,
                history_tokens=[1, 2, 3],
                return_cache=True,
            )
        )
        events = await _collect(seq)
        call = gens[0].insert_calls[0]
        assert call["caches"] == [cache]
        assert call["all_tokens"] == [[1, 2, 3]]
        # Done tokens cover history + suffix + generated.
        assert events[-1]["tokens"] == [1, 2, 3, 4, 5, 9]

    async def test_fresh_request_inserts_none_cache(self):
        from olmlx.engine.batching import BatchRequest

        sched, _, gens = _make_scheduler([[(9, "length")]])
        await _collect(await sched.submit(BatchRequest(tokens=[1], max_tokens=8)))
        call = gens[0].insert_calls[0]
        assert call["caches"] == [None]
        assert call["all_tokens"] == [[]]

    async def test_cancel_with_want_cache_returns_cache(self):
        from olmlx.engine.batching import BatchRequest, BatchScheduler

        gate = threading.Event()
        long_script = [(i, None) for i in range(100)] + [(0, "stop")]

        def hold(gen):
            if gen.next_calls == 3:
                gate.wait(timeout=5.0)

        sched, _, _ = _make_scheduler([long_script], on_next=hold)
        seq = await sched.submit(
            BatchRequest(tokens=[1], max_tokens=200, return_cache=True)
        )
        while True:
            ev = await asyncio.wait_for(seq.out.get(), timeout=5.0)
            if ev["type"] == "token":
                break
        BatchScheduler.cancel(seq)
        gate.set()
        events = await _collect(seq)
        done = events[-1]
        assert done["reason"] == "cancelled"
        assert isinstance(done["cache"][0], FakeCache)
        assert done["tokens"][0] == 1  # prompt prefix preserved

    async def test_cancel_after_want_cache_cleared_omits_cache(self):
        from olmlx.engine.batching import BatchRequest, BatchScheduler

        gate = threading.Event()
        long_script = [(i, None) for i in range(100)] + [(0, "stop")]

        def hold(gen):
            if gen.next_calls == 3:
                gate.wait(timeout=5.0)

        sched, _, _ = _make_scheduler([long_script], on_next=hold)
        seq = await sched.submit(
            BatchRequest(tokens=[1], max_tokens=200, return_cache=True)
        )
        while True:
            ev = await asyncio.wait_for(seq.out.get(), timeout=5.0)
            if ev["type"] == "token":
                break
        # Consumer declines the cache (timeout / disconnect path).
        seq.want_cache = False
        BatchScheduler.cancel(seq)
        gate.set()
        events = await _collect(seq)
        assert events[-1]["reason"] == "cancelled"
        assert "cache" not in events[-1]


class TestSchedulerStats:
    async def test_counters_after_drain(self):
        from olmlx.engine.batching import BatchRequest

        # Sequential submits → two busy periods, each with a fresh FakeGen
        # consuming script index 0 (2 steps each).
        sched, _, _ = _make_scheduler([[(10, None), (0, "stop")]])
        await _collect(await sched.submit(BatchRequest(tokens=[1], max_tokens=8)))
        await _collect(await sched.submit(BatchRequest(tokens=[2], max_tokens=8)))
        s = sched.stats()
        assert s["batch_inserts"] == 2
        # Every generation response counts: 2 steps per period.
        assert s["batch_tokens"] == 4
        assert s["batch_active_sequences"] == 0
        assert s["batch_queued"] == 0

    async def test_active_gauge_nonzero_mid_period(self):
        from olmlx.engine.batching import BatchRequest

        gate = threading.Event()
        observed = {}

        def hold(gen):
            if gen.next_calls == 3:
                observed["active"] = holder["sched"].stats()["batch_active_sequences"]
                gate.wait(timeout=5.0)

        holder = {}
        sched, _, _ = _make_scheduler(
            [[(i, None) for i in range(50)] + [(0, "stop")]], on_next=hold
        )
        holder["sched"] = sched
        seq = await sched.submit(BatchRequest(tokens=[1], max_tokens=100))
        while True:
            ev = await asyncio.wait_for(seq.out.get(), timeout=5.0)
            if ev["type"] == "token":
                break
        gate.set()
        await _collect(seq)
        assert observed["active"] == 1
        assert sched.stats()["batch_active_sequences"] == 0


# ---------------------------------------------------------------------------
# Consumer round trip (Phase 2): _stream_completion_batched + prompt cache
# ---------------------------------------------------------------------------


class FakeDetok:
    """Streaming-detokenizer stand-in: each token decodes to ``<id>``."""

    def __init__(self):
        self.last_segment = ""

    def add_token(self, t):
        self.last_segment = f"<{t}>"

    def finalize(self):
        self.last_segment = ""


class FakeTokenizer:
    @property
    def detokenizer(self):
        return FakeDetok()


class TrimmableFakeCache(FakeCache):
    """FakeCache that satisfies mlx-lm's trim_prompt_cache contract."""

    def __init__(self, tag=None):
        super().__init__(tag)
        self.trimmed = 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trimmed += n
        return n


def _consumer_lm():
    from olmlx.engine.model_manager import LoadedModel
    from olmlx.engine.prompt_cache.store import PromptCacheStore

    return LoadedModel(
        name="consumer-test",
        hf_path="x",
        model=object(),
        tokenizer=FakeTokenizer(),
        prompt_cache_store=PromptCacheStore(max_slots=4),
        supports_cache_persistence=True,
    )


async def _run_batched(lm, sched, monkeypatch, prompt, gen_kwargs=None, **kw):
    """Drive _stream_completion_batched against a scheduler; return chunks."""
    from olmlx.engine import inference
    from olmlx.utils.timing import TimingStats

    monkeypatch.setattr(inference, "_get_batch_scheduler", lambda _lm: sched)
    monkeypatch.setattr(inference, "_batched_kv_preflight", lambda *a, **k: None)
    chunks = []
    async for chunk in inference._stream_completion_batched(
        lm, prompt, 32, gen_kwargs or {}, TimingStats(), **kw
    ):
        chunks.append(chunk)
    return chunks


class TestConsumerCacheRoundTrip:
    async def test_fresh_request_stores_prompt_plus_generated(self, monkeypatch):
        lm = _consumer_lm()
        sched, _, _ = _make_scheduler([[(10, None), (0, "stop")]])
        chunks = await _run_batched(
            lm, sched, monkeypatch, [1, 2, 3], use_prompt_cache=True, cache_id="cid"
        )
        assert chunks[0] == {
            "cache_info": True,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 3,
        }
        assert [c["text"] for c in chunks[1:-1]] == ["<10>"]
        assert chunks[-1]["done"] is True
        stored = lm.prompt_cache_store.peek("cid")
        assert stored is not None
        # mlx-lm semantics: stored tokens = prompt + generated (incl. the
        # finish-step EOS token).
        assert stored.tokens == [1, 2, 3, 10, 0]

    async def test_prefix_hit_seeds_sequence_and_restores(self, monkeypatch):
        from olmlx.engine.model_manager import CachedPromptState

        lm = _consumer_lm()
        seed_cache = [TrimmableFakeCache("seed")]
        lm.prompt_cache_store.set(
            "cid", CachedPromptState(tokens=[1, 2, 3, 10, 0], cache=seed_cache)
        )
        sched, _, gens = _make_scheduler([[(20, "length")]])
        chunks = await _run_batched(
            lm,
            sched,
            monkeypatch,
            [1, 2, 3, 10, 0, 7, 8],
            use_prompt_cache=True,
            cache_id="cid",
        )
        assert chunks[0] == {
            "cache_info": True,
            "cache_read_tokens": 5,
            "cache_creation_tokens": 2,
        }
        call = gens[0].insert_calls[0]
        assert call["prompts"] == [[7, 8]]
        assert call["caches"] == [seed_cache]
        assert call["all_tokens"] == [[1, 2, 3, 10, 0]]
        stored = lm.prompt_cache_store.peek("cid")
        assert stored is not None
        assert stored.tokens == [1, 2, 3, 10, 0, 7, 8, 20]

    async def test_exact_match_trims_one_and_resubmits_last_token(self, monkeypatch):
        from olmlx.engine.model_manager import CachedPromptState

        lm = _consumer_lm()
        seed_cache = [TrimmableFakeCache("seed")]
        lm.prompt_cache_store.set(
            "cid", CachedPromptState(tokens=[1, 2, 3], cache=seed_cache)
        )
        sched, _, gens = _make_scheduler([[(9, "length")]])
        chunks = await _run_batched(
            lm, sched, monkeypatch, [1, 2, 3], use_prompt_cache=True, cache_id="cid"
        )
        # Exact match backs up one position (the sequence needs a token).
        assert chunks[0]["cache_read_tokens"] == 2
        assert seed_cache[0].trimmed == 1
        assert gens[0].insert_calls[0]["prompts"] == [[3]]

    async def test_move_semantics_entry_leaves_store_during_flight(self, monkeypatch):
        """The taken entry must not be reachable while the sequence runs
        (no shared mutable cache object across lockless consumers)."""
        from olmlx.engine.model_manager import CachedPromptState

        lm = _consumer_lm()
        lm.prompt_cache_store.set(
            "cid",
            CachedPromptState(tokens=[1, 2, 3], cache=[TrimmableFakeCache()]),
        )
        observed = {}
        gate = threading.Event()

        def spy(gen):
            if gen.next_calls == 2:
                observed["mid_flight"] = lm.prompt_cache_store.peek("cid")
                gate.set()

        sched, _, _ = _make_scheduler([[(9, None), (0, "stop")]], on_next=spy)
        await _run_batched(
            lm, sched, monkeypatch, [1, 2, 3, 4], use_prompt_cache=True, cache_id="cid"
        )
        assert gate.wait(timeout=5.0)
        assert observed["mid_flight"] is None
        # ...and it is back after the round trip.
        assert lm.prompt_cache_store.peek("cid") is not None

    async def test_stop_sequence_cancellation_still_stores(self, monkeypatch):
        lm = _consumer_lm()
        long_script = [(i, None) for i in range(10, 60)] + [(0, "stop")]
        sched, _, _ = _make_scheduler([long_script])
        chunks = await _run_batched(
            lm,
            sched,
            monkeypatch,
            [1, 2],
            gen_kwargs={"stop": ["<12>"]},
            use_prompt_cache=True,
            cache_id="cid",
        )
        assert chunks[-1]["done_reason"] == "stop"
        text = "".join(
            c.get("text", "") for c in chunks[:-1] if not c.get("cache_info")
        )
        assert text == "<10><11>"
        stored = lm.prompt_cache_store.peek("cid")
        assert stored is not None
        assert stored.tokens[:2] == [1, 2]

    async def test_disconnect_does_not_store(self, monkeypatch):
        from olmlx.engine import inference
        from olmlx.utils.timing import TimingStats

        lm = _consumer_lm()
        long_script = [(i, None) for i in range(10, 60)] + [(0, "stop")]
        sched, _, _ = _make_scheduler([long_script])
        monkeypatch.setattr(inference, "_get_batch_scheduler", lambda _lm: sched)
        monkeypatch.setattr(inference, "_batched_kv_preflight", lambda *a, **k: None)
        agen = inference._stream_completion_batched(
            lm, [1, 2], 100, {}, TimingStats(), use_prompt_cache=True, cache_id="cid"
        )
        async for chunk in agen:
            if chunk.get("text"):
                break
        await agen.aclose()
        assert lm.prompt_cache_store.peek("cid") is None

    async def test_queue_timeout_restores_taken_entry(self, monkeypatch):
        """A batch queue timeout (503) must not eat the cache entry — the
        exclusive path's queue timeout fires before cache setup, so a
        retry there still gets its prefix; match that."""
        from olmlx.engine import inference
        from olmlx.engine.batching import BatchSequence
        from olmlx.engine.model_manager import CachedPromptState
        from olmlx.utils.timing import TimingStats

        lm = _consumer_lm()
        lm.inference_queue_timeout = 0.05
        seed_cache = [TrimmableFakeCache("seed")]
        lm.prompt_cache_store.set(
            "cid", CachedPromptState(tokens=[1, 2, 3], cache=seed_cache)
        )

        class _StalledScheduler:
            """Accepts the submit but never produces events (saturated
            batch); cancel responds so the disconnect drain terminates."""

            async def submit(self, req):
                return BatchSequence(req, asyncio.get_running_loop())

            @staticmethod
            def cancel(seq):
                seq.cancelled.set()
                seq.out.put_nowait({"type": "done", "reason": "cancelled"})

        monkeypatch.setattr(
            inference, "_get_batch_scheduler", lambda _lm: _StalledScheduler()
        )
        monkeypatch.setattr(inference, "_batched_kv_preflight", lambda *a, **k: None)

        with pytest.raises(inference.ServerBusyError):
            async for _ in inference._stream_completion_batched(
                lm,
                [1, 2, 3, 4],
                32,
                {},
                TimingStats(),
                use_prompt_cache=True,
                cache_id="cid",
            ):
                pass

        restored = lm.prompt_cache_store.peek("cid")
        assert restored is not None
        assert restored.cache is seed_cache
        # Post-trim coverage: the prefix the trimmed cache represents.
        assert restored.tokens == [1, 2, 3]

    async def test_no_prompt_cache_keeps_phase1_behavior(self, monkeypatch):
        lm = _consumer_lm()
        sched, _, gens = _make_scheduler([[(10, "length")]])
        chunks = await _run_batched(lm, sched, monkeypatch, [1, 2, 3])
        # No cache_info chunk, nothing stored.
        assert not any(c.get("cache_info") for c in chunks)
        assert len(lm.prompt_cache_store) == 0
        assert gens[0].insert_calls[0]["caches"] == [None]

    async def test_non_persistable_model_reports_but_skips_store(self, monkeypatch):
        lm = _consumer_lm()
        lm.supports_cache_persistence = False
        sched, _, gens = _make_scheduler([[(10, "length")]])
        chunks = await _run_batched(
            lm, sched, monkeypatch, [1, 2, 3], use_prompt_cache=True, cache_id="cid"
        )
        assert chunks[0]["cache_info"] is True
        assert chunks[0]["cache_creation_tokens"] == 3
        assert len(lm.prompt_cache_store) == 0
        assert gens[0].insert_calls[0]["caches"] == [None]


class TestFullCompletionBatched:
    async def test_aggregates_stream_into_result_dict(self, monkeypatch):
        from olmlx.engine import inference
        from olmlx.utils.timing import TimingStats

        lm = _consumer_lm()
        lm.batch_convertible = True
        sched, _, _ = _make_scheduler([[(10, None), (11, None), (0, "stop")]])
        monkeypatch.setattr(inference, "_get_batch_scheduler", lambda _lm: sched)
        monkeypatch.setattr(inference, "_batched_kv_preflight", lambda *a, **k: None)
        monkeypatch.setattr(inference.settings, "batching", True)

        stats = TimingStats()
        result = await inference._full_completion(
            lm,
            [1, 2, 3],
            32,
            {},
            stats,
            use_prompt_cache=True,
            prompt_tokens=[1, 2, 3],
            cache_id="cid",
        )
        assert result["text"] == "<10><11>"
        assert result["done"] is True
        assert result["stats"] is stats
        assert result["cache_read_tokens"] == 0
        assert result["cache_creation_tokens"] == 3
        # Round trip reached the store from the non-streaming path too.
        assert lm.prompt_cache_store.peek("cid") is not None

    async def test_stop_sequence_sets_finish_reason(self, monkeypatch):
        from olmlx.engine import inference
        from olmlx.utils.timing import TimingStats

        lm = _consumer_lm()
        lm.batch_convertible = True
        long_script = [(i, None) for i in range(10, 60)] + [(0, "stop")]
        sched, _, _ = _make_scheduler([long_script])
        monkeypatch.setattr(inference, "_get_batch_scheduler", lambda _lm: sched)
        monkeypatch.setattr(inference, "_batched_kv_preflight", lambda *a, **k: None)
        monkeypatch.setattr(inference.settings, "batching", True)

        result = await inference._full_completion(
            lm, [1, 2], 100, {"stop": ["<12>"]}, TimingStats()
        )
        assert result["text"] == "<10><11>"
        assert result["finish_reason"] == "stop"
        assert result["done_reason"] == "stop"


# ---------------------------------------------------------------------------
# Batched KV preflight
# ---------------------------------------------------------------------------


class TestBatchedKvPreflight:
    def _lm(self):
        from types import SimpleNamespace

        return SimpleNamespace(model=object(), kv_cache_quant=None)

    def test_rejects_when_over_limit(self, monkeypatch):
        from olmlx.engine import inference

        monkeypatch.setattr(
            inference.memory_utils, "get_system_memory_bytes", lambda: 100 * 1024**3
        )
        monkeypatch.setattr(
            inference.memory_utils, "get_metal_memory", lambda: 80 * 1024**3
        )
        monkeypatch.setattr(
            inference,
            "estimate_kv_cache_bytes",
            lambda model, n, kv_cache_quant=None: 30 * 1024**3,
        )
        with pytest.raises(MemoryError, match="prompt too long"):
            inference._batched_kv_preflight(self._lm(), 50_000, 1024)

    def test_allows_when_fits(self, monkeypatch):
        from olmlx.engine import inference

        monkeypatch.setattr(
            inference.memory_utils, "get_system_memory_bytes", lambda: 100 * 1024**3
        )
        monkeypatch.setattr(
            inference.memory_utils, "get_metal_memory", lambda: 10 * 1024**3
        )
        monkeypatch.setattr(
            inference,
            "estimate_kv_cache_bytes",
            lambda model, n, kv_cache_quant=None: 5 * 1024**3,
        )
        inference._batched_kv_preflight(self._lm(), 50_000, 1024)

    def test_estimator_failure_degrades_open(self, monkeypatch):
        """Estimation failure must not block requests (matches the
        exclusive preflight's warn-and-continue)."""
        from olmlx.engine import inference

        monkeypatch.setattr(
            inference.memory_utils, "get_system_memory_bytes", lambda: 100 * 1024**3
        )

        def boom(model, n, kv_cache_quant=None):
            raise RuntimeError("no estimate")

        monkeypatch.setattr(inference, "estimate_kv_cache_bytes", boom)
        inference._batched_kv_preflight(self._lm(), 50_000, 1024)


# ---------------------------------------------------------------------------
# Eligibility predicate
# ---------------------------------------------------------------------------


def _eligible_lm(**overrides):
    """A LoadedModel-shaped stub that passes every eligibility check."""
    from types import SimpleNamespace

    base = dict(
        name="stub",
        is_vlm=False,
        is_whisper=False,
        is_tts=False,
        is_reranker=False,
        is_distributed=False,
        is_flash=False,
        is_flash_moe=False,
        is_speculative=False,
        kv_cache_quant=None,
        template_caps=SimpleNamespace(has_channel_format=False),
        batch_convertible=True,  # preset: skip the model probe
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestBatchEligible:
    @pytest.fixture(autouse=True)
    def _batching_on(self, monkeypatch):
        from olmlx.config import settings

        monkeypatch.setattr(settings, "batching", True)

    def _eligible(self, lm, gen_kwargs=None, **kw):
        from olmlx.engine.inference import _batch_eligible

        defaults = dict(max_tokens=64, images=None, audio=None)
        defaults.update(kw)
        return _batch_eligible(lm, gen_kwargs or {}, **defaults)

    def test_eligible_baseline(self):
        assert self._eligible(_eligible_lm())

    def test_disabled_by_default(self, monkeypatch):
        from olmlx.config import settings

        monkeypatch.setattr(settings, "batching", False)
        assert not self._eligible(_eligible_lm())

    def test_mock_settings_do_not_enable(self, monkeypatch):
        """Tests across the suite patch inference.settings with a
        MagicMock; its truthy .batching must not engage the batch path."""
        from unittest.mock import MagicMock

        from olmlx.engine import inference

        monkeypatch.setattr(inference, "settings", MagicMock())
        assert not self._eligible(_eligible_lm())

    def test_mock_model_probe_is_false(self):
        """A Mock model's make_cache() iterates empty — the probe must
        memoize False, never True."""
        from unittest.mock import MagicMock

        lm = _eligible_lm(batch_convertible=None, model=MagicMock())
        assert not self._eligible(lm)
        assert lm.batch_convertible is False

    @pytest.mark.parametrize(
        "flag",
        [
            "is_vlm",
            "is_whisper",
            "is_tts",
            "is_reranker",
            "is_distributed",
            "is_flash",
            "is_flash_moe",
            "is_speculative",
        ],
    )
    def test_model_kind_flags_disqualify(self, flag):
        assert not self._eligible(_eligible_lm(**{flag: True}))

    def test_kv_quant_disqualifies(self):
        assert not self._eligible(_eligible_lm(kv_cache_quant="turboquant:4"))

    def test_images_audio_disqualify(self):
        assert not self._eligible(_eligible_lm(), images=["img"])
        assert not self._eligible(_eligible_lm(), audio=["clip"])

    def test_grammar_processors_do_not_disqualify(self):
        """Phase 2 lifts the grammar exclusion: per-sequence processors
        ride logits_processors into the batch (GenerationBatch._step
        calls them with [1, vocab] rows + per-sequence token history)."""
        assert self._eligible(
            _eligible_lm(), {"logits_processors": [lambda toks, logits: logits]}
        )

    def test_seed_disqualifies(self):
        assert not self._eligible(_eligible_lm(), {"seed": 42})

    @pytest.mark.parametrize("mt", [0, -1, -2])
    def test_nonpositive_max_tokens_disqualifies(self, mt):
        """Ollama num_predict -1/-2 mean unlimited; BatchGenerator's
        ``generated >= max_tokens`` would finish after one token, so these
        must stay on the exclusive path (stream_generate treats them as
        infinite)."""
        assert not self._eligible(_eligible_lm(), max_tokens=mt)

    def test_channel_format_disqualifies(self):
        from types import SimpleNamespace

        lm = _eligible_lm(template_caps=SimpleNamespace(has_channel_format=True))
        assert not self._eligible(lm)

    def test_probe_failure_memoized_false(self):
        # batch_convertible=None forces the probe; the stub has no .model
        # so _make_prompt_cache_for_lm raises → memoized False.
        lm = _eligible_lm(batch_convertible=None)
        assert not self._eligible(lm)
        assert lm.batch_convertible is False

    def test_probe_result_memoized(self):
        lm = _eligible_lm(batch_convertible=False)
        assert not self._eligible(lm)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestBatchingSettings:
    def test_defaults(self, monkeypatch):
        for var in (
            "OLMLX_BATCHING",
            "OLMLX_BATCH_COMPLETION_SIZE",
            "OLMLX_BATCH_PREFILL_SIZE",
            "OLMLX_BATCH_PREFILL_STEP",
        ):
            monkeypatch.delenv(var, raising=False)
        s = Settings()
        assert s.batching is False
        assert s.batch_completion_size == 8
        assert s.batch_prefill_size == 4
        assert s.batch_prefill_step == 2048

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_BATCHING", "true")
        monkeypatch.setenv("OLMLX_BATCH_COMPLETION_SIZE", "16")
        monkeypatch.setenv("OLMLX_BATCH_PREFILL_SIZE", "2")
        monkeypatch.setenv("OLMLX_BATCH_PREFILL_STEP", "1024")
        s = Settings()
        assert s.batching is True
        assert s.batch_completion_size == 16
        assert s.batch_prefill_size == 2
        assert s.batch_prefill_step == 1024

    @pytest.mark.parametrize(
        "var",
        [
            "OLMLX_BATCH_COMPLETION_SIZE",
            "OLMLX_BATCH_PREFILL_SIZE",
            "OLMLX_BATCH_PREFILL_STEP",
        ],
    )
    def test_sizes_reject_zero(self, monkeypatch, var):
        monkeypatch.setenv(var, "0")
        with pytest.raises(ValueError):
            Settings()
