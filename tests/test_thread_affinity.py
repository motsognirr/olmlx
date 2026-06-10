"""Tests for the thread-affinity tripwires that make the implicit
"all mutations happen on one thread" contract explicit (issue #463).

Three mutation-affinity contracts are enforced:

- ``ThreadAffinityGuard``: registry / model-manager / prompt-cache-store
  mutations all happen on a single thread (the asyncio event-loop thread in
  the server, the main thread in the CLI and tests). The guard pins the
  owning thread on first mutation and raises on any later cross-thread call.
- ``SerializedMutationGuard``: ``_SpecCacheStore`` mutations are serialized
  by the inference lock but legitimately roam across decode worker threads,
  so the tripwire there forbids *overlap*, not thread changes.
"""

import asyncio
import threading

import pytest

from olmlx.engine.model_manager import (
    CachedPromptState,
    ModelManager,
    PromptCacheStore,
)
from olmlx.engine.speculative import _SpecCacheStore
from olmlx.utils.affinity import SerializedMutationGuard, ThreadAffinityGuard


def _run_in_thread(fn):
    """Run ``fn`` in a fresh thread, re-raising any exception here."""
    raised: list[BaseException] = []

    def _target():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            raised.append(exc)

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    if raised:
        raise raised[0]


class TestThreadAffinityGuard:
    def test_repeated_same_thread_calls_pass(self):
        guard = ThreadAffinityGuard("test guard")
        guard.check()
        guard.check()

    def test_cross_thread_call_raises(self):
        guard = ThreadAffinityGuard("test guard")
        guard.check()  # pin to this thread
        with pytest.raises(RuntimeError, match="test guard"):
            _run_in_thread(guard.check)

    def test_pins_to_first_calling_thread(self):
        guard = ThreadAffinityGuard("test guard")
        _run_in_thread(guard.check)  # pin to the worker thread
        with pytest.raises(RuntimeError, match="test guard"):
            guard.check()


class TestSerializedMutationGuard:
    def test_sequential_entries_pass(self):
        guard = SerializedMutationGuard("test store")
        with guard:
            pass
        with guard:
            pass

    def test_sequential_entries_from_different_threads_pass(self):
        guard = SerializedMutationGuard("test store")
        with guard:
            pass

        def _enter():
            with guard:
                pass

        _run_in_thread(_enter)  # thread roaming is allowed

    def test_overlapping_entry_raises(self):
        guard = SerializedMutationGuard("test store")
        with guard:
            with pytest.raises(RuntimeError, match="test store"):
                with guard:
                    pass

    def test_released_after_exception(self):
        guard = SerializedMutationGuard("test store")
        with pytest.raises(ValueError):
            with guard:
                raise ValueError("boom")
        with guard:  # must not raise — the lock was released
            pass


class TestRegistryAffinity:
    def test_add_mapping_off_thread_raises(self, registry, tmp_path, monkeypatch):
        models_json = tmp_path / "models2.json"
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_json)
        registry.add_mapping("pinned", "org/pinned-model")  # pin this thread
        with pytest.raises(RuntimeError, match="ModelRegistry"):
            _run_in_thread(lambda: registry.add_mapping("other", "org/other-model"))
        # The mapping must not have been applied.
        assert registry.resolve("other") is None

    def test_remove_off_thread_raises(self, registry, tmp_path, monkeypatch):
        models_json = tmp_path / "models2.json"
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_json)
        registry.add_mapping("pinned", "org/pinned-model")
        with pytest.raises(RuntimeError, match="ModelRegistry"):
            _run_in_thread(lambda: registry.remove("pinned"))
        assert registry.resolve("pinned") is not None

    def test_add_alias_off_thread_raises(self, registry, tmp_path):
        registry._aliases_path = tmp_path / "aliases.json"
        registry.add_alias("q3", "qwen3")  # pin this thread
        with pytest.raises(RuntimeError, match="ModelRegistry"):
            _run_in_thread(lambda: registry.add_alias("q3b", "qwen3"))


class TestPromptCacheStoreAffinity:
    def test_set_off_thread_raises(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", CachedPromptState(tokens=[1], cache=["c1"]))
        with pytest.raises(RuntimeError, match="PromptCacheStore"):
            _run_in_thread(
                lambda: store.set("b", CachedPromptState(tokens=[2], cache=["c2"]))
            )
        assert store.peek("b") is None

    def test_get_off_thread_raises(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", CachedPromptState(tokens=[1], cache=["c1"]))
        with pytest.raises(RuntimeError, match="PromptCacheStore"):
            _run_in_thread(lambda: store.get("a"))

    def test_remove_off_thread_raises(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", CachedPromptState(tokens=[1], cache=["c1"]))
        with pytest.raises(RuntimeError, match="PromptCacheStore"):
            _run_in_thread(lambda: store.remove("a"))
        assert store.peek("a") is not None

    def test_clear_off_thread_is_allowed(self):
        # _close_loaded_model runs clear() in an asyncio.to_thread worker
        # during model teardown, after the LoadedModel is popped from
        # _loaded — exclusive access is guaranteed, so clear() is exempt
        # from the affinity tripwire.
        store = PromptCacheStore(max_slots=4)
        store.set("a", CachedPromptState(tokens=[1], cache=["c1"]))
        _run_in_thread(store.clear)
        assert len(store) == 0


class TestModelManagerAffinity:
    def test_unload_off_thread_raises(self, registry):
        manager = ModelManager(registry)
        # Pin on this thread: unload of a model that isn't loaded returns
        # False but still passes through the guarded entry point.
        assert asyncio.run(manager.unload("qwen3")) is False
        with pytest.raises(RuntimeError, match="ModelManager"):
            _run_in_thread(lambda: asyncio.run(manager.unload("qwen3")))

    def test_expire_stale_off_thread_raises(self, registry):
        manager = ModelManager(registry)
        asyncio.run(manager._expire_stale())  # pin on this thread
        with pytest.raises(RuntimeError, match="ModelManager"):
            _run_in_thread(lambda: asyncio.run(manager._expire_stale()))


class TestSpecCacheStoreSerialization:
    def test_sequential_cross_thread_use_passes(self):
        # The spec store's mutations run on a different decode worker thread
        # each request (serialized by the inference lock) — thread roaming
        # must NOT trip the guard, only overlapping access.
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 2, 3], "payload-a")
        _run_in_thread(lambda: store.insert([4, 5, 6], "payload-b"))
        _run_in_thread(lambda: store.find([1, 2, 3]))
        assert store.find([4, 5, 6]) is not None

    def test_overlapping_mutation_raises(self):
        store = _SpecCacheStore(capacity=2)
        with store._guard:  # simulate an in-flight mutation
            with pytest.raises(RuntimeError, match="_SpecCacheStore"):
                store.insert([1, 2, 3], "payload")

    def test_overlapping_find_raises(self):
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 2, 3], "payload")
        with store._guard:
            with pytest.raises(RuntimeError, match="_SpecCacheStore"):
                store.find([1, 2, 3])
