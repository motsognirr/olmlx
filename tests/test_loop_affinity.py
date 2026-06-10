"""Tests for the explicit event-loop-affinity contract (issue #463).

Registry mappings, the model manager's loaded dict, and the prompt-cache
stores are mutated without locks on the assumption that every mutator runs
on the single asyncio event-loop thread.  ``olmlx.utils.loop_affinity``
makes that contract enforced: the app lifespan binds the loop thread and
mutating entry points raise immediately when called from any other thread.

The speculative ``_SpecCacheStore`` is the exception: it legitimately runs
on the decode worker thread, serialized by the inference lock — its contract
is "no concurrent access", enforced by a non-blocking guard.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from olmlx.engine.model_manager import CachedPromptState, PromptCacheStore
from olmlx.engine.prompt_cache.vlm_state import VlmPromptCacheStore
from olmlx.engine.speculative import _SpecCacheStore
from olmlx.utils import loop_affinity


@pytest.fixture(autouse=True)
def _unbound():
    """Every test starts unbound and leaves no binding behind."""
    loop_affinity.unbind_loop_thread()
    yield
    loop_affinity.unbind_loop_thread()


def _run_in_thread(fn):
    """Run ``fn()`` in a worker thread; return the exception it raised (or None)."""
    holder: list = [None]

    def runner():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 — captured for assertion
            holder[0] = exc

    t = threading.Thread(target=runner)
    t.start()
    t.join()
    return holder[0]


def _make_state(token_id: int = 1) -> CachedPromptState:
    return CachedPromptState(tokens=[token_id], cache=[f"cache_{token_id}"])


class TestAssertLoopThread:
    def test_noop_when_unbound(self):
        loop_affinity.assert_loop_thread("op")
        assert _run_in_thread(lambda: loop_affinity.assert_loop_thread("op")) is None

    def test_bound_same_thread_passes(self):
        loop_affinity.bind_loop_thread()
        loop_affinity.assert_loop_thread("op")

    def test_bound_other_thread_raises_naming_operation(self):
        loop_affinity.bind_loop_thread()
        exc = _run_in_thread(
            lambda: loop_affinity.assert_loop_thread("ModelRegistry.add_mapping")
        )
        assert isinstance(exc, RuntimeError)
        assert "ModelRegistry.add_mapping" in str(exc)

    def test_unbind_restores_noop(self):
        loop_affinity.bind_loop_thread()
        loop_affinity.unbind_loop_thread()
        assert _run_in_thread(lambda: loop_affinity.assert_loop_thread("op")) is None


class TestRegistryAffinity:
    def test_add_mapping_off_loop_raises_before_mutating(self, registry):
        loop_affinity.bind_loop_thread()
        exc = _run_in_thread(lambda: registry.add_mapping("newmodel", "org/new"))
        assert isinstance(exc, RuntimeError)
        assert "newmodel:latest" not in registry._mappings

    def test_remove_off_loop_raises_before_mutating(self, registry):
        loop_affinity.bind_loop_thread()
        exc = _run_in_thread(lambda: registry.remove("qwen3:latest"))
        assert isinstance(exc, RuntimeError)
        assert "qwen3:latest" in registry._mappings

    def test_add_alias_off_loop_raises(self, registry):
        loop_affinity.bind_loop_thread()
        exc = _run_in_thread(lambda: registry.add_alias("ali", "qwen3:latest"))
        assert isinstance(exc, RuntimeError)
        assert "ali:latest" not in registry._aliases

    def test_add_mapping_on_bound_thread_ok(self, registry):
        loop_affinity.bind_loop_thread()
        registry.add_mapping("newmodel", "org/new")
        assert "newmodel:latest" in registry._mappings

    def test_unbound_cli_style_off_thread_ok(self, registry):
        # CLI processes (`olmlx models pull`) never bind, so mutation from
        # whatever thread they use stays allowed.
        assert (
            _run_in_thread(lambda: registry.add_mapping("newmodel", "org/new")) is None
        )


class TestModelManagerAffinity:
    def test_unload_off_loop_raises_before_pop(self, mock_manager):
        loop_affinity.bind_loop_thread()
        exc = _run_in_thread(lambda: asyncio.run(mock_manager.unload("qwen3:latest")))
        assert isinstance(exc, RuntimeError)
        assert "event-loop thread" in str(exc)
        assert "qwen3:latest" in mock_manager._loaded

    def test_expire_stale_off_loop_raises(self, mock_manager):
        loop_affinity.bind_loop_thread()
        exc = _run_in_thread(lambda: asyncio.run(mock_manager._expire_stale()))
        assert isinstance(exc, RuntimeError)
        assert "event-loop thread" in str(exc)


class TestPromptCacheStoreAffinity:
    def test_mutators_off_loop_raise(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state(1))
        loop_affinity.bind_loop_thread()
        mutators = [
            ("set", lambda: store.set("b", _make_state(2))),
            ("get", lambda: store.get("a")),
            ("remove", lambda: store.remove("a")),
            ("evict_all_to_disk", lambda: store.evict_all_to_disk()),
            ("takeover", lambda: store.takeover("a", "b")),
            ("fetch_nearest", lambda: store.fetch_nearest([1, 2])),
            ("insert_checkpoint", lambda: store.insert_checkpoint(_make_state(3))),
        ]
        for name, fn in mutators:
            exc = _run_in_thread(fn)
            assert isinstance(exc, RuntimeError), f"{name} did not raise off-loop"
        # Nothing mutated: the original entry is intact and alone.
        assert store.get("a") is not None
        assert len(store) == 1

    def test_async_mutators_off_loop_raise(self):
        store = PromptCacheStore(max_slots=4)
        loop_affinity.bind_loop_thread()
        for name, coro_fn in [
            ("async_set", lambda: store.async_set("a", _make_state(1))),
            ("async_get", lambda: store.async_get("a")),
            ("async_evict_all_to_disk", lambda: store.async_evict_all_to_disk()),
        ]:
            exc = _run_in_thread(lambda fn=coro_fn: asyncio.run(fn()))
            assert isinstance(exc, RuntimeError), f"{name} did not raise off-loop"

    def test_mutators_on_bound_thread_ok(self):
        store = PromptCacheStore(max_slots=4)
        loop_affinity.bind_loop_thread()
        store.set("a", _make_state(1))
        assert store.get("a") is not None
        store.remove("a")
        assert store.get("a") is None

    def test_clear_allowed_off_loop(self):
        # clear()'s only production caller is _close_loaded_model, which runs
        # on a worker thread via asyncio.to_thread during unload/eviction/
        # expiry — by then the LoadedModel is popped from _loaded, so the
        # closer owns the store exclusively and loop affinity is not its
        # contract.
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state(1))
        loop_affinity.bind_loop_thread()
        assert _run_in_thread(lambda: store.clear()) is None
        assert len(store) == 0


class TestVlmPromptCacheStoreAffinity:
    def test_mutators_off_loop_raise(self):
        store = VlmPromptCacheStore(capacity=2)
        store.insert("a", object())
        loop_affinity.bind_loop_thread()
        for name, fn in [
            ("insert", lambda: store.insert("b", object())),
            ("get", lambda: store.get("a")),
        ]:
            exc = _run_in_thread(fn)
            assert isinstance(exc, RuntimeError), f"{name} did not raise off-loop"
        loop_affinity.unbind_loop_thread()
        assert store.get("a") is not None

    def test_mutators_on_bound_thread_ok(self):
        store = VlmPromptCacheStore(capacity=2)
        loop_affinity.bind_loop_thread()
        store.insert("a", object())
        assert store.get("a") is not None

    def test_clear_allowed_off_loop(self):
        # Same exclusive-ownership contract as PromptCacheStore.clear():
        # called from the worker-thread close path in _close_loaded_model.
        store = VlmPromptCacheStore(capacity=2)
        store.insert("a", object())
        loop_affinity.bind_loop_thread()
        assert _run_in_thread(lambda: store.clear()) is None


class TestWorkerThreadClosePath:
    def test_close_loaded_model_clears_stores_off_loop(
        self, mock_manager, mock_loaded_model
    ):
        """Regression: _close_loaded_model runs via asyncio.to_thread on every
        unload/eviction/expiry and clears both prompt-cache stores there.  A
        loop-affinity assert on clear() turns every production unload into a
        logged error that leaves the cache (incl. disk entries) uncleared."""
        mock_loaded_model.prompt_cache_store.set("a", _make_state(1))
        loop_affinity.bind_loop_thread()
        exc = _run_in_thread(
            lambda: mock_manager._close_loaded_model(mock_loaded_model)
        )
        assert exc is None  # an ExceptionGroup here means a close step failed
        # clear() succeeded, so the store was nulled (success-only nulling).
        assert mock_loaded_model.prompt_cache_store is None


class TestSpecCacheStoreSerializedAccess:
    def test_sequential_cross_thread_access_ok(self):
        # The store runs on the decode worker thread, serialized by the
        # inference lock — sequential access from different threads is legal
        # even when the loop thread is bound.
        loop_affinity.bind_loop_thread()
        store = _SpecCacheStore(capacity=2)
        assert _run_in_thread(lambda: store.insert([1, 2, 3], "payload")) is None
        hit = store.find([1, 2, 3, 4])
        assert hit is not None
        entry, common = hit
        assert common == 3

    def test_concurrent_access_raises(self):
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 2, 3], "payload")
        with store._serialized():
            exc = _run_in_thread(lambda: store.find([1, 2, 3]))
        assert isinstance(exc, RuntimeError)
        assert "serialized" in str(exc)
        # Guard released — access works again.
        assert store.find([1, 2, 3]) is not None


class TestLifespanBinding:
    @pytest.mark.asyncio
    async def test_lifespan_binds_and_unbinds(self, tmp_path, monkeypatch):
        from olmlx.app import lifespan

        monkeypatch.setattr("olmlx.app.settings.models_dir", tmp_path / "models")
        monkeypatch.setattr(
            "olmlx.app.settings.models_config", tmp_path / "models.json"
        )
        (tmp_path / "models.json").write_text("{}")

        app = MagicMock()
        app.state = MagicMock()

        async with lifespan(app):
            # Bound to the loop thread: passes here, raises off-thread.
            loop_affinity.assert_loop_thread("op")
            exc = _run_in_thread(lambda: loop_affinity.assert_loop_thread("op"))
            assert isinstance(exc, RuntimeError)
        # Unbound after shutdown so repeated create_app() in tests is clean.
        assert _run_in_thread(lambda: loop_affinity.assert_loop_thread("op")) is None
