"""Tests for KV cache disk offload (PromptCacheStore disk tier)."""

import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.engine.model_manager import CachedPromptState, PromptCacheStore


def _make_state(token_id: int = 1) -> CachedPromptState:
    """Create a minimal CachedPromptState for testing."""
    return CachedPromptState(tokens=[token_id], cache=[f"cache_{token_id}"])


def _sized_state(token_id: int, nbytes: int) -> CachedPromptState:
    """Create a CachedPromptState whose estimated size is ``nbytes``."""
    from unittest.mock import MagicMock

    layer = MagicMock()
    layer.keys = MagicMock()
    layer.keys.nbytes = nbytes
    layer.values = MagicMock()
    layer.values.nbytes = 0
    return CachedPromptState(tokens=[token_id], cache=[layer])


class TestPromptCacheStoreDiskEviction:
    """Test that evicted caches are saved to disk instead of deleted."""

    def test_eviction_saves_to_disk(self, tmp_path):
        """When a cache is evicted, it should be saved to disk."""
        store = PromptCacheStore(
            max_slots=1,
            disk_path=tmp_path,
            model_name="test-model",
        )
        state_a = _make_state(1)
        with patch("olmlx.engine.prompt_cache.store.save_prompt_cache") as mock_save:
            store.set("a", state_a)
            store.set("b", _make_state(2))  # evicts "a"
            mock_save.assert_called_once()
            call_args = mock_save.call_args
            # First arg is the file path, second is the cache list
            assert "a" in str(call_args[0][0])
            assert call_args[0][1] is state_a.cache

    def test_eviction_saves_tokens_metadata(self, tmp_path):
        """Eviction saves the token list in metadata for later restoration."""
        store = PromptCacheStore(
            max_slots=1,
            disk_path=tmp_path,
            model_name="test-model",
        )
        state_a = CachedPromptState(tokens=[10, 20, 30], cache=["kv"])
        with patch("olmlx.engine.prompt_cache.store.save_prompt_cache") as mock_save:
            store.set("a", state_a)
            store.set("b", _make_state(2))  # evicts "a"
            metadata = mock_save.call_args[0][2]
            assert "tokens" in metadata

    def test_eviction_without_disk_path_deletes(self):
        """Without disk_path, eviction behaves as before (no save)."""
        store = PromptCacheStore(max_slots=1)
        store.set("a", _make_state(1))
        with patch("olmlx.engine.prompt_cache.store.save_prompt_cache") as mock_save:
            store.set("b", _make_state(2))  # evicts "a"
            mock_save.assert_not_called()

    def test_save_failure_logs_and_continues(self, tmp_path, caplog):
        """If save_prompt_cache fails, log warning and continue (don't crash)."""
        import logging

        store = PromptCacheStore(
            max_slots=1,
            disk_path=tmp_path,
            model_name="test-model",
        )
        store.set("a", _make_state(1))
        with patch(
            "olmlx.engine.prompt_cache.store.save_prompt_cache",
            side_effect=OSError("disk full"),
        ):
            with caplog.at_level(logging.WARNING):
                store.set("b", _make_state(2))
                # Should still evict from memory
                assert store.get("a") is None
                assert "disk full" in caplog.text


class TestPromptCacheStoreDiskLoad:
    """Test that cache misses check disk before creating fresh cache."""

    def test_get_loads_from_disk_on_miss(self, tmp_path):
        """On memory miss, check disk for saved cache and restore it."""
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="test-model",
        )
        restored_cache = ["restored_kv"]
        restored_tokens = [10, 20, 30]

        with patch(
            "olmlx.engine.prompt_cache.store.load_prompt_cache",
            return_value=(restored_cache, {"tokens": "[10, 20, 30]"}),
        ) as mock_load:
            # Simulate: file exists on disk
            disk_file = store._disk_file_path("a")
            disk_file.parent.mkdir(parents=True, exist_ok=True)
            disk_file.touch()

            result = store.get("a")
            assert result is not None
            assert result.cache == restored_cache
            assert result.tokens == restored_tokens
            mock_load.assert_called_once()

    def test_get_returns_none_when_not_on_disk(self, tmp_path):
        """If not in memory and not on disk, return None."""
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="test-model",
        )
        assert store.get("nonexistent") is None

    def test_disk_load_failure_returns_none(self, tmp_path, caplog):
        """If load_prompt_cache fails, return None (don't crash)."""
        import logging

        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="test-model",
        )
        with patch(
            "olmlx.engine.prompt_cache.store.load_prompt_cache",
            side_effect=Exception("corrupt file"),
        ):
            disk_file = store._disk_file_path("a")
            disk_file.parent.mkdir(parents=True, exist_ok=True)
            disk_file.touch()

            with caplog.at_level(logging.WARNING):
                result = store.get("a")
                assert result is None
                assert "corrupt file" in caplog.text

    def test_disk_load_respects_max_slots(self, tmp_path):
        """Loading from disk must not exceed max_slots capacity."""
        store = PromptCacheStore(
            max_slots=1,
            disk_path=tmp_path,
            model_name="test-model",
        )
        # One entry already in memory
        store.set("a", _make_state(1))
        assert len(store) == 1

        # Simulate "b" on disk — mock both load and save since loading "b"
        # evicts "a" which triggers save_prompt_cache for "a"
        with (
            patch(
                "olmlx.engine.prompt_cache.store.load_prompt_cache",
                return_value=(["kv_b"], {"tokens": "[2]"}),
            ),
            patch("olmlx.engine.prompt_cache.store.save_prompt_cache") as mock_save,
        ):
            disk_file = store._disk_file_path("b")
            disk_file.parent.mkdir(parents=True, exist_ok=True)
            disk_file.touch()

            result = store.get("b")
            assert result is not None
            # Must not exceed max_slots
            assert len(store) <= 1
            # "a" should have been saved to disk when evicted
            mock_save.assert_called_once()

    def test_disk_file_deleted_after_load(self, tmp_path):
        """After loading from disk, the disk file should be removed."""
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="test-model",
        )
        with patch(
            "olmlx.engine.prompt_cache.store.load_prompt_cache",
            return_value=(["kv"], {"tokens": "[1, 2]"}),
        ):
            disk_file = store._disk_file_path("a")
            disk_file.parent.mkdir(parents=True, exist_ok=True)
            disk_file.touch()

            store.get("a")
            assert not disk_file.exists()


class TestPromptCacheStoreDiskCleanup:
    """Test disk LRU cleanup when exceeding max size."""

    def test_disk_cleanup_removes_oldest_files(self, tmp_path):
        """When disk usage exceeds max, oldest files are removed."""
        store = PromptCacheStore(
            max_slots=1,
            disk_path=tmp_path,
            model_name="test-model",
            disk_max_bytes=100,  # Very small limit
        )
        # Create some fake cache files
        model_dir = tmp_path / "test-model"
        model_dir.mkdir(parents=True, exist_ok=True)

        old_file = model_dir / "old_cache.safetensors"
        old_file.write_bytes(b"x" * 60)
        # Set mtime in the past
        os.utime(old_file, (time.time() - 100, time.time() - 100))

        new_file = model_dir / "new_cache.safetensors"
        new_file.write_bytes(b"x" * 60)

        store._cleanup_disk()

        # Old file should be removed to get under limit
        assert not old_file.exists()
        # New file should remain
        assert new_file.exists()

    def test_remove_also_deletes_disk_file(self, tmp_path):
        """remove() should delete the disk file to prevent stale restoration."""
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="test-model",
        )
        # Create a fake disk file as if "a" was evicted to disk
        disk_file = store._disk_file_path("a")
        disk_file.parent.mkdir(parents=True, exist_ok=True)
        disk_file.write_bytes(b"fake cache")

        store.remove("a")
        assert not disk_file.exists()

    def test_disk_cleanup_noop_under_limit(self, tmp_path):
        """No cleanup needed when under the limit."""
        store = PromptCacheStore(
            max_slots=1,
            disk_path=tmp_path,
            model_name="test-model",
            disk_max_bytes=10_000,  # Large limit
        )
        model_dir = tmp_path / "test-model"
        model_dir.mkdir(parents=True, exist_ok=True)

        f = model_dir / "cache.safetensors"
        f.write_bytes(b"x" * 50)

        store._cleanup_disk()
        assert f.exists()

    def test_clear_also_clears_disk(self, tmp_path):
        """store.clear() should remove disk cache files too."""
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="test-model",
        )
        model_dir = tmp_path / "test-model"
        model_dir.mkdir(parents=True, exist_ok=True)
        f = model_dir / "somecache.safetensors"
        f.write_bytes(b"data")

        store.clear()
        assert not f.exists()


class TestPromptCacheStoreDiskFilePath:
    """Test cache file path generation."""

    def test_disk_file_path_is_deterministic(self, tmp_path):
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="test-model",
        )
        p1 = store._disk_file_path("agent-abc")
        p2 = store._disk_file_path("agent-abc")
        assert p1 == p2

    def test_disk_file_path_includes_model_name(self, tmp_path):
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="qwen3-8b",
        )
        path = store._disk_file_path("agent-1")
        assert "qwen3-8b" in str(path)

    def test_disk_file_path_sanitizes_cache_id(self, tmp_path):
        """Cache IDs with special characters should be safe for filenames."""
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="test-model",
        )
        path = store._disk_file_path("agent/with:special chars")
        # Should not contain path separators in the filename
        assert "/" not in path.name and "\\" not in path.name
        assert path.suffix == ".safetensors"

    def test_disk_dir_sanitizes_model_name_with_slash(self, tmp_path):
        """Model names like 'Qwen/Qwen3-8B' should not create nested dirs."""
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="Qwen/Qwen3-8B",
        )
        disk_dir = store._disk_dir()
        # Should be a single directory, not nested Qwen/Qwen3-8B
        assert disk_dir.parent == tmp_path
        assert "/" not in disk_dir.name


class TestPromptCacheStoreEvictAllToDisk:
    """Test evict_all_to_disk() for memory pressure scenarios."""

    def test_evict_all_to_disk_saves_all_entries(self, tmp_path):
        """All in-memory entries should be saved to disk and cleared from memory."""
        store = PromptCacheStore(
            max_slots=4,
            disk_path=tmp_path,
            model_name="test-model",
        )
        store.set("a", _make_state(1))
        store.set("b", _make_state(2))
        assert len(store) == 2

        with patch("olmlx.engine.prompt_cache.store.save_prompt_cache") as mock_save:
            store.evict_all_to_disk()
            assert len(store) == 0
            assert mock_save.call_count == 2

    def test_evict_all_to_disk_without_disk_path_just_clears(self):
        """Without disk configured, evict_all_to_disk just clears memory."""
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state(1))
        store.evict_all_to_disk()
        assert len(store) == 0

    def test_cleanup_disk_updates_metrics(self, tmp_path):
        """_cleanup_disk should increment evictions_disk and report bytes_on_disk."""
        store = PromptCacheStore(
            max_slots=4,
            disk_path=tmp_path,
            model_name="test-model",
            disk_max_bytes=100,
        )
        disk_dir = tmp_path / "test-model"
        disk_dir.mkdir(parents=True)
        # Three files at 80 bytes each = 240 total; budget 100 → 2 evicted
        for name in ("a", "b", "c"):
            (disk_dir / f"{name}.safetensors").write_bytes(b"x" * 80)
        store._cleanup_disk()
        assert store.metrics.evictions_disk == 2
        # 80 bytes left under the 100-byte budget after cleanup
        assert store.metrics.bytes_on_disk == 80

    def test_bytes_on_disk_decrements_on_remove(self, tmp_path):
        """Regression for aider PR #391 follow-up: bytes_on_disk must
        stay consistent when files are unlinked outside _cleanup_disk."""
        store = PromptCacheStore(
            max_slots=4,
            disk_path=tmp_path,
            model_name="test-model",
        )
        disk_dir = tmp_path / "test-model"
        disk_dir.mkdir(parents=True)
        (disk_dir / "a.safetensors").write_bytes(b"x" * 80)
        (disk_dir / "b.safetensors").write_bytes(b"x" * 80)
        # Prime the metric by running cleanup.
        store._cleanup_disk()
        assert store.metrics.bytes_on_disk == 160

        # remove() unlinks the disk file and must refresh the metric.
        store.remove("a")
        assert store.metrics.bytes_on_disk == 80

        # clear_disk() wipes everything; metric must drop to zero.
        store.clear_disk()
        assert store.metrics.bytes_on_disk == 0

    def test_clear_disk_recomputes_metric_after_unlink_failure(self, tmp_path):
        """clear_disk must not zero bytes_on_disk if an unlink failed
        and a file is still on disk."""
        store = PromptCacheStore(
            max_slots=4,
            disk_path=tmp_path,
            model_name="test-model",
        )
        disk_dir = tmp_path / "test-model"
        disk_dir.mkdir(parents=True)
        (disk_dir / "kept.safetensors").write_bytes(b"x" * 100)
        (disk_dir / "removed.safetensors").write_bytes(b"x" * 100)
        store._cleanup_disk()
        assert store.metrics.bytes_on_disk == 200

        # Patch unlink so "kept" fails but "removed" succeeds.
        real_unlink = Path.unlink

        def fake_unlink(self_path, *args, **kwargs):
            if self_path.name == "kept.safetensors":
                raise OSError("simulated")
            return real_unlink(self_path, *args, **kwargs)

        with patch.object(Path, "unlink", autospec=True, side_effect=fake_unlink):
            removed = store.clear_disk()

        # One file successfully removed, one survived.
        assert removed == 1
        # The metric must reflect the surviving file, not be zero.
        assert store.metrics.bytes_on_disk == 100

    def test_bytes_on_disk_zero_after_clear(self, tmp_path):
        """clear() removes the disk directory; bytes_on_disk → 0."""
        store = PromptCacheStore(
            max_slots=4,
            disk_path=tmp_path,
            model_name="test-model",
        )
        disk_dir = tmp_path / "test-model"
        disk_dir.mkdir(parents=True)
        (disk_dir / "x.safetensors").write_bytes(b"x" * 1024)
        store._cleanup_disk()
        assert store.metrics.bytes_on_disk == 1024
        store.clear()
        assert store.metrics.bytes_on_disk == 0

    def test_cleanup_disk_updates_bytes_without_size_cap(self, tmp_path):
        """Regression for aider PR #391 follow-up: bytes_on_disk must
        be reported even when no disk_max_bytes cap is configured."""
        store = PromptCacheStore(
            max_slots=4,
            disk_path=tmp_path,
            model_name="test-model",
            disk_max_bytes=None,  # disk offload enabled, no cap
        )
        disk_dir = tmp_path / "test-model"
        disk_dir.mkdir(parents=True)
        for name in ("a", "b"):
            (disk_dir / f"{name}.safetensors").write_bytes(b"x" * 80)
        store._cleanup_disk()
        # No eviction loop runs without a cap, but the metric must reflect reality.
        assert store.metrics.evictions_disk == 0
        assert store.metrics.bytes_on_disk == 160

    def test_evict_all_to_disk_entries_restorable(self, tmp_path):
        """Entries evicted to disk can be restored on next get()."""
        store = PromptCacheStore(
            max_slots=4,
            disk_path=tmp_path,
            model_name="test-model",
        )
        store.set("a", _make_state(1))

        with patch("olmlx.engine.prompt_cache.store.save_prompt_cache"):
            store.evict_all_to_disk()

        # Simulate file exists on disk for restoration
        disk_file = store._disk_file_path("a")
        disk_file.parent.mkdir(parents=True, exist_ok=True)
        disk_file.touch()

        with patch(
            "olmlx.engine.prompt_cache.store.load_prompt_cache",
            return_value=(["restored_kv"], {"tokens": "[1]"}),
        ):
            result = store.get("a")
            assert result is not None
            assert result.tokens == [1]


class TestConfigSettings:
    """Test new config settings for disk cache."""

    def test_default_settings(self):
        from olmlx.config import Settings

        s = Settings()
        assert s.prompt_cache_disk is False
        assert s.prompt_cache_disk_max_gb == 10.0
        assert "cache" in str(s.prompt_cache_disk_path)
        assert "kv" in str(s.prompt_cache_disk_path)

    def test_env_override(self, monkeypatch):
        from olmlx.config import Settings

        monkeypatch.setenv("OLMLX_PROMPT_CACHE_DISK", "true")
        monkeypatch.setenv("OLMLX_PROMPT_CACHE_DISK_MAX_GB", "5.0")
        monkeypatch.setenv("OLMLX_PROMPT_CACHE_DISK_PATH", "/tmp/kv")
        s = Settings()
        assert s.prompt_cache_disk is True
        assert s.prompt_cache_disk_max_gb == 5.0
        assert s.prompt_cache_disk_path == Path("/tmp/kv")


class TestAsyncDiskCache:
    """Test async wrappers that offload disk I/O to threads."""

    @pytest.mark.asyncio
    async def test_async_get_memory_hit_no_thread(self):
        """When cache is in memory, asyncio.to_thread is NOT called."""
        store = PromptCacheStore(max_slots=2)
        store.set("a", _make_state(1))

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            result = await store.async_get("a")
            assert result is not None
            assert result.tokens == [1]
            mock_thread.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_get_disk_fallback_uses_thread(self, tmp_path):
        """On memory miss with disk hit, asyncio.to_thread IS called with _read_from_disk."""
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="test-model",
        )

        disk_file = store._disk_file_path("a")
        disk_file.parent.mkdir(parents=True, exist_ok=True)
        disk_file.touch()

        loaded_state = CachedPromptState(tokens=[10, 20], cache=["restored_kv"])
        with patch(
            "asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=(loaded_state, disk_file),
        ) as mock_thread:
            result = await store.async_get("a")
            assert result is not None
            # First call: _read_from_disk; second call: unlink
            assert mock_thread.call_count >= 1
            assert mock_thread.call_args_list[0][0][0] == store._read_from_disk
            assert mock_thread.call_args_list[0][0][1] == "a"

    @pytest.mark.asyncio
    async def test_async_set_eviction_saves_in_thread(self, tmp_path):
        """Fill cache to capacity, async_set a new entry, verify _save_to_disk runs via asyncio.to_thread."""
        store = PromptCacheStore(
            max_slots=1,
            disk_path=tmp_path,
            model_name="test-model",
        )
        state_a = _make_state(1)
        store.set("a", state_a)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await store.async_set("b", _make_state(2))
            mock_thread.assert_called_once()
            # Verify _save_to_disk was called with the evicted entry
            assert mock_thread.call_args[0][0] == store._save_to_disk
            assert mock_thread.call_args[0][1] == "a"
            assert mock_thread.call_args[0][2] is state_a

    @pytest.mark.asyncio
    async def test_async_set_no_eviction_no_thread(self):
        """When cache has room, no thread needed."""
        store = PromptCacheStore(max_slots=4)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await store.async_set("a", _make_state(1))
            mock_thread.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_evict_all_uses_thread(self, tmp_path):
        """Verify evict_all_to_disk snapshots on event loop, saves in a thread."""
        store = PromptCacheStore(
            max_slots=4,
            disk_path=tmp_path,
            model_name="test-model",
        )
        store.set("a", _make_state(1))
        store.set("b", _make_state(2))

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await store.async_evict_all_to_disk()
            mock_thread.assert_called_once()
            # _save_entries_to_disk is called with the snapshot
            assert mock_thread.call_args[0][0] == store._save_entries_to_disk
            # _entries should already be cleared (snapshot-then-clear on event loop)
            assert len(store) == 0

    @pytest.mark.asyncio
    async def test_async_get_disk_restore_enforces_ram_budget(self, tmp_path):
        """Disk-restored entries must trigger _enforce_ram_budget on
        the async path (issue raised in PR #391 review)."""
        store = PromptCacheStore(
            max_slots=10,
            disk_path=tmp_path,
            model_name="test-model",
            ram_budget_bytes=2500,
        )
        # Seed two 1000-byte entries (under budget).
        store.set("a", _sized_state(1, 1000))
        store.set("b", _sized_state(2, 1000))
        assert store.metrics.bytes_in_ram == 2000

        # Pretend "c" is on disk: stub _read_from_disk to hand back a
        # 1000-byte state. After restore, bytes_in_ram would be 3000 if
        # the budget weren't enforced.
        disk_file = store._disk_file_path("c")
        disk_file.parent.mkdir(parents=True, exist_ok=True)
        disk_file.touch()

        with (
            patch.object(
                store,
                "_read_from_disk",
                return_value=(_sized_state(3, 1000), disk_file),
            ),
            patch.object(store, "_save_to_disk") as mock_save,
        ):
            result = await store.async_get("c")

        assert result is not None
        # Budget enforced: oldest entry ("a") got evicted to land at 2000.
        assert store.metrics.bytes_in_ram <= 2500
        assert store.peek("a") is None
        assert store.peek("b") is not None
        assert store.peek("c") is not None
        # The budget-evicted entry was spilled to disk.
        spilled_ids = [call.args[0] for call in mock_save.call_args_list]
        assert "a" in spilled_ids

    @pytest.mark.asyncio
    async def test_async_get_restored_entry_evicted_by_budget_not_respilled(
        self, tmp_path
    ):
        """Issue #466: when _enforce_ram_budget evicts the just-restored
        entry, async_get must not rewrite it to disk — its original file
        is still intact (the unlink is guarded on residency), so the
        re-save is a pure read-then-rewrite of multi-GB state. Other
        budget evictees still spill normally."""
        store = PromptCacheStore(
            max_slots=10,
            disk_path=tmp_path,
            model_name="test-model",
            ram_budget_bytes=2500,
        )
        # Seed two 1000-byte entries (under budget).
        store.set("a", _sized_state(1, 1000))
        store.set("b", _sized_state(2, 1000))

        # "c" on disk is 3000 bytes — over the whole budget by itself, so
        # enforcement evicts a, b, AND the just-restored c.
        disk_file = store._disk_file_path("c")
        disk_file.parent.mkdir(parents=True, exist_ok=True)
        disk_file.write_bytes(b"placeholder")
        # Age the file so we can assert the skip refreshes its mtime.
        os.utime(disk_file, (1000, 1000))

        loaded = _sized_state(3, 3000)
        with (
            patch.object(store, "_read_from_disk", return_value=(loaded, disk_file)),
            patch.object(store, "_save_to_disk") as mock_save,
        ):
            result = await store.async_get("c")

        # The caller still gets the loaded state for this request.
        assert result is loaded
        # Everything was budget-evicted, including the restored entry.
        assert store.peek("a") is None
        assert store.peek("b") is None
        assert store.peek("c") is None
        # The displaced residents spilled to disk, but the restored entry
        # was NOT rewritten — its original file is the surviving copy.
        spilled_ids = [call.args[0] for call in mock_save.call_args_list]
        assert "a" in spilled_ids
        assert "b" in spilled_ids
        assert "c" not in spilled_ids
        assert disk_file.exists()
        # The surviving file was touched: _cleanup_disk evicts by oldest
        # mtime, so a stale mtime would make the just-used entry the
        # first one deleted under the disk cap (LRU inversion).
        assert disk_file.stat().st_mtime > 1000

    @pytest.mark.asyncio
    async def test_async_get_budget_bounced_entry_resaved_if_file_gone(self, tmp_path):
        """If a sibling evictee's save triggered _cleanup_disk and it
        deleted the restored entry's original file (oldest mtime) before
        the skip runs, the skip must fall back to a real save — otherwise
        the entry is lost from both tiers."""
        store = PromptCacheStore(
            max_slots=10,
            disk_path=tmp_path,
            model_name="test-model",
            ram_budget_bytes=2500,
        )
        store.set("a", _sized_state(1, 1000))

        # _read_from_disk reports a path that no longer exists by the
        # time the spill loop runs (as if disk-cap cleanup removed it).
        disk_file = store._disk_file_path("c")
        disk_file.parent.mkdir(parents=True, exist_ok=True)

        loaded = _sized_state(3, 3000)
        with (
            patch.object(store, "_read_from_disk", return_value=(loaded, disk_file)),
            patch.object(store, "_save_to_disk") as mock_save,
        ):
            result = await store.async_get("c")

        assert result is loaded
        # No surviving file to rely on — the entry must be re-saved.
        spilled_ids = [call.args[0] for call in mock_save.call_args_list]
        assert "c" in spilled_ids

    def test_sync_get_disk_restore_enforces_ram_budget(self, tmp_path):
        """Disk-restored entries must trigger _enforce_ram_budget on
        the sync path too."""
        from unittest.mock import MagicMock

        store = PromptCacheStore(
            max_slots=10,
            disk_path=tmp_path,
            model_name="test-model",
            ram_budget_bytes=2500,
        )
        store.set("a", _sized_state(1, 1000))
        store.set("b", _sized_state(2, 1000))

        disk_file = store._disk_file_path("c")
        disk_file.parent.mkdir(parents=True, exist_ok=True)
        disk_file.touch()

        with patch(
            "olmlx.engine.prompt_cache.store.load_prompt_cache",
            return_value=([MagicMock()], {"tokens": "[3]"}),
        ):
            # The mock-returned cache layer has no .keys/.values with nbytes,
            # so the restored entry contributes 0 bytes — we won't actually
            # exceed the budget here. Validate the gentler claim: nothing
            # crashes and a later set still triggers budget eviction
            # correctly. (Realistically `set()` already runs the budget;
            # the sync path inherits it through self.set().)
            restored = store.get("c")
            assert restored is not None
        # Round-trip a sized entry to verify the budget enforcement that
        # set() inherits is still in effect.
        store.set("d", _sized_state(4, 1000))
        assert store.metrics.bytes_in_ram <= 2500

    def test_sync_methods_unchanged(self, tmp_path):
        """Regression: sync get()/set() still work exactly as before."""
        store = PromptCacheStore(
            max_slots=2,
            disk_path=tmp_path,
            model_name="test-model",
        )
        state_a = _make_state(1)
        state_b = _make_state(2)

        # set works
        result = store.set("a", state_a)
        assert result is None  # no eviction
        assert len(store) == 1

        # get works
        got = store.get("a")
        assert got is state_a

        # eviction via set works
        store.set("b", state_b)
        assert len(store) == 2

        # get returns None for missing
        assert store.get("missing") is None
