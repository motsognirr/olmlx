"""Tests for KV cache disk offload (PromptCacheStore disk tier)."""

import os
import time
from pathlib import Path
from unittest.mock import patch

from olmlx.engine.model_manager import CachedPromptState, PromptCacheStore


def _make_state(token_id: int = 1) -> CachedPromptState:
    """Create a minimal CachedPromptState for testing."""
    return CachedPromptState(tokens=[token_id], cache=[f"cache_{token_id}"])


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
        with patch("olmlx.engine.model_manager.save_prompt_cache") as mock_save:
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
        with patch("olmlx.engine.model_manager.save_prompt_cache") as mock_save:
            store.set("a", state_a)
            store.set("b", _make_state(2))  # evicts "a"
            metadata = mock_save.call_args[0][2]
            assert "tokens" in metadata

    def test_eviction_without_disk_path_deletes(self):
        """Without disk_path, eviction behaves as before (no save)."""
        store = PromptCacheStore(max_slots=1)
        store.set("a", _make_state(1))
        with patch("olmlx.engine.model_manager.save_prompt_cache") as mock_save:
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
            "olmlx.engine.model_manager.save_prompt_cache",
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
            "olmlx.engine.model_manager.load_prompt_cache",
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
            "olmlx.engine.model_manager.load_prompt_cache",
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
                "olmlx.engine.model_manager.load_prompt_cache",
                return_value=(["kv_b"], {"tokens": "[2]"}),
            ),
            patch("olmlx.engine.model_manager.save_prompt_cache") as mock_save,
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
            "olmlx.engine.model_manager.load_prompt_cache",
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

        with patch("olmlx.engine.model_manager.save_prompt_cache") as mock_save:
            store.evict_all_to_disk()
            assert len(store) == 0
            assert mock_save.call_count == 2

    def test_evict_all_to_disk_without_disk_path_just_clears(self):
        """Without disk configured, evict_all_to_disk just clears memory."""
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state(1))
        store.evict_all_to_disk()
        assert len(store) == 0

    def test_evict_all_to_disk_entries_restorable(self, tmp_path):
        """Entries evicted to disk can be restored on next get()."""
        store = PromptCacheStore(
            max_slots=4,
            disk_path=tmp_path,
            model_name="test-model",
        )
        store.set("a", _make_state(1))

        with patch("olmlx.engine.model_manager.save_prompt_cache"):
            store.evict_all_to_disk()

        # Simulate file exists on disk for restoration
        disk_file = store._disk_file_path("a")
        disk_file.parent.mkdir(parents=True, exist_ok=True)
        disk_file.touch()

        with patch(
            "olmlx.engine.model_manager.load_prompt_cache",
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
