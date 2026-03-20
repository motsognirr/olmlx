"""Tests for olmlx.utils.memory — consolidated Metal/system memory helpers."""

from unittest.mock import patch

import olmlx.utils.memory as memory_mod


class TestGetMetalMemory:
    def test_returns_active_plus_cache(self):
        with patch.object(memory_mod, "mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 1000
            mock_mx.get_cache_memory.return_value = 500
            assert memory_mod.get_metal_memory() == 1500


class TestGetSystemMemoryBytes:
    def test_returns_page_size_times_pages(self):
        with patch.object(memory_mod, "os") as mock_os:
            mock_os.sysconf.side_effect = lambda key: {
                "SC_PAGE_SIZE": 4096,
                "SC_PHYS_PAGES": 1000,
            }[key]
            assert memory_mod.get_system_memory_bytes() == 4096 * 1000


    def test_returns_zero_on_os_error(self):
        with patch.object(memory_mod, "os") as mock_os:
            mock_os.sysconf.side_effect = OSError("unsupported")
            assert memory_mod.get_system_memory_bytes() == 0



class TestIsMemoryPressureHigh:
    def test_returns_true_when_above_threshold(self):
        with (
            patch.object(
                memory_mod, "get_system_memory_bytes", return_value=32 * 1024**3
            ),
            patch.object(memory_mod, "get_metal_memory", return_value=23 * 1024**3),
        ):
            assert memory_mod.is_memory_pressure_high(0.75, 0.9) is True

    def test_returns_false_when_below_threshold(self):
        with (
            patch.object(
                memory_mod, "get_system_memory_bytes", return_value=32 * 1024**3
            ),
            patch.object(memory_mod, "get_metal_memory", return_value=10 * 1024**3),
        ):
            assert memory_mod.is_memory_pressure_high(0.75, 0.9) is False

    def test_returns_false_when_system_memory_zero(self):
        with patch.object(memory_mod, "get_system_memory_bytes", return_value=0):
            assert memory_mod.is_memory_pressure_high(0.75, 0.9) is False

    def test_returns_false_on_exception(self):
        with (
            patch.object(
                memory_mod, "get_system_memory_bytes", return_value=32 * 1024**3
            ),
            patch.object(
                memory_mod, "get_metal_memory", side_effect=RuntimeError("boom")
            ),
        ):
            assert memory_mod.is_memory_pressure_high(0.75, 0.9) is False
