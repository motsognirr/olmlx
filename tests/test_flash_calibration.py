"""Tests for C4 dataset calibration (Paper §3.1)."""

from unittest.mock import MagicMock, patch

from olmlx.engine.flash.prepare import (
    _get_c4_calibration_data,
    _get_calibration_data,
)


class TestC4Calibration:
    def test_fallback_to_synthetic_when_datasets_unavailable(self):
        """When 'datasets' is not installed, fall back to synthetic data."""
        with patch.dict("sys.modules", {"datasets": None}):
            result = _get_c4_calibration_data(100)
        # Should return synthetic data (capped at 256 max from synthetic)
        assert len(result) > 0
        assert all(isinstance(t, str) for t in result)

    def test_c4_streaming_returns_correct_count(self):
        """Mock streaming dataset returns the requested number of samples."""
        fake_data = [
            {
                "text": f"Sample text number {i} with enough length to pass the filter."
                * 3
            }
            for i in range(50)
        ]

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_data))

        mock_load = MagicMock(return_value=mock_ds)
        mock_datasets = MagicMock()
        mock_datasets.load_dataset = mock_load

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            result = _get_c4_calibration_data(20)

        assert len(result) == 20
        mock_load.assert_called_once()

    def test_short_docs_skipped(self):
        """Documents shorter than 100 chars are filtered out."""
        fake_data = [
            {"text": "short"},
            {"text": "x" * 200},
            {"text": "tiny"},
            {"text": "y" * 300},
        ]

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_data))

        mock_load = MagicMock(return_value=mock_ds)
        mock_datasets = MagicMock()
        mock_datasets.load_dataset = mock_load

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            result = _get_c4_calibration_data(10)

        # Only the 2 long docs should pass
        assert len(result) == 2

    def test_synthetic_calibration_still_works(self):
        """The original synthetic generator still works."""
        result = _get_calibration_data(50)
        assert len(result) == 50
        assert all(isinstance(t, str) for t in result)

    def test_network_error_falls_back_to_synthetic(self):
        """Network errors during C4 streaming fall back to synthetic data."""

        def raise_connection_error(*args, **kwargs):
            raise ConnectionError("Network unreachable")

        mock_datasets = MagicMock()
        mock_datasets.load_dataset = raise_connection_error

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            result = _get_c4_calibration_data(100)

        assert len(result) > 0
        assert all(isinstance(t, str) for t in result)

    def test_long_docs_truncated(self):
        """Documents longer than 2048 chars are truncated."""
        fake_data = [{"text": "a" * 5000}]

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_data))

        mock_load = MagicMock(return_value=mock_ds)
        mock_datasets = MagicMock()
        mock_datasets.load_dataset = mock_load

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            result = _get_c4_calibration_data(1)

        assert len(result) == 1
        assert len(result[0]) == 2048
