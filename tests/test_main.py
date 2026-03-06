"""Tests for mlx_ollama.__main__."""

from unittest.mock import MagicMock, patch

from mlx_ollama.__main__ import main


class TestMain:
    def test_main_delegates_to_cli_main(self):
        with patch("mlx_ollama.__main__.cli_main") as mock_cli:
            main()
            mock_cli.assert_called_once()
