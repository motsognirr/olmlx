"""Tests for mlx_ollama.__main__."""

from unittest.mock import MagicMock, patch

from mlx_ollama.__main__ import main


class TestMain:
    def test_main_starts_uvicorn(self):
        with patch("mlx_ollama.__main__.uvicorn") as mock_uvicorn:
            with patch("mlx_ollama.__main__.settings") as mock_settings:
                mock_settings.host = "0.0.0.0"
                mock_settings.port = 11434
                main()
                mock_uvicorn.run.assert_called_once()
                call_kwargs = mock_uvicorn.run.call_args
                assert "mlx_ollama.app:create_app" in str(call_kwargs)
