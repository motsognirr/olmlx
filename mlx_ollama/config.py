from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "MLX_OLLAMA_", "env_file": ".env"}

    host: str = "0.0.0.0"
    port: int = 11434
    models_dir: Path = Path.home() / ".mlx_ollama" / "models"
    models_config: Path = Path.home() / ".mlx_ollama" / "models.json"
    default_keep_alive: str = "5m"
    max_loaded_models: int = 1


settings = Settings()
