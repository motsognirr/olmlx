from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "OLMLX_", "env_file": ".env"}

    host: str = "0.0.0.0"
    port: int = 11434
    models_dir: Path = Path.home() / ".olmlx" / "models"
    models_config: Path = Path.home() / ".olmlx" / "models.json"
    default_keep_alive: str = "5m"
    max_loaded_models: int = 1
    cors_origins: list[str] = ["http://localhost:*", "http://127.0.0.1:*"]


settings = Settings()
