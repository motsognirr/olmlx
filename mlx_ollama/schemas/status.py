from pydantic import BaseModel

from mlx_ollama.schemas.models import ModelDetails


class RunningModel(BaseModel):
    name: str
    model: str = ""
    size: int = 0
    digest: str = ""
    details: ModelDetails = ModelDetails()
    expires_at: str = ""
    size_vram: int = 0


class PsResponse(BaseModel):
    models: list[RunningModel]


class VersionResponse(BaseModel):
    version: str
