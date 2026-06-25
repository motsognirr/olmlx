from pydantic import BaseModel

from olmlx.schemas.models import ModelDetails


class RunningModel(BaseModel):
    name: str
    model: str = ""
    size: int = 0
    digest: str = ""
    details: ModelDetails = ModelDetails()
    expires_at: str = ""
    size_vram: int = 0
    active_refs: int = 0
    cache_metrics: dict[str, int] = {}
    batch_metrics: dict[str, int] = {}
    #: Base model name when this row is a LoRA adapter (issue #362); None for
    #: ordinary base models.
    adapter_base: str | None = None


class PsResponse(BaseModel):
    models: list[RunningModel]


class VersionResponse(BaseModel):
    version: str
