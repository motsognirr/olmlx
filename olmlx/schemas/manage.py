from pydantic import BaseModel

from olmlx.schemas.common import ModelName


class CopyRequest(BaseModel):
    source: ModelName
    destination: ModelName


class DeleteRequest(BaseModel):
    model: ModelName


class CreateRequest(BaseModel):
    model: ModelName
    modelfile: str | None = None
    stream: bool = True
    path: str | None = None
    quantize: str | None = None


class WarmupRequest(BaseModel):
    model: ModelName
    keep_alive: str | None = None


class AbortRequest(BaseModel):
    model: ModelName


class UnloadRequest(BaseModel):
    model: ModelName
