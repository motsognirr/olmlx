from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from olmlx.schemas.common import ModelName


class CopyRequest(BaseModel):
    source: ModelName
    destination: ModelName


class DeleteRequest(BaseModel):
    model: ModelName


class CreateRequest(BaseModel):
    """Both the legacy ``modelfile`` shape and Ollama's newer structured
    shape (``from``/``system``/``parameters``, #760). Structured fields
    override the corresponding Modelfile instruction."""

    model_config = ConfigDict(populate_by_name=True)

    model: ModelName
    modelfile: str | None = None
    stream: bool = True
    path: str | None = None
    quantize: str | None = None
    from_: str | None = Field(default=None, alias="from")
    system: str | None = None
    parameters: dict[str, Any] | None = None
    # Accepted so they can be rejected with a clear 400 rather than dropped.
    template: str | None = None
    files: dict[str, str] | None = None
    adapters: dict[str, str] | None = None
    messages: list[dict[str, Any]] | None = None
    license: str | list[str] | None = None


class WarmupRequest(BaseModel):
    model: ModelName
    keep_alive: int | str | None = None


class AbortRequest(BaseModel):
    model: ModelName


class UnloadRequest(BaseModel):
    model: ModelName
