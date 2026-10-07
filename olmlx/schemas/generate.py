from typing import Any

from pydantic import BaseModel, Field, field_validator

from olmlx.utils.images import ensure_image_data_uris
from olmlx.schemas.common import (
    ModelName,
    ModelOptions,
)


class GenerateRequest(BaseModel):
    model: ModelName
    # Empty/missing is Ollama's load (or, with keep_alive 0, unload) request
    # (#760); the router answers it before any inference.
    prompt: str = Field("", max_length=1_000_000)
    suffix: str | None = None
    images: list[str] | None = None
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    stream: bool = True
    raw: bool = False
    # Ollama accepts either ``"json"`` (any JSON value) or a JSON Schema
    # dict (strict adherence). Both are passed through to xgrammar.
    format: str | dict[str, Any] | None = None
    think: bool | str | None = None
    options: ModelOptions | None = None
    keep_alive: int | str | None = None

    @field_validator("images")
    @classmethod
    def wrap_raw_base64_images(cls, v: list[str] | None) -> list[str] | None:
        # Ollama sends raw base64; mlx_vlm would open() it as a path (#714).
        return ensure_image_data_uris(v)


class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    thinking: str | None = None
    done: bool
    done_reason: str | None = None
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
