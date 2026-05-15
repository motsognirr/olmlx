from pydantic import BaseModel, field_validator

from olmlx.schemas.common import ModelName


class EmbedRequest(BaseModel):
    model: ModelName
    input: str | list[str]
    truncate: bool = True
    options: dict | None = None
    keep_alive: str | None = None

    @field_validator("input")
    @classmethod
    def validate_input_non_empty(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str):
            if not v:
                raise ValueError("input cannot be empty")
        else:
            if not v:
                raise ValueError("input cannot be an empty list")
            if any(not s for s in v):
                raise ValueError("input cannot contain empty strings")
        return v


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None


class EmbeddingsRequest(BaseModel):
    model: ModelName
    prompt: str
    options: dict | None = None
    keep_alive: str | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("prompt cannot be empty")
        return v


class EmbeddingsResponse(BaseModel):
    embedding: list[float]
