from pydantic import BaseModel, field_validator

from olmlx.schemas.common import ModelName, validate_non_empty_text_input


class EmbedRequest(BaseModel):
    model: ModelName
    input: str | list[str]
    truncate: bool = True
    options: dict | None = None
    keep_alive: int | str | None = None

    @field_validator("input")
    @classmethod
    def validate_input_non_empty(cls, v: str | list[str]) -> str | list[str]:
        return validate_non_empty_text_input(v, "input")


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
    keep_alive: int | str | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt_non_empty(cls, v: str) -> str:
        return validate_non_empty_text_input(v, "prompt")


class EmbeddingsResponse(BaseModel):
    embedding: list[float]
