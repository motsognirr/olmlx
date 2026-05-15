from pydantic import BaseModel, Field, field_validator

from olmlx.schemas.common import (
    ModelName,
    ModelOptions,
    validate_non_empty_text_input,
)


class GenerateRequest(BaseModel):
    model: ModelName
    prompt: str = Field(..., max_length=1_000_000)
    suffix: str | None = None
    images: list[str] | None = None
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    stream: bool = True
    raw: bool = False
    format: str | None = None
    options: ModelOptions | None = None
    keep_alive: str | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt_non_empty(cls, v: str) -> str:
        return validate_non_empty_text_input(v, "prompt")


class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    done_reason: str | None = None
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
