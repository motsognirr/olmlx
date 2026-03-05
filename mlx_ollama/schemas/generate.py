from pydantic import BaseModel

from mlx_ollama.schemas.common import ModelOptions


class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
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
