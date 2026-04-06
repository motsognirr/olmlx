from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelName = Annotated[str, Field(min_length=1, max_length=256)]


def validate_token_limit(v: int, field_name: str) -> int:
    from olmlx.config import settings

    if v > settings.max_tokens_limit:
        raise ValueError(
            f"{field_name} {v} exceeds configured limit {settings.max_tokens_limit} "
            f"(set OLMLX_MAX_TOKENS_LIMIT to increase)"
        )
    return v


class ModelOptions(BaseModel):
    """Ollama model options / parameters."""

    model_config = ConfigDict(extra="allow")

    num_keep: int | None = None
    seed: int | None = None
    num_predict: int | None = Field(None, ge=-2)  # -1=infinite, -2=fill context

    @field_validator("num_predict")
    @classmethod
    def validate_num_predict(cls, v: int | None) -> int | None:
        if v is None or v < 0:
            return v  # special values -1 (infinite) and -2 (fill context)
        return validate_token_limit(v, "num_predict")

    top_k: int | None = Field(None, ge=0)
    top_p: float | None = Field(None, ge=0, le=1)
    min_p: float | None = Field(None, ge=0, le=1)
    tfs_z: float | None = None
    typical_p: float | None = None
    repeat_last_n: int | None = Field(None, ge=-1)
    temperature: float | None = Field(None, ge=0)  # no upper bound (Ollama compat)
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    penalize_newline: bool | None = None
    stop: list[str] | None = None
    numa: bool | None = None
    num_ctx: int | None = Field(None, ge=1)
    num_batch: int | None = None
    num_gpu: int | None = None
    main_gpu: int | None = None
    low_vram: bool | None = None
    vocab_only: bool | None = None
    use_mmap: bool | None = None
    use_mlock: bool | None = None
    num_thread: int | None = None
