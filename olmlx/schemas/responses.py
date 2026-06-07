from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from olmlx.schemas.common import ModelName


class ResponsesRequest(BaseModel):
    model: ModelName
    # `input` is either a plain string (one user turn) or a list of input
    # items (messages / function_call / function_call_output). Items are kept
    # as freeform dicts and validated during translation (router raises
    # ValueError -> 400/422 for unknown shapes).
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    stream: bool = False
    max_output_tokens: int | None = Field(None, ge=1)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    previous_response_id: str | None = None
    store: bool = True
    reasoning: dict[str, Any] | None = None  # {"effort": "low"|"medium"|"high"}
    text: dict[str, Any] | None = None       # {"format": {"type": ...}}
    metadata: dict[str, Any] | None = None
    seed: int | None = None

    parallel_tool_calls: bool = True

    @field_validator("input")
    @classmethod
    def _validate_input_non_empty(
        cls, v: str | list[dict[str, Any]]
    ) -> str | list[dict[str, Any]]:
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("input string cannot be empty")
        elif not v:
            raise ValueError("input list cannot be empty")
        return v

    @field_validator("max_output_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int | None) -> int | None:
        if v is None:
            return v
        from olmlx.schemas.common import validate_token_limit

        return validate_token_limit(v, "max_output_tokens")


class ResponsesResponse(BaseModel):
    """Top-level Responses object. Output items and usage are kept as dicts so
    the router builds them directly; response_model_exclude_none trims absent
    fields the SDK treats as optional."""

    id: str
    object: str = "response"
    created_at: int
    status: str
    model: str
    output: list[dict[str, Any]]
    usage: dict[str, Any] | None = None
    previous_response_id: str | None = None
    incomplete_details: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, Any] | None = None
    # Echoes ResponsesRequest.parallel_tool_calls (set by the router).
    parallel_tool_calls: bool = True
    temperature: float | None = None
    top_p: float | None = None
    tool_choice: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
