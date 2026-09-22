"""OpenAI-compatible ``/v1/images/generations`` schemas (#723)."""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from olmlx.schemas.common import ModelName

_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")


class ImageGenerationRequest(BaseModel):
    """OpenAI images request plus olmlx/mflux extensions.

    Unknown OpenAI fields (``quality``, ``style``, ``user``, ``background``
    ...) are accepted and ignored. v1 limits: ``n`` must be 1 and only
    ``b64_json`` is returned (there is no static file server for ``url``).
    """

    model: ModelName
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: Literal["b64_json", "url"] = "b64_json"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    # --- olmlx extensions (mflux knobs) ---
    seed: int | None = Field(default=None, ge=0, lt=2**32)
    #: Diffusion steps; unset uses the variant's default (Qwen-Image: 4,
    #: Qwen-Image-2.1: 40).
    steps: int | None = Field(default=None, ge=1, le=200)
    #: CFG scale; unset uses the variant's default (4.0 / 1.0).
    guidance: float | None = Field(default=None, ge=0.0, le=50.0)
    #: Only effective with CFG (guidance > 1); Qwen-Image-2.1's default
    #: guidance of 1.0 ignores it.
    negative_prompt: str | None = None
    keep_alive: str | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt cannot be empty or blank")
        return v

    @field_validator("n")
    @classmethod
    def validate_n(cls, v: int) -> int:
        if v != 1:
            raise ValueError("only n=1 is supported")
        return v

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v: str) -> str:
        if v != "b64_json":
            raise ValueError("response_format 'url' is not supported; use 'b64_json'")
        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: str) -> str:
        m = _SIZE_RE.match(v)
        if m is None:
            raise ValueError("size must be 'WIDTHxHEIGHT', e.g. '1024x1024'")
        w, h = int(m.group(1)), int(m.group(2))
        if w % 16 or h % 16 or w < 64 or h < 64:
            raise ValueError("size dimensions must be multiples of 16 and at least 64")
        return v

    @field_validator("guidance")
    @classmethod
    def validate_guidance(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("guidance must be finite")
        return v

    @property
    def dimensions(self) -> tuple[int, int]:
        """``(width, height)`` parsed from ``size``."""
        w, h = self.size.split("x")
        return int(w), int(h)


class ImageData(BaseModel):
    b64_json: str
    revised_prompt: str | None = None
    #: olmlx extension: the seed actually used (for reproducibility).
    seed: int | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
    output_format: str
    size: str
