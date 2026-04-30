from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .settings import settings


class EmbedRequest(BaseModel):
    input: str | list[str] = Field(
        ...,
        description="A single string or list of strings to embed.",
    )
    input_type: Literal["query", "document"] = Field(
        default="document",
        description="`query` wraps each input with the Qwen3 instruction template; "
        "`document` embeds the raw text.",
    )
    instruction: str | None = Field(
        default=None,
        max_length=2_000,
        description="Custom task instruction for query-side inputs. "
        "Ignored when input_type='document'. Defaults to a generic retrieval prompt.",
    )
    normalize: bool = Field(
        default=True,
        description="L2-normalize the returned vectors (recommended for cosine search).",
    )

    @field_validator("input")
    @classmethod
    def _validate_input(cls, v: str | list[str]) -> str | list[str]:
        max_chars = settings.max_chars_per_input
        max_items = settings.max_inputs_per_request

        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                raise ValueError("input must be a non-empty string.")
            if len(v) > max_chars:
                raise ValueError(
                    f"input exceeds max_chars_per_input ({max_chars}). Got {len(v)} chars."
                )
            return v

        if not v:
            raise ValueError("input list must contain at least one item.")
        if len(v) > max_items:
            raise ValueError(
                f"input list exceeds max_inputs_per_request ({max_items}). Got {len(v)} items."
            )
        for i, s in enumerate(v):
            if not isinstance(s, str):
                raise ValueError(f"input[{i}] must be a string.")
            if not s.strip():
                raise ValueError(f"input[{i}] must be non-empty.")
            if len(s) > max_chars:
                raise ValueError(
                    f"input[{i}] exceeds max_chars_per_input ({max_chars}). Got {len(s)} chars."
                )
        return v


class EmbedItem(BaseModel):
    index: int
    embedding: list[float]


class EmbedResponse(BaseModel):
    model: str
    dim: int
    input_type: Literal["query", "document"]
    normalized: bool
    data: list[EmbedItem]


class ModelInfoResponse(BaseModel):
    model: str
    dim: int
    n_ctx: int
