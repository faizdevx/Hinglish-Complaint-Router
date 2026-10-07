from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ModelKey = Literal["lstm", "transformer"]


class PredictRequest(BaseModel):
    text: str = Field(..., description="Message to classify", max_length=5000)
    model: ModelKey | None = Field(None, description="lstm | transformer (default: transformer if available)")
    language: str | None = Field(None, description="Optional user-supplied language tag; shown in the result, not used by the model")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Enter at least one non-empty message.")
        return v


class CompareRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    language: str | None = None

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Enter at least one non-empty message.")
        return v


class PredictResponse(BaseModel):
    intent: str
    urgency: str
    escalation_probability: float = Field(..., ge=0.0, le=1.0)
    language: str
    route: str
    model: str | None = None
    inference_ms: float | None = None
