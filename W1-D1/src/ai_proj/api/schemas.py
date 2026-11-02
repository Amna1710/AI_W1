
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class UsageInfo(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None

    currency: str = "USD"
    pricing_available: bool = False


class AIRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    provider: Literal["ollama", "openai"] = "ollama"

    use_case: Literal[
        "question_answering",
        "extraction",
        "classification",
        "generation",
    ] = "generation"

    prompt: str = Field(
        min_length=1,
        max_length=10_000,
    )

    model: str = Field(
        default="qwen3.5:4b",
        min_length=1,
        max_length=100,
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )

    max_tokens: int = Field(
        default=500,
        gt=0,
        le=4000,
    )


class AIResponse(BaseModel):
    request_id: UUID
    status: Literal["success", "error"]
    output: str
    model: str
    usage: UsageInfo

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class APIError(BaseModel):
    code: str
    message: str
    request_id: UUID | None = None


class BatchAIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: list[AIRequest] = Field(
        min_length=1,
        max_length=50,
    )


class BatchAIResult(BaseModel):
    index: int
    status: Literal["success", "error"]
    output: str | None = None
    model: str
    usage: UsageInfo | None = None
    error: str | None = None


class BatchAIResponse(BaseModel):
    request_id: UUID

    status: Literal[
        "success",
        "partial_success",
        "error",
    ]

    total: int
    succeeded: int
    failed: int
    results: list[BatchAIResult]

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
