"""Pydantic request/response models for the text-to-SQL service."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
]


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, description="natural-language question")
    db_id: str = Field(..., min_length=1, description="target database id")
    max_new_tokens: int | None = Field(
        None, ge=1, le=1024, description="optional decode length override"
    )
    include_raw: bool = Field(False, description="return the raw model output too")


class GenerateResponse(BaseModel):
    sql: str
    db_id: str
    request_id: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    model: str
    raw_output: str | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter: str | None
    device: str
    uptime_s: float


class MetricsResponse(BaseModel):
    total_requests: int
    error_rate: float
    empty_sql_rate: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    avg_prompt_tokens: float
    avg_completion_tokens: float


class ErrorResponse(BaseModel):
    error: str
    request_id: str | None = None