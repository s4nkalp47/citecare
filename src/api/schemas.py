"""Request/response models of the HTTP API. Answers and sources reuse src.core.schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    # No min/max length here: empty and over-long questions are the input guard's job, so they
    # get the same blocked response as every other input block.
    question: str
    source_type: str | None = Field(default=None, max_length=64,
                                    description="Only search this source type, e.g. 'guideline'")


class BlockedResponse(BaseModel):
    """HTTP 400 body for a guarded request. Never carries the guard's reason, score or rationale."""

    blocked: Literal[True] = True
    stage: Literal["input", "output"]
    category: str | None
    message: str


class DeleteSourceResponse(BaseModel):
    source_id: str
    chunks_removed: int


class ErrorResponse(BaseModel):
    detail: str


class HealthModels(BaseModel):
    llm: str
    prompt_guard: str
    safety: str
    topic: str
    embed: str
    rerank: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    project: str
    qdrant_reachable: bool
    collection: str
    points: int | None  # None when Qdrant is unreachable
    models: HealthModels
