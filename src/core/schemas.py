"""Structured query results shared by the service, API, UI and evals."""

from typing import Literal

from pydantic import BaseModel

Confidence = Literal["high", "medium", "low", "insufficient_evidence"]


class Citation(BaseModel):
    """A retrieved chunk the answer cites. All fields come from node metadata, never the LLM."""

    ref: int  # the [n] marker used inline in the answer
    title: str
    page: int | None = None
    url: str | None = None
    source_type: str
    snippet: str
    score: float | None = None  # retrieval similarity


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    confidence: Confidence
    contexts: list[str] = []  # text of every retrieved chunk, cited or not (used by evals)
    latency_ms: int = 0
