"""Guard verdicts. `reason`, `score` and `rationale` are internal: for logs, never for API clients."""

from pydantic import BaseModel, Field

from src.core.schemas import RAGResponse


class GuardResult(BaseModel):
    """Outcome of one guard check.

    Only `allowed` and `category` are public. The internal fields are excluded from `model_dump()`
    and JSON so a blocked request can't learn the classifier score or the policy model's reasoning;
    API responses carry only the stage ("input"/"output") and `category`.
    """

    allowed: bool
    category: str | None = None  # e.g. "prompt_injection", "blocked_topic", "uncited", "guard_error"
    reason: str = Field(default="", exclude=True)
    score: float | None = Field(default=None, exclude=True)  # Prompt Guard probability
    rationale: str | None = Field(default=None, exclude=True)  # safeguard model's explanation


class OutputGuardResult(GuardResult):
    # The response to return to the client (with `disclaimer` set); None when blocked.
    response: RAGResponse | None = None
