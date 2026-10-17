"""OutputGuardrail: deterministic citation policy, then a policy model against the project policy."""

from __future__ import annotations

from src.config import Settings, get_settings
from src.core.llm import get_groq_client
from src.core.projects import ProjectConfig
from src.core.schemas import RAGResponse
from src.guardrails.common import ask_safeguard, blocked, guard_error
from src.guardrails.models import OutputGuardResult

_CONFIDENT = ("high", "medium")


def format_for_review(question: str, response: RAGResponse) -> str:
    """The safeguard sees the cited snippets too, so it can judge claims against their sources."""
    sources = []
    for c in response.citations:
        header = f"[{c.ref}] {c.title}" + (f", page {c.page}" if c.page is not None else "")
        sources.append(f"{header}: {c.snippet}")
    return (f"Question:\n{question}\n\nAnswer (confidence: {response.confidence}):\n"
            f"{response.answer}\n\nCited sources:\n" + ("\n".join(sources) or "(none)"))


class OutputGuardrail:
    def __init__(self, project: ProjectConfig, client=None, settings: Settings | None = None):
        self.project = project
        self.settings = settings or get_settings()
        self.client = client or get_groq_client()

    def check(self, question: str, response: RAGResponse) -> OutputGuardResult:
        if response.confidence == "insufficient_evidence":
            # A refusal: nothing to verify, and no disclaimer needed.
            return OutputGuardResult(allowed=True, reason="insufficient evidence answer",
                                     response=response)

        if not response.citations and (self.project.require_citations
                                       or response.confidence in _CONFIDENT):
            return blocked("output", OutputGuardResult(
                allowed=False, category="uncited",
                reason=f"{response.confidence}-confidence answer has no citations"))

        result = OutputGuardResult(allowed=True, reason="passed output checks")
        if self.project.safety_policy:
            try:
                verdict = ask_safeguard(self.client, self.settings, self.settings.groq_safety_model,
                                        self.project.safety_policy,
                                        format_for_review(question, response))
            except Exception as exc:  # noqa: BLE001 - any guard failure is handled by the fail mode
                result = guard_error("output", exc, self.settings.guard_fail_mode,
                                     result_cls=OutputGuardResult)
                if not result.allowed:
                    return result
            else:
                if verdict.violation:
                    return blocked("output", OutputGuardResult(
                        allowed=False, category=f"policy:{verdict.category or 'violation'}",
                        reason=f"safety policy violation: {verdict.category}",
                        rationale=verdict.rationale))
                result.rationale = verdict.rationale

        # A copy, so the caller's response is never mutated.
        result.response = response.model_copy(
            update={"disclaimer": self.project.disclaimer or None})
        return result
