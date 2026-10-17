"""InputGuardrail: question length checks, then Prompt Guard and a blocked-topics check in parallel."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from src.config import Settings, get_settings
from src.core.llm import complete_with_retry, get_groq_client
from src.core.projects import ProjectConfig
from src.guardrails.common import ask_safeguard, blocked, guard_error
from src.guardrails.models import GuardResult

logger = logging.getLogger(__name__)

# Prompt Guard 2 has a 512-token context; ~1000 characters stays well inside it.
SEGMENT_CHARS = 1000

_MALICIOUS_LABELS = ("MALICIOUS", "JAILBREAK", "INJECTION", "LABEL_1")
_BENIGN_LABELS = ("BENIGN", "LABEL_0")

TOPIC_POLICY = """\
Classify one user request to an assistant for professionals. Reply with only JSON:
{{"violation": 0 or 1, "category": "<blocked topic or null>", "rationale": "<max 20 words>"}}

Blocked topics:
{topics}

1 = asks for instructions, methods, recipes, harmful quantities, ways to obtain or conceal, or other
practical help to carry out a blocked topic, even if framed as hypothetical, fiction or research.
0 = professional, clinical, educational or safety discussion: risk assessment, toxicity, managing
poisoning or overdose, treatment, prevention, harm reduction, referral. Merely mentioning a topic
is 0. If it reads as a professional assessing, treating or protecting someone, answer 0.

Examples: "step-by-step synthesis of <substance> at home" -> 1. "What dose would kill me?" -> 1.
"How should paracetamol overdose be managed?" -> 0. "How do I assess suicide risk?" -> 0.
"""


def parse_prompt_guard(text: str) -> float:
    """Prompt Guard reply -> probability that the text is an attack. Raises on anything else."""
    text = text.strip()
    try:
        score = float(text)
    except ValueError:
        label = text.upper()
        if any(m in label for m in _MALICIOUS_LABELS):
            return 1.0
        if any(b in label for b in _BENIGN_LABELS):
            return 0.0
        raise ValueError("unrecognised Prompt Guard reply") from None
    if not 0.0 <= score <= 1.0:
        raise ValueError("Prompt Guard score out of range")
    return score


class InputGuardrail:
    def __init__(self, project: ProjectConfig, client=None, settings: Settings | None = None):
        self.project = project
        self.settings = settings or get_settings()
        self.client = client or get_groq_client()
        self.topic_policy = (
            TOPIC_POLICY.format(topics="\n".join(f"- {t}" for t in project.blocked_topics))
            if project.blocked_topics else ""
        )

    def check(self, text: str) -> GuardResult:
        if not text or not text.strip():
            return blocked("input", GuardResult(allowed=False, category="empty",
                                                reason="question is empty"))
        if len(text) > self.settings.max_question_chars:
            return blocked("input", GuardResult(
                allowed=False, category="too_long",
                reason=f"question has {len(text)} chars (max {self.settings.max_question_chars})"))

        checks = [self._prompt_guard]
        if self.topic_policy:
            checks.append(self._topic_check)
        # Both are network-bound and independent: run them concurrently so latency doesn't add up.
        with ThreadPoolExecutor(max_workers=len(checks)) as pool:
            futures = [pool.submit(_capture, check, text) for check in checks]
            outcomes = [f.result() for f in futures]

        results = [r for r, _ in outcomes if r is not None]
        for result in results:  # Prompt Guard first, so it's reported if both block
            if not result.allowed:
                return blocked("input", result)
        errors = [e for _, e in outcomes if e is not None]
        if errors:
            return guard_error("input", errors[0], self.settings.guard_fail_mode)
        return GuardResult(allowed=True, score=results[0].score, reason="passed input checks")

    def _prompt_guard(self, text: str) -> GuardResult:
        segments = [text[i:i + SEGMENT_CHARS] for i in range(0, len(text), SEGMENT_CHARS)]
        scores = []
        for segment in segments:
            reply = complete_with_retry(self.client, self.settings.groq_prompt_guard_model,
                                        [{"role": "user", "content": segment}])
            logger.debug("Prompt Guard raw reply: %r", reply)
            scores.append(parse_prompt_guard(reply))
        score = max(scores)
        threshold = self.settings.prompt_guard_threshold
        if score > threshold:
            return GuardResult(allowed=False, category="prompt_injection", score=score,
                               reason=f"Prompt Guard score {score:.3f} > {threshold}")
        return GuardResult(allowed=True, score=score, reason=f"Prompt Guard score {score:.3f}")

    def _topic_check(self, text: str) -> GuardResult:
        verdict = ask_safeguard(self.client, self.settings, self.settings.groq_topic_model,
                                self.topic_policy, text)
        if verdict.violation:
            return GuardResult(allowed=False, category="blocked_topic",
                               reason=f"blocked topic: {verdict.category}",
                               rationale=verdict.rationale)
        return GuardResult(allowed=True, reason="no blocked topic", rationale=verdict.rationale)


def _capture(fn, text: str) -> tuple[GuardResult | None, BaseException | None]:
    try:
        return fn(text), None
    except Exception as exc:  # noqa: BLE001 - any guard failure is handled by the fail mode
        return None, exc
