"""Helpers shared by the input and output guardrails: safeguard calls, fail mode, block logging."""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel

from src.config import Settings
from src.core.llm import complete_with_retry, describe_error
from src.guardrails.models import GuardResult

logger = logging.getLogger("src.guardrails")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")

# Reply budget for policy checks: low-effort reasoning plus a one-line JSON verdict. Groq counts
# max_tokens towards the per-minute token limit (unset, it reserves ~1.2K), so keep it explicit:
# prompt (~1K) + this stays well under 8K TPM even with both checks in the same minute.
SAFEGUARD_MAX_TOKENS = 768


class SafeguardVerdict(BaseModel):
    violation: bool  # the policy asks for 0/1; pydantic accepts both ints and bools
    category: str | None = None
    rationale: str | None = None


def parse_safeguard_json(text: str) -> SafeguardVerdict:
    """Parse the safeguard reply, tolerating ``` fences and text around the JSON object."""
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in safeguard reply")
    return SafeguardVerdict.model_validate(json.loads(text[start:end + 1]))


def ask_safeguard(client, settings: Settings, model: str, policy: str,
                  content: str) -> SafeguardVerdict:
    """Judge `content` against `policy` (system message) with a policy-following model."""
    kwargs = {"temperature": 0, "max_tokens": SAFEGUARD_MAX_TOKENS}
    if settings.groq_reasoning_effort:
        kwargs["reasoning_effort"] = settings.groq_reasoning_effort
    reply = complete_with_retry(
        client, model,
        [{"role": "system", "content": policy}, {"role": "user", "content": content}],
        **kwargs,
    )
    return parse_safeguard_json(reply)


def blocked(stage: str, result: GuardResult) -> GuardResult:
    logger.info("%s guard blocked: category=%s score=%s reason=%s",
                stage, result.category, result.score, result.reason)
    return result


def guard_error(stage: str, exc: BaseException, fail_mode: str,
                result_cls: type[GuardResult] = GuardResult, **fields) -> GuardResult:
    """A guard call failed: block when fail-closed, allow when fail-open.

    Logs the error type, HTTP status and Groq error code, never the message content.
    """
    allowed = fail_mode == "open"
    error = describe_error(exc)
    logger.warning("%s guard call failed (%s); failing %s", stage, error, fail_mode)
    result = result_cls(allowed=allowed, category="guard_error",
                        reason=f"guard call failed: {error}", **fields)
    return result if allowed else blocked(stage, result)
