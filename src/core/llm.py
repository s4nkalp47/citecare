"""Groq LLM via LlamaIndex, with backoff on free-tier rate limits (HTTP 429)."""

import logging
import re
from functools import lru_cache

import groq
from llama_index.core.llms import LLM, ChatMessage
from llama_index.llms.groq import Groq
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_before_delay,
    wait_exponential,
)

from src.config import get_settings

logger = logging.getLogger(__name__)

# Give up rather than sleep past this many seconds in total (first call to last retry).
MAX_RETRY_SECONDS = 60
# Groq 429 messages say e.g. "Please try again in 7.66s", "in 1m2.5s" or "in 520ms".
_TRY_AGAIN = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)(ms|s)\b")


def _require_api_key() -> str:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is empty - add it to .env")
    return settings.groq_api_key


@lru_cache
def get_llm() -> Groq:
    settings = get_settings()
    api_key = _require_api_key()
    additional_kwargs = {}
    if settings.groq_reasoning_effort:
        # LlamaIndex only sends its own `reasoning_effort` field for OpenAI o-series models.
        additional_kwargs["reasoning_effort"] = settings.groq_reasoning_effort
    return Groq(
        model=settings.groq_llm_model,
        api_key=api_key,
        temperature=0,
        max_retries=0,  # retries are handled by chat_with_retry, not the openai client
        additional_kwargs=additional_kwargs,
    )


@lru_cache
def get_groq_client() -> groq.Groq:
    """Raw Groq SDK client, for classifier-style models (guardrails) that aren't chat LLMs."""
    return groq.Groq(api_key=_require_api_key(), max_retries=0)  # retries: complete_with_retry


def describe_error(exc: BaseException) -> str:
    """Exception type, HTTP status and Groq error type/code. Never the message, which can echo
    request details (organisation id, prompt sizes)."""
    parts = [type(exc).__name__]
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"status={status}")
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        parts += [f"{key}={error[key]}" for key in ("type", "code") if error.get(key)]
    return " ".join(parts)


def _is_rate_limit(exc: BaseException) -> bool:
    # Only 429. A 413 "rate_limit_exceeded" means one request exceeds the per-minute token limit
    # on its own, so retrying can never succeed.
    return getattr(exc, "status_code", None) == 429


def retry_after_seconds(exc: BaseException) -> float | None:
    """How long Groq asks us to wait: the retry-after header, else "try again in Ns" in the error."""
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass  # an HTTP date; fall through to the message
    match = _TRY_AGAIN.search(str(exc))
    if match:
        minutes, amount, unit = match.groups()
        return int(minutes or 0) * 60 + float(amount) / (1000 if unit == "ms" else 1)
    return None


_backoff = wait_exponential(multiplier=2, min=2, max=30)


def _wait_for_rate_limit(retry_state: RetryCallState) -> float:
    hint = retry_after_seconds(retry_state.outcome.exception())
    if hint is None:
        return _backoff(retry_state)
    return hint + 0.5  # small margin so the limit window has really reset


def _log_retry(retry_state: RetryCallState) -> None:
    logger.warning("Groq rate limited (%s); retrying in %.1fs (attempt %d)",
                   describe_error(retry_state.outcome.exception()),
                   retry_state.upcoming_sleep, retry_state.attempt_number)


retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=_wait_for_rate_limit,
    # Stop when the next sleep would take us past the total budget, not after sleeping into it.
    stop=stop_after_attempt(5) | stop_before_delay(MAX_RETRY_SECONDS),
    before_sleep=_log_retry,
    reraise=True,
)


@retry_on_rate_limit
def chat_with_retry(llm: LLM, messages: list[ChatMessage]) -> str:
    """Send a chat request and return the reply text, retrying on HTTP 429."""
    return llm.chat(messages).message.content or ""


@retry_on_rate_limit
def complete_with_retry(client: groq.Groq, model: str, messages: list[dict], **kwargs) -> str:
    """Groq SDK chat completion returning the reply text, retrying on HTTP 429."""
    reply = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return reply.choices[0].message.content or ""
