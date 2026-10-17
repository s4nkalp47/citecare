"""Groq LLM via LlamaIndex, with exponential backoff on free-tier rate limits (HTTP 429)."""

import logging
from functools import lru_cache

from llama_index.core.llms import LLM, ChatMessage
from llama_index.llms.groq import Groq
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_llm() -> Groq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is empty - add it to .env")
    additional_kwargs = {}
    if settings.groq_reasoning_effort:
        # LlamaIndex only sends its own `reasoning_effort` field for OpenAI o-series models.
        additional_kwargs["reasoning_effort"] = settings.groq_reasoning_effort
    return Groq(
        model=settings.groq_llm_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_retries=0,  # retries are handled by chat_with_retry, not the openai client
        additional_kwargs=additional_kwargs,
    )


def _is_rate_limit(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 429


@retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def chat_with_retry(llm: LLM, messages: list[ChatMessage]) -> str:
    """Send a chat request and return the reply text, retrying on HTTP 429."""
    return llm.chat(messages).message.content or ""
