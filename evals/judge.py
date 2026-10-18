"""Phase 10: a Groq model as the DeepEval judge.

Defaults to settings.groq_judge_model (qwen/qwen3.8-27b): its rate limit is separate from the
generator and guardrail models, and it's from a different model family than the gpt-oss generator,
so the judge doesn't grade answers written in its own style.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel, ValidationError

from evals import JUDGE_DELAY_SECONDS
from src.config import get_settings
from src.core.llm import get_groq_client, retry_on_rate_limit

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass
class JudgeUsage:
    """Tokens used by the judge so far. Calls whose reply had no `usage` are estimated."""

    prompt: int = 0
    completion: int = 0
    calls: int = 0
    estimated_calls: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion


def clean_reply(text: str) -> str:
    """Drop <think> blocks and ``` fences around the reply."""
    return _FENCE.sub("", _THINK.sub("", text).strip())


def extract_json(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in judge reply")
    return json.loads(text[start:end + 1])


@retry_on_rate_limit
def _create(client, model: str, messages: list[dict], **kwargs):
    """Groq chat completion (full reply, for `usage`), retrying on HTTP 429."""
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model: str | None = None, *, client=None, delay: float = JUDGE_DELAY_SECONDS,
                 reasoning_effort: str | None = None):
        settings = get_settings()
        # DeepEval's `name` drops the "qwen/" prefix; Groq needs the full id.
        self.model_name = model or settings.groq_judge_model
        self.delay = delay
        self.reasoning_effort = (settings.groq_judge_reasoning_effort if reasoning_effort is None
                                 else reasoning_effort)
        self.usage = JudgeUsage()
        self._client = client
        super().__init__(self.model_name)

    def load_model(self):
        return self._client or get_groq_client()

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        kwargs = {"temperature": 0}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        reply = _create(self.model, self.model_name, [{"role": "user", "content": prompt}],
                        **kwargs)
        text = clean_reply(reply.choices[0].message.content or "")
        self._count(prompt, text, getattr(reply, "usage", None))
        if self.delay:
            time.sleep(self.delay)
        if schema is None:
            return text
        try:
            return schema.model_validate(extract_json(text))
        except (ValueError, ValidationError):
            return text  # DeepEval falls back to its own JSON parsing (and reports the error)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name

    def _count(self, prompt: str, text: str, usage) -> None:
        self.usage.calls += 1
        if usage is not None and getattr(usage, "prompt_tokens", None) is not None:
            self.usage.prompt += usage.prompt_tokens
            self.usage.completion += usage.completion_tokens or 0
        else:  # ~4 characters per token
            self.usage.prompt += len(prompt) // 4
            self.usage.completion += len(text) // 4
            self.usage.estimated_calls += 1
