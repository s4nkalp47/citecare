"""Shared test doubles: a deterministic keyword embedding and a scripted LLM (no network)."""

import math
import re
import zlib

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole

_STOPWORDS = {"a", "an", "the", "is", "are", "for", "with", "to", "of", "and", "in", "or", "what",
              "how", "which", "should", "be", "as", "on", "who", "adults"}


class KeywordEmbedding(BaseEmbedding):
    """Bag-of-words hashed into 384 dims, so similarity reflects word overlap.

    Unlike MockEmbedding (constant vectors, every score 1.0) this makes cutoffs testable.
    """

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * 384
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word not in _STOPWORDS:
                vec[zlib.crc32(word.encode()) % 384] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)


class FakeLLM:
    """Stands in for the Groq LLM: returns scripted replies and records every chat call."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        self.calls.append(list(messages))
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT,
                                                content=self.replies.pop(0)))
