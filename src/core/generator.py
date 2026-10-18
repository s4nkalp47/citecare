"""Generator: retrieved chunks + question -> JSON answer from the LLM -> RAGResponse.

The LLM only chooses which numbered context blocks it used; citation metadata (title, page, url)
always comes from the retrieved nodes so it can't be hallucinated.
"""

from __future__ import annotations

import json
import logging
import re

from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, ValidationError

from src.core.llm import chat_with_retry, get_llm
from src.core.projects import ProjectConfig
from src.core.schemas import Citation, Confidence, RAGResponse

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 300

JSON_INSTRUCTIONS = """\

Respond with a single JSON object and nothing else, in this format:
{"answer": "<your answer, citing context blocks inline as [n]>",
 "cited_sources": [<numbers of the context blocks the answer relies on>],
 "confidence": "high" | "medium" | "low" | "insufficient_evidence"}
The answer MUST put an inline marker such as [2] right after each claim, using the number of the
context block it came from. Every number in cited_sources must appear as a [n] marker in the
answer, and every [n] marker in the answer must be listed in cited_sources."""

RETRY_MESSAGE = ("Your previous reply was not a valid JSON object in the required format. "
                 "Reply again with only the JSON object.")

INVALID_OUTPUT_ANSWER = "The model did not return a valid structured answer. Please try again."

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_MARKER = re.compile(r"\[\d+\]")
# gpt-oss sometimes writes 【2】 or 【2†L3-L5】 instead of [2].
_FULLWIDTH_MARKER = re.compile(r"【(\d+)(?:†[^】]*)?】")


class LLMAnswer(BaseModel):
    answer: str
    cited_sources: list[int] = []
    confidence: Confidence


def parse_llm_answer(text: str) -> LLMAnswer:
    """Parse the LLM reply, tolerating ``` fences and text around the JSON object."""
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in reply")
    return LLMAnswer.model_validate(json.loads(text[start:end + 1]))


def format_context(nodes: list[NodeWithScore]) -> str:
    blocks = []
    for i, node in enumerate(nodes, start=1):
        meta = node.node.metadata
        header = f"[{i}] {meta.get('title', 'Untitled')}"
        if meta.get("page") is not None:
            header += f", page {meta['page']}"
        blocks.append(f"{header}\n{node.node.get_content()}")
    return "\n\n".join(blocks)


def _citation(ref: int, node: NodeWithScore) -> Citation:
    meta = node.node.metadata
    text = " ".join(node.node.get_content().split())
    return Citation(
        ref=ref,
        title=meta.get("title", "Untitled"),
        page=meta.get("page"),
        url=meta.get("url") or None,
        source_type=meta.get("source_type", "unknown"),
        snippet=text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else ""),
        score=node.score,
    )


class Generator:
    def __init__(self, project: ProjectConfig, llm: LLM | None = None):
        self.project = project
        self.llm = llm or get_llm()

    def generate(self, question: str, nodes: list[NodeWithScore]) -> RAGResponse:
        contexts = [n.node.get_content() for n in nodes]
        user_prompt = self.project.qa_prompt.format(context_str=format_context(nodes),
                                                    query_str=question) + JSON_INSTRUCTIONS
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=self.project.system_prompt),
                    ChatMessage(role=MessageRole.USER, content=user_prompt)]

        parsed = None
        for attempt in range(2):  # one retry on invalid JSON
            reply = chat_with_retry(self.llm, messages)
            try:
                parsed = parse_llm_answer(reply)
                break
            except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
                logger.warning("Invalid JSON from LLM (attempt %d): %s", attempt + 1, exc)
                messages += [ChatMessage(role=MessageRole.ASSISTANT, content=reply),
                             ChatMessage(role=MessageRole.USER, content=RETRY_MESSAGE)]
        if parsed is None:
            return RAGResponse(answer=INVALID_OUTPUT_ANSWER, confidence="insufficient_evidence",
                               contexts=contexts)

        # Keep only block numbers that exist, once each, in the order the LLM cited them.
        refs = [r for r in dict.fromkeys(parsed.cited_sources) if 1 <= r <= len(nodes)]
        answer = _FULLWIDTH_MARKER.sub(r"[\1]", parsed.answer)
        if refs and not _MARKER.search(answer):
            # The model cited sources but left no inline markers: append them so they still show.
            answer = answer.rstrip() + " " + "".join(f"[{r}]" for r in refs)
        return RAGResponse(
            answer=answer,
            citations=[_citation(r, nodes[r - 1]) for r in refs],
            confidence=parsed.confidence,
            contexts=contexts,
        )
