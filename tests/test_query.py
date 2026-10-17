"""Query-side tests: retriever, generator, query engine, service, LLM retries.

In-memory Qdrant, keyword embedding and a scripted LLM: no Groq calls.
"""

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from llama_index.core import Document
from llama_index.core.base.response.schema import PydanticResponse
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore, TextNode
from qdrant_client import QdrantClient
from tenacity import wait_none

from src.config import Settings
from src.core import llm as llm_module
from src.core.generator import INVALID_OUTPUT_ANSWER, Generator, parse_llm_answer
from src.core.indexer import Indexer
from src.core.llm import chat_with_retry, get_llm
from src.core.projects import get_project
from src.core.query_engine import NO_EVIDENCE_ANSWER, RAGQueryEngine
from src.core.retriever import Retriever
from src.core.service import RAGService
from tests.conftest import FakeLLM, KeywordEmbedding

# Fixed cutoff so these tests don't depend on the calibrated MedRAG value.
PROJECT = dataclasses.replace(get_project("medrag"), similarity_cutoff=0.3)
NG136_URL = "https://www.nice.org.uk/guidance/ng136"


def _doc(doc_id: str, text: str, **metadata) -> Document:
    doc = Document(text=text, id_=doc_id, metadata={"source_id": doc_id, **metadata})
    doc.excluded_embed_metadata_keys = list(doc.metadata)  # score on text only
    return doc


DOCS = [
    _doc("ng136:p12", "Offer an ACE inhibitor or ARB to people with hypertension and type 2 "
         "diabetes.", source_type="guideline", title="NICE NG136 Hypertension", page=12,
         url=NG136_URL),
    _doc("ng28:p5", "Offer metformin as first-line treatment for type 2 diabetes.",
         source_type="guideline", title="NICE NG28 Type2 Diabetes", page=5,
         url="https://www.nice.org.uk/guidance/ng28"),
    _doc("pmid-1", "SGLT2 inhibitors reduce heart failure admissions in type 2 diabetes.",
         source_type="pubmed", title="SGLT2 inhibitors: a meta-analysis",
         url="https://pubmed.ncbi.nlm.nih.gov/1/"),
]


def _reply(answer="Offer metformin [1].", cited=(1,), confidence="high") -> str:
    return json.dumps({"answer": answer, "cited_sources": list(cited), "confidence": confidence})


def _nodes() -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text=d.text, metadata=d.metadata), score=0.9 - i / 10)
            for i, d in enumerate(DOCS)]


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def retriever(client, tmp_path: Path) -> Retriever:
    embed = KeywordEmbedding()
    Indexer(PROJECT, client=client, embed_model=embed,
            storage_dir=tmp_path / "pipeline_storage").run(DOCS)
    return Retriever(PROJECT, client=client, embed_model=embed)


def _service(retriever: Retriever, llm: FakeLLM) -> RAGService:
    return RAGService("medrag", retriever=retriever, generator=Generator(PROJECT, llm=llm))


# ---------- Retriever ----------

def test_retriever_ranks_relevant_chunk_first(retriever):
    nodes = retriever.retrieve("metformin first-line treatment type 2 diabetes")
    assert nodes[0].node.metadata["title"] == "NICE NG28 Type2 Diabetes"
    assert all(n.score >= PROJECT.similarity_cutoff for n in nodes)


def test_retriever_source_type_filter(retriever):
    nodes = retriever.search("type 2 diabetes treatment", source_type="pubmed")
    assert nodes and {n.node.metadata["source_type"] for n in nodes} == {"pubmed"}


def test_off_topic_query_falls_below_cutoff(retriever):
    assert retriever.search("zebra stripes savanna")  # something comes back...
    assert retriever.retrieve("zebra stripes savanna") == []  # ...but nothing passes


def test_retriever_missing_collection_returns_empty(client):
    retriever = Retriever(PROJECT, client=client, embed_model=KeywordEmbedding())
    assert retriever.search("metformin") == []
    assert retriever.retrieve("metformin") == []


# ---------- Generator ----------

def test_citations_come_from_node_metadata():
    llm = FakeLLM(json.dumps({
        "answer": "Offer an ACE inhibitor [1].", "cited_sources": [1],
        "confidence": "high", "title": "Made-up title", "url": "https://evil.example",
    }))
    response = Generator(PROJECT, llm=llm).generate("BP drug in diabetes?", _nodes())

    [citation] = response.citations
    assert citation.ref == 1
    assert citation.title == "NICE NG136 Hypertension"
    assert citation.page == 12
    assert citation.url == NG136_URL
    assert citation.source_type == "guideline"
    assert citation.snippet.startswith("Offer an ACE inhibitor")
    assert citation.score == pytest.approx(0.9)
    assert response.contexts == [d.text for d in DOCS]


def test_cited_sources_map_by_number_and_drop_invalid():
    llm = FakeLLM(_reply(cited=[3, 9, 0, 3, 2]))
    response = Generator(PROJECT, llm=llm).generate("q", _nodes())

    assert [c.ref for c in response.citations] == [3, 2]
    assert response.citations[0].title == "SGLT2 inhibitors: a meta-analysis"
    assert response.citations[0].page is None


def test_missing_inline_markers_are_appended():
    llm = FakeLLM(_reply(answer="Offer metformin first-line.", cited=[2, 1]))
    response = Generator(PROJECT, llm=llm).generate("q", _nodes())
    assert response.answer == "Offer metformin first-line. [2][1]"


def test_existing_inline_markers_are_kept():
    llm = FakeLLM(_reply(answer="Offer metformin [2].", cited=[2]))
    response = Generator(PROJECT, llm=llm).generate("q", _nodes())
    assert response.answer == "Offer metformin [2]."


def test_no_markers_appended_without_citations():
    llm = FakeLLM(_reply(answer="Not covered.", cited=[], confidence="insufficient_evidence"))
    response = Generator(PROJECT, llm=llm).generate("q", _nodes())
    assert response.answer == "Not covered."


def test_prompt_has_numbered_context_and_system_prompt():
    llm = FakeLLM(_reply())
    Generator(PROJECT, llm=llm).generate("What about metformin?", _nodes())

    system, user = llm.calls[0]
    assert system.content == PROJECT.system_prompt
    assert "[1] NICE NG136 Hypertension, page 12" in user.content
    assert "[3] SGLT2 inhibitors: a meta-analysis\n" in user.content
    assert "What about metformin?" in user.content
    assert "cited_sources" in user.content
    assert "MUST put an inline marker" in user.content


def test_invalid_json_is_retried_once():
    llm = FakeLLM("Sure! Metformin is first-line.", _reply())
    response = Generator(PROJECT, llm=llm).generate("q", _nodes())

    assert len(llm.calls) == 2
    assert llm.calls[1][-2].content == "Sure! Metformin is first-line."  # bad reply sent back
    assert response.confidence == "high"
    assert [c.ref for c in response.citations] == [1]


def test_invalid_json_twice_returns_insufficient_evidence():
    llm = FakeLLM("not json", '{"answer": "x", "confidence": "very sure"}')
    response = Generator(PROJECT, llm=llm).generate("q", _nodes())

    assert len(llm.calls) == 2
    assert response.confidence == "insufficient_evidence"
    assert response.answer == INVALID_OUTPUT_ANSWER
    assert response.citations == []


def test_parse_llm_answer_accepts_fenced_json():
    parsed = parse_llm_answer(f"```json\n{_reply(cited=[2])}\n```")
    assert parsed.cited_sources == [2]
    assert parsed.confidence == "high"


# ---------- Query engine and service ----------

def test_query_engine_returns_structured_response(retriever):
    llm = FakeLLM(_reply())
    engine = RAGQueryEngine(retriever=retriever, generator=Generator(PROJECT, llm=llm))

    result = engine.query("metformin first-line treatment type 2 diabetes")

    assert isinstance(result, PydanticResponse)
    assert result.response.citations[0].title == "NICE NG28 Type2 Diabetes"
    assert result.source_nodes[0].node.metadata["title"] == "NICE NG28 Type2 Diabetes"


def test_service_ask_end_to_end(retriever):
    response = _service(retriever, FakeLLM(_reply())).ask(
        "metformin first-line treatment type 2 diabetes")

    assert response.confidence == "high"
    assert response.citations[0].url == "https://www.nice.org.uk/guidance/ng28"
    assert response.contexts
    assert response.latency_ms >= 0


def test_service_insufficient_evidence_skips_llm(retriever):
    llm = FakeLLM()
    response = _service(retriever, llm).ask("zebra stripes savanna")

    assert llm.calls == []
    assert response.confidence == "insufficient_evidence"
    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.citations == [] and response.contexts == []


def test_service_source_type_filter(retriever):
    service = _service(retriever, FakeLLM(_reply()))
    response = service.ask("heart failure SGLT2 type 2 diabetes", source_type="pubmed")

    assert {c.source_type for c in response.citations} == {"pubmed"}
    assert service.query_engine.source_type is None  # the shared engine is not mutated


# ---------- LLM ----------

class _RateLimited(Exception):
    status_code = 429


class _FlakyLLM:
    def __init__(self, error: Exception, failures: int):
        self.error, self.failures, self.calls = error, failures, 0

    def chat(self, messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return FakeLLM("ok").chat(messages)


def test_chat_with_retry_retries_on_429():
    llm = _FlakyLLM(_RateLimited(), failures=2)
    fast = chat_with_retry.retry_with(wait=wait_none())
    assert fast(llm, [ChatMessage(content="hi")]) == "ok"
    assert llm.calls == 3


def test_chat_with_retry_does_not_retry_other_errors():
    llm = _FlakyLLM(ValueError("bad request"), failures=1)
    with pytest.raises(ValueError):
        chat_with_retry.retry_with(wait=wait_none())(llm, [ChatMessage(content="hi")])
    assert llm.calls == 1


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_get_llm_configures_groq():
    get_llm.cache_clear()
    with patch.object(llm_module, "get_settings",
                      return_value=_settings(groq_api_key="test", groq_llm_model="m")):
        llm = get_llm()
    get_llm.cache_clear()

    assert llm.model == "m"
    assert llm.temperature == 0
    assert llm.max_retries == 0
    assert llm.additional_kwargs == {"reasoning_effort": "low"}


def test_get_llm_requires_api_key():
    get_llm.cache_clear()
    with (patch.object(llm_module, "get_settings", return_value=_settings(groq_api_key="")),
          pytest.raises(RuntimeError, match="GROQ_API_KEY")):
        get_llm()
    get_llm.cache_clear()
