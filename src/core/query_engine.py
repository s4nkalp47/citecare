"""RAGQueryEngine: a LlamaIndex query engine that retrieves, then generates a structured answer."""

from __future__ import annotations

from llama_index.core.base.response.schema import PydanticResponse
from llama_index.core.query_engine import CustomQueryEngine

from src.core.generator import Generator
from src.core.retriever import Retriever
from src.core.schemas import RAGResponse

NO_EVIDENCE_ANSWER = "No sufficiently relevant sources were found to answer this question."


class RAGQueryEngine(CustomQueryEngine):
    """`query(question)` returns a PydanticResponse whose `.response` is a RAGResponse."""

    retriever: Retriever
    generator: Generator
    # custom_query only receives the question string, so the metadata filter is engine state.
    source_type: str | None = None

    def custom_query(self, query_str: str) -> PydanticResponse:
        nodes = self.retriever.retrieve(query_str, source_type=self.source_type)
        if not nodes:  # nothing passed the similarity cutoff: don't spend an LLM call
            result = RAGResponse(answer=NO_EVIDENCE_ANSWER, confidence="insufficient_evidence")
        else:
            result = self.generator.generate(query_str, nodes)
        return PydanticResponse(response=result, source_nodes=nodes)
