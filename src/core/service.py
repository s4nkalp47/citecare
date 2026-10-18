"""RAGService: the single entry point used by the API and the CLI."""

from __future__ import annotations

import time

from llama_index.core.schema import NodeWithScore

from src.core.generator import Generator
from src.core.projects import get_project
from src.core.query_engine import RAGQueryEngine
from src.core.retriever import Retriever
from src.core.schemas import RAGResponse, SourceInfo
from src.core.sources import SourceManager


class RAGService:
    def __init__(self, project_name: str, *, retriever: Retriever | None = None,
                 generator: Generator | None = None, sources: SourceManager | None = None):
        self.project = get_project(project_name)
        self._sources = sources
        self.query_engine = RAGQueryEngine(
            retriever=retriever or Retriever(self.project),
            generator=generator or Generator(self.project),
        )

    def ask(self, question: str, source_type: str | None = None) -> RAGResponse:
        return self.ask_with_nodes(question, source_type)[0]

    def ask_with_nodes(self, question: str, source_type: str | None = None,
                       ) -> tuple[RAGResponse, list[NodeWithScore]]:
        """`ask` plus the retrieved nodes with their metadata (evals check which chunks were
        retrieved; `RAGResponse.contexts` only has their text)."""
        start = time.perf_counter()
        engine = self.query_engine
        if source_type:
            # Shallow copy: shares the retriever and generator, only the filter differs.
            engine = engine.model_copy(update={"source_type": source_type})
        output = engine.query(question)
        result: RAGResponse = output.response
        result.latency_ms = round((time.perf_counter() - start) * 1000)
        return result, list(output.source_nodes)

    # ---------- Sources (Phase 5) ----------

    @property
    def sources(self) -> SourceManager:
        # Built on first use, so ask()-only callers never create the ingestor.
        if self._sources is None:
            self._sources = SourceManager(self.project)
        return self._sources

    def upload_source(self, filename: str, data: bytes) -> SourceInfo:
        return self.sources.upload(filename, data)

    def list_sources(self) -> list[SourceInfo]:
        return self.sources.list()

    def delete_source(self, source_id: str) -> int:
        return self.sources.delete(source_id)
