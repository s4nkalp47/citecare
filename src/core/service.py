"""RAGService: the single entry point used by the API and the CLI."""

from __future__ import annotations

import time

from src.core.generator import Generator
from src.core.projects import get_project
from src.core.query_engine import RAGQueryEngine
from src.core.retriever import Retriever
from src.core.schemas import RAGResponse


class RAGService:
    def __init__(self, project_name: str, *, retriever: Retriever | None = None,
                 generator: Generator | None = None):
        self.project = get_project(project_name)
        self.query_engine = RAGQueryEngine(
            retriever=retriever or Retriever(self.project),
            generator=generator or Generator(self.project),
        )

    def ask(self, question: str, source_type: str | None = None) -> RAGResponse:
        start = time.perf_counter()
        engine = self.query_engine
        if source_type:
            # Shallow copy: shares the retriever and generator, only the filter differs.
            engine = engine.model_copy(update={"source_type": source_type})
        result: RAGResponse = engine.query(question).response
        result.latency_ms = round((time.perf_counter() - start) * 1000)
        return result
