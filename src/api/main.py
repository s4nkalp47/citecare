"""FastAPI app: input guard -> RAGService -> output guard, plus source management and health.

Run from the project root:
    uvicorn src.api.main:app --reload        # then open http://localhost:8000/docs

The service, guards and Qdrant client are FastAPI dependencies, built once on first use and
overridable in tests via `app.dependency_overrides`.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from src.api.schemas import (
    AskRequest,
    BlockedResponse,
    DeleteSourceResponse,
    ErrorResponse,
    HealthModels,
    HealthResponse,
)
from src.config import Settings, get_settings
from src.core.llm import describe_error
from src.core.projects import get_project
from src.core.schemas import RAGResponse, SourceInfo
from src.core.service import RAGService
from src.core.sources import InvalidSourceError, SourceNotFoundError
from src.core.vector_store import count_points, get_qdrant_client
from src.guardrails import InputGuardrail, OutputGuardrail

logger = logging.getLogger(__name__)

# Generic texts for blocked requests: the client learns the stage and category, nothing more.
BLOCKED_MESSAGES = {
    "input": "This question can't be processed. Please rephrase it as a clinical question.",
    "output": "An answer was generated but withheld because it did not pass the safety checks.",
}


# ---------- Dependencies ----------

@lru_cache
def get_service() -> RAGService:
    return RAGService(get_settings().api_project)


@lru_cache
def get_input_guard() -> InputGuardrail:
    return InputGuardrail(get_project(get_settings().api_project))


@lru_cache
def get_output_guard() -> OutputGuardrail:
    return OutputGuardrail(get_project(get_settings().api_project))


def get_qdrant() -> QdrantClient:
    return get_qdrant_client()


ServiceDep = Annotated[RAGService, Depends(get_service)]


def _blocked(stage: str, category: str | None) -> JSONResponse:
    body = BlockedResponse(stage=stage, category=category, message=BLOCKED_MESSAGES[stage])
    return JSONResponse(status_code=400, content=body.model_dump())


# ---------- App ----------

def create_app() -> FastAPI:
    app = FastAPI(title="MedRAG API", version="0.1.0",
                  description="Clinical-guideline question answering behind input/output guardrails.")

    # Plain `def` endpoints: the service, guards and Qdrant calls are blocking, so FastAPI runs
    # them in its threadpool instead of on the event loop.

    @app.post("/ask", response_model=RAGResponse,
              responses={400: {"model": BlockedResponse, "description": "Blocked by a guardrail"}})
    def ask(request: AskRequest, service: ServiceDep,
            input_guard: Annotated[InputGuardrail, Depends(get_input_guard)],
            output_guard: Annotated[OutputGuardrail, Depends(get_output_guard)]):
        checked = input_guard.check(request.question)
        if not checked.allowed:
            return _blocked("input", checked.category)
        response = service.ask(request.question, source_type=request.source_type)
        result = output_guard.check(request.question, response)
        if not result.allowed or result.response is None:
            return _blocked("output", result.category)
        return result.response

    @app.post("/sources", response_model=SourceInfo, status_code=201,
              responses={400: {"model": ErrorResponse, "description": "Rejected upload"}})
    def upload_source(file: Annotated[UploadFile, File(description="A PDF guideline")],
                      service: ServiceDep) -> SourceInfo:
        if not file.filename:
            raise HTTPException(400, "The upload has no file name")
        try:
            return service.upload_source(file.filename, file.file.read())
        except InvalidSourceError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/sources", response_model=list[SourceInfo])
    def list_sources(service: ServiceDep) -> list[SourceInfo]:
        return service.list_sources()

    @app.delete("/sources/{source_id}", response_model=DeleteSourceResponse,
                responses={404: {"model": ErrorResponse, "description": "Unknown source"}})
    def delete_source(source_id: str, service: ServiceDep) -> DeleteSourceResponse:
        try:
            removed = service.delete_source(source_id)
        except SourceNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        return DeleteSourceResponse(source_id=source_id, chunks_removed=removed)

    @app.get("/health", response_model=HealthResponse,
             responses={503: {"model": HealthResponse, "description": "Qdrant unreachable"}})
    def health(settings: Annotated[Settings, Depends(get_settings)],
               client: Annotated[QdrantClient, Depends(get_qdrant)]):
        project = get_project(settings.api_project)
        try:
            points = count_points(project, client)
            reachable = True
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            logger.warning("Health check: Qdrant unreachable (%s)", describe_error(exc))
            points, reachable = None, False
        body = HealthResponse(
            status="ok" if reachable else "degraded",
            project=project.name,
            qdrant_reachable=reachable,
            collection=project.collection_name,
            points=points,
            # Model names only: never keys or URLs with credentials.
            models=HealthModels(llm=settings.groq_llm_model,
                                prompt_guard=settings.groq_prompt_guard_model,
                                safety=settings.groq_safety_model,
                                topic=settings.groq_topic_model,
                                embed=settings.embed_model,
                                rerank=settings.rerank_model),
        )
        return JSONResponse(status_code=200 if reachable else 503, content=body.model_dump())

    return app


app = create_app()
