"""Thin httpx client for the MedRAG API (src/api/main.py), returning the API's own pydantic models.

A guardrail block (HTTP 400 with `blocked: true`) is an expected outcome, so `ask` returns it as a
`BlockedResponse` instead of raising. Everything else that isn't a success raises `APIError`.
"""

from __future__ import annotations

from typing import Self
from urllib.parse import quote

import httpx

from src.api.schemas import BlockedResponse, DeleteSourceResponse, HealthResponse
from src.config import get_settings
from src.core.schemas import RAGResponse, SourceInfo

# /ask runs two guard stages, retrieval, reranking and the LLM, with 429 retries on the free tier;
# /sources parses the PDF with LlamaParse before indexing it. Both can take a while.
ASK_TIMEOUT = 180.0
UPLOAD_TIMEOUT = 600.0
DEFAULT_TIMEOUT = 15.0


class APIError(Exception):
    """The API answered with an unexpected status, or could not be reached (status_code None)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class APIUnavailableError(APIError):
    """Connection refused, DNS failure or timeout: the API isn't running or is too slow."""


class MedRAGClient:
    def __init__(self, base_url: str | None = None, *,
                 transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or get_settings().api_url).rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=DEFAULT_TIMEOUT,
                                  transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- Endpoints ----------

    def health(self) -> HealthResponse:
        # 503 still carries a HealthResponse ("degraded": Qdrant unreachable).
        response = self._request("GET", "/health", ok=(200, 503))
        return HealthResponse.model_validate(response.json())

    def ask(self, question: str, source_type: str | None = None) -> RAGResponse | BlockedResponse:
        payload = {"question": question, "source_type": source_type or None}
        response = self._request("POST", "/ask", ok=(200, 400), json=payload, timeout=ASK_TIMEOUT)
        if response.status_code == 400:
            body = _json_or_none(response)
            if isinstance(body, dict) and body.get("blocked") is True:
                return BlockedResponse.model_validate(body)
            raise APIError(_detail(response), response.status_code)
        return RAGResponse.model_validate(response.json())

    def list_sources(self) -> list[SourceInfo]:
        response = self._request("GET", "/sources")
        return [SourceInfo.model_validate(item) for item in response.json()]

    def upload_source(self, file_name: str, content: bytes) -> SourceInfo:
        files = {"file": (file_name, content, "application/pdf")}
        response = self._request("POST", "/sources", ok=(201,), files=files,
                                 timeout=UPLOAD_TIMEOUT)
        return SourceInfo.model_validate(response.json())

    def delete_source(self, source_id: str) -> DeleteSourceResponse:
        # Source ids contain ':' (e.g. "pdf:abc123"); quote everything to keep it one path segment.
        response = self._request("DELETE", f"/sources/{quote(source_id, safe='')}")
        return DeleteSourceResponse.model_validate(response.json())

    # ---------- Internals ----------

    def _request(self, method: str, path: str, *, ok: tuple[int, ...] = (200,),
                 **kwargs) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise APIUnavailableError(f"The API at {self.base_url} timed out") from exc
        except httpx.TransportError as exc:
            raise APIUnavailableError(f"The API at {self.base_url} is not reachable") from exc
        if response.status_code not in ok:
            raise APIError(_detail(response), response.status_code)
        return response


def _json_or_none(response: httpx.Response):
    try:
        return response.json()
    except ValueError:
        return None


def _detail(response: httpx.Response) -> str:
    """FastAPI's `{"detail": ...}` message when there is one, else the status line."""
    body = _json_or_none(response)
    if isinstance(body, dict) and isinstance(body.get("detail"), str):
        return body["detail"]
    return f"HTTP {response.status_code} {response.reason_phrase}".strip()
