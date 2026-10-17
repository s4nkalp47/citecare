"""Qdrant access: one collection per project, chunks carry the Document metadata as payload."""

from functools import lru_cache

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from src.config import get_settings
from src.core.projects import ProjectConfig


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, timeout=30)


def get_vector_store(project: ProjectConfig, client: QdrantClient | None = None) -> QdrantVectorStore:
    # The collection is created on the first insert, sized from the embedding dimension.
    return QdrantVectorStore(collection_name=project.collection_name,
                             client=client or get_qdrant_client())


def count_points(project: ProjectConfig, client: QdrantClient | None = None,
                 source_id: str | None = None) -> int:
    """Number of chunks in the project's collection (0 if it doesn't exist yet)."""
    client = client or get_qdrant_client()
    if not client.collection_exists(project.collection_name):
        return 0
    count_filter = _source_filter(source_id) if source_id else None
    return client.count(project.collection_name, count_filter=count_filter, exact=True).count


def delete_by_source_id(project: ProjectConfig, source_id: str,
                        client: QdrantClient | None = None) -> int:
    """Delete every chunk of one source (e.g. a guideline PDF). Returns how many were removed."""
    client = client or get_qdrant_client()
    n = count_points(project, client, source_id=source_id)
    if n:
        client.delete(project.collection_name,
                      points_selector=rest.FilterSelector(filter=_source_filter(source_id)),
                      wait=True)
    return n


def _source_filter(source_id: str) -> rest.Filter:
    return rest.Filter(must=[rest.FieldCondition(key="source_id",
                                                 match=rest.MatchValue(value=source_id))])
