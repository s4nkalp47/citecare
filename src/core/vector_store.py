"""Qdrant access: one collection per project, chunks carry the Document metadata as payload."""

import json
from collections import Counter
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
    # The collection is created on the first insert, sized from the embedding dimension. With
    # `project.hybrid` it gets named dense + sparse (BM25) vectors, so switching hybrid on or off
    # needs a rebuilt collection (scripts.build_index --rebuild).
    if project.hybrid:
        return QdrantVectorStore(collection_name=project.collection_name,
                                 client=client or get_qdrant_client(), enable_hybrid=True,
                                 fastembed_sparse_model=get_settings().sparse_model)
    return QdrantVectorStore(collection_name=project.collection_name,
                             client=client or get_qdrant_client())


def count_points(project: ProjectConfig, client: QdrantClient | None = None,
                 source_id: str | None = None) -> int:
    """Number of chunks in the project's collection (0 if it doesn't exist yet)."""
    client = client or get_qdrant_client()
    if not client.collection_exists(project.collection_name):
        return 0
    count_filter = source_filter(source_id) if source_id else None
    return client.count(project.collection_name, count_filter=count_filter, exact=True).count


def delete_by_source_id(project: ProjectConfig, source_id: str,
                        client: QdrantClient | None = None) -> int:
    """Delete every chunk of one source (e.g. a guideline PDF). Returns how many were removed."""
    client = client or get_qdrant_client()
    n = count_points(project, client, source_id=source_id)
    if n:
        client.delete(project.collection_name,
                      points_selector=rest.FilterSelector(filter=source_filter(source_id)),
                      wait=True)
    return n


def source_filter(source_id: str) -> rest.Filter:
    return rest.Filter(must=[rest.FieldCondition(key="source_id",
                                                 match=rest.MatchValue(value=source_id))])


def count_duplicate_chunks(project: ProjectConfig, client: QdrantClient | None = None) -> int:
    """Points whose (doc_id, page, chunk text) another point already has; 0 for a clean index.

    The page is part of the key because one Document can span pages (a whole NICE guideline)
    and repeat a sentence on two of them.
    """
    client = client or get_qdrant_client()
    if not client.collection_exists(project.collection_name):
        return 0
    seen: Counter[tuple[str, object, str]] = Counter()
    offset = None
    while True:
        points, offset = client.scroll(project.collection_name, limit=1000, offset=offset,
                                       with_payload=["doc_id", "page", "_node_content"],
                                       with_vectors=False)
        for point in points:
            payload = point.payload or {}
            text = json.loads(payload.get("_node_content") or "{}").get("text", "")
            seen[(payload.get("doc_id", ""), payload.get("page"), text)] += 1
        if offset is None:
            return sum(n - 1 for n in seen.values())
