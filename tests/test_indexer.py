"""Indexer tests: in-memory Qdrant and a mock embedding model, no network."""

from pathlib import Path

import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from qdrant_client import QdrantClient

from src.core.indexer import Indexer
from src.core.projects import get_project
from src.core.vector_store import count_points, delete_by_source_id

PROJECT = get_project("medrag")


def _doc(doc_id: str, source_id: str, text: str, **metadata) -> Document:
    return Document(text=text, id_=doc_id, metadata={
        "source_type": "guideline", "source_id": source_id, "title": f"Guideline {source_id}",
        **metadata,
    })


def _docs() -> list[Document]:
    long_text = " ".join(f"Sentence {i} about blood pressure targets." for i in range(300))
    return [
        _doc("a:p1", "a", long_text, page=1),
        _doc("a:p2", "a", "Offer an ACE inhibitor to adults under 55.", page=2),
        _doc("b", "b", "Metformin is first-line for type 2 diabetes."),
    ]


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def make_indexer(client, tmp_path: Path):
    def make() -> Indexer:  # a fresh Indexer reloads the persisted docstore, like a new run
        return Indexer(PROJECT, client=client, embed_model=MockEmbedding(embed_dim=384),
                       storage_dir=tmp_path / "pipeline_storage")
    return make


def _payloads(client: QdrantClient) -> list[dict]:
    points, _ = client.scroll(PROJECT.collection_name, limit=1000, with_payload=True)
    return [p.payload for p in points]


def test_count_points_without_collection(client):
    assert count_points(PROJECT, client) == 0


def test_run_writes_chunks_with_metadata(make_indexer, client):
    stats = make_indexer().run(_docs())

    assert stats.documents_in == 3
    assert stats.chunks_written > 3  # the long page is split into several chunks
    assert stats.total_points == stats.chunks_written == count_points(PROJECT, client)
    payloads = _payloads(client)
    assert {p["source_id"] for p in payloads} == {"a", "b"}
    page2 = next(p for p in payloads if p.get("page") == 2)
    assert page2["title"] == "Guideline a"
    assert page2["source_type"] == "guideline"
    assert page2["doc_id"] == "a:p2"


def test_rerun_is_idempotent(make_indexer, client):
    first = make_indexer().run(_docs())
    second = make_indexer().run(_docs())

    assert second.chunks_written == 0
    assert second.total_points == first.total_points


def test_changed_document_replaces_its_chunks(make_indexer, client):
    make_indexer().run(_docs())
    before = count_points(PROJECT, client)

    changed = _docs()
    changed[2] = _doc("b", "b", "Updated: metformin plus SGLT2 inhibitor if CVD.")
    stats = make_indexer().run(changed)

    assert stats.chunks_written == 1
    assert count_points(PROJECT, client) == before
    texts = [p["_node_content"] for p in _payloads(client) if p["source_id"] == "b"]
    assert len(texts) == 1 and "SGLT2" in texts[0]


def test_missing_documents_are_kept(make_indexer, client):
    first = make_indexer().run(_docs())
    make_indexer().run(_docs()[:1])  # e.g. a --no-pubmed run
    assert count_points(PROJECT, client) == first.total_points


def test_delete_by_source_id_removes_only_that_source(make_indexer, client):
    make_indexer().run(_docs())
    n_b = count_points(PROJECT, client, source_id="b")

    removed = delete_by_source_id(PROJECT, "a", client)

    assert removed > 0
    assert count_points(PROJECT, client, source_id="a") == 0
    assert count_points(PROJECT, client) == n_b == 1
    assert delete_by_source_id(PROJECT, "missing", client) == 0


def test_delete_source_allows_reindexing(make_indexer, client):
    first = make_indexer().run(_docs())
    make_indexer().delete_source("a")
    stats = make_indexer().run(_docs())

    assert stats.chunks_written == first.chunks_written - 1  # only source "a" re-added
    assert stats.total_points == first.total_points


def test_wiped_collection_triggers_full_reindex(make_indexer, client):
    first = make_indexer().run(_docs())
    client.delete_collection(PROJECT.collection_name)

    stats = make_indexer().run(_docs())
    assert stats.chunks_written == first.chunks_written
