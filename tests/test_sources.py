"""SourceManager tests: in-memory Qdrant, mock embeddings, mocked LlamaParse (no credits)."""

from pathlib import Path
from unittest.mock import patch

import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from qdrant_client import QdrantClient

from src.config import Settings
from src.core.generator import Generator
from src.core.indexer import Indexer
from src.core.projects import get_project
from src.core.retriever import Retriever
from src.core.service import RAGService
from src.core.sources import (
    InvalidSourceError,
    SourceManager,
    SourceNotFoundError,
    safe_filename,
)
from src.core.vector_store import count_points
from src.plugins.medrag.ingestor import MedRAGIngestor, file_hash
from tests.conftest import FakeLLM

PROJECT = get_project("medrag")
PDF = b"%PDF-1.4 fake hypertension guideline"
FAKE_PAGES = [
    {"page": 1, "markdown": "# Hypertension\n1.4.2 Offer an ACE inhibitor or ARB."},
    {"page": 2, "markdown": "1.4.3 Offer a calcium-channel blocker to people over 55."},
]


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def ingestor(tmp_path: Path) -> MedRAGIngestor:
    settings = Settings(data_dir=tmp_path, llama_cloud_api_key="test", _env_file=None)
    return MedRAGIngestor(PROJECT, settings)


@pytest.fixture
def manager(ingestor, client, tmp_path: Path) -> SourceManager:
    return SourceManager(PROJECT, ingestor=ingestor, client=client,
                         embed_model=MockEmbedding(embed_dim=384),
                         storage_dir=tmp_path / "pipeline_storage", max_upload_bytes=1024)


@pytest.fixture
def parse_pdf():
    with patch.object(MedRAGIngestor, "_parse_pdf", return_value=FAKE_PAGES) as mock:
        yield mock


def test_medrag_registers_its_ingestor():
    assert PROJECT.ingestor_factory is MedRAGIngestor


def test_upload_list_delete_roundtrip(manager, ingestor, client, parse_pdf):
    info = manager.upload("nice_ng136_hypertension.pdf", PDF)

    assert info.source_id == file_hash(ingestor.guidelines_dir / "nice_ng136_hypertension.pdf")
    assert info.title == "NICE NG136 Hypertension"
    assert info.source_type == "guideline"
    assert info.file_name == "nice_ng136_hypertension.pdf"
    assert info.chunks == 2 == count_points(PROJECT, client)
    assert manager.list() == [info]
    assert list(ingestor.cache_dir.glob(f"*-{info.source_id}.json"))

    assert manager.delete(info.source_id) == 2

    assert manager.list() == []
    assert count_points(PROJECT, client) == 0
    assert not (ingestor.guidelines_dir / "nice_ng136_hypertension.pdf").exists()
    assert not list(ingestor.cache_dir.glob("*.json"))
    with pytest.raises(SourceNotFoundError):
        manager.delete(info.source_id)


def test_deleted_source_can_be_uploaded_again(manager, parse_pdf):
    info = manager.upload("guide.pdf", PDF)
    manager.delete(info.source_id)
    assert manager.upload("guide.pdf", PDF).chunks == 2  # docstore entries were removed too


def test_reupload_same_file_is_idempotent(manager, client, parse_pdf):
    first = manager.upload("guide.pdf", PDF)
    second = manager.upload("guide.pdf", PDF)

    assert second == first
    assert count_points(PROJECT, client) == 2
    assert parse_pdf.call_count == 1  # the parse cache was reused


def test_same_content_under_another_name_is_rejected(manager, ingestor, parse_pdf):
    manager.upload("guide.pdf", PDF)
    with pytest.raises(InvalidSourceError, match="already indexed"):
        manager.upload("copy.pdf", PDF)
    assert not (ingestor.guidelines_dir / "copy.pdf").exists()


def test_different_file_with_existing_name_is_rejected(manager, parse_pdf):
    manager.upload("guide.pdf", PDF)
    with pytest.raises(InvalidSourceError, match="already exists"):
        manager.upload("guide.pdf", PDF + b" v2")


@pytest.mark.parametrize("filename, data, match", [
    ("notes.txt", b"plain text", "Only PDF"),
    ("fake.pdf", b"MZ\x90\x00 not a pdf", "not a valid PDF"),
    ("empty.pdf", b"", "empty"),
    ("big.pdf", b"%PDF-" + b"x" * 2048, "limit"),
])
def test_upload_rejects_invalid_files(manager, ingestor, parse_pdf, filename, data, match):
    with pytest.raises(InvalidSourceError, match=match):
        manager.upload(filename, data)
    parse_pdf.assert_not_called()
    assert not list(ingestor.guidelines_dir.glob("*"))


def test_failed_parse_removes_saved_file(manager, ingestor, client):
    with (patch.object(MedRAGIngestor, "_parse_pdf", side_effect=RuntimeError("LlamaParse down")),
          pytest.raises(RuntimeError)):
        manager.upload("guide.pdf", PDF)
    assert not (ingestor.guidelines_dir / "guide.pdf").exists()
    assert count_points(PROJECT, client) == 0


def test_pdf_without_text_is_rejected(manager, ingestor):
    with (patch.object(MedRAGIngestor, "_parse_pdf", return_value=[{"page": 1, "markdown": " "}]),
          pytest.raises(InvalidSourceError, match="No text")):
        manager.upload("scan.pdf", PDF)
    assert not (ingestor.guidelines_dir / "scan.pdf").exists()


def test_safe_filename_strips_paths_and_odd_characters():
    assert safe_filename("../../etc/My Guide (v2).PDF") == "My_Guide_v2.pdf"
    assert safe_filename("C:\\Users\\x\\nice_ng28.pdf") == "nice_ng28.pdf"
    with pytest.raises(InvalidSourceError):
        safe_filename("../???.pdf")


def test_list_includes_sources_indexed_by_build_index(manager, client, tmp_path, parse_pdf):
    # e.g. PubMed abstracts added by scripts.build_index, not by an upload.
    pubmed = Document(text="SGLT2 inhibitors reduce admissions.", id_="pmid-1", metadata={
        "source_id": "pmid-1", "source_type": "pubmed", "title": "SGLT2 meta-analysis"})
    Indexer(PROJECT, client=client, embed_model=MockEmbedding(embed_dim=384),
            storage_dir=tmp_path / "pipeline_storage").run([pubmed])
    manager.upload("guide.pdf", PDF)

    sources = manager.list()
    assert [(s.source_type, s.title, s.chunks) for s in sources] == [
        ("guideline", "Guide", 2), ("pubmed", "SGLT2 meta-analysis", 1)]

    # Deleting a non-file source removes its chunks and leaves the PDF alone.
    assert manager.delete("pmid-1") == 1
    assert [s.source_type for s in manager.list()] == ["guideline"]


def test_list_without_collection_is_empty(manager):
    assert manager.list() == []


def test_service_delegates_to_source_manager(manager, client, parse_pdf):
    service = RAGService("medrag", sources=manager,
                         retriever=Retriever(PROJECT, client=client,
                                             embed_model=MockEmbedding(embed_dim=384)),
                         generator=Generator(PROJECT, llm=FakeLLM()))

    info = service.upload_source("guide.pdf", PDF)
    assert service.list_sources() == [info]
    assert service.delete_source(info.source_id) == 2
    assert service.list_sources() == []
