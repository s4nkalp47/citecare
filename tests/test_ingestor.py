"""Ingestor tests. LlamaParse and PubMed are mocked: no network, no credits."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document

from src.config import Settings
from src.core.projects import get_project
from src.plugins.medrag.ingestor import (
    MedRAGIngestor,
    pages_from_parse_result,
    title_from_filename,
    url_from_filename,
)

FAKE_PAGES = [
    {"page": 1, "markdown": "# Hypertension\n1.4.2 Offer an ACE inhibitor..."},
    {"page": 2, "markdown": "   "},  # blank page should be skipped
    {"page": 3, "markdown": "1.4.3 Offer a calcium-channel blocker..."},
]


@pytest.fixture
def ingestor(tmp_path: Path) -> MedRAGIngestor:
    (tmp_path / "guidelines").mkdir()
    (tmp_path / "bootstrap").mkdir()
    settings = Settings(data_dir=tmp_path, llama_cloud_api_key="test", _env_file=None)
    return MedRAGIngestor(get_project("medrag"), settings)


def _fake_pdf(ingestor: MedRAGIngestor, name: str = "nice_ng136_hypertension.pdf") -> Path:
    path = ingestor.guidelines_dir / name
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def test_title_from_filename():
    assert title_from_filename(Path("nice_ng136_hypertension.pdf")) == "NICE NG136 Hypertension"
    assert title_from_filename(Path("nice_ng28_type2_diabetes.pdf")) == "NICE NG28 Type2 Diabetes"


def test_url_from_filename():
    nice = "https://www.nice.org.uk/guidance/"
    assert url_from_filename(Path("nice_ng136_hypertension.pdf")) == nice + "ng136"
    assert url_from_filename(Path("nice_ng28_type2_diabetes.pdf")) == nice + "ng28"
    assert url_from_filename(Path("NICE-CG191-pneumonia.pdf")) == nice + "cg191"
    assert url_from_filename(Path("who_tb_guideline.pdf")) == ""  # not NICE
    assert url_from_filename(Path("nice_overview.pdf")) == ""  # no guideline code


def test_load_guidelines_builds_page_documents(ingestor):
    _fake_pdf(ingestor)
    with patch.object(MedRAGIngestor, "_parse_pdf", return_value=FAKE_PAGES):
        docs = ingestor.load_guidelines()

    assert [d.metadata["page"] for d in docs] == [1, 3]
    first = docs[0]
    assert first.metadata["source_type"] == "guideline"
    assert first.metadata["title"] == "NICE NG136 Hypertension"
    assert first.metadata["url"] == "https://www.nice.org.uk/guidance/ng136"
    assert first.id_ == f"{first.metadata['source_id']}:p1"
    assert "source_id" in first.excluded_embed_metadata_keys


NICE_PAGES = [
    {"page": 1, "markdown": "# NICE guideline\n\n# Hypertension in adults"},
    {"page": 2, "markdown": "\n".join(f"1.4.{i} Offer treatment option {i}." for i in range(1, 8))},
    {"page": 3, "markdown": "\n".join(f"1.4.{i} Review option {i}." for i in range(8, 14))},
]


def test_nice_guideline_becomes_one_document(ingestor):
    _fake_pdf(ingestor)
    with patch.object(MedRAGIngestor, "_parse_pdf", return_value=NICE_PAGES):
        [doc] = ingestor.load_guidelines()

    assert doc.id_ == doc.metadata["source_id"]
    assert doc.metadata["doc_format"] == "nice_guideline"
    assert "page" not in doc.metadata  # set per chunk by the parser
    assert "<!-- page 3 -->" in doc.text
    assert "doc_format" in doc.excluded_embed_metadata_keys


def test_nice_guideline_pages_when_not_joined(ingestor):
    _fake_pdf(ingestor)
    with patch.object(MedRAGIngestor, "_parse_pdf", return_value=NICE_PAGES):
        docs = ingestor.load_guidelines(join_nice=False)
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]


def test_guideline_parse_is_cached(ingestor):
    _fake_pdf(ingestor)
    with patch.object(MedRAGIngestor, "_parse_pdf", return_value=FAKE_PAGES) as parse:
        ingestor.load_guidelines()
        ingestor.load_guidelines()
    parse.assert_called_once()
    assert len(list(ingestor.cache_dir.glob("*.json"))) == 1


def test_document_ids_are_stable(ingestor):
    _fake_pdf(ingestor)
    with patch.object(MedRAGIngestor, "_parse_pdf", return_value=FAKE_PAGES):
        ids_first = [d.id_ for d in ingestor.load_guidelines()]
        ids_second = [d.id_ for d in ingestor.load_guidelines()]
    assert ids_first == ids_second


def _pubmed_doc(pmid: str, abstract: str = "Background: ACE inhibitors lower BP.") -> Document:
    title = "ACE inhibitors in hypertension"
    return Document(text=f"{title}\nBMJ (2020)\n\n{abstract}".strip(), metadata={
        "pmid": pmid, "title": title, "journal": "BMJ", "year": "2020",
        "publication_types": "Meta-Analysis", "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    })


def test_load_pubmed_normalizes_and_dedupes(ingestor):
    reader = MagicMock()
    reader.load_data.return_value = [_pubmed_doc("123")]
    with patch("src.plugins.medrag.ingestor.PubMedAbstractReader", return_value=reader):
        docs = ingestor.load_pubmed(queries=["q1", "q2"])  # same article from both queries

    assert len(docs) == 1
    assert docs[0].metadata["source_id"] == "pmid-123"
    assert docs[0].metadata["title"] == "ACE inhibitors in hypertension"
    assert docs[0].metadata["source_type"] == "pubmed"
    assert docs[0].metadata["year"] == "2020"
    # The search query is provenance: kept as metadata, never embedded or shown to the LLM.
    assert docs[0].metadata["query"] == "q1"
    assert "query" in docs[0].excluded_embed_metadata_keys
    assert "query" in docs[0].excluded_llm_metadata_keys


def test_load_pubmed_skips_records_without_abstract(ingestor):
    reader = MagicMock()
    reader.load_data.return_value = [_pubmed_doc("1", abstract=""), _pubmed_doc("2")]
    with patch("src.plugins.medrag.ingestor.PubMedAbstractReader", return_value=reader):
        docs = ingestor.load_pubmed(queries=["q1"])
    assert [d.metadata["source_id"] for d in docs] == ["pmid-2"]


def test_load_pubmed_survives_failed_query(ingestor):
    reader = MagicMock()
    reader.load_data.side_effect = RuntimeError("NCBI down")
    with patch("src.plugins.medrag.ingestor.PubMedAbstractReader", return_value=reader):
        assert ingestor.load_pubmed(queries=["q1"]) == []


def test_pages_from_parse_result_skips_failed_pages():
    result = SimpleNamespace(
        job=SimpleNamespace(status="COMPLETED", error_message=None),
        markdown=SimpleNamespace(pages=[
            SimpleNamespace(success=True, page_number=1, markdown="# Page one"),
            SimpleNamespace(success=False, page_number=2, error="OCR failed"),
        ]),
    )
    assert pages_from_parse_result(result, "x.pdf") == [{"page": 1, "markdown": "# Page one"}]


def test_pages_from_parse_result_raises_on_failed_job():
    result = SimpleNamespace(job=SimpleNamespace(status="FAILED", error_message="bad file"),
                             markdown=None)
    with pytest.raises(RuntimeError, match="FAILED"):
        pages_from_parse_result(result, "x.pdf")


def test_load_bootstrap(ingestor):
    assert ingestor.load_bootstrap() == []
    (ingestor.bootstrap_dir / "scope.md").write_text("MedRAG covers adult guidelines.")
    docs = ingestor.load_bootstrap()
    assert len(docs) == 1
    assert docs[0].metadata["source_type"] == "bootstrap"
