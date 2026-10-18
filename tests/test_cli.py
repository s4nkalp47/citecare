"""Phase 9: the Typer CLI with a mocked service, ingestor and indexer (no Groq, Qdrant or PubMed)."""

import dataclasses
import json
import sys
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from src import cli
from src.core.indexer import IndexStats
from src.core.projects import get_project
from src.core.schemas import Citation, RAGResponse, SourceInfo
from src.core.sources import SourceNotFoundError

RESPONSE = RAGResponse(
    answer="Offer an ACE inhibitor or ARB [1].",
    citations=[Citation(ref=1, title="NG136 Hypertension", page=12, url="https://nice.org.uk/ng136",
                        source_type="guideline", snippet="Offer an ACE inhibitor", score=1.3e-4),
               Citation(ref=2, title="NG28 Type 2 diabetes", source_type="guideline",
                        snippet="Blood pressure targets", score=0.02)],
    confidence="high", contexts=["a", "b", "c"], latency_ms=420)


class FakeService:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.calls: list[tuple] = []
        self.sources = [SourceInfo(source_id="ng136", title="NG136 Hypertension",
                                   source_type="guideline", chunks=40, file_name="ng136.pdf"),
                        SourceInfo(source_id="pubmed-123", title="A trial", source_type="pubmed",
                                   chunks=2)]

    def ask(self, question, source_type=None):
        self.calls.append(("ask", question, source_type))
        return RESPONSE

    def list_sources(self):
        return self.sources

    def delete_source(self, source_id):
        self.calls.append(("delete", source_id))
        if source_id != "ng136":
            raise SourceNotFoundError(f"No source with id '{source_id}'")
        return 40


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def service(monkeypatch):
    services: list[FakeService] = []

    def make(project_name):
        services.append(FakeService(project_name))
        return services[-1]

    monkeypatch.setattr(cli, "make_service", make)
    return services


# ---------- ask ----------

def test_ask_prints_answer_citations_and_bypass_notice(runner, service):
    result = runner.invoke(cli.app, ["ask", "BP drug in diabetes?"])
    assert result.exit_code == 0, result.output
    assert service[0].calls == [("ask", "BP drug in diabetes?", None)]
    assert service[0].project_name == "medrag"
    assert cli.GUARDRAILS_NOTICE in result.stderr
    assert "Offer an ACE inhibitor or ARB [1]." in result.stdout
    assert "Confidence: high" in result.stdout
    assert "[1] NG136 Hypertension, page 12  (top match)" in result.stdout
    assert "https://nice.org.uk/ng136" in result.stdout
    assert "[2] NG28 Type 2 diabetes\n" in result.stdout  # no page, no label
    # Never show the reranker score outside --json.
    assert "1.3e-04" not in result.stdout and "0.0001" not in result.stdout


def test_ask_json_is_parseable_and_passes_filter(runner, service):
    result = runner.invoke(cli.app, ["--project", "other", "ask", "q", "--source-type",
                                     "guideline", "--json"])
    assert result.exit_code == 0, result.output
    assert service[0].project_name == "other"
    assert service[0].calls == [("ask", "q", "guideline")]
    assert RAGResponse.model_validate(json.loads(result.stdout)) == RESPONSE
    assert cli.GUARDRAILS_NOTICE in result.stderr


def test_ask_requires_question(runner, service):
    result = runner.invoke(cli.app, ["ask"])
    assert result.exit_code == 2
    assert service == []


# ---------- sources ----------

def test_sources_list(runner, service):
    result = runner.invoke(cli.app, ["sources", "list"])
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0].startswith("ng136 ") and "40 chunks" in lines[0] and "NG136" in lines[0]
    assert lines[1].startswith("pubmed-123") and "pubmed" in lines[1]
    assert "2 sources, 42 chunks" in result.stdout


def test_sources_list_empty(runner, service, monkeypatch):
    monkeypatch.setattr(FakeService, "list_sources", lambda self: [])
    result = runner.invoke(cli.app, ["sources", "list"])
    assert result.exit_code == 0
    assert "No sources indexed." in result.stdout


def test_sources_delete_with_yes(runner, service):
    result = runner.invoke(cli.app, ["sources", "delete", "ng136", "-y"])
    assert result.exit_code == 0, result.output
    assert service[0].calls == [("delete", "ng136")]
    assert "Deleted ng136 (40 chunks removed)" in result.stdout


def test_sources_delete_asks_and_aborts(runner, service):
    result = runner.invoke(cli.app, ["sources", "delete", "ng136"], input="n\n")
    assert result.exit_code == 1
    assert service == []  # nothing built, nothing deleted


def test_sources_delete_confirmed(runner, service):
    result = runner.invoke(cli.app, ["sources", "delete", "ng136"], input="y\n")
    assert result.exit_code == 0, result.output
    assert service[0].calls == [("delete", "ng136")]


def test_sources_delete_unknown(runner, service):
    result = runner.invoke(cli.app, ["sources", "delete", "nope", "--yes"])
    assert result.exit_code == 1
    assert "No source with id 'nope'" in result.stderr


# ---------- ingest ----------

class FakeIngestor:
    instances: ClassVar[list["FakeIngestor"]] = []

    def __init__(self, project):
        self.project = project
        self.calls: list[dict] = []
        FakeIngestor.instances.append(self)

    def load_all(self, include_pubmed=True, pubmed_max_results=5):
        self.calls.append({"include_pubmed": include_pubmed, "max_results": pubmed_max_results})
        return ["doc1", "doc2"]


class FakeIndexer:
    instances: ClassVar[list["FakeIndexer"]] = []

    def __init__(self, project):
        self.project = project
        self.client = object()
        self.reset_called = False
        self.ran_with = None
        FakeIndexer.instances.append(self)

    def reset(self):
        self.reset_called = True

    def run(self, docs):
        self.ran_with = docs
        return IndexStats(documents_in=len(docs), chunks_written=7, total_points=99)


@pytest.fixture
def ingest_mocks(monkeypatch):
    FakeIngestor.instances.clear()
    FakeIndexer.instances.clear()
    project = dataclasses.replace(get_project("medrag"), ingestor_factory=FakeIngestor)
    monkeypatch.setattr(cli, "get_project", lambda name: project)
    monkeypatch.setattr(cli, "Indexer", FakeIndexer)
    monkeypatch.setattr(cli, "count_duplicate_chunks", lambda project, client: 0)
    return project


def test_ingest_defaults(runner, ingest_mocks):
    result = runner.invoke(cli.app, ["ingest"])
    assert result.exit_code == 0, result.output
    assert FakeIngestor.instances[0].calls == [{"include_pubmed": True, "max_results": 5}]
    indexer = FakeIndexer.instances[0]
    assert indexer.ran_with == ["doc1", "doc2"] and not indexer.reset_called
    assert "Documents in:     2" in result.stdout
    assert "Chunks written:   7" in result.stdout
    assert "Total points:     99" in result.stdout


def test_ingest_flags(runner, ingest_mocks):
    result = runner.invoke(cli.app, ["ingest", "--no-pubmed", "--max-results", "3", "--rebuild"])
    assert result.exit_code == 0, result.output
    assert FakeIngestor.instances[0].calls == [{"include_pubmed": False, "max_results": 3}]
    assert FakeIndexer.instances[0].reset_called


def test_ingest_rejects_bad_max_results(runner, ingest_mocks):
    result = runner.invoke(cli.app, ["ingest", "--max-results", "0"])
    assert result.exit_code == 2
    assert FakeIngestor.instances == []


def test_ingest_without_ingestor(runner, monkeypatch):
    project = dataclasses.replace(get_project("medrag"), ingestor_factory=None)
    monkeypatch.setattr(cli, "get_project", lambda name: project)
    result = runner.invoke(cli.app, ["ingest"])
    assert result.exit_code == 1
    assert "no ingestor_factory" in result.stderr


def test_build_index_wraps_ingest(monkeypatch):
    from scripts import build_index

    seen = {}
    monkeypatch.setattr(build_index, "app", lambda args, prog_name: seen.update(args=args))
    monkeypatch.setattr(sys, "argv", ["build_index", "--no-pubmed", "--project", "x",
                                      "--rebuild"])
    build_index.main()
    assert seen["args"] == ["--project", "x", "ingest", "--no-pubmed", "--rebuild"]


# ---------- eval ----------

def test_eval_not_wired_yet(runner):
    result = runner.invoke(cli.app, ["eval"])
    assert result.exit_code == 1
    assert "Phase 10" in result.stderr


def test_no_args_shows_help(runner):
    result = runner.invoke(cli.app, [])
    assert "ingest" in result.output and "sources" in result.output
