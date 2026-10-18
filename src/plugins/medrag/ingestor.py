"""MedRAGIngestor: turns guideline PDFs, PubMed articles and bootstrap docs into
LlamaIndex Documents with consistent metadata.

Every Document carries:
    source_type  "guideline" | "pubmed" | "bootstrap"
    source_id    stable id of the whole source (file hash or PMC id), used to
                 dedupe and to delete a source's chunks later
    title        human-readable name used in citations
    page         page number (guidelines only; for NICE guidelines set per chunk by the parser)
    url          link to the original (when known; NICE guideline urls come from the file name)

NICE guidelines become one Document per guideline (pages joined with markers, doc_format
"nice_guideline") so `nice.NiceGuidelineParser` can chunk by recommendation across page breaks.
Other PDFs stay one Document per page.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

from src.config import Settings, get_settings
from src.core.projects import ProjectConfig
from src.plugins.medrag.nice import DOC_FORMAT, is_nice_guideline, join_pages
from src.plugins.medrag.pubmed import PubMedAbstractReader

logger = logging.getLogger(__name__)

# Keys the embedding model and LLM don't need to see. PubMed's `query` (the search that found the
# abstract) is provenance: embedded, it made every abstract look relevant to its query's words.
HIDDEN_FROM_EMBED = ["source_id", "url", "file_name", "query", "doc_format"]
HIDDEN_FROM_LLM = ["source_id", "file_name", "query", "doc_format"]

_GUIDELINE_CODE = re.compile(r"^[a-z]{1,3}\d+$")  # ng136, cg191
_ACRONYMS = {"nice", "who", "ada", "cdc", "idsa", "ats"}
_NICE_GUIDANCE_URL = "https://www.nice.org.uk/guidance/{code}"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def title_from_filename(path: Path) -> str:
    """nice_ng136_hypertension.pdf -> 'NICE NG136 Hypertension'."""
    words = path.stem.replace("-", "_").split("_")
    return " ".join(
        w.upper() if _GUIDELINE_CODE.match(w) or w in _ACRONYMS else w.capitalize()
        for w in words
    )


def url_from_filename(path: Path) -> str:
    """nice_ng136_hypertension.pdf -> 'https://www.nice.org.uk/guidance/ng136' ('' if not NICE)."""
    words = path.stem.lower().replace("-", "_").split("_")
    if "nice" not in words:
        return ""
    code = next((w for w in words if _GUIDELINE_CODE.match(w)), None)
    return _NICE_GUIDANCE_URL.format(code=code) if code else ""


def pages_from_parse_result(result, file_name: str) -> list[dict]:
    """Convert a LlamaParse v2 result into [{'page': int, 'markdown': str}], skipping failed pages."""
    if result.job.status != "COMPLETED":
        raise RuntimeError(f"LlamaParse job for {file_name} {result.job.status}: "
                           f"{result.job.error_message}")
    if result.markdown is None:
        raise RuntimeError(f"LlamaParse returned no markdown for {file_name}")
    pages = []
    for page in result.markdown.pages:
        if not page.success:
            logger.warning("LlamaParse failed on %s page %d: %s", file_name, page.page_number,
                           page.error)
            continue
        pages.append({"page": page.page_number, "markdown": page.markdown})
    return pages


def _make_doc(text: str, doc_id: str, **metadata) -> Document:
    doc = Document(text=text, id_=doc_id, metadata=metadata)
    doc.excluded_embed_metadata_keys = list(HIDDEN_FROM_EMBED)
    doc.excluded_llm_metadata_keys = list(HIDDEN_FROM_LLM)
    return doc


class MedRAGIngestor:
    def __init__(self, project: ProjectConfig, settings: Settings | None = None):
        self.project = project
        self.settings = settings or get_settings()
        self.guidelines_dir = self.settings.data_dir / "guidelines"
        self.bootstrap_dir = self.settings.data_dir / "bootstrap"
        self.cache_dir = self.settings.data_dir / "parsed"
        self.upload_dir = self.guidelines_dir  # uploaded PDFs are guidelines

    # ---------- Uploaded sources (SourceIngestor protocol, used by SourceManager) ----------

    def source_id_for(self, path: Path) -> str:
        return file_hash(path)

    def load_file(self, path: Path) -> list[Document]:
        return self.load_guidelines([path])

    def remove_source_files(self, source_id: str) -> list[Path]:
        """Delete the guideline PDF(s) and parse cache(s) of one source."""
        removed = list(self.cache_dir.glob(f"*-{source_id}.json"))
        removed += [p for p in self.guidelines_dir.glob("*.pdf") if file_hash(p) == source_id]
        for path in removed:
            path.unlink()
        if removed:
            logger.info("Removed %s", ", ".join(p.name for p in removed))
        return removed

    # ---------- Guideline PDFs (LlamaParse, cached) ----------

    def load_guidelines(self, paths: list[Path] | None = None,
                        join_nice: bool = True) -> list[Document]:
        """One Document per NICE guideline (`join_nice`), else one per page."""
        paths = paths if paths is not None else sorted(self.guidelines_dir.glob("*.pdf"))
        docs: list[Document] = []
        for path in paths:
            source_id = file_hash(path)
            title = title_from_filename(path)
            url = url_from_filename(path)
            pages = [p for p in self._parsed_pages(path, source_id) if p["markdown"].strip()]
            if join_nice and is_nice_guideline(pages):
                docs.append(_make_doc(
                    join_pages(pages),
                    doc_id=source_id,
                    source_type="guideline",
                    source_id=source_id,
                    title=title,
                    file_name=path.name,
                    url=url,
                    doc_format=DOC_FORMAT,
                ))
                continue
            for page in pages:
                docs.append(_make_doc(
                    page["markdown"],
                    doc_id=f"{source_id}:p{page['page']}",
                    source_type="guideline",
                    source_id=source_id,
                    title=title,
                    page=page["page"],
                    file_name=path.name,
                    url=url,
                ))
        logger.info("Loaded %d guideline documents from %d PDFs", len(docs), len(paths))
        return docs

    def _parsed_pages(self, path: Path, source_id: str) -> list[dict]:
        """Return [{'page': int, 'markdown': str}], parsing with LlamaParse only on cache miss."""
        cache_file = self.cache_dir / f"{path.stem}-{source_id}.json"
        if cache_file.exists():
            logger.info("Using cached parse for %s", path.name)
            return json.loads(cache_file.read_text(encoding="utf-8"))

        logger.info("Parsing %s with LlamaParse (uses credits)", path.name)
        pages = self._parse_pdf(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
        return pages

    def _parse_pdf(self, path: Path) -> list[dict]:
        """Parse a PDF with the LlamaParse v2 API (`llama-cloud` SDK)."""
        # Imported lazily so tests and other loaders don't need LlamaParse configured.
        from llama_cloud import LlamaCloud

        if not self.settings.llama_cloud_api_key:
            raise RuntimeError("LLAMA_CLOUD_API_KEY is empty - add it to .env")
        client = LlamaCloud(api_key=self.settings.llama_cloud_api_key)
        uploaded = client.files.create(file=path, purpose="parse")
        result = client.parsing.parse(
            file_id=uploaded.id,
            tier=self.settings.llama_parse_tier,
            version="latest",
            expand=["markdown"],
        )
        return pages_from_parse_result(result, path.name)

    # ---------- PubMed abstracts ----------

    def load_pubmed(self, queries: list[str] | None = None, max_results: int = 5) -> list[Document]:
        queries = queries if queries is not None else self.project.pubmed_queries
        reader = PubMedAbstractReader(api_key=self.settings.ncbi_api_key)
        docs: list[Document] = []
        seen: set[str] = set()
        for query in queries:
            try:
                raw_docs = reader.load_data(query, max_results=max_results,
                                            extra_filter=self.project.pubmed_filter)
            except Exception:  # network / NCBI errors shouldn't kill the whole ingest
                logger.exception("PubMed query failed: %s", query)
                continue
            for raw in raw_docs:
                source_id = f"pmid-{raw.metadata['pmid']}"
                # Skip duplicates across queries and records with no abstract.
                if source_id in seen or "\n\n" not in raw.text:
                    continue
                seen.add(source_id)
                docs.append(_make_doc(
                    raw.text,
                    doc_id=source_id,
                    source_type="pubmed",
                    source_id=source_id,
                    title=raw.metadata["title"],
                    journal=raw.metadata["journal"],
                    year=raw.metadata["year"],
                    publication_types=raw.metadata["publication_types"],
                    query=query,
                    url=raw.metadata["url"],
                ))
        logger.info("Loaded %d PubMed abstracts from %d queries", len(docs), len(queries))
        return docs

    # ---------- Bootstrap docs (local .md / .txt) ----------

    def load_bootstrap(self) -> list[Document]:
        files = [p for p in self.bootstrap_dir.glob("*") if p.suffix in {".md", ".txt"}]
        if not files:
            return []
        raw_docs = SimpleDirectoryReader(input_files=[str(f) for f in files]).load_data()
        docs = []
        for raw in raw_docs:
            path = Path(raw.metadata["file_path"])
            source_id = file_hash(path)
            docs.append(_make_doc(
                raw.text,
                doc_id=source_id,
                source_type="bootstrap",
                source_id=source_id,
                title=title_from_filename(path),
                file_name=path.name,
                url="",
            ))
        return docs

    # ---------- Everything ----------

    def load_all(self, include_pubmed: bool = True, pubmed_max_results: int = 5) -> list[Document]:
        docs = self.load_guidelines() + self.load_bootstrap()
        if include_pubmed:
            docs += self.load_pubmed(max_results=pubmed_max_results)
        return docs
