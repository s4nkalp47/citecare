"""SourceManager: upload, list and delete a project's sources at runtime.

Uploads are saved to the ingestor's upload dir, ingested and indexed on their own. Listing
aggregates chunk payloads in Qdrant, so it also shows sources added by scripts.build_index.
Deleting removes the Qdrant points, the docstore entries and the plugin's files (the uploaded
file and its parse cache), so the source doesn't come back on the next build_index run.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding
from qdrant_client import QdrantClient

from src.config import get_settings
from src.core.indexer import Indexer
from src.core.projects import ProjectConfig, SourceIngestor
from src.core.schemas import SourceInfo
from src.core.vector_store import get_qdrant_client, source_filter

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".pdf"}
_PDF_MAGIC = b"%PDF-"
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_PAYLOAD_KEYS = ["source_id", "title", "source_type", "file_name"]


class InvalidSourceError(ValueError):
    """The upload was rejected (wrong type, too big, duplicate...). Maps to HTTP 400."""


class SourceNotFoundError(LookupError):
    """No source with that id. Maps to HTTP 404."""


def safe_filename(filename: str) -> str:
    """Strip directories and unusual characters: '../My Guide (v2).PDF' -> 'My_Guide_v2_.pdf'."""
    name = Path(filename.replace("\\", "/")).name
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    stem = _UNSAFE_CHARS.sub("_", stem).strip("._")
    if not stem:
        raise InvalidSourceError(f"Invalid file name: {filename!r}")
    return stem + suffix


class SourceManager:
    def __init__(self, project: ProjectConfig, *, ingestor: SourceIngestor | None = None,
                 client: QdrantClient | None = None, embed_model: BaseEmbedding | None = None,
                 storage_dir: Path | None = None, max_upload_bytes: int | None = None):
        self.project = project
        if ingestor is None and project.ingestor_factory is not None:
            ingestor = project.ingestor_factory(project)
        self.ingestor = ingestor
        self.client = client or get_qdrant_client()
        self.embed_model = embed_model  # None -> the Indexer loads the shared FastEmbed model
        self.storage_dir = storage_dir
        self.max_upload_bytes = max_upload_bytes or get_settings().max_upload_mb * 1024 * 1024
        # Uploads and deletes rewrite the docstore file; don't let two run at once.
        self._lock = threading.Lock()

    # ---------- Upload ----------

    def upload(self, filename: str, data: bytes) -> SourceInfo:
        """Save one PDF, ingest and index just that file. Re-uploading the same file is a no-op."""
        ingestor = self._require_ingestor()
        name = safe_filename(filename)
        self._validate(name, data)

        with self._lock:
            dest = ingestor.upload_dir / name
            if dest.exists() and dest.read_bytes() != data:
                raise InvalidSourceError(f"A different file named '{name}' already exists; "
                                         "delete that source first")
            created = not dest.exists()
            if created:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            try:
                source_id = ingestor.source_id_for(dest)
                existing = self._aggregate(source_id).get(source_id)
                if existing and existing.file_name not in (None, name):
                    raise InvalidSourceError(f"This file is already indexed as "
                                             f"'{existing.file_name}' ({source_id})")
                docs = ingestor.load_file(dest)
                if not docs:
                    raise InvalidSourceError(f"No text could be extracted from '{name}'")
                self._indexer().run(docs)
            except BaseException:
                if created:
                    dest.unlink(missing_ok=True)
                raise

        info = self._aggregate(source_id)[source_id]
        logger.info("Uploaded %s as %s (%d chunks)", name, source_id, info.chunks)
        return info

    def _validate(self, name: str, data: bytes) -> None:
        if Path(name).suffix not in ALLOWED_SUFFIXES:
            raise InvalidSourceError(f"Only PDF files are accepted, got '{name}'")
        if not data:
            raise InvalidSourceError("The uploaded file is empty")
        if len(data) > self.max_upload_bytes:
            raise InvalidSourceError(f"File is larger than the "
                                     f"{self.max_upload_bytes // (1024 * 1024)} MB limit")
        if not data.startswith(_PDF_MAGIC):
            raise InvalidSourceError(f"'{name}' is not a valid PDF")

    # ---------- List ----------

    def list(self) -> list[SourceInfo]:
        """Every indexed source with its chunk count, sorted by type then title."""
        return sorted(self._aggregate().values(), key=lambda s: (s.source_type, s.title.lower()))

    def _aggregate(self, source_id: str | None = None) -> dict[str, SourceInfo]:
        """Group chunk payloads by source_id (optionally just one source)."""
        collection = self.project.collection_name
        if not self.client.collection_exists(collection):
            return {}
        scroll_filter = source_filter(source_id) if source_id else None

        sources: dict[str, SourceInfo] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(collection, scroll_filter=scroll_filter,
                                                limit=1000, offset=offset,
                                                with_payload=_PAYLOAD_KEYS, with_vectors=False)
            for point in points:
                payload = point.payload or {}
                sid = payload.get("source_id")
                if not sid:
                    continue
                if sid in sources:
                    sources[sid].chunks += 1
                else:
                    sources[sid] = SourceInfo(source_id=sid, title=payload.get("title") or sid,
                                              source_type=payload.get("source_type") or "unknown",
                                              chunks=1, file_name=payload.get("file_name"))
            if offset is None:
                return sources

    # ---------- Delete ----------

    def delete(self, source_id: str) -> int:
        """Remove a source's chunks, docstore entries and files. Returns chunks removed."""
        with self._lock:
            indexer = self._indexer()
            in_docstore = any(doc.metadata.get("source_id") == source_id
                              for doc in indexer.docstore.docs.values())
            removed_chunks = indexer.delete_source(source_id)
            removed_files = (self.ingestor.remove_source_files(source_id)
                             if self.ingestor is not None else [])
        if not (removed_chunks or in_docstore or removed_files):
            raise SourceNotFoundError(f"No source with id '{source_id}'")
        logger.info("Deleted source %s (%d chunks, %d files)", source_id, removed_chunks,
                    len(removed_files))
        return removed_chunks

    # ---------- Helpers ----------

    def _indexer(self) -> Indexer:
        # A fresh Indexer per operation reloads the docstore, picking up build_index runs.
        return Indexer(self.project, client=self.client, embed_model=self.embed_model,
                       storage_dir=self.storage_dir)

    def _require_ingestor(self) -> SourceIngestor:
        if self.ingestor is None:
            raise RuntimeError(f"Project '{self.project.name}' has no ingestor_factory; "
                               "uploads are not supported")
        return self.ingestor
