"""Indexer: Documents -> chunks -> embeddings -> Qdrant, idempotently.

A docstore persisted at data/pipeline_storage/<project>/ remembers each Document's hash, so
re-running skips unchanged Documents and replaces the chunks of changed ones (UPSERTS).
Documents missing from a run are kept (a --no-pubmed run must not delete PubMed chunks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient

from src.config import get_settings
from src.core.embeddings import get_embed_model
from src.core.projects import ProjectConfig
from src.core.vector_store import (
    count_points,
    delete_by_source_id,
    get_qdrant_client,
    get_vector_store,
)

logger = logging.getLogger(__name__)

DOCSTORE_FILE = "docstore.json"


@dataclass(frozen=True)
class IndexStats:
    documents_in: int
    chunks_written: int
    total_points: int


class Indexer:
    def __init__(self, project: ProjectConfig, client: QdrantClient | None = None,
                 embed_model: BaseEmbedding | None = None, storage_dir: Path | None = None):
        self.project = project
        self.client = client or get_qdrant_client()
        self.embed_model = embed_model or get_embed_model()
        self.storage_dir = storage_dir or get_settings().data_dir / "pipeline_storage" / project.name
        self.docstore = self._load_docstore()

    def run(self, documents: list[Document]) -> IndexStats:
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=self.project.chunk_size,
                                 chunk_overlap=self.project.chunk_overlap),
                self.embed_model,
            ],
            vector_store=get_vector_store(self.project, self.client),
            docstore=self.docstore,
            docstore_strategy=DocstoreStrategy.UPSERTS,
            disable_cache=True,  # the docstore already skips unchanged documents
        )
        nodes = pipeline.run(documents=documents, show_progress=len(documents) > 20)
        self._persist_docstore()
        stats = IndexStats(documents_in=len(documents), chunks_written=len(nodes),
                           total_points=count_points(self.project, self.client))
        logger.info("Indexed %s", stats)
        return stats

    def delete_source(self, source_id: str) -> int:
        """Remove a source from Qdrant and the docstore, so a later run can re-add it."""
        removed = delete_by_source_id(self.project, source_id, self.client)
        # With UPSERTS the docstore holds the input Documents (and their hashes), not chunks.
        doc_ids = [doc_id for doc_id, doc in self.docstore.docs.items()
                   if doc.metadata.get("source_id") == source_id]
        for doc_id in doc_ids:
            self.docstore.delete_document(doc_id, raise_error=False)
        self._persist_docstore()
        return removed

    def _load_docstore(self) -> SimpleDocumentStore:
        path = self.storage_dir / DOCSTORE_FILE
        if not path.exists():
            return SimpleDocumentStore()
        docstore = SimpleDocumentStore.from_persist_path(str(path))
        # If the collection was wiped (e.g. Qdrant volume removed), the docstore would make the
        # pipeline skip every document. Start fresh so everything is re-indexed.
        if docstore.docs and count_points(self.project, self.client) == 0:
            logger.warning("Collection '%s' is empty but the docstore is not; re-indexing all",
                           self.project.collection_name)
            return SimpleDocumentStore()
        return docstore

    def _persist_docstore(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.docstore.persist(str(self.storage_dir / DOCSTORE_FILE))
