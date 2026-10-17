"""Retriever: question -> top-k chunks from the project's Qdrant collection."""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from qdrant_client import QdrantClient

from src.core.embeddings import get_embed_model
from src.core.projects import ProjectConfig
from src.core.vector_store import get_qdrant_client, get_vector_store


class Retriever:
    def __init__(self, project: ProjectConfig, client: QdrantClient | None = None,
                 embed_model: BaseEmbedding | None = None):
        self.project = project
        self.client = client or get_qdrant_client()
        # Must be the same embedding model the indexer used.
        self.index = VectorStoreIndex.from_vector_store(
            get_vector_store(project, self.client),
            embed_model=embed_model or get_embed_model(),
        )

    def search(self, question: str, source_type: str | None = None) -> list[NodeWithScore]:
        """Top `similarity_top_k` chunks, best first, without the similarity cutoff."""
        if not self.client.collection_exists(self.project.collection_name):
            return []
        filters = None
        if source_type:
            filters = MetadataFilters(filters=[MetadataFilter(key="source_type", value=source_type)])
        retriever = self.index.as_retriever(similarity_top_k=self.project.similarity_top_k,
                                            filters=filters)
        return retriever.retrieve(question)

    def retrieve(self, question: str, source_type: str | None = None) -> list[NodeWithScore]:
        """Chunks scoring at least `project.similarity_cutoff`; [] means no usable evidence."""
        return [n for n in self.search(question, source_type)
                if n.score is not None and n.score >= self.project.similarity_cutoff]
