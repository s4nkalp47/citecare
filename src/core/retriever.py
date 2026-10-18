"""Retriever: question -> top-k chunks from the project's Qdrant collection.

With `project.rerank_candidates` set, the vector search fetches that many candidates and a
cross-encoder reranker keeps the best `similarity_top_k`. `project.neighbour_mode` then adds
adjacent same-section chunks (Phase 4b).
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from qdrant_client import QdrantClient

from src.core.embeddings import get_embed_model
from src.core.neighbours import NeighbourExpander
from src.core.projects import ProjectConfig
from src.core.vector_store import get_qdrant_client, get_vector_store


class Retriever:
    def __init__(self, project: ProjectConfig, client: QdrantClient | None = None,
                 embed_model: BaseEmbedding | None = None,
                 reranker: BaseNodePostprocessor | None = None):
        self.project = project
        self.client = client or get_qdrant_client()
        # Must be the same embedding model the indexer used.
        vector_store = get_vector_store(project, self.client)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model or get_embed_model(),
        )
        self.expander = NeighbourExpander(vector_store)
        self.reranker = None
        if project.rerank_candidates > 0:
            if reranker is None:
                from src.core.reranker import get_reranker  # loads the ONNX model

                reranker = get_reranker(project.similarity_top_k)
            self.reranker = reranker

    def search(self, question: str, source_type: str | None = None) -> list[NodeWithScore]:
        """Top `similarity_top_k` chunks, best first, without the similarity cutoff."""
        return self.search_with_dense_score(question, source_type)[0]

    def search_with_dense_score(self, question: str, source_type: str | None = None
                                ) -> tuple[list[NodeWithScore], float | None]:
        """`search()` plus the best vector-search score, before the reranker replaces it."""
        if not self.client.collection_exists(self.project.collection_name):
            return [], None
        filters = None
        if source_type:
            filters = MetadataFilters(filters=[MetadataFilter(key="source_type", value=source_type)])
        top_k = self.project.rerank_candidates if self.reranker else self.project.similarity_top_k
        kwargs = {}
        if self.project.hybrid:
            kwargs = {"vector_store_query_mode": "hybrid", "sparse_top_k": top_k,
                      "hybrid_top_k": top_k}
        retriever = self.index.as_retriever(similarity_top_k=top_k, filters=filters, **kwargs)
        nodes = retriever.retrieve(question)
        dense_best = _best(nodes)
        if self.reranker:
            nodes = self.reranker.postprocess_nodes(nodes, query_str=question)
        if self.project.neighbour_mode != "off":
            nodes = self.expander.expand(nodes[:self.project.similarity_top_k])
            if self.project.neighbour_mode == "budget":
                nodes = nodes[:self.project.similarity_top_k]
        return nodes, dense_best

    def retrieve(self, question: str, source_type: str | None = None) -> list[NodeWithScore]:
        """All of `search()`, or [] (no usable evidence) when the question fails the gate.

        A question is rejected only if its best final score is below `project.similarity_cutoff`
        and, when `project.dense_cutoff` is set, its best vector-search score is below that too.
        The reranker alone misses some paraphrases ("ramipril ... dry cough" scores 1.7e-5, under
        off-topic questions) that the bi-encoder still places in the domain.

        The gate doesn't prune the context: cross-encoder scores are extreme (relevant neighbours
        or second recommendations can score 1e-4 next to a 0.9 top chunk), and top_k already
        bounds the context.
        """
        nodes, dense_best = self.search_with_dense_score(question, source_type)
        best = _best(nodes)
        if best is None:
            return []
        if best >= self.project.similarity_cutoff:
            return nodes
        dense_cutoff = self.project.dense_cutoff
        if dense_cutoff is not None and dense_best is not None and dense_best >= dense_cutoff:
            return nodes
        return []


def _best(nodes: list[NodeWithScore]) -> float | None:
    return max((n.score for n in nodes if n.score is not None), default=None)
