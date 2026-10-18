"""Embedding model shared by indexing and querying (must be the same model for both)."""

from functools import lru_cache

from llama_index.core.bridge.pydantic import Field
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from src.config import get_settings


class InstructedFastEmbedEmbedding(FastEmbedEmbedding):
    """FastEmbedEmbedding that prepends a retrieval instruction to queries only.

    FastEmbed's `query_embed` for bge models is plain `embed()`, so the bge query instruction
    ("Represent this sentence for searching relevant passages: ") is never added on its own.
    Documents are embedded without it, as bge expects.
    """

    query_instruction: str = Field(default="", description="Prefix added to every query.")

    @classmethod
    def class_name(cls) -> str:
        return "InstructedFastEmbedEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        # The async variant delegates here too.
        return super()._get_query_embedding(self.query_instruction + query)


@lru_cache
def get_embed_model() -> InstructedFastEmbedEmbedding:
    # Downloads the ONNX model on first use (~130 MB for bge-small), then runs locally.
    settings = get_settings()
    return InstructedFastEmbedEmbedding(model_name=settings.embed_model,
                                        query_instruction=settings.embed_query_instruction)
