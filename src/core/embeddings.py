"""Embedding model shared by indexing and querying (must be the same model for both)."""

from functools import lru_cache

from llama_index.embeddings.fastembed import FastEmbedEmbedding

from src.config import get_settings


@lru_cache
def get_embed_model() -> FastEmbedEmbedding:
    # Downloads the ONNX model on first use (~130 MB for bge-small), then runs locally.
    return FastEmbedEmbedding(model_name=get_settings().embed_model)
