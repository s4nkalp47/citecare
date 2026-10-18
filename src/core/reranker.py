"""Reranker: a local FastEmbed cross-encoder that re-scores retrieved chunks against the question.

Bi-encoder cosine scores on this corpus are bunched together (0.69-0.74 for one question), so the
retriever fetches a wider candidate pool and the cross-encoder, which reads question and chunk
together, picks the top `similarity_top_k`. Scores are sigmoid(logit), in (0, 1).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

from src.config import get_settings


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class FastEmbedRerank(BaseNodePostprocessor):
    model_name: str = Field(description="FastEmbed TextCrossEncoder model name.")
    top_n: int = Field(default=4, description="Number of nodes to keep.")

    _model: Any = PrivateAttr()

    def __init__(self, model_name: str, top_n: int = 4, **kwargs: Any):
        super().__init__(model_name=model_name, top_n=top_n, **kwargs)
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        # Downloads the ONNX model on first use (~80 MB for MiniLM-L-6), then runs locally.
        self._model = TextCrossEncoder(model_name=model_name)

    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedRerank"

    def _postprocess_nodes(self, nodes: list[NodeWithScore],
                           query_bundle: QueryBundle | None = None) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("FastEmbedRerank needs the query")
        if not nodes:
            return []
        texts = [n.node.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
        scores = self._model.rerank(query_bundle.query_str, texts)
        reranked = [NodeWithScore(node=n.node, score=_sigmoid(float(s)))
                    for n, s in zip(nodes, scores)]
        reranked.sort(key=lambda n: n.score, reverse=True)
        return reranked[:self.top_n]


@lru_cache
def get_reranker(top_n: int) -> FastEmbedRerank:
    return FastEmbedRerank(model_name=get_settings().rerank_model, top_n=top_n)
