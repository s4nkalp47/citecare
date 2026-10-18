"""Query instruction and cross-encoder reranker, with FastEmbed's ONNX models stubbed out."""

from unittest.mock import patch

import numpy as np
import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.core.embeddings import InstructedFastEmbedEmbedding
from src.core.reranker import FastEmbedRerank

INSTRUCTION = "Represent this sentence for searching relevant passages: "


class _StubTextEmbedding:
    """Records what FastEmbed would embed; query_embed is plain embed, as for bge."""

    def __init__(self, model_name: str, **kwargs):
        self.seen: list[str] = []

    def embed(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else list(texts)
        self.seen += texts
        return [np.ones(3) for _ in texts]

    query_embed = embed


class _StubCrossEncoder:
    """Logit = minus the document length, so shorter chunks rank higher."""

    def __init__(self, model_name: str, **kwargs):
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents, **kwargs):
        documents = list(documents)
        self.calls.append((query, documents))
        return [-float(len(d)) for d in documents]


def _embed_model() -> InstructedFastEmbedEmbedding:
    with patch("fastembed.TextEmbedding", _StubTextEmbedding):
        return InstructedFastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5",
                                            query_instruction=INSTRUCTION)


def test_query_instruction_is_added_to_queries_only():
    model = _embed_model()
    model.get_query_embedding("first-line treatment for hypertension")
    model.get_text_embedding("Offer an ACE inhibitor or an ARB.")

    assert model._model.seen == [INSTRUCTION + "first-line treatment for hypertension",
                                 "Offer an ACE inhibitor or an ARB."]


async def _aquery(model, text):
    return await model.aget_query_embedding(text)


def test_query_instruction_on_async_path():
    import asyncio

    model = _embed_model()
    asyncio.run(_aquery(model, "metformin"))
    assert model._model.seen == [INSTRUCTION + "metformin"]


def _rerank(top_n: int) -> FastEmbedRerank:
    with patch("fastembed.rerank.cross_encoder.TextCrossEncoder", _StubCrossEncoder):
        return FastEmbedRerank(model_name="Xenova/ms-marco-MiniLM-L-6-v2", top_n=top_n)


def _nodes(*texts: str) -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text=t, metadata={"title": "NG136", "page": i}),
                          score=0.7) for i, t in enumerate(texts)]


def test_rerank_sorts_truncates_and_uses_sigmoid_scores():
    reranker = _rerank(top_n=2)
    nodes = _nodes("a much longer chunk of text", "short", "medium text")

    out = reranker.postprocess_nodes(nodes, query_str="bp target")

    assert [n.node.get_content() for n in out] == ["short", "medium text"]
    assert all(0 < n.score < 1 for n in out)
    assert out[0].score > out[1].score
    query, docs = reranker._model.calls[0]
    assert query == "bp target"
    assert "title: NG136" in docs[1]  # scored with the embed metadata, like LlamaIndex rerankers
    assert nodes[0].score == 0.7  # input nodes untouched


def test_rerank_empty_and_missing_query():
    reranker = _rerank(top_n=4)
    assert reranker.postprocess_nodes([], query_str="q") == []
    with pytest.raises(ValueError):
        reranker.postprocess_nodes(_nodes("x"))
