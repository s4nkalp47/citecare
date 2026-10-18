"""NICE recommendation-aware chunking (src/plugins/medrag/nice.py). No network."""

import dataclasses
from pathlib import Path

from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import MetadataMode
from qdrant_client import QdrantClient

from src.core.indexer import Indexer
from src.core.projects import get_project
from src.plugins.medrag.nice import (
    DOC_FORMAT,
    NiceGuidelineParser,
    is_nice_guideline,
    join_pages,
)

TITLE = "NICE NG136 Hypertension"

PAGES = [
    {"page": 1, "markdown": "# NICE guideline\n\n# Hypertension in adults\n\n# **Your responsibility**\n\n"
                            "The recommendations in this guideline represent the view of NICE."},
    {"page": 2, "markdown": "# Contents\n\nOverview 4\n\n1.1 Measuring blood pressure 5\n\n"
                            "1.4 Treating and monitoring hypertension 10\n\nTerms used 25"},
    {"page": 3, "markdown": "# 1.4 Treating and monitoring hypertension\n\n"
                            "## **Monitoring treatment and blood pressure targets**\n\n"
                            "## Table 1: Clinic blood pressure targets\n\n"
                            "<table><tr><th>Age</th><th>Target</th></tr>"
                            "<tr><td><b>Under 80</b></td><td>Below 140/90 mmHg</td></tr></table>\n\n"
                            "1.4.20 For adults with hypertension aged under 80, reduce clinic blood "
                            "pressure to below 140/90 mmHg and ensure that it is maintained below "
                            "that level. **[2019]**\n\n"
                            "## **Step 1 treatment**\n\n"
                            "1.4.32 Offer an <u>ACE inhibitor</u> or an ARB to adults starting step 1 "
                            "antihypertensive"},
    {"page": 4, "markdown": "treatment who:\n\n- have type 2 diabetes and are of any age or family "
                            "origin **or**\n- are aged under 55. **[2019]**\n\n"
                            "## 1.4.33\n\nIf an ACE inhibitor is not tolerated, for example because "
                            "of cough, offer an ARB to treat hypertension.\n\n**[2019]**\n\n"
                            "1.4.34 " + " ".join(f"Monitoring sentence {i} about potassium and renal "
                                                  "function." for i in range(60))},
]


def _guideline() -> Document:
    return Document(text=join_pages(PAGES), id_="src1", metadata={
        "source_type": "guideline", "source_id": "src1", "title": TITLE, "url": "",
        "doc_format": DOC_FORMAT})


def _chunks(**kwargs) -> list:
    return NiceGuidelineParser(**kwargs).get_nodes_from_documents([_guideline()])


def _by_rec(nodes, rec: str) -> list:
    return [n for n in nodes if rec in n.metadata["rec_ids"]]


def test_recommendation_split_by_page_break_is_one_chunk():
    [chunk] = _by_rec(_chunks(), "1.4.32")
    text = chunk.get_content()
    assert "starting step 1 antihypertensive treatment who:" in text  # joined with a space
    assert "have type 2 diabetes" in text
    assert "<u>" not in text and "**" not in text
    assert (chunk.metadata["page"], chunk.metadata["page_end"]) == (3, 4)


def test_heading_path_prefix_and_table_scope():
    nodes = _chunks()
    [rec] = _by_rec(nodes, "1.4.32")
    assert rec.get_content().startswith(
        f"{TITLE} > 1.4 Treating and monitoring hypertension > Step 1 treatment\n\n1.4.32 Offer")
    # The table heading applies to the table only, not to the recommendation after it.
    [target] = _by_rec(nodes, "1.4.20")
    assert target.get_content().startswith(
        f"{TITLE} > 1.4 Treating and monitoring hypertension > "
        "Monitoring treatment and blood pressure targets\n\n1.4.20")
    [table] = [n for n in nodes if not n.metadata["rec_ids"] and "Table 1" in n.get_content()]
    assert "Under 80 | Below 140/90 mmHg" in table.get_content()  # HTML flattened
    assert "<td>" not in table.get_content()


def test_heading_style_rec_number_and_orphan_tag_merge():
    [chunk] = _by_rec(_chunks(), "1.4.33")
    text = chunk.get_content()
    assert "1.4.33\n\nIf an ACE inhibitor is not tolerated" in text
    assert text.rstrip().endswith("[2019]")  # the orphan tag stays with its recommendation
    assert not any(n.get_content().strip().endswith("> [2019]") for n in _chunks())


def test_long_recommendation_is_split_with_prefix_on_each_piece():
    parser_nodes = _chunks(max_chunk_tokens=120)
    pieces = _by_rec(parser_nodes, "1.4.34")
    assert len(pieces) > 1
    prefix = f"{TITLE} > 1.4 Treating and monitoring hypertension > Step 1 treatment\n\n"
    assert all(p.get_content().startswith(prefix) for p in pieces)


def test_boilerplate_and_contents_are_dropped():
    texts = " ".join(n.get_content() for n in _chunks())
    assert "represent the view of NICE" not in texts
    assert "Terms used 25" not in texts


def test_parser_metadata_is_hidden_from_embedding_and_llm():
    [chunk] = _by_rec(_chunks(), "1.4.32")
    embed = chunk.get_content(metadata_mode=MetadataMode.EMBED)
    llm = chunk.get_content(metadata_mode=MetadataMode.LLM)
    for key in ("rec_ids", "page_end", "doc_format"):
        assert key not in embed and key not in llm
    assert "page: 3" in embed and f"title: {TITLE}" in embed


def test_other_documents_use_sentence_splitter():
    doc = Document(text="Metformin is first-line. " * 200, id_="pmid-1",
                   metadata={"source_type": "pubmed", "title": "Metformin review"})
    nodes = NiceGuidelineParser(chunk_size=128, chunk_overlap=16).get_nodes_from_documents([doc])
    assert len(nodes) > 1
    assert all("rec_ids" not in n.metadata for n in nodes)
    assert all(n.metadata["title"] == "Metformin review" for n in nodes)


def test_is_nice_guideline():
    rec_lines = "\n".join(f"1.1.{i} Do something." for i in range(1, 12))
    assert is_nice_guideline([{"page": 1, "markdown": "**NICE guideline**"},
                              {"page": 2, "markdown": rec_lines}])
    assert not is_nice_guideline([{"page": 1, "markdown": "WHO guideline\n" + rec_lines}])
    assert not is_nice_guideline([{"page": 1, "markdown": "NICE guideline\n1.1.1 Only one."}])
    assert not is_nice_guideline([])


def test_indexer_uses_the_project_node_parser(tmp_path: Path):
    client = QdrantClient(":memory:")
    project = dataclasses.replace(get_project("medrag"), hybrid=False)
    Indexer(project, client=client, embed_model=MockEmbedding(embed_dim=384),
            storage_dir=tmp_path).run([_guideline()])

    points, _ = client.scroll(project.collection_name, limit=100, with_payload=True)
    payloads = [p.payload for p in points]
    assert any(p["rec_ids"] == ["1.4.32"] and p["page"] == 3 for p in payloads)
    assert all(p["source_id"] == "src1" for p in payloads)


# ---------- Neighbour expansion (src/core/neighbours.py) over rec chunks ----------

def _indexed_retriever(tmp_path: Path, **overrides):
    from src.core.retriever import Retriever
    from tests.conftest import KeywordEmbedding

    client = QdrantClient(":memory:")
    settings = {"hybrid": False, "rerank_candidates": 0, "similarity_cutoff": 0.0,
                "neighbour_mode": "off", **overrides}
    project = dataclasses.replace(get_project("medrag"), **settings)
    embed = KeywordEmbedding()
    Indexer(project, client=client, embed_model=embed, storage_dir=tmp_path).run([_guideline()])
    return Retriever(project, client=client, embed_model=embed)


def _recs(nodes) -> list[list[str]]:
    return [n.node.metadata["rec_ids"] for n in nodes]


def test_neighbours_follow_their_chunk_within_the_section(tmp_path):
    retriever = _indexed_retriever(tmp_path, similarity_top_k=1)
    [seed] = retriever.search("cough ARB tolerated")  # expansion off: just 1.4.33
    assert seed.node.metadata["rec_ids"] == ["1.4.33"]

    expanded = retriever.expander.expand([seed])
    assert _recs(expanded) == [["1.4.33"], ["1.4.34"], ["1.4.32"]]  # next, then previous
    assert all(n.score == seed.score for n in expanded)


def test_neighbours_stop_at_section_boundary(tmp_path):
    retriever = _indexed_retriever(tmp_path, similarity_top_k=1)
    [seed] = retriever.search("ACE inhibitor ARB type 2 diabetes family origin")
    assert seed.node.metadata["rec_ids"] == ["1.4.32"]
    # The previous chunk (1.4.20) is in another section, so only the next one is added.
    assert _recs(retriever.expander.expand([seed])) == [["1.4.32"], ["1.4.33"]]


def test_budget_and_window_modes(tmp_path):
    budget = _indexed_retriever(tmp_path / "b", similarity_top_k=2, neighbour_mode="budget")
    window = _indexed_retriever(tmp_path / "w", similarity_top_k=2, neighbour_mode="window")
    question = "cough ARB tolerated"

    assert len(budget.search(question)) == 2  # neighbours take top-k slots
    nodes = window.search(question)
    assert len(nodes) > 2 and _recs(nodes)[:2] == [["1.4.33"], ["1.4.34"]]
    assert len({n.node.node_id for n in nodes}) == len(nodes)  # no duplicates
    assert window.retrieve(question)  # neighbours carry a score, so the cutoff keeps them


def test_chunks_without_section_are_not_expanded(tmp_path):
    from llama_index.core.schema import NodeWithScore, TextNode

    from src.core.neighbours import NeighbourExpander

    node = NodeWithScore(node=TextNode(text="abstract", metadata={"title": "PubMed"}), score=0.9)
    assert NeighbourExpander(vector_store=None).expand([node]) == [node]
