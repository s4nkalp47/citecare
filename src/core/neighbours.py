"""Neighbour expansion: add the chunks just before and after each retrieved chunk (Phase 4b).

A question often needs two consecutive recommendations ("offer X; if X is not tolerated, offer
Y"), but small chunks put them in separate chunks. Expansion follows the PREVIOUS/NEXT
relationships the node parser records, and only adds a neighbour that shares the retrieved
chunk's `group_key` metadata (e.g. the same guideline section). Chunks without that key are never
expanded. Neighbours inherit the retrieved chunk's score, so the similarity cutoff treats them
alike.

Modes (ProjectConfig.neighbour_mode):
  budget  neighbours take slots in the top-k: each retrieved chunk is followed by its neighbours,
          then the list is cut at similarity_top_k
  window  the top-k retrieved chunks each bring their neighbours along (up to 3 x top-k chunks)
"""

from __future__ import annotations

from llama_index.core.schema import NodeRelationship, NodeWithScore
from llama_index.core.vector_stores.types import BasePydanticVectorStore


class NeighbourExpander:
    def __init__(self, vector_store: BasePydanticVectorStore, group_key: str = "section"):
        self.vector_store = vector_store
        self.group_key = group_key

    def _neighbour_ids(self, node: NodeWithScore) -> list[str]:
        if self.group_key not in node.node.metadata:
            return []
        rels = node.node.relationships
        # Next first: the recommendation after a match is more often its continuation.
        return [rels[r].node_id for r in (NodeRelationship.NEXT, NodeRelationship.PREVIOUS)
                if r in rels]

    def expand(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """Each node followed by its same-group neighbours, without duplicates."""
        wanted = {i for n in nodes for i in self._neighbour_ids(n)}
        wanted -= {n.node.node_id for n in nodes}
        fetched = {n.node_id: n for n in self.vector_store.get_nodes(node_ids=list(wanted))} \
            if wanted else {}

        out: list[NodeWithScore] = []
        seen: set[str] = set()
        retrieved = {n.node.node_id: n for n in nodes}
        for node in nodes:
            if node.node.node_id in seen:
                continue
            out.append(node)
            seen.add(node.node.node_id)
            group = node.node.metadata.get(self.group_key)
            for nid in self._neighbour_ids(node):
                if nid in seen or nid in retrieved:
                    continue  # a retrieved chunk keeps its own place and score
                neighbour = fetched.get(nid)
                if neighbour is not None and neighbour.metadata.get(self.group_key) == group:
                    out.append(NodeWithScore(node=neighbour, score=node.score))
                    seen.add(nid)
        return out
