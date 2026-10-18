"""Project registry: the domain-agnostic core looks up domain plugins here."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from llama_index.core import Document
    from llama_index.core.node_parser import NodeParser


class SourceIngestor(Protocol):
    """What the core needs from a plugin's ingestor to manage uploaded sources (Phase 5)."""

    upload_dir: Path  # where uploaded files are saved

    def source_id_for(self, path: Path) -> str:
        """The `source_id` the Documents of this file will carry."""

    def load_file(self, path: Path) -> list[Document]:
        """Turn one uploaded file into Documents."""

    def remove_source_files(self, source_id: str) -> list[Path]:
        """Delete the uploaded file and any cached artefacts of a source; return what was removed."""


@dataclass(frozen=True)
class ProjectConfig:
    """Everything domain-specific a RAG project needs. The core never hardcodes these."""

    name: str
    collection_name: str
    description: str
    system_prompt: str
    qa_prompt: str
    pubmed_queries: list[str] = field(default_factory=list)
    # Appended to every PubMed query (PubMed search syntax).
    pubmed_filter: str = ""
    chunk_size: int = 512
    chunk_overlap: int = 64
    similarity_top_k: int = 4
    # Candidates fetched for the cross-encoder reranker (settings.rerank_model), which keeps the
    # best similarity_top_k. 0 = no reranking: the vector search returns similarity_top_k itself.
    rerank_candidates: int = 0
    # Dense + BM25 sparse (settings.sparse_model) search. Needs a collection built with it on.
    hybrid: bool = False
    # Add each retrieved chunk's neighbours from the same `section` (src/core/neighbours.py):
    # "off" | "budget" (within similarity_top_k) | "window" (attached, up to 3 x top_k chunks).
    neighbour_mode: Literal["off", "budget", "window"] = "off"
    # A question whose best chunk scores below this gets insufficient_evidence (the context is not
    # filtered). It applies to the final score: the reranker's when rerank_candidates > 0, else
    # cosine.
    similarity_cutoff: float = 0.5
    # Optional second gate on the best vector-search (cosine) score, before reranking: a question
    # passes if either score reaches its cutoff. None = final score only.
    dense_cutoff: float | None = None
    # Output policy enforced by the output guardrail (Phase 6).
    require_citations: bool = True
    disclaimer: str = ""
    blocked_topics: list[str] = field(default_factory=list)
    # Policy text the output guardrail's safeguard model judges answers against (Phase 6).
    safety_policy: str = ""
    # Builds the ingestor used for runtime uploads, so the core never imports the plugin.
    ingestor_factory: Callable[[ProjectConfig], SourceIngestor] | None = None
    # Builds the chunker; None = SentenceSplitter(chunk_size, chunk_overlap). (Phase 4b)
    node_parser_factory: Callable[[ProjectConfig], NodeParser] | None = None


_REGISTRY: dict[str, ProjectConfig] = {}


def register_project(config: ProjectConfig) -> None:
    if config.name in _REGISTRY:
        raise ValueError(f"Project '{config.name}' is already registered")
    _REGISTRY[config.name] = config


def get_project(name: str) -> ProjectConfig:
    _load_plugins()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown project '{name}'. Available: {list_projects()}") from None


def list_projects() -> list[str]:
    _load_plugins()
    return sorted(_REGISTRY)


def _load_plugins() -> None:
    # Importing a plugin package registers it. Add new plugins here.
    if not _REGISTRY:
        from src.plugins.medrag.config import MEDRAG_CONFIG

        register_project(MEDRAG_CONFIG)
