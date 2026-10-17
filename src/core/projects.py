"""Project registry: the domain-agnostic core looks up domain plugins here."""

from dataclasses import dataclass, field


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
    # Retrieved chunks scoring below this are dropped; none left -> insufficient_evidence.
    similarity_cutoff: float = 0.5
    # Output policy enforced by the output guardrail (Phase 6).
    require_citations: bool = True
    disclaimer: str = ""
    blocked_topics: list[str] = field(default_factory=list)


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
