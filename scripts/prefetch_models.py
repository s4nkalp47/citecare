"""Phase 11: download every FastEmbed model the registered projects need.

Run at image build time (the Dockerfile's `models` stage) so containers start without
downloading anything. The cache directory comes from FASTEMBED_CACHE_PATH, which FastEmbed reads
itself for every model class (QdrantVectorStore doesn't forward a cache_dir to the sparse model,
so a setting couldn't reach all three). Model names come from settings, as at runtime.

    python -m scripts.prefetch_models
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from src.config import Settings, get_settings
from src.core.projects import ProjectConfig, get_project, list_projects

logger = logging.getLogger(__name__)


def models_to_fetch(settings: Settings, projects: Iterable[ProjectConfig]) -> list[tuple[str, str]]:
    """(kind, model name): the embedding model always, the others only if a project uses them."""
    projects = list(projects)
    models = [("embedding", settings.embed_model)]
    if any(p.rerank_candidates > 0 for p in projects):
        models.append(("reranker", settings.rerank_model))
    if any(p.hybrid for p in projects):
        models.append(("sparse", settings.sparse_model))
    return models


def fetch(kind: str, model_name: str) -> None:
    """Instantiate the FastEmbed model once, which downloads it into the cache."""
    if kind == "embedding":
        from fastembed import TextEmbedding

        TextEmbedding(model_name=model_name)
    elif kind == "reranker":
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        TextCrossEncoder(model_name=model_name)
    elif kind == "sparse":
        from fastembed import SparseTextEmbedding

        SparseTextEmbedding(model_name=model_name)
    else:
        raise ValueError(f"Unknown model kind '{kind}'")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    projects = [get_project(name) for name in list_projects()]
    for kind, model_name in models_to_fetch(get_settings(), projects):
        logger.info("Fetching %s model %s", kind, model_name)
        fetch(kind, model_name)


if __name__ == "__main__":
    main()
