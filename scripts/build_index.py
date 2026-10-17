"""Phase 3: load documents with the project's ingestor and index them into Qdrant.

Safe to re-run: unchanged documents are skipped. Also used as the Docker one-shot indexer.

Run from the project root:
    python -m scripts.build_index                    # guidelines + bootstrap + PubMed
    python -m scripts.build_index --no-pubmed
    python -m scripts.build_index --max-results 3    # PubMed results per query
"""

import argparse
import logging

from src.core.indexer import Indexer
from src.core.projects import get_project
from src.plugins.medrag.ingestor import MedRAGIngestor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="medrag")
    parser.add_argument("--no-pubmed", action="store_true")
    parser.add_argument("--max-results", type=int, default=5, help="PubMed results per query")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    project = get_project(args.project)

    docs = MedRAGIngestor(project).load_all(include_pubmed=not args.no_pubmed,
                                            pubmed_max_results=args.max_results)
    stats = Indexer(project).run(docs)

    print(f"\nCollection:     {project.collection_name}")
    print(f"Documents in:   {stats.documents_in}")
    print(f"Chunks written: {stats.chunks_written}")
    print(f"Total points:   {stats.total_points}")


if __name__ == "__main__":
    main()
