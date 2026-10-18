"""Phase 3: load documents with the project's ingestor and index them into Qdrant.

Since Phase 9 a thin wrapper around `medrag ingest` (src/cli.py), kept for existing habits and
the Docker one-shot indexer. Safe to re-run: unchanged documents are skipped.

Run from the project root:
    python -m scripts.build_index                    # guidelines + bootstrap + PubMed
    python -m scripts.build_index --no-pubmed
    python -m scripts.build_index --max-results 3    # PubMed results per query
    python -m scripts.build_index --rebuild          # drop the collection and re-index everything
                                                     # (after changing chunking or hybrid)
    python -m scripts.build_index --project medrag
"""

import sys

from src.cli import app


def main() -> None:
    # --project is a top-level CLI option; everything else belongs to `ingest`.
    args = sys.argv[1:]
    top: list[str] = []
    if "--project" in args:
        i = args.index("--project")
        top, args = args[i:i + 2], args[:i] + args[i + 2:]
    app(args=[*top, "ingest", *args], prog_name="build_index")


if __name__ == "__main__":
    main()
