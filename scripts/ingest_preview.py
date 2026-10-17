"""Phase 2 check: run ingestion and print what was loaded (nothing is indexed yet).

Run from the project root:
    python -m scripts.ingest_preview                 # guidelines + bootstrap + PubMed
    python -m scripts.ingest_preview --no-pubmed     # skip PubMed
    python -m scripts.ingest_preview --only pubmed --max-results 2
"""

import argparse
import logging
from collections import Counter

from src.core.projects import get_project
from src.plugins.medrag.ingestor import MedRAGIngestor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["guidelines", "pubmed", "bootstrap"])
    parser.add_argument("--no-pubmed", action="store_true")
    parser.add_argument("--max-results", type=int, default=3, help="PubMed results per query")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingestor = MedRAGIngestor(get_project("medrag"))

    if args.only == "guidelines":
        docs = ingestor.load_guidelines()
    elif args.only == "pubmed":
        docs = ingestor.load_pubmed(max_results=args.max_results)
    elif args.only == "bootstrap":
        docs = ingestor.load_bootstrap()
    else:
        docs = ingestor.load_all(include_pubmed=not args.no_pubmed,
                                 pubmed_max_results=args.max_results)

    print(f"\n{len(docs)} documents")
    print("By source type:", dict(Counter(d.metadata["source_type"] for d in docs)))
    print("By title:")
    for title, n in Counter(d.metadata["title"] for d in docs).most_common():
        print(f"  {n:4d}  {title}")

    if docs:
        sample = docs[0]
        print("\nSample document")
        print("  id:      ", sample.id_)
        print("  metadata:", sample.metadata)
        print("  text:    ", sample.text[:400].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
