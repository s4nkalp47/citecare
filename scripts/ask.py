"""Phase 4: ask a question and pretty-print the structured, cited answer (calls Groq once or twice).

Run from the project root:
    python -m scripts.ask "What is the first-line treatment for hypertension in type 2 diabetes?"
    python -m scripts.ask "..." --source-type guideline
    python -m scripts.ask "..." --json               # raw RAGResponse JSON
    python -m scripts.ask "..." --retrieve-only      # retrieval scores only, no LLM call
"""

import argparse
import logging

from src.core.projects import get_project
from src.core.retriever import Retriever
from src.core.service import RAGService


def print_retrieval(project_name: str, question: str, source_type: str | None) -> None:
    project = get_project(project_name)
    nodes = Retriever(project).search(question, source_type=source_type)
    if not nodes:
        print("  (collection empty or missing)")
        return
    # The cutoff gates the question on its best score (Phase 4b), it doesn't filter chunks.
    verdict = "PASS" if nodes[0].score >= project.similarity_cutoff else "REJECT (no evidence)"
    print(f"\n{len(nodes)} chunks; best {nodes[0].score:.2e} vs cutoff "
          f"{project.similarity_cutoff}: {verdict}")
    for node in nodes:
        meta = node.node.metadata
        page = f" p.{meta['page']}" if meta.get("page") is not None else ""
        recs = f" [{', '.join(meta['rec_ids'])}]" if meta.get("rec_ids") else ""
        print(f"  {node.score:.2e}  {meta.get('title')}{page}{recs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--project", default="medrag")
    parser.add_argument("--source-type", help="e.g. guideline | pubmed | bootstrap")
    parser.add_argument("--json", action="store_true", help="print the raw RAGResponse as JSON")
    parser.add_argument("--retrieve-only", action="store_true",
                        help="print retrieval scores without calling the LLM")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.retrieve_only:
        print_retrieval(args.project, args.question, args.source_type)
        return

    response = RAGService(args.project).ask(args.question, source_type=args.source_type)
    if args.json:
        print(response.model_dump_json(indent=2))
        return

    print(f"\n{response.answer}\n")
    print(f"Confidence: {response.confidence}    Latency: {response.latency_ms} ms")
    print(f"Retrieved:  {len(response.contexts)} chunks, cited {len(response.citations)}")
    for c in response.citations:
        page = f", page {c.page}" if c.page is not None else ""
        # No raw score: reranker scores are often near 0 even for correct sources. Block 1 is
        # the top-ranked chunk (use --retrieve-only or --json for scores).
        label = "  (top match)" if c.ref == 1 else ""
        print(f"\n  [{c.ref}] {c.title}{page}{label}")
        if c.url:
            print(f"      {c.url}")
        print(f"      \"{c.snippet}\"")


if __name__ == "__main__":
    main()
