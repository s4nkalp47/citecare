"""Phase 9: developer CLI. Calls RAGService directly, so it bypasses the guardrails by design.

After `pip install -e .`:
    medrag ingest [--no-pubmed] [--max-results N] [--rebuild]
    medrag ask "What is the first-line treatment for hypertension in type 2 diabetes?"
    medrag ask "..." --source-type guideline --json
    medrag sources list
    medrag sources delete <source_id> [--yes]
    medrag eval [--limit N] [--no-llm-metrics] [--metrics a,b] [--resume FILE]
                [--judge-delay S] [--threshold NAME=VALUE]   # Phase 10, see evals/run_evals.py
    medrag --project <name> ...      # default: medrag

Without the entry point: python -m src.cli ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from evals import JUDGE_DELAY_SECONDS, run_evals
from src.core.indexer import Indexer
from src.core.projects import get_project
from src.core.schemas import RAGResponse
from src.core.service import RAGService
from src.core.sources import SourceNotFoundError
from src.core.vector_store import count_duplicate_chunks

GUARDRAILS_NOTICE = ("Note: the CLI calls RAGService directly; input/output guardrails are "
                     "bypassed (use the API or UI for guarded answers).")

app = typer.Typer(help="MedRAG developer CLI (no guardrails).", no_args_is_help=True,
                  add_completion=False)
sources_app = typer.Typer(help="List or delete indexed sources.", no_args_is_help=True)
app.add_typer(sources_app, name="sources")


def make_service(project_name: str) -> RAGService:
    """Replaced in tests."""
    return RAGService(project_name)


@app.callback()
def main(ctx: typer.Context,
         project: Annotated[str, typer.Option(help="Registered project name.")] = "medrag",
         ) -> None:
    ctx.obj = {"project": project}
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


# ---------- ingest ----------

@app.command()
def ingest(ctx: typer.Context,
           no_pubmed: Annotated[bool, typer.Option("--no-pubmed",
                                                   help="Skip the PubMed queries.")] = False,
           max_results: Annotated[int, typer.Option(min=1,
                                                    help="PubMed results per query.")] = 5,
           rebuild: Annotated[bool, typer.Option(
               "--rebuild", help="Drop the collection and docstore first, then index "
                                 "everything (after changing chunking or hybrid).")] = False,
           ) -> None:
    """Load the project's documents and index them into Qdrant (unchanged ones are skipped)."""
    logging.getLogger().setLevel(logging.INFO)
    project = get_project(ctx.obj["project"])
    if project.ingestor_factory is None:
        _fail(f"Project '{project.name}' has no ingestor_factory; nothing to ingest")

    docs = project.ingestor_factory(project).load_all(include_pubmed=not no_pubmed,
                                                      pubmed_max_results=max_results)
    indexer = Indexer(project)
    if rebuild:
        indexer.reset()
    stats = indexer.run(docs)

    typer.echo(f"\nCollection:       {project.collection_name}")
    typer.echo(f"Documents in:     {stats.documents_in}")
    typer.echo(f"Chunks written:   {stats.chunks_written}")
    typer.echo(f"Total points:     {stats.total_points}")
    typer.echo(f"Duplicate chunks: {count_duplicate_chunks(project, indexer.client)}")


# ---------- ask ----------

@app.command()
def ask(ctx: typer.Context,
        question: Annotated[str, typer.Argument(help="The clinical question.")],
        source_type: Annotated[str | None, typer.Option(
            help="Only retrieve this source type, e.g. guideline | pubmed | bootstrap.")] = None,
        as_json: Annotated[bool, typer.Option("--json",
                                              help="Print the raw RAGResponse JSON.")] = False,
        ) -> None:
    """Answer a question with citations (calls Groq; no guardrails)."""
    # stderr, so --json output stays parseable.
    typer.echo(GUARDRAILS_NOTICE, err=True)
    response = make_service(ctx.obj["project"]).ask(question, source_type=source_type)
    if as_json:
        typer.echo(response.model_dump_json(indent=2))
    else:
        _print_response(response)


def _print_response(response: RAGResponse) -> None:
    typer.echo(f"\n{response.answer}\n")
    typer.echo(f"Confidence: {response.confidence}    Latency: {response.latency_ms} ms")
    typer.echo(f"Retrieved:  {len(response.contexts)} chunks, cited {len(response.citations)}")
    for c in response.citations:
        page = f", page {c.page}" if c.page is not None else ""
        # No raw score (reranker scores are often near 0 even for correct sources); block 1 is
        # the top-ranked chunk. --json keeps the score for debugging.
        label = "  (top match)" if c.ref == 1 else ""
        typer.echo(f"\n  [{c.ref}] {c.title}{page}{label}")
        if c.url:
            typer.echo(f"      {c.url}")
        typer.echo(f"      \"{c.snippet}\"")


# ---------- sources ----------

@sources_app.command("list")
def sources_list(ctx: typer.Context) -> None:
    """Every indexed source with its chunk count."""
    sources = make_service(ctx.obj["project"]).list_sources()
    if not sources:
        typer.echo("No sources indexed.")
        return
    width = max(len(s.source_id) for s in sources)
    for s in sources:
        typer.echo(f"{s.source_id:<{width}}  {s.source_type:<9}  {s.chunks:>5} chunks  {s.title}")
    typer.echo(f"\n{len(sources)} sources, {sum(s.chunks for s in sources)} chunks")


@sources_app.command("delete")
def sources_delete(ctx: typer.Context,
                   source_id: Annotated[str, typer.Argument(help="From `medrag sources list`.")],
                   yes: Annotated[bool, typer.Option("--yes", "-y",
                                                     help="Don't ask for confirmation.")] = False,
                   ) -> None:
    """Remove a source's chunks, docstore entries and files."""
    if not yes:
        typer.confirm(f"Delete source '{source_id}'?", abort=True)
    try:
        removed = make_service(ctx.obj["project"]).delete_source(source_id)
    except SourceNotFoundError as exc:
        _fail(str(exc))
    typer.echo(f"Deleted {source_id} ({removed} chunks removed)")


# ---------- eval ----------

@app.command("eval")
def eval_command(ctx: typer.Context,
                 limit: Annotated[int | None, typer.Option(
                     min=1, help="Only the first N golden-set cases.")] = None,
                 no_llm_metrics: Annotated[bool, typer.Option(
                     "--no-llm-metrics", help="No judge calls: retrieval hits and refusals "
                                              "only.")] = False,
                 metrics: Annotated[str, typer.Option(
                     help="Comma-separated LLM metrics.")] = ",".join(run_evals.METRIC_NAMES),
                 resume: Annotated[Path | None, typer.Option(
                     help="Continue this results file (evals/results/*.json).")] = None,
                 judge_delay: Annotated[float, typer.Option(
                     min=0, help="Seconds to wait after each judge call.")] = JUDGE_DELAY_SECONDS,
                 threshold: Annotated[list[str] | None, typer.Option(
                     help="Override a threshold, NAME=VALUE (repeatable).")] = None,
                 ) -> None:
    """Score the golden set: retrieval hits, refusals and DeepEval metrics (calls Groq).

    Exit code 0 when every threshold is met, 1 below a threshold, 2 stopped on a rate limit
    (continue with --resume)."""
    try:
        metric_names = [] if no_llm_metrics else run_evals.parse_metrics(metrics)
        thresholds = run_evals.parse_thresholds(threshold or [])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        code = run_evals.run(ctx.obj["project"], limit=limit, metrics=metric_names, resume=resume,
                             judge_delay=judge_delay, thresholds=thresholds,
                             service=make_service(ctx.obj["project"]))
    except run_evals.EvalError as exc:
        _fail(str(exc))
    raise typer.Exit(code=code)


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
