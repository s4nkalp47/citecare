# CiteCare

Clinical-guideline question answering with LlamaIndex, Qdrant and Groq.

The Python package, `medrag` CLI command, project name and Qdrant collection (`medrag_guidelines`) keep
the internal name `medrag`, so existing indexes, parse caches and docstores stay valid.

## 1. Get the guideline PDFs

The PDFs are not in the repo (`data/guidelines/*.pdf` is gitignored). Download each from its NICE
page (**Tools and resources → Download guidance (PDF)**) and save it in `data/guidelines/` under
exactly this name. The file name sets the citation title and link, and the parse cache is keyed on
the name and file hash:

| Guideline | Page | Save as |
|---|---|---|
| NG136 Hypertension in adults | https://www.nice.org.uk/guidance/ng136 | `nice_ng136_hypertension.pdf` |
| NG28 Type 2 diabetes in adults | https://www.nice.org.uk/guidance/ng28 | `nice_ng28_type2_diabetes.pdf` |
| NG250 Pneumonia | https://www.nice.org.uk/guidance/ng250 | `nice_ng250_pneumonia.pdf` |

> **LlamaParse credits:** the first ingest on a machine without `data/parsed/` parses every PDF
> with LlamaParse, which spends credits and needs `LLAMA_CLOUD_API_KEY`. The result is cached in
> `data/parsed/<name>-<hash>.json` and later runs (local or Docker) reuse it. A different
> version of a PDF has a different hash, so it is parsed again. Copy `data/parsed/` from another
> machine to skip parsing.

## 2. Configure

```powershell
copy .env.example .env      # then add GROQ_API_KEY (and LLAMA_CLOUD_API_KEY for the first parse)
```

## 3a. Run everything in Docker

```powershell
docker compose up --build
```

- `qdrant` starts (healthcheck on `/readyz`), then the one-shot `indexer` runs `medrag ingest`
  over the mounted `./data` and exits. Unchanged documents are skipped, and a PubMed (NCBI)
  outage only drops the abstracts. Then `api` starts, then `ui`.
- UI: http://localhost:8501 · API docs: http://localhost:8000/docs · health:
  http://localhost:8000/health
- Any CLI command against the stack: `docker compose run --rm indexer sources list`,
  `docker compose run --rm indexer ingest --no-pubmed --rebuild`.
- Only `indexer` and `api` read `.env`. The UI gets just `API_URL`. Inside the network Compose
  points `QDRANT_URL` at `http://qdrant:6333`.
- FastEmbed models (embedding, reranker) are downloaded at build time, as named in `.env`
  (`EMBED_MODEL`, `RERANK_MODEL`, `SPARSE_MODEL`). Rebuild after changing them.
- The Qdrant volume is the same one the local workflow uses, so an existing index is reused.

## 3b. Or run locally (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
docker compose up -d qdrant
python -m scripts.check_setup
medrag ingest                                   # or: python -m scripts.build_index
uvicorn src.api.main:app --reload               # http://localhost:8000/docs
streamlit run src/ui/app.py                     # http://localhost:8501 (needs the API)
```

CLI (calls `RAGService` directly, **no guardrails**):

```powershell
medrag ask "What is the first-line treatment for hypertension in type 2 diabetes?"
medrag sources list
medrag sources delete <source_id>
```

## Tests and evals

```powershell
pytest -q                                        # no API keys needed: Groq, LlamaParse, PubMed mocked
ruff check .
medrag eval --limit 3                            # calls Groq (generator + judge)
medrag eval --no-llm-metrics                     # retrieval hits and refusals only
```

## CI/CD (GitHub Actions)

- `.github/workflows/ci.yml` runs on every push and PR: `ruff check`, `pytest`, and a gitleaks
  secret scan over the full history. On a push to `main` it then builds the `api`, `ui` and
  `indexer` images and pushes them to GHCR as `ghcr.io/<owner>/<repo>-<target>:latest` (and
  `:sha-…`) with `GITHUB_TOKEN`.
- `.github/workflows/eval.yml` runs only by hand (Actions → eval → Run workflow, default
  `--limit 5`). It needs repository secrets `GROQ_API_KEY` and `QDRANT_URL`, an already-indexed
  Qdrant reachable from GitHub without an API key, because CI can't build the index itself. It
  stops at once without them.

## Progress

- [x] Phase 1: Foundation (config, project registry, MedRAG plugin config, Qdrant)
- [x] Phase 2: Ingestion (LlamaParse, PubmedReader, bootstrap docs)
- [x] Phase 3: Indexing (chunking, FastEmbed, Qdrant)
- [x] Phase 4: Query engine and RAGService
- [x] Phase 4b: Retrieval quality (reranker, smaller chunks, bge query instruction)
- [x] Phase 5: Source manager
- [x] Phase 6: Input/output guardrails
- [x] Phase 7: FastAPI
- [x] Phase 8: Streamlit UI
- [x] Phase 9: CLI
- [x] Phase 10: Evals (DeepEval)
- [x] Phase 11: Docker Compose and CI/CD
