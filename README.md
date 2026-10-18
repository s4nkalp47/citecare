# MedRAG

Clinical-guideline question answering with LlamaIndex, Qdrant and Groq.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env      # then add your GROQ_API_KEY
docker compose up -d qdrant
python -m scripts.check_setup
pytest
```

## Progress

- [x] Phase 1: Foundation (config, project registry, MedRAG plugin config, Qdrant)
- [x] Phase 2: Ingestion (LlamaParse, PubmedReader, bootstrap docs)
- [x] Phase 3: Indexing (chunking, FastEmbed, Qdrant)
- [x] Phase 4: Query engine and RAGService
- [x] Phase 4b: Retrieval quality (reranker, smaller chunks, bge query instruction)
- [x] Phase 5: Source manager
- [x] Phase 6: Input/output guardrails
- [ ] Phase 7: FastAPI
- [ ] Phase 8: Streamlit UI
- [ ] Phase 9: CLI
- [ ] Phase 10: Evals (DeepEval)
- [ ] Phase 11: Docker Compose and CI/CD
