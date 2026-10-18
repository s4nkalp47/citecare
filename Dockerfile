# Phase 11: one image definition, three targets (docker compose builds them all).
#   docker build --target api .        FastAPI, FastEmbed models baked in
#   docker build --target indexer .    one-shot `medrag ingest`, same models
#   docker build --target ui .         Streamlit; no models (it only talks to the API)

# ---------- deps: Python packages and the source ----------
FROM python:3.12-slim AS deps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first, from a stub package, so a source change doesn't reinstall them.
COPY pyproject.toml ./
RUN mkdir src evals && touch src/__init__.py evals/__init__.py \
    && pip install . \
    && pip uninstall -y medrag \
    && rm -rf src evals build *.egg-info

COPY src ./src
COPY evals ./evals
COPY scripts ./scripts
# Editable, so src/config.py's ROOT_DIR is /app (a site-packages install would put data_dir there).
RUN pip install --no-deps -e . \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /opt/fastembed \
    && chown app:app /app/data /opt/fastembed
# data/ is mounted at runtime (docker-compose.yml); .env is never in the image (.dockerignore).
ENV DATA_DIR=/app/data
USER app

# ---------- models: FastEmbed models downloaded at build time ----------
FROM deps AS models
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
ARG SPARSE_MODEL=Qdrant/bm25
# FastEmbed reads FASTEMBED_CACHE_PATH for every model class, at build and at runtime.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed \
    EMBED_MODEL=${EMBED_MODEL} \
    RERANK_MODEL=${RERANK_MODEL} \
    SPARSE_MODEL=${SPARSE_MODEL}
RUN python -m scripts.prefetch_models

# ---------- api ----------
FROM models AS api
EXPOSE 8000
# /health is 503 when Qdrant is unreachable, which urlopen raises on.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- indexer ----------
FROM models AS indexer
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["ingest"]

# ---------- ui ----------
FROM deps AS ui
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]
CMD ["streamlit", "run", "src/ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", \
     "--server.headless", "true", "--browser.gatherUsageStats", "false"]
