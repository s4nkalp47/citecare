"""Phase 11: Docker Compose, Dockerfile, .dockerignore and workflow structure; model prefetch."""

import dataclasses
import re

import pytest
import yaml

from scripts import prefetch_models
from src.config import ROOT_DIR, Settings
from src.core.projects import get_project


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT_DIR / path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def services() -> dict:
    return _yaml("docker-compose.yml")["services"]


def _env(service: dict) -> dict:
    env = service.get("environment", {})
    return env if isinstance(env, dict) else dict(item.split("=", 1) for item in env)


# ---------- docker-compose.yml ----------

def test_compose_has_the_stack(services):
    assert set(services) == {"qdrant", "indexer", "api", "ui"}
    assert "8000:8000" in services["api"]["ports"]
    assert "8501:8501" in services["ui"]["ports"]
    assert "qdrant_storage:/qdrant/storage" in services["qdrant"]["volumes"]


def test_startup_order(services):
    assert "healthcheck" in services["qdrant"]
    assert services["indexer"]["restart"] == "no"
    assert services["indexer"]["depends_on"]["qdrant"]["condition"] == "service_healthy"
    api_deps = services["api"]["depends_on"]
    assert api_deps["indexer"]["condition"] == "service_completed_successfully"
    assert api_deps["qdrant"]["condition"] == "service_healthy"
    assert services["ui"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_qdrant_healthcheck_needs_no_curl(services):
    # qdrant/qdrant has no curl/wget/nc; the check uses bash's /dev/tcp.
    test = services["qdrant"]["healthcheck"]["test"]
    assert test[:3] == ["CMD", "bash", "-c"]
    assert "/dev/tcp/127.0.0.1/6333" in test[3] and "/readyz" in test[3]


def test_build_targets(services):
    for name in ("indexer", "api", "ui"):
        assert services[name]["build"]["target"] == name
        assert services[name]["build"]["context"] == "."


def test_only_indexer_and_api_get_secrets(services):
    for name in ("indexer", "api"):
        assert services[name]["env_file"] == ".env"
        assert _env(services[name])["QDRANT_URL"] == "http://qdrant:6333"
        assert "./data:/app/data" in services[name]["volumes"]
    ui = services["ui"]
    assert "env_file" not in ui
    assert _env(ui) == {"API_URL": "http://api:8000"}


def test_no_inline_secrets(services):
    for service in services.values():
        assert not any(re.search(r"(API_KEY|TOKEN|SECRET)", key) for key in _env(service))


# ---------- Dockerfile and .dockerignore ----------

def test_dockerfile_targets_and_model_cache():
    text = (ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")
    stages = re.findall(r"^FROM \S+ AS (\w+)$", text, flags=re.MULTILINE)
    assert {"api", "ui", "indexer"} <= set(stages)
    assert "FASTEMBED_CACHE_PATH=/opt/fastembed" in text
    assert "python -m scripts.prefetch_models" in text
    # The UI image has no models.
    assert re.search(r"^FROM deps AS ui$", text, flags=re.MULTILINE)


def test_dockerignore_excludes_secrets_and_state():
    lines = {line.strip().rstrip("/")
             for line in (ROOT_DIR / ".dockerignore").read_text(encoding="utf-8").splitlines()}
    assert {".env", ".venv", ".git", "data"} <= lines


# ---------- workflows ----------

def _triggers(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))  # PyYAML reads a bare `on` key as True


def test_ci_workflow():
    ci = _yaml(".github/workflows/ci.yml")
    assert set(_triggers(ci)) == {"push", "pull_request"}
    assert {"lint", "test", "secrets", "images"} <= set(ci["jobs"])
    images = ci["jobs"]["images"]
    assert images["permissions"] == {"contents": "read", "packages": "write"}
    assert set(images["needs"]) == {"lint", "test", "secrets"}
    assert "refs/heads/main" in images["if"]
    assert images["strategy"]["matrix"]["target"] == ["api", "ui", "indexer"]


def test_eval_workflow_is_manual_only():
    ev = _yaml(".github/workflows/eval.yml")
    assert set(_triggers(ev)) == {"workflow_dispatch"}
    assert _triggers(ev)["workflow_dispatch"]["inputs"]["limit"]["default"] == "5"


# ---------- scripts/prefetch_models.py ----------

SETTINGS = Settings(_env_file=None, embed_model="e", rerank_model="r", sparse_model="s")


def test_models_to_fetch_follows_projects():
    base = dataclasses.replace(get_project("medrag"), rerank_candidates=0, hybrid=False)
    assert prefetch_models.models_to_fetch(SETTINGS, [base]) == [("embedding", "e")]
    reranked = dataclasses.replace(base, name="b", rerank_candidates=20)
    hybrid = dataclasses.replace(base, name="c", hybrid=True)
    assert prefetch_models.models_to_fetch(SETTINGS, [base, reranked, hybrid]) == [
        ("embedding", "e"), ("reranker", "r"), ("sparse", "s")]


def test_medrag_needs_the_reranker():
    kinds = [k for k, _ in prefetch_models.models_to_fetch(SETTINGS, [get_project("medrag")])]
    assert kinds[:2] == ["embedding", "reranker"]


def test_main_instantiates_each_model(monkeypatch):
    loaded: list[tuple[str, str]] = []

    def recorder(kind):
        return lambda model_name: loaded.append((kind, model_name))

    monkeypatch.setattr("fastembed.TextEmbedding", recorder("embedding"))
    monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", recorder("reranker"))
    monkeypatch.setattr("fastembed.SparseTextEmbedding", recorder("sparse"))
    monkeypatch.setattr(prefetch_models, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(prefetch_models, "list_projects", lambda: ["medrag"])
    prefetch_models.main()
    assert loaded == prefetch_models.models_to_fetch(SETTINGS, [get_project("medrag")])


def test_fetch_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown model kind"):
        prefetch_models.fetch("colbert", "x")
