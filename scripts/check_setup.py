"""Phase 1 smoke test: config loads, Qdrant is reachable, Groq key works.

Run from the project root:  python -m scripts.check_setup
"""

import sys

from groq import Groq
from qdrant_client import QdrantClient

from src.config import get_settings
from src.core.projects import get_project, list_projects


def check(label: str, fn) -> bool:
    try:
        detail = fn()
        print(f"[OK]   {label}: {detail}")
        return True
    except Exception as exc:  # noqa: BLE001 - report any failure
        print(f"[FAIL] {label}: {exc}")
        return False


def check_registry() -> str:
    project = get_project("medrag")
    return f"projects={list_projects()}, collection={project.collection_name}"


def check_qdrant() -> str:
    client = QdrantClient(url=get_settings().qdrant_url, timeout=5)
    names = [c.name for c in client.get_collections().collections]
    return f"connected, collections={names}"


def check_groq() -> str:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is empty - add it to .env")
    client = Groq(api_key=settings.groq_api_key)
    reply = client.chat.completions.create(
        model=settings.groq_llm_model,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        max_tokens=5,
        temperature=0,
    )
    return f"{settings.groq_llm_model} replied '{reply.choices[0].message.content.strip()}'"


if __name__ == "__main__":
    results = [
        check("Project registry", check_registry),
        check("Qdrant", check_qdrant),
        check("Groq", check_groq),
    ]
    sys.exit(0 if all(results) else 1)
