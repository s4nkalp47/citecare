"""Application settings, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    groq_api_key: str = ""
    groq_llm_model: str = "openai/gpt-oss-120b"
    groq_prompt_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    # Guardrail policy models: output safety check and input blocked-topics check (Phase 6).
    # gpt-oss-safeguard-20b also works but its free tier is 3 RPM / 2K TPM, too tight for both.
    groq_safety_model: str = "openai/gpt-oss-20b"
    groq_topic_model: str = "openai/gpt-oss-20b"
    groq_judge_model: str = "openai/gpt-oss-20b"
    # Sent to reasoning models (gpt-oss) as `reasoning_effort`; empty = don't send it.
    groq_reasoning_effort: str = "low"

    llama_cloud_api_key: str = ""
    # LlamaParse v2 tier: fast | cost_effective | agentic | agentic_plus ("fast" has no markdown)
    llama_parse_tier: str = "cost_effective"

    qdrant_url: str = "http://localhost:6333"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    # Prepended to questions (not documents) before embedding. bge v1.5 retrieval instruction;
    # FastEmbed does not add it itself. Empty = off. (Phase 4b)
    embed_query_instruction: str = "Represent this sentence for searching relevant passages: "
    # Local FastEmbed cross-encoder that re-scores the retrieved candidates (Phase 4b).
    # Projects opt in with ProjectConfig.rerank_candidates.
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # FastEmbed sparse model for hybrid search (projects with ProjectConfig.hybrid).
    sparse_model: str = "Qdrant/bm25"

    ncbi_api_key: str = ""

    data_dir: Path = ROOT_DIR / "data"
    # Largest source file accepted by SourceManager.upload (Phase 5).
    max_upload_mb: int = 25

    # Guardrails (Phase 6). "closed" blocks when a guard call fails; "open" lets the request through.
    guard_fail_mode: Literal["closed", "open"] = "closed"
    # Prompt Guard 2 probability above which a question is blocked as injection/jailbreak.
    prompt_guard_threshold: float = 0.5
    max_question_chars: int = 2000

    # Project served by the FastAPI app (Phase 7).
    api_project: str = "medrag"


@lru_cache
def get_settings() -> Settings:
    return Settings()
