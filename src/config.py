"""Application settings, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    groq_api_key: str = ""
    groq_llm_model: str = "openai/gpt-oss-120b"
    groq_prompt_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    groq_safety_model: str = "openai/gpt-oss-safeguard-20b"
    groq_judge_model: str = "openai/gpt-oss-20b"
    # Sent to reasoning models (gpt-oss) as `reasoning_effort`; empty = don't send it.
    groq_reasoning_effort: str = "low"

    llama_cloud_api_key: str = ""
    # LlamaParse v2 tier: fast | cost_effective | agentic | agentic_plus ("fast" has no markdown)
    llama_parse_tier: str = "cost_effective"

    qdrant_url: str = "http://localhost:6333"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    ncbi_api_key: str = ""

    data_dir: Path = ROOT_DIR / "data"
    # Largest source file accepted by SourceManager.upload (Phase 5).
    max_upload_mb: int = 25


@lru_cache
def get_settings() -> Settings:
    return Settings()
