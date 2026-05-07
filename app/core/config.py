from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Minimal config layer.

    Design choice: keep everything as env-vars so the app is Docker-friendly.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # RAG
    index_dir: str = "storage"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4

    # LLM
    llm_provider: str = "openai"  # openai (OpenAI-compatible)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_timeout_s: int = 60

    # API
    log_level: str = "INFO"


settings = Settings()

