"""
Application configuration loaded from environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings. Every value comes from env vars or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Application ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    log_level: str = "INFO"

    # --- PostgreSQL ---
    postgres_user: str = "scider"
    postgres_password: str = "scider_dev_password"
    postgres_db: str = "scider_rag"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    database_url: str = (
        "postgresql+asyncpg://scider:scider_dev_password@postgres:5432/scider_rag"
    )

    # --- Redis ---
    redis_url: str = "redis://redis:6379/0"

    # --- Qdrant ---
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "scider_chunks"

    # --- OpenAI ---
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout_seconds: int = 30

    # --- Rate Limiting ---
    rate_limit_per_minute: int = 60

    # --- Ingestion ---
    max_file_size_mb: int = 50
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Agent ---
    agent_max_iterations: int = 5
    retrieval_top_k: int = 10

    # --- Sandbox ---
    sandbox_timeout_seconds: int = 10
    sandbox_max_memory_mb: int = 256

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance. Call once, reuse everywhere."""
    return Settings()
