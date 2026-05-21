"""
Central configuration for the Stylist Concierge.
All environment variables are loaded here.
Switching LLM provider = change LLM_PROVIDER + LLM_MODEL in .env — nothing else.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────
#  Enums for type-safe config
# ─────────────────────────────────────────────

class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    LOCAL = "local"  # sentence-transformers


class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# ─────────────────────────────────────────────
#  Main Settings
# ─────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    secret_key: str = "change-me-in-production"
    debug: bool = True
    app_name: str = "Stylist Concierge API"
    app_version: str = "1.0.0"

    # ── LLM Provider ─────────────────────────
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    agent_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    agent_max_tokens: int = Field(default=2048, ge=128)
    agent_max_iterations: int = Field(default=10, ge=1)

    # ── Provider Keys ─────────────────────────
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # ── Embeddings ───────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # ── Qdrant ───────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "fashion_catalog"

    # ── PostgreSQL ───────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "stylist"
    postgres_password: str = "stylistpass"
    postgres_db: str = "stylist_db"
    database_url: str = "postgresql+asyncpg://stylist:stylistpass@localhost:5432/stylist_db"

    # ── Redis ────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_url: str = "redis://localhost:6379/0"

    # ── Cache ────────────────────────────────
    cache_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    cache_ttl_seconds: int = 3600

    # ── Scraper ──────────────────────────────
    scraper_headless: bool = True
    scraper_timeout: int = 30000         # ms
    scraper_max_retries: int = 3
    scraper_delay_min: float = 1.0       # seconds between requests
    scraper_delay_max: float = 3.0
    scraper_concurrent_limit: int = 3

    # ── Rate Limiting ─────────────────────────
    rate_limit_per_minute: int = 30

    # ─────────────────────────────────────────
    #  Derived helpers (not env vars)
    # ─────────────────────────────────────────

    @property
    def active_api_key(self) -> Optional[str]:
        """Returns the API key for the currently configured LLM provider."""
        mapping = {
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.GOOGLE: self.google_api_key,
        }
        return mapping.get(self.llm_provider)

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnvironment.PRODUCTION

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @field_validator("agent_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


# ─────────────────────────────────────────────
#  LLM Factory — pluggable provider
# ─────────────────────────────────────────────

def get_llm(settings: Optional[Settings] = None):
    """
    Returns a LangChain-compatible LLM object for the configured provider.
    Switch providers by changing LLM_PROVIDER + LLM_MODEL in .env.
    """
    cfg = settings or get_settings()

    if cfg.llm_provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=cfg.llm_model,
            api_key=cfg.groq_api_key,
            temperature=cfg.agent_temperature,
            max_tokens=cfg.agent_max_tokens,
        )

    elif cfg.llm_provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg.llm_model,
            api_key=cfg.openai_api_key,
            temperature=cfg.agent_temperature,
            max_tokens=cfg.agent_max_tokens,
        )

    elif cfg.llm_provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=cfg.llm_model,
            api_key=cfg.anthropic_api_key,
            temperature=cfg.agent_temperature,
            max_tokens=cfg.agent_max_tokens,
        )

    elif cfg.llm_provider == LLMProvider.GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            google_api_key=cfg.google_api_key,
            temperature=cfg.agent_temperature,
            max_output_tokens=cfg.agent_max_tokens,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {cfg.llm_provider}")


def get_embedding_model(settings: Optional[Settings] = None):
    """
    Returns a LangChain-compatible embedding model for the configured provider.
    """
    cfg = settings or get_settings()

    if cfg.embedding_provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=cfg.embedding_model,
            api_key=cfg.openai_api_key,
        )

    elif cfg.embedding_provider == EmbeddingProvider.LOCAL:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {cfg.embedding_provider}")


# ─────────────────────────────────────────────
#  Singleton accessor (cached)
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()