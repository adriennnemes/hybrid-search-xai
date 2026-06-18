"""
Centralized configuration.

Why this file exists:
- Keep ALL env-based configuration in ONE place.
- Avoid os.getenv() scattered across the codebase.
- Provide typed defaults + safe parsing (int/float).

Fallback:
This ensures the app always has valid configuration, even if environment variables are missing.

Each embedding model must use its own Chroma collection. This prevents mixing vector spaces across models and ensures fair evaluation.
"""

import os
from functools import lru_cache
from pydantic import BaseModel

class Settings(BaseModel):
    # Runtime mode
    ENV: str = "dev"

    # ChromaDB connection settings
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "xai_papers"

    # Embedding model configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Search defaults
    TOP_K_DEFAULT: int = 10
    ALPHA_DEFAULT: float = 0.5

    # Logging (file-only)
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "/app/logs/app.log"


def _read_env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def _read_int(key: str, default: int) -> int:
    raw = _read_env(key, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_float(key: str, default: float) -> float:
    raw = _read_env(key, None)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

@lru_cache(maxsize=1)
def get_settings() -> Settings: 
    return Settings(
        ENV=_read_env("ENV", "dev"),

        CHROMA_HOST=_read_env("CHROMA_HOST", "chromadb"),
        CHROMA_PORT=_read_int("CHROMA_PORT", 8000),

        CHROMA_COLLECTION=_read_env("CHROMA_COLLECTION", "xai_papers"),
        EMBEDDING_MODEL=_read_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),

        TOP_K_DEFAULT=_read_int("TOP_K_DEFAULT", 10),
        ALPHA_DEFAULT=_read_float("ALPHA_DEFAULT", 0.5),

        LOG_LEVEL=_read_env("LOG_LEVEL", "INFO"),
        LOG_FILE_PATH=_read_env("LOG_FILE_PATH", "/app/logs/app.log"),
    )

settings = get_settings()