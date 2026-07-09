"""Application configuration.

Everything runs against a local Ollama instance, so there are no secrets to
manage. Values are read from environment variables with working defaults, so
the app runs with zero configuration locally — while still allowing a
deployment platform to override HOST/PORT at runtime.
"""

import os


class Settings:
    """Centralized settings. Every value has a working default."""

    # Ollama / LLM
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))

    # API
    API_TITLE: str = "RAG Document Q&A API"
    API_VERSION: str = "0.1.0"
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
