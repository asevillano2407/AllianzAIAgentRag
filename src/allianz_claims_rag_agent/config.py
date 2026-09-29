"""Runtime configuration loaded from environment variables."""

from enum import StrEnum
from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Supported application environments."""

    LOCAL = "local"
    TEST = "test"


class LogLevel(StrEnum):
    """Supported logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """Validated configuration for local application components."""

    model_config = SettingsConfigDict(
        env_prefix="ALLIANZ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Environment.LOCAL
    log_level: LogLevel = LogLevel.INFO
    manual_path: Path = Path("data/raw/Manual-cide-ascide-y-cicos.pdf")
    qdrant_path: Path = Path("data/qdrant")
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    embedding_model: str = Field(default="qwen3-embedding:0.6b", min_length=1)
    llm_model: str = Field(default="qwen3:4b", min_length=1)
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    max_agent_retries: int = Field(default=1, ge=0, le=3)
