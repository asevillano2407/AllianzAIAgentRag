"""Environment-based application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["local", "test", "production"] = "local"
    log_level: str = "INFO"
    manual_path: Path = Path("data/raw/Manual-cide-ascide-y-cicos.pdf")
    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "cide_ascide_cicos"
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    top_k: int = Field(default=6, ge=1, le=20)
    max_distance: float = Field(default=0.65, ge=0.0, le=2.0)
    llm_provider: Literal["vertexai"] = "vertexai"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    google_cloud_project: str | None = None
    google_cloud_location: str = "europe-west1"
    max_input_characters: int = Field(default=6000, ge=200, le=20000)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Return an uppercase logging level."""

        return value.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()
