"""Tests for validated application settings."""

import pytest
from pydantic import ValidationError

from allianz_claims_rag_agent.config import Environment, Settings


def test_settings_use_safe_local_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment is Environment.LOCAL
    assert settings.manual_path.name == "Manual-cide-ascide-y-cicos.pdf"
    assert settings.retrieval_top_k == 3
    assert settings.max_agent_retries == 1
    assert settings.qdrant_collection_prefix == "allianz_manual"
    assert settings.embedding_batch_size == 8
    assert settings.llm_model == "llama3.2:3b"
    assert settings.ollama_timeout_seconds == 300.0


def test_settings_can_be_overridden_with_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLIANZ_ENVIRONMENT", "test")
    monkeypatch.setenv("ALLIANZ_RETRIEVAL_TOP_K", "4")

    settings = Settings(_env_file=None)

    assert settings.environment is Environment.TEST
    assert settings.retrieval_top_k == 4


def test_chunk_window_can_be_overridden_with_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLIANZ_CHUNK_SIZE", "800")
    monkeypatch.setenv("ALLIANZ_CHUNK_OVERLAP", "100")

    settings = Settings(_env_file=None)

    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100


def test_settings_reject_invalid_retrieval_limit() -> None:
    with pytest.raises(ValidationError):
        Settings(retrieval_top_k=0, _env_file=None)


def test_settings_reject_overlap_larger_than_half_chunk() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size=400, chunk_overlap=201, _env_file=None)
