"""Tests for the Ollama embedding API adapter."""

import json
import math

import httpx
import pytest

from allianz_claims_rag_agent.embeddings import OllamaEmbeddingProvider
from allianz_claims_rag_agent.errors import EmbeddingError


def _provider(handler: httpx.MockTransport) -> OllamaEmbeddingProvider:
    client = httpx.Client(base_url="http://ollama.test", transport=handler)
    return OllamaEmbeddingProvider(
        base_url="http://unused.test",
        model_name="test-embedding",
        client=client,
    )


def test_embed_documents_sends_batch_without_silent_truncation() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/embed"
        assert payload == {
            "model": "test-embedding",
            "input": ["CIDE", "ASCIDE"],
            "truncate": False,
        }
        return httpx.Response(200, json={"embeddings": [[1, 0.5], [0.2, 1]]})

    provider = _provider(httpx.MockTransport(handle))

    assert provider.embed_documents(["CIDE", "ASCIDE"]) == [[1.0, 0.5], [0.2, 1.0]]


def test_embed_query_returns_the_single_vector() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"embeddings": [[0.1, 0.9]]})
    )

    assert _provider(transport).embed_query("¿Cuándo se aplica CIDE?") == [0.1, 0.9]


def test_empty_document_batch_does_not_call_ollama() -> None:
    def unexpected_call(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Ollama should not be called")

    assert _provider(httpx.MockTransport(unexpected_call)).embed_documents([]) == []


def test_provider_reports_unavailable_ollama() -> None:
    def refuse_connection(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    provider = _provider(httpx.MockTransport(refuse_connection))

    with pytest.raises(EmbeddingError, match="Cannot connect"):
        provider.embed_query("CIDE")


def test_provider_reports_missing_model_without_exposing_response_body() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(404, json={"error": "model not found"})
    )

    with pytest.raises(EmbeddingError, match="HTTP 404"):
        _provider(transport).embed_query("CIDE")


def test_provider_reports_timeout_with_configuration_guidance() -> None:
    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow model", request=request)

    with pytest.raises(EmbeddingError, match="ALLIANZ_EMBEDDING_BATCH_SIZE"):
        _provider(httpx.MockTransport(time_out)).embed_query("CIDE")


@pytest.mark.parametrize(
    "embeddings, message",
    [
        ([], "unexpected number"),
        ([[]], "empty or invalid"),
        ([[True]], "only numbers"),
        ([[math.nan]], "only finite"),
    ],
)
def test_provider_rejects_invalid_embedding_payloads(
    embeddings: list[list[object]],
    message: str,
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=json.dumps({"embeddings": embeddings}),
        )
    )

    with pytest.raises(EmbeddingError, match=message):
        _provider(transport).embed_query("CIDE")
