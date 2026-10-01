"""Integration tests for the embedded Qdrant vector store."""

import math

import pytest
from qdrant_client import QdrantClient

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.errors import RetrievalError
from allianz_claims_rag_agent.retrieval import QdrantVectorStore


def _chunk(chunk_id: str, text: str, page: int = 1) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        text=text,
        source="manual.pdf",
        page=page,
        section="Convenios",
    )


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(
        collection_name="test_manual",
        client=QdrantClient(location=":memory:"),
    )


def test_store_indexes_and_ranks_chunks_by_cosine_similarity() -> None:
    store = _store()
    cide = _chunk("cide", "Convenio CIDE")
    ascide = _chunk("ascide", "Convenio ASCIDE", page=2)

    store.upsert([cide, ascide], [[1.0, 0.0], [0.0, 1.0]])
    results = store.search([0.9, 0.1], limit=2)

    assert [result.chunk_id for result in results] == ["cide", "ascide"]
    assert results[0].score is not None
    assert results[0].score > results[1].score
    assert results[0].section == "Convenios"


def test_store_upsert_replaces_a_chunk_with_the_same_deterministic_id() -> None:
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(collection_name="test_manual", client=client)

    store.upsert([_chunk("same-id", "original")], [[1.0, 0.0]])
    store.upsert([_chunk("same-id", "updated")], [[1.0, 0.0]])

    assert client.count("test_manual", exact=True).count == 1
    assert store.search([1.0, 0.0], limit=1)[0].text == "updated"


def test_store_rejects_vectors_with_different_sizes() -> None:
    store = _store()

    with pytest.raises(RetrievalError, match="same size"):
        store.upsert([_chunk("one", "one"), _chunk("two", "two")], [[1.0], [1.0, 2.0]])


def test_store_does_not_recreate_collection_for_a_different_model_dimension() -> None:
    store = _store()
    store.upsert([_chunk("one", "one")], [[1.0, 0.0]])

    with pytest.raises(RetrievalError, match="does not match"):
        store.upsert([_chunk("two", "two")], [[1.0, 0.0, 0.0]])


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_store_rejects_non_finite_vector_values(invalid_value: float) -> None:
    with pytest.raises(RetrievalError, match="finite"):
        _store().upsert([_chunk("one", "one")], [[invalid_value]])


def test_store_wraps_search_for_a_missing_collection() -> None:
    with pytest.raises(RetrievalError, match="semantic search"):
        _store().search([1.0, 0.0], limit=1)
