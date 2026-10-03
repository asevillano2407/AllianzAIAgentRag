"""Tests for provider-independent indexing and retrieval services."""

from collections.abc import Sequence

import pytest

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.errors import EmbeddingError
from allianz_claims_rag_agent.retrieval.services import (
    IndexingProgress,
    IndexingService,
    SemanticRetriever,
)


class FakeEmbeddingProvider:
    """Small deterministic provider that never loads a real ML model."""

    model_name = "fake-embedding-v1"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class RecordingVectorStore:
    def __init__(self) -> None:
        self.upsert_batches: list[tuple[list[SourceChunk], list[list[float]]]] = []
        self.search_vector: list[float] | None = None
        self.search_limit: int | None = None

    def upsert(
        self,
        chunks: Sequence[SourceChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self.upsert_batches.append((list(chunks), [list(vector) for vector in vectors]))

    def search(self, query_vector: Sequence[float], limit: int) -> list[SourceChunk]:
        self.search_vector = list(query_vector)
        self.search_limit = limit
        return [_chunk("result", "evidence")]


class SequencedVectorStore(RecordingVectorStore):
    def __init__(self, results: list[list[SourceChunk]]) -> None:
        super().__init__()
        self.results = results
        self.search_limits: list[int] = []

    def search(self, query_vector: Sequence[float], limit: int) -> list[SourceChunk]:
        self.search_limits.append(limit)
        return self.results[len(self.search_limits) - 1]


class InvalidCountProvider(FakeEmbeddingProvider):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return []


def _chunk(chunk_id: str, text: str) -> SourceChunk:
    return SourceChunk(chunk_id=chunk_id, text=text, source="manual.pdf", page=1)


def test_indexing_service_processes_bounded_batches() -> None:
    store = RecordingVectorStore()
    progress: list[IndexingProgress] = []
    service = IndexingService(
        FakeEmbeddingProvider(), store, batch_size=2, on_progress=progress.append
    )
    chunks = [_chunk("one", "a"), _chunk("two", "bb"), _chunk("three", "ccc")]

    summary = service.index(chunks)

    assert [len(batch[0]) for batch in store.upsert_batches] == [2, 1]
    assert store.upsert_batches[0][1] == [[1.0, 1.0], [2.0, 1.0]]
    assert summary.model_name == "fake-embedding-v1"
    assert summary.indexed_chunks == 3
    assert summary.vector_size == 2
    assert progress == [
        IndexingProgress(indexed_chunks=2, total_chunks=3),
        IndexingProgress(indexed_chunks=3, total_chunks=3),
    ]


def test_indexing_service_rejects_provider_vector_count_mismatch() -> None:
    service = IndexingService(InvalidCountProvider(), RecordingVectorStore())

    with pytest.raises(EmbeddingError, match="unexpected vector count"):
        service.index([_chunk("one", "text")])


def test_semantic_retriever_embeds_normalized_query_and_passes_limit() -> None:
    store = RecordingVectorStore()
    retriever = SemanticRetriever(FakeEmbeddingProvider(), store)

    results = retriever.retrieve("  CIDE  ", limit=4)

    assert store.search_vector == [4.0, 1.0]
    assert store.search_limit == 4
    assert results[0].chunk_id == "result"


def test_semantic_retriever_rejects_empty_query() -> None:
    retriever = SemanticRetriever(FakeEmbeddingProvider(), RecordingVectorStore())

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.retrieve("   ", limit=4)


def test_semantic_retriever_fuses_multiple_query_rankings_and_removes_duplicates() -> None:
    first = _chunk("first", "first evidence")
    shared = _chunk("shared", "shared evidence")
    last = _chunk("last", "last evidence")
    store = SequencedVectorStore([[first, shared], [shared, last]])
    retriever = SemanticRetriever(FakeEmbeddingProvider(), store)

    results = retriever.retrieve_many(["original", "technical"], limit=3)

    assert [chunk.chunk_id for chunk in results] == ["shared", "first", "last"]
    assert store.search_limits == [6, 6]
    assert results[0].score is not None


def test_semantic_retriever_overfetches_single_query_before_final_cut() -> None:
    chunks = [_chunk(str(index), f"evidence {index}") for index in range(6)]
    store = SequencedVectorStore([chunks])
    retriever = SemanticRetriever(FakeEmbeddingProvider(), store)

    results = retriever.retrieve_many(["original"], limit=3)

    assert store.search_limits == [6]
    assert len(results) == 3


def test_semantic_retriever_rejects_empty_query_collection() -> None:
    retriever = SemanticRetriever(FakeEmbeddingProvider(), RecordingVectorStore())

    with pytest.raises(ValueError, match="non-empty query"):
        retriever.retrieve_many(["", "   "], limit=3)
