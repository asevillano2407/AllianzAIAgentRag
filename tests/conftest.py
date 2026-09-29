"""Shared test doubles."""

from collections.abc import Sequence

from allianz_rag.models import DocumentChunk, RetrievedChunk


class FakeVectorStore:
    """In-memory store with deterministic results."""

    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = results or []
        self.chunks: list[DocumentChunk] = []

    def upsert(self, chunks: Sequence[DocumentChunk]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        return self.results[:top_k]

    def count(self) -> int:
        return len(self.chunks)
