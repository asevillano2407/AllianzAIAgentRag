"""Storage-independent contract for a vector index."""

from collections.abc import Sequence
from typing import Protocol

from allianz_claims_rag_agent.domain import SourceChunk


class VectorStore(Protocol):
    """Persist chunk vectors and recover their nearest neighbours."""

    def upsert(self, chunks: Sequence[SourceChunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert new points or replace existing points with the same IDs."""

    def search(self, query_vector: Sequence[float], limit: int) -> list[SourceChunk]:
        """Return the closest chunks ordered by descending similarity."""
