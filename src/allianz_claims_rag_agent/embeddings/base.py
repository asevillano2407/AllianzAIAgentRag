"""Provider-independent contract for local embedding models."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Convert documents and queries into vectors in the same semantic space."""

    @property
    def model_name(self) -> str:
        """Return the stable identifier of the embedding model."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document fragments for indexing."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one user query for retrieval."""
