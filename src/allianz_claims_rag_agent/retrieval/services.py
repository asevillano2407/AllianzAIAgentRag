"""Application services for indexing and semantic retrieval."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.embeddings import EmbeddingProvider
from allianz_claims_rag_agent.errors import EmbeddingError
from allianz_claims_rag_agent.retrieval.base import VectorStore


@dataclass(frozen=True, slots=True)
class IndexingSummary:
    """Observable result of one indexing operation."""

    model_name: str
    indexed_chunks: int
    vector_size: int


@dataclass(frozen=True, slots=True)
class IndexingProgress:
    """Progress reported after one batch has been safely persisted."""

    indexed_chunks: int
    total_chunks: int


class IndexingService:
    """Embed chunks in bounded batches and persist them in a vector store."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        batch_size: int = 32,
        on_progress: Callable[[IndexingProgress], None] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._batch_size = batch_size
        self._on_progress = on_progress

    def index(self, chunks: Sequence[SourceChunk]) -> IndexingSummary:
        """Index all chunks while keeping model inference memory bounded."""
        if not chunks:
            raise ValueError("At least one chunk is required for indexing")

        vector_size: int | None = None
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors = self._embedding_provider.embed_documents([chunk.text for chunk in batch])
            if len(vectors) != len(batch):
                raise EmbeddingError("The embedding provider returned an unexpected vector count")
            if vectors:
                current_size = len(vectors[0])
                if vector_size is not None and current_size != vector_size:
                    raise EmbeddingError("The embedding vector size changed between batches")
                vector_size = current_size
            self._vector_store.upsert(batch, vectors)
            if self._on_progress is not None:
                self._on_progress(
                    IndexingProgress(
                        indexed_chunks=min(start + len(batch), len(chunks)),
                        total_chunks=len(chunks),
                    )
                )

        if vector_size is None:
            raise EmbeddingError("The embedding provider returned no vectors")
        return IndexingSummary(
            model_name=self._embedding_provider.model_name,
            indexed_chunks=len(chunks),
            vector_size=vector_size,
        )


class SemanticRetriever:
    """Embed a validated question and recover its closest source chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def retrieve(self, query: str, limit: int) -> list[SourceChunk]:
        """Return semantic evidence for a non-empty user query."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        query_vector = self._embedding_provider.embed_query(normalized_query)
        return self._vector_store.search(query_vector, limit)
