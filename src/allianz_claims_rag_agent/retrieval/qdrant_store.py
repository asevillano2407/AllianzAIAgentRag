"""Qdrant implementation of the vector store contract."""

import math
from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.errors import RetrievalError


class QdrantVectorStore:
    """Store and search source chunks in an embedded local Qdrant database."""

    def __init__(
        self,
        collection_name: str,
        path: Path | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")
        if client is None and path is None:
            raise ValueError("path is required when no Qdrant client is provided")

        self.collection_name = collection_name
        self._client = client or QdrantClient(path=str(path))

    def upsert(
        self,
        chunks: Sequence[SourceChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Validate and persist aligned chunks and vectors."""
        if not chunks:
            raise RetrievalError("At least one chunk is required for indexing")
        if len(chunks) != len(vectors):
            raise RetrievalError("The number of chunks and vectors must match")

        vector_size = self._validate_vectors(vectors)
        self._ensure_collection(vector_size)

        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=list(vector),
                payload=chunk.model_dump(mode="json", exclude_none=True),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise RetrievalError("Qdrant could not persist the embedded chunks") from exc

    def search(self, query_vector: Sequence[float], limit: int) -> list[SourceChunk]:
        """Search by cosine similarity and reconstruct validated domain objects."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._validate_vectors([query_vector])

        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=list(query_vector),
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            raise RetrievalError("Qdrant could not execute the semantic search") from exc

        results: list[SourceChunk] = []
        for point in response.points:
            if point.payload is None:
                raise RetrievalError("Qdrant returned a point without source metadata")
            results.append(SourceChunk.model_validate({**point.payload, "score": point.score}))
        return results

    def _ensure_collection(self, vector_size: int) -> None:
        try:
            if not self._client.collection_exists(self.collection_name):
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                return

            collection = self._client.get_collection(self.collection_name)
            vector_config = collection.config.params.vectors
            if not isinstance(vector_config, models.VectorParams):
                raise RetrievalError("Named vectors are not supported by this store")
            if vector_config.size != vector_size:
                raise RetrievalError(
                    "The collection vector size does not match the embedding model "
                    f"({vector_config.size} != {vector_size})"
                )
            if vector_config.distance != models.Distance.COSINE:
                raise RetrievalError("The collection must use cosine distance")
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError("Qdrant collection setup failed") from exc

    @staticmethod
    def _validate_vectors(vectors: Sequence[Sequence[float]]) -> int:
        if not vectors or not vectors[0]:
            raise RetrievalError("Embedding vectors cannot be empty")

        vector_size = len(vectors[0])
        for vector in vectors:
            if len(vector) != vector_size:
                raise RetrievalError("All embedding vectors must have the same size")
            if not all(math.isfinite(value) for value in vector):
                raise RetrievalError("Embedding vectors must contain only finite numbers")
        return vector_size
