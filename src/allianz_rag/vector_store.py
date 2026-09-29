"""Vector-store boundary and Chroma implementation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from allianz_rag.models import DocumentChunk, RetrievedChunk


class VectorStore(Protocol):
    """Storage contract used by ingestion, graph and evaluation."""

    def upsert(self, chunks: Sequence[DocumentChunk]) -> int:
        """Persist chunks and return the number written."""

    def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """Return the nearest chunks ordered by ascending distance."""

    def count(self) -> int:
        """Return the collection size."""


class ChromaVectorStore:
    """Persistent Chroma collection with local multilingual embeddings."""

    def __init__(self, path: Path, collection_name: str, embedding_model: str) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Chroma dependencies are missing. Install the project with `pip install -e .`."
            ) from exc

        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path))
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: Sequence[DocumentChunk]) -> int:
        """Persist chunks in batches to avoid oversized Chroma operations."""

        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[
                    {
                        "source": chunk.source,
                        "page": chunk.page,
                        "section": chunk.section,
                    }
                    for chunk in batch
                ],
            )
        return len(chunks)

    def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """Return Chroma matches as typed domain models."""

        result = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                source=str(metadata["source"]),
                page=int(metadata["page"]),
                section=str(metadata["section"]),
                text=document,
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return int(self._collection.count())
