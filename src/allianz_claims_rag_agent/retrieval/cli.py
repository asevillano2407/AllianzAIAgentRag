"""Command-line entry points for local vector indexing and search."""

import argparse
import json
import sys
from pathlib import Path

from allianz_claims_rag_agent.config import Settings
from allianz_claims_rag_agent.embeddings import OllamaEmbeddingProvider
from allianz_claims_rag_agent.errors import ApplicationError
from allianz_claims_rag_agent.ingestion.serialization import read_chunks_jsonl
from allianz_claims_rag_agent.retrieval.naming import collection_name_for_model
from allianz_claims_rag_agent.retrieval.qdrant_store import QdrantVectorStore
from allianz_claims_rag_agent.retrieval.services import (
    IndexingProgress,
    IndexingService,
    SemanticRetriever,
)

DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def build_index_parser() -> argparse.ArgumentParser:
    """Create the vector indexing command parser."""
    parser = argparse.ArgumentParser(description="Embed chunks and index them in local Qdrant.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--model", help="Ollama model; defaults to ALLIANZ_EMBEDDING_MODEL.")
    parser.add_argument("--qdrant-path", type=Path, help="Defaults to ALLIANZ_QDRANT_PATH.")
    return parser


def build_search_parser() -> argparse.ArgumentParser:
    """Create the semantic search command parser."""
    parser = argparse.ArgumentParser(description="Search the locally indexed manual.")
    parser.add_argument("query", help="Question or description to search for.")
    parser.add_argument("--model", help="Must match the model used during indexing.")
    parser.add_argument("--qdrant-path", type=Path, help="Defaults to ALLIANZ_QDRANT_PATH.")
    parser.add_argument("--top-k", type=_positive_int, help="Defaults to ALLIANZ_RETRIEVAL_TOP_K.")
    return parser


def index_main() -> int:
    """Read, embed, and persist all processed chunks."""
    args = build_index_parser().parse_args()
    settings = Settings()
    model_name = args.model or settings.embedding_model
    qdrant_path = args.qdrant_path or settings.qdrant_path
    collection_name = collection_name_for_model(settings.qdrant_collection_prefix, model_name)

    try:
        chunks = read_chunks_jsonl(args.input)
        store = QdrantVectorStore(collection_name=collection_name, path=qdrant_path)
        with _build_provider(settings, model_name) as provider:
            summary = IndexingService(
                embedding_provider=provider,
                vector_store=store,
                batch_size=settings.embedding_batch_size,
                on_progress=_report_indexing_progress,
            ).index(chunks)
    except (ApplicationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "model": summary.model_name,
                "collection": collection_name,
                "indexed_chunks": summary.indexed_chunks,
                "vector_size": summary.vector_size,
                "qdrant_path": str(qdrant_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


def search_main() -> int:
    """Embed one query and print its closest source chunks."""
    args = build_search_parser().parse_args()
    settings = Settings()
    model_name = args.model or settings.embedding_model
    qdrant_path = args.qdrant_path or settings.qdrant_path
    top_k = args.top_k if args.top_k is not None else settings.retrieval_top_k
    collection_name = collection_name_for_model(settings.qdrant_collection_prefix, model_name)

    try:
        store = QdrantVectorStore(collection_name=collection_name, path=qdrant_path)
        with _build_provider(settings, model_name) as provider:
            results = SemanticRetriever(provider, store).retrieve(args.query, top_k)
    except (ApplicationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        json.dumps(
            [result.model_dump(mode="json", exclude_none=True) for result in results],
            ensure_ascii=False,
        )
    )
    return 0


def _build_provider(settings: Settings, model_name: str) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        base_url=str(settings.ollama_base_url),
        model_name=model_name,
        timeout_seconds=settings.ollama_timeout_seconds,
    )


def _report_indexing_progress(progress: IndexingProgress) -> None:
    print(
        f"Indexed {progress.indexed_chunks}/{progress.total_chunks} chunks",
        file=sys.stderr,
        flush=True,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed
