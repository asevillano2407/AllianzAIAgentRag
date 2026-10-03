"""Manual end-to-end retrieval and structured-generation command."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from allianz_claims_rag_agent.config import Settings
from allianz_claims_rag_agent.domain import QueryType
from allianz_claims_rag_agent.embeddings import OllamaEmbeddingProvider
from allianz_claims_rag_agent.errors import ApplicationError
from allianz_claims_rag_agent.generation.ollama import OllamaStructuredLlm
from allianz_claims_rag_agent.generation.service import AnswerGenerator
from allianz_claims_rag_agent.retrieval.factory import build_evidence_retriever
from allianz_claims_rag_agent.retrieval.naming import collection_name_for_model
from allianz_claims_rag_agent.retrieval.qdrant_store import QdrantVectorStore
from allianz_claims_rag_agent.retrieval.query_expansion import build_retrieval_queries


def build_parser() -> argparse.ArgumentParser:
    """Create the manual grounded-generation parser."""
    parser = argparse.ArgumentParser(
        description="Retrieve manual evidence and generate one structured local answer."
    )
    parser.add_argument("query", help="Question or accident description in Spanish.")
    parser.add_argument(
        "--query-type",
        choices=[query_type.value for query_type in QueryType],
        default=QueryType.MANUAL_QUESTION.value,
        help="Passed explicitly until the deterministic router is added.",
    )
    parser.add_argument("--embedding-model", help="Defaults to ALLIANZ_EMBEDDING_MODEL.")
    parser.add_argument("--llm-model", help="Defaults to ALLIANZ_LLM_MODEL.")
    parser.add_argument("--qdrant-path", type=Path, help="Defaults to ALLIANZ_QDRANT_PATH.")
    parser.add_argument("--top-k", type=_positive_int, help="Defaults to ALLIANZ_RETRIEVAL_TOP_K.")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable local cross-encoder reranking.",
    )
    parser.add_argument("--reranker-model", help="Defaults to ALLIANZ_RERANKER_MODEL.")
    parser.add_argument(
        "--candidate-k",
        type=_positive_int,
        help="Defaults to configured candidate-k.",
    )
    return parser


def main() -> int:
    """Retrieve evidence, generate JSON, validate it, and print the result."""
    args = build_parser().parse_args()
    settings = Settings()
    embedding_model = args.embedding_model or settings.embedding_model
    llm_model = args.llm_model or settings.llm_model
    qdrant_path = args.qdrant_path or settings.qdrant_path
    top_k = args.top_k if args.top_k is not None else settings.retrieval_top_k
    candidate_k = (
        args.candidate_k
        if args.candidate_k is not None
        else settings.retrieval_candidate_k
    )
    query_type = QueryType(args.query_type)
    retrieval_queries = build_retrieval_queries(args.query, query_type)
    collection_name = collection_name_for_model(
        settings.qdrant_collection_prefix,
        embedding_model,
    )

    try:
        store = QdrantVectorStore(collection_name=collection_name, path=qdrant_path)
        with OllamaEmbeddingProvider(
            base_url=str(settings.ollama_base_url),
            model_name=embedding_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        ) as embedding_provider:
            retriever = build_evidence_retriever(
                embedding_provider,
                store,
                reranker_model=(
                    args.reranker_model or settings.reranker_model
                    if args.rerank
                    else None
                ),
                candidate_k=candidate_k,
                reranker_batch_size=settings.reranker_batch_size,
            )
            chunks = retriever.retrieve_many(retrieval_queries, top_k)

        with OllamaStructuredLlm(
            base_url=str(settings.ollama_base_url),
            model_name=llm_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        ) as llm:
            generated = AnswerGenerator(llm).generate(args.query, query_type, chunks)
    except (ApplicationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "response": generated.response.model_dump(mode="json"),
                "generation": {
                    key: value for key, value in asdict(generated).items() if key != "response"
                },
                "retrieval_queries": retrieval_queries,
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed
