"""End-to-end CLI for the automatically routed LangGraph workflow."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from allianz_claims_rag_agent.config import Settings
from allianz_claims_rag_agent.embeddings import OllamaEmbeddingProvider
from allianz_claims_rag_agent.errors import ApplicationError
from allianz_claims_rag_agent.generation import AnswerGenerator, OllamaStructuredLlm
from allianz_claims_rag_agent.orchestration import (
    AgentDependencies,
    DeterministicQueryRouter,
    build_claims_graph,
)
from allianz_claims_rag_agent.retrieval import (
    QdrantVectorStore,
    build_evidence_retriever,
    collection_name_for_model,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the agent CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the bounded local claims RAG agent with automatic routing."
    )
    parser.add_argument("query", help="Manual question or accident description in Spanish.")
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
    """Build local dependencies, execute the graph, and print observable JSON."""
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
    collection_name = collection_name_for_model(
        settings.qdrant_collection_prefix,
        embedding_model,
    )

    try:
        store = QdrantVectorStore(collection_name=collection_name, path=qdrant_path)
        with (
            OllamaEmbeddingProvider(
                base_url=str(settings.ollama_base_url),
                model_name=embedding_model,
                timeout_seconds=settings.ollama_timeout_seconds,
            ) as embedding_provider,
            OllamaStructuredLlm(
                base_url=str(settings.ollama_base_url),
                model_name=llm_model,
                timeout_seconds=settings.ollama_timeout_seconds,
            ) as llm,
        ):
            graph = build_claims_graph(
                AgentDependencies(
                    retriever=build_evidence_retriever(
                        embedding_provider,
                        store,
                        reranker_model=(
                            args.reranker_model or settings.reranker_model
                            if args.rerank
                            else None
                        ),
                        candidate_k=candidate_k,
                        reranker_batch_size=settings.reranker_batch_size,
                    ),
                    generator=AnswerGenerator(llm),
                    router=DeterministicQueryRouter(),
                    top_k=top_k,
                    max_retries=settings.max_agent_retries,
                )
            )
            result = graph.invoke({"query": args.query})
    except (ApplicationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    generated = result.get("generated")
    print(
        json.dumps(
            {
                "response": result["response"].model_dump(mode="json"),
                "agent": {
                    "query_type": result["query_type"].value,
                    "status": result["status"],
                    "fallback_used": result["fallback_used"],
                    "retry_count": result["retry_count"],
                    "execution_path": result["execution_path"],
                    "last_error": result.get("error"),
                },
                "generation": (
                    {
                        key: value
                        for key, value in asdict(generated).items()
                        if key != "response"
                    }
                    if generated is not None
                    else None
                ),
                "retrieval_queries": result["retrieval_queries"],
                "retrieved_chunk_ids": [
                    chunk.chunk_id for chunk in result.get("chunks", [])
                ],
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
