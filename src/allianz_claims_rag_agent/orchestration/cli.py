"""End-to-end CLI for the automatically routed LangGraph workflow."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from allianz_claims_rag_agent.config import Settings
from allianz_claims_rag_agent.errors import ApplicationError
from allianz_claims_rag_agent.orchestration.runtime import run_local_agent


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
    try:
        result = run_local_agent(
            args.query,
            settings=settings,
            rerank=args.rerank,
            embedding_model=embedding_model,
            llm_model=llm_model,
            qdrant_path=qdrant_path,
            top_k=top_k,
            candidate_k=candidate_k,
            reranker_model=args.reranker_model,
        )
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
