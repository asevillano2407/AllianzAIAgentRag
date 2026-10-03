"""Reproducible sequential runner for the five interview demonstration cases."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from allianz_claims_rag_agent.config import Settings
from allianz_claims_rag_agent.embeddings import OllamaEmbeddingProvider
from allianz_claims_rag_agent.evaluation.cases import (
    DemoCase,
    evaluate_business_response,
    load_demo_cases,
    page_group_recall,
)
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

DEFAULT_CASES_PATH = Path("evaluation/datasets/demo_accident_cases.jsonl")


def build_parser() -> argparse.ArgumentParser:
    """Create the reproducible demo evaluation parser."""
    parser = argparse.ArgumentParser(description="Run all interview cases sequentially.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, help="Optional JSONL results file.")
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
    """Build the local graph once and emit one JSON object per evaluated case."""
    args = build_parser().parse_args()
    settings = Settings()
    cases = load_demo_cases(args.cases)
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
    store = QdrantVectorStore(collection_name=collection_name, path=qdrant_path)
    records: list[dict[str, object]] = []

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
        for case in cases:
            result = graph.invoke({"query": case.query})
            generated = result.get("generated")
            record = {
                "case_id": case.case_id,
                "query": case.query,
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
                "retrieval_evaluation": _retrieval_evaluation(
                    case,
                    [chunk.page for chunk in result.get("chunks", [])],
                ),
                "business_evaluation": evaluate_business_response(
                    case,
                    result["response"],
                ),
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _retrieval_evaluation(case: DemoCase, retrieved_pages: list[int]) -> dict[str, object]:
    return {
        "expected_page_groups": case.expected_page_groups,
        "retrieved_pages": retrieved_pages,
        "page_group_recall": page_group_recall(
            case.expected_page_groups,
            retrieved_pages,
        ),
    }
