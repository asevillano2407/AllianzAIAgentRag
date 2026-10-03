"""Fair local LLM benchmark using one frozen retrieval context per case."""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from allianz_claims_rag_agent.config import Settings
from allianz_claims_rag_agent.domain import QueryType, SourceChunk
from allianz_claims_rag_agent.embeddings import OllamaEmbeddingProvider
from allianz_claims_rag_agent.errors import GenerationError, OutputValidationError
from allianz_claims_rag_agent.evaluation.cases import (
    DemoCase,
    evaluate_business_response,
    load_demo_cases,
    page_group_recall,
)
from allianz_claims_rag_agent.generation import AnswerGenerator, OllamaStructuredLlm
from allianz_claims_rag_agent.orchestration import DeterministicQueryRouter
from allianz_claims_rag_agent.retrieval import (
    CandidateRetriever,
    QdrantVectorStore,
    build_evidence_retriever,
    collection_name_for_model,
)
from allianz_claims_rag_agent.retrieval.query_expansion import build_retrieval_queries

DEFAULT_CASES_PATH = Path("evaluation/datasets/demo_accident_cases.jsonl")
DEFAULT_MODELS = ("llama3.2:3b", "qwen3:4b", "qwen3:8b")


@dataclass(frozen=True, slots=True)
class PreparedCase:
    """A case whose routing and retrieval evidence are frozen for every LLM."""

    case: DemoCase
    query_type: QueryType
    retrieval_queries: list[str]
    chunks: list[SourceChunk]
    reranker_model: str | None = None


def prepare_cases(
    cases: list[DemoCase],
    retriever: CandidateRetriever,
    router: DeterministicQueryRouter,
    top_k: int,
    reranker_model: str | None = None,
) -> list[PreparedCase]:
    """Run deterministic routing and retrieval exactly once per evaluation case."""
    prepared: list[PreparedCase] = []
    for case in cases:
        query_type = router.route(case.query)
        retrieval_queries = build_retrieval_queries(case.query, query_type)
        prepared.append(
            PreparedCase(
                case=case,
                query_type=query_type,
                retrieval_queries=retrieval_queries,
                chunks=retriever.retrieve_many(retrieval_queries, top_k),
                reranker_model=reranker_model,
            )
        )
    return prepared


def evaluate_model(
    model_name: str,
    prepared_cases: list[PreparedCase],
    settings: Settings,
    on_record: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    """Evaluate first-pass structured generation for one model on frozen evidence."""
    records: list[dict[str, object]] = []
    with OllamaStructuredLlm(
        base_url=str(settings.ollama_base_url),
        model_name=model_name,
        timeout_seconds=settings.ollama_timeout_seconds,
    ) as llm:
        generator = AnswerGenerator(llm)
        for prepared in prepared_cases:
            try:
                generated = generator.generate(
                    prepared.case.query,
                    prepared.query_type,
                    prepared.chunks,
                )
            except (GenerationError, OutputValidationError) as exc:
                record = _failed_record(model_name, prepared, exc)
                records.append(record)
                if on_record is not None:
                    on_record(record)
                continue
            record = {
                **_base_record(model_name, prepared),
                "status": "completed",
                "error": None,
                "response": generated.response.model_dump(mode="json"),
                "business_evaluation": evaluate_business_response(
                    prepared.case,
                    generated.response,
                ),
                "generation": {
                    key: value
                    for key, value in asdict(generated).items()
                    if key != "response"
                },
            }
            records.append(record)
            if on_record is not None:
                on_record(record)
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate technical reliability and latency by model."""
    model_names = list(dict.fromkeys(str(record["model_name"]) for record in records))
    summaries: list[dict[str, object]] = []
    for model_name in model_names:
        model_records = [record for record in records if record["model_name"] == model_name]
        completed = [record for record in model_records if record["status"] == "completed"]
        latencies = [
            generation["total_duration_ms"]
            for record in completed
            if isinstance((generation := record.get("generation")), dict)
            and isinstance(generation.get("total_duration_ms"), int | float)
        ]
        business_correct = sum(
            bool(evaluation.get("business_correct"))
            for record in model_records
            if isinstance((evaluation := record.get("business_evaluation")), dict)
        )
        summaries.append(
            {
                "model_name": model_name,
                "cases": len(model_records),
                "completed": len(completed),
                "technical_success_rate": (
                    len(completed) / len(model_records) if model_records else 0.0
                ),
                "business_accuracy": (
                    business_correct / len(model_records) if model_records else 0.0
                ),
                "average_generation_duration_ms": mean(latencies) if latencies else None,
            }
        )
    return summaries


def build_parser() -> argparse.ArgumentParser:
    """Create the fair multi-model benchmark parser."""
    parser = argparse.ArgumentParser(
        description="Compare local LLMs using identical retrieved evidence."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--case-id",
        dest="case_ids",
        action="append",
        help="Evaluate only this case id; repeat the option to select several cases.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Ollama model; repeat the option. Defaults to three local candidates.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSONL result file.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Prepare and score frozen evidence without calling an LLM.",
    )
    parser.add_argument("--embedding-model", help="Defaults to ALLIANZ_EMBEDDING_MODEL.")
    parser.add_argument("--qdrant-path", type=Path, help="Defaults to ALLIANZ_QDRANT_PATH.")
    parser.add_argument("--top-k", type=_positive_int, help="Defaults to configured top-k.")
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
    """Retrieve once, evaluate each selected LLM, and emit detailed JSONL records."""
    args = build_parser().parse_args()
    settings = Settings()
    cases = load_demo_cases(args.cases)
    if args.case_ids:
        requested_case_ids = set(args.case_ids)
        available_case_ids = {case.case_id for case in cases}
        unknown_case_ids = requested_case_ids - available_case_ids
        if unknown_case_ids:
            raise ValueError(
                "Unknown case ids: " + ", ".join(sorted(unknown_case_ids))
            )
        cases = [case for case in cases if case.case_id in requested_case_ids]
    models = list(dict.fromkeys(args.models or DEFAULT_MODELS))
    embedding_model = args.embedding_model or settings.embedding_model
    qdrant_path = args.qdrant_path or settings.qdrant_path
    top_k = args.top_k if args.top_k is not None else settings.retrieval_top_k
    candidate_k = (
        args.candidate_k
        if args.candidate_k is not None
        else settings.retrieval_candidate_k
    )
    reranker_model = (
        args.reranker_model or settings.reranker_model if args.rerank else None
    )
    store = QdrantVectorStore(
        collection_name=collection_name_for_model(
            settings.qdrant_collection_prefix,
            embedding_model,
        ),
        path=qdrant_path,
    )

    with OllamaEmbeddingProvider(
        base_url=str(settings.ollama_base_url),
        model_name=embedding_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    ) as embedding_provider:
        prepared_cases = prepare_cases(
            cases,
            build_evidence_retriever(
                embedding_provider,
                store,
                reranker_model=reranker_model,
                candidate_k=candidate_k,
                reranker_batch_size=settings.reranker_batch_size,
            ),
            DeterministicQueryRouter(),
            top_k,
            reranker_model=reranker_model,
        )

    if args.retrieval_only:
        records = [_prepared_record(prepared) for prepared in prepared_cases]
        _emit_and_optionally_write(records, args.output)
        return 0

    records: list[dict[str, object]] = []
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")

    def persist_record(record: dict[str, object]) -> None:
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.output is not None:
            with args.output.open("a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    for model_name in models:
        model_records = evaluate_model(
            model_name,
            prepared_cases,
            settings,
            on_record=persist_record,
        )
        records.extend(model_records)

    for summary in summarize(records):
        print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)

    return 0


def _base_record(model_name: str, prepared: PreparedCase) -> dict[str, object]:
    retrieved_pages = [chunk.page for chunk in prepared.chunks]
    return {
        "model_name": model_name,
        "case_id": prepared.case.case_id,
        "query_type": prepared.query_type.value,
        "retrieval_queries": prepared.retrieval_queries,
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in prepared.chunks],
        "retrieved_pages": retrieved_pages,
        "expected_page_groups": prepared.case.expected_page_groups,
        "retrieval_page_group_recall": page_group_recall(
            prepared.case.expected_page_groups,
            retrieved_pages,
        ),
        "reranker_model": prepared.reranker_model,
    }


def _failed_record(
    model_name: str,
    prepared: PreparedCase,
    error: Exception,
) -> dict[str, object]:
    return {
        **_base_record(model_name, prepared),
        "status": "failed",
        "error": str(error),
        "response": None,
        "business_evaluation": evaluate_business_response(prepared.case, None),
        "generation": None,
    }


def _prepared_record(prepared: PreparedCase) -> dict[str, object]:
    return {
        **_base_record("frozen-retrieval", prepared),
        "query": prepared.case.query,
    }


def _emit_and_optionally_write(
    records: list[dict[str, object]],
    output: Path | None,
) -> None:
    for record in records:
        print(json.dumps(record, ensure_ascii=False), flush=True)
    _write_records(records, output)


def _write_records(records: list[dict[str, object]], output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed
