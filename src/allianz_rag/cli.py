"""Command-line entry point for ingestion, querying and evaluation."""

import argparse
import json
from pathlib import Path

from allianz_rag.config import get_settings
from allianz_rag.evaluation import evaluate, load_dataset, write_result
from allianz_rag.ingestion import ingest_manual
from allianz_rag.service import build_service
from allianz_rag.vector_store import ChromaVectorStore


def _store() -> ChromaVectorStore:
    settings = get_settings()
    return ChromaVectorStore(
        settings.chroma_path,
        settings.chroma_collection,
        settings.embedding_model,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="allianz-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Index the configured PDF manual")

    ask_parser = subparsers.add_parser("ask", help="Run one grounded analysis")
    ask_parser.add_argument("query")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate retrieval")
    eval_parser.add_argument(
        "--dataset", type=Path, default=Path("evaluation/golden_dataset.jsonl")
    )
    eval_parser.add_argument(
        "--output", type=Path, default=Path("evaluation/results/latest.json")
    )
    eval_parser.add_argument("--include-generation", action="store_true")
    return parser


def main() -> None:
    """Execute the selected command."""

    args = build_parser().parse_args()
    settings = get_settings()
    store = _store()

    if args.command == "ingest":
        count = ingest_manual(settings.manual_path, store)
        print(json.dumps({"indexed_chunks": count, "collection_size": store.count()}))
        return

    if args.command == "ask":
        response = build_service(settings).analyze(args.query)
        print(response.model_dump_json(indent=2))
        return

    cases = load_dataset(args.dataset)
    service = build_service(settings) if args.include_generation else None
    result = evaluate(cases, store, top_k=settings.top_k, service=service)
    write_result(result, args.output)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
