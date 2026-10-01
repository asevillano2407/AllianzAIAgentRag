"""Command-line retrieval evaluation against a labelled local dataset."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from allianz_claims_rag_agent.config import Settings
from allianz_claims_rag_agent.embeddings import OllamaEmbeddingProvider
from allianz_claims_rag_agent.errors import ApplicationError
from allianz_claims_rag_agent.evaluation.retrieval import evaluate_rankings, read_retrieval_cases
from allianz_claims_rag_agent.retrieval.naming import collection_name_for_model
from allianz_claims_rag_agent.retrieval.qdrant_store import QdrantVectorStore
from allianz_claims_rag_agent.retrieval.services import SemanticRetriever

DEFAULT_DATASET_PATH = Path("evaluation/datasets/retrieval_cases.jsonl")


def build_parser() -> argparse.ArgumentParser:
    """Create the retrieval evaluation parser."""
    parser = argparse.ArgumentParser(description="Evaluate one indexed embedding model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", help="Must match the indexed Ollama embedding model.")
    parser.add_argument("--qdrant-path", type=Path, help="Defaults to ALLIANZ_QDRANT_PATH.")
    parser.add_argument("--top-k", type=int, help="Defaults to ALLIANZ_RETRIEVAL_TOP_K.")
    return parser


def main() -> int:
    """Run every labelled query and print aggregate retrieval metrics."""
    args = build_parser().parse_args()
    settings = Settings()
    model_name = args.model or settings.embedding_model
    qdrant_path = args.qdrant_path or settings.qdrant_path
    top_k = args.top_k if args.top_k is not None else settings.retrieval_top_k
    collection_name = collection_name_for_model(settings.qdrant_collection_prefix, model_name)

    if top_k < 1:
        print("top-k must be at least 1", file=sys.stderr)
        return 1

    try:
        cases = read_retrieval_cases(args.dataset)
        store = QdrantVectorStore(collection_name=collection_name, path=qdrant_path)
        with OllamaEmbeddingProvider(
            base_url=str(settings.ollama_base_url),
            model_name=model_name,
            timeout_seconds=settings.ollama_timeout_seconds,
        ) as provider:
            retriever = SemanticRetriever(provider, store)
            rankings = {case.case_id: retriever.retrieve(case.query, top_k) for case in cases}
        metrics = evaluate_rankings(cases, rankings, top_k)
    except (ApplicationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps({"model": model_name, **asdict(metrics)}, ensure_ascii=False))
    return 0
