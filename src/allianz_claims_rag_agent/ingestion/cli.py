"""Command-line entry point for local PDF ingestion."""

import argparse
import json
import sys
from pathlib import Path

from allianz_claims_rag_agent.config import Settings
from allianz_claims_rag_agent.errors import ApplicationError
from allianz_claims_rag_agent.ingestion.pipeline import IngestionPipeline
from allianz_claims_rag_agent.ingestion.serialization import write_chunks_jsonl


def build_parser() -> argparse.ArgumentParser:
    """Create the ingestion command parser."""
    parser = argparse.ArgumentParser(description="Extract and chunk the configured PDF.")
    parser.add_argument("--input", type=Path, help="PDF path; defaults to ALLIANZ_MANUAL_PATH.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Destination JSONL file.",
    )
    return parser


def main() -> int:
    """Run ingestion and print a machine-readable summary."""
    args = build_parser().parse_args()
    settings = Settings()
    document_path = args.input or settings.manual_path
    pipeline = IngestionPipeline(settings.chunk_size, settings.chunk_overlap)
    try:
        result = pipeline.run(document_path)
    except ApplicationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    write_chunks_jsonl(result.chunks, args.output)

    summary = {
        "source": str(document_path),
        "pages": len(result.pages),
        "pages_without_text": list(result.pages_without_text),
        "pages_without_chunks": list(result.pages_without_chunks),
        "chunks": len(result.chunks),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
