"""Serialization of processed chunks for inspection and later indexing."""

import json
from pathlib import Path

from pydantic import ValidationError

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.errors import DocumentProcessingError


def read_chunks_jsonl(input_path: Path) -> tuple[SourceChunk, ...]:
    """Read and validate source chunks from a UTF-8 JSON Lines file."""
    if not input_path.is_file():
        raise DocumentProcessingError(f"Chunk file does not exist: {input_path}")

    chunks: list[SourceChunk] = []
    try:
        with input_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    chunks.append(SourceChunk.model_validate_json(line))
                except (ValueError, ValidationError) as exc:
                    raise DocumentProcessingError(
                        f"Invalid chunk at {input_path}:{line_number}"
                    ) from exc
    except OSError as exc:
        raise DocumentProcessingError(f"Cannot read chunk file: {input_path}") from exc

    if not chunks:
        raise DocumentProcessingError(f"Chunk file is empty: {input_path}")
    return tuple(chunks)


def write_chunks_jsonl(chunks: tuple[SourceChunk, ...], output_path: Path) -> None:
    """Write chunks atomically as UTF-8 JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            payload = chunk.model_dump(mode="json", exclude_none=True)
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    temporary_path.replace(output_path)
