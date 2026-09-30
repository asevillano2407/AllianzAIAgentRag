"""Serialization of processed chunks for inspection and later indexing."""

import json
from pathlib import Path

from allianz_claims_rag_agent.domain import SourceChunk


def write_chunks_jsonl(chunks: tuple[SourceChunk, ...], output_path: Path) -> None:
    """Write chunks atomically as UTF-8 JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            payload = chunk.model_dump(mode="json", exclude_none=True)
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    temporary_path.replace(output_path)
