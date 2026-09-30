"""Tests for inspectable chunk serialization."""

import json
from pathlib import Path

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.ingestion.serialization import write_chunks_jsonl


def test_write_chunks_jsonl_preserves_unicode_and_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "chunks.jsonl"
    chunks = (
        SourceChunk(
            chunk_id="manual-p001-c001-a1b2",
            text="Colisión en una rotonda.",
            source="manual.pdf",
            page=1,
            section="1. Ámbito",
        ),
    )

    write_chunks_jsonl(chunks, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["text"] == "Colisión en una rotonda."
    assert payload["page"] == 1
    assert not output_path.with_suffix(".jsonl.tmp").exists()
