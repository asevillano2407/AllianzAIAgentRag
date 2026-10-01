"""Tests for inspectable chunk serialization."""

import json
from pathlib import Path

import pytest

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.errors import DocumentProcessingError
from allianz_claims_rag_agent.ingestion.serialization import read_chunks_jsonl, write_chunks_jsonl


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


def test_read_chunks_jsonl_restores_validated_chunks(tmp_path: Path) -> None:
    output_path = tmp_path / "chunks.jsonl"
    chunks = (
        SourceChunk(chunk_id="chunk-1", text="CIDE", source="manual.pdf", page=7),
        SourceChunk(chunk_id="chunk-2", text="ASCIDE", source="manual.pdf", page=8),
    )
    write_chunks_jsonl(chunks, output_path)

    assert read_chunks_jsonl(output_path) == chunks


def test_read_chunks_jsonl_reports_invalid_line_number(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    input_path.write_text(
        '{"chunk_id":"one","text":"CIDE","source":"manual.pdf","page":1}\ninvalid\n',
        encoding="utf-8",
    )

    with pytest.raises(DocumentProcessingError, match=r"chunks\.jsonl:2"):
        read_chunks_jsonl(input_path)


def test_read_chunks_jsonl_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DocumentProcessingError, match="does not exist"):
        read_chunks_jsonl(tmp_path / "missing.jsonl")


def test_read_chunks_jsonl_rejects_empty_file(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    input_path.write_text("\n", encoding="utf-8")

    with pytest.raises(DocumentProcessingError, match="is empty"):
        read_chunks_jsonl(input_path)
