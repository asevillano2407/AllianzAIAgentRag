"""Tests for deterministic page-aware chunking."""

import pytest

from allianz_claims_rag_agent.domain import DocumentPage
from allianz_claims_rag_agent.ingestion.chunking import PageChunker


def test_chunker_never_combines_pages_and_carries_section() -> None:
    pages = [
        DocumentPage(
            source="manual.pdf",
            page=10,
            text="3. Alcoholemia\n\n" + "Texto relevante. " * 15,
        ),
        DocumentPage(source="manual.pdf", page=11, text="Continuación de la regla."),
    ]
    chunker = PageChunker(chunk_size=140, chunk_overlap=20)

    chunks = chunker.chunk(pages)

    assert {chunk.page for chunk in chunks} == {10, 11}
    assert all(chunk.section == "3. Alcoholemia" for chunk in chunks)
    assert all(len(chunk.text) <= 140 for chunk in chunks)


def test_chunk_ids_are_deterministic_and_content_sensitive() -> None:
    chunker = PageChunker(chunk_size=200, chunk_overlap=20)
    original = [DocumentPage(source="manual.pdf", page=1, text="Contenido estable")]
    changed = [DocumentPage(source="manual.pdf", page=1, text="Contenido modificado")]

    first_id = chunker.chunk(original)[0].chunk_id
    repeated_id = chunker.chunk(original)[0].chunk_id
    changed_id = chunker.chunk(changed)[0].chunk_id

    assert first_id == repeated_id
    assert first_id != changed_id


def test_chunker_skips_empty_pages() -> None:
    chunks = PageChunker(200, 20).chunk(
        [DocumentPage(source="manual.pdf", page=32, text="")]
    )

    assert chunks == []


def test_chunker_skips_page_with_only_a_section_heading() -> None:
    chunks = PageChunker(200, 20).chunk(
        [DocumentPage(source="manual.pdf", page=31, text="15. D.A.A. (Continuación)")]
    )

    assert chunks == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(99, 10), (200, -1), (200, 101)],
)
def test_chunker_rejects_invalid_windows(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        PageChunker(chunk_size, chunk_overlap)
