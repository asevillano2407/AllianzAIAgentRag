"""Tests for deterministic text processing."""

import pytest

from allianz_rag.text import chunk_pages, clean_text


def test_clean_text_removes_page_numbers_and_damaged_glyphs() -> None:
    raw = "10\n3. Alcoholemia\nNo se considera la conducci�n una exclusi�n.\n"

    result = clean_text(raw)

    assert result == "3. Alcoholemia\nNo se considera la conduccin una exclusin."


def test_chunking_preserves_page_boundaries_and_is_stable() -> None:
    pages = ["1. Primera seccion\n" + "Texto de prueba. " * 40, "Segunda pagina"]

    first = chunk_pages(pages, source="manual.pdf", max_characters=220, overlap_characters=30)
    second = chunk_pages(pages, source="manual.pdf", max_characters=220, overlap_characters=30)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert {chunk.page for chunk in first} == {1, 2}
    assert all(chunk.source == "manual.pdf" for chunk in first)


@pytest.mark.parametrize(
    ("max_characters", "overlap"),
    [(100, 10), (300, -1), (300, 300)],
)
def test_chunking_rejects_invalid_sizes(max_characters: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_pages(
            ["text"],
            source="manual.pdf",
            max_characters=max_characters,
            overlap_characters=overlap,
        )
