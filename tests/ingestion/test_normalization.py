"""Tests for conservative page text normalization."""

from allianz_claims_rag_agent.domain import DocumentPage
from allianz_claims_rag_agent.ingestion.normalization import normalize_page


def test_normalization_removes_matching_printed_page_number() -> None:
    page = DocumentPage(
        source="manual.pdf",
        page=8,
        text=" 9\n2. Adelantamientos\n\nTexto de la sección.\n",
    )

    normalized = normalize_page(page)

    assert normalized.text == "2. Adelantamientos\n\nTexto de la sección."


def test_normalization_keeps_number_that_is_not_page_header() -> None:
    page = DocumentPage(source="manual.pdf", page=8, text="8\n\nContenido")

    normalized = normalize_page(page)

    assert normalized.text == "8\n\nContenido"


def test_normalization_collapses_spacing_but_preserves_paragraphs() -> None:
    page = DocumentPage(
        source="manual.pdf",
        page=1,
        text="Primera   línea\r\nsegunda línea\r\n\r\nOtro\u00a0párrafo.\u00ad",
    )

    normalized = normalize_page(page)

    assert normalized.text == "Primera línea segunda línea\n\nOtro párrafo."
