"""Tests for page-preserving PDF extraction."""

from pathlib import Path

import pytest
from pypdf import PdfWriter

from allianz_claims_rag_agent.errors import DocumentProcessingError
from allianz_claims_rag_agent.ingestion.pdf import PdfTextExtractor


def test_pdf_extractor_preserves_blank_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank-pages.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.add_blank_page(width=595, height=842)
    with pdf_path.open("wb") as stream:
        writer.write(stream)

    pages = PdfTextExtractor().extract(pdf_path)

    assert [page.page for page in pages] == [1, 2]
    assert [page.text for page in pages] == ["", ""]
    assert all(page.source == "blank-pages.pdf" for page in pages)


def test_pdf_extractor_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DocumentProcessingError, match="PDF not found"):
        PdfTextExtractor().extract(tmp_path / "missing.pdf")


def test_pdf_extractor_rejects_non_pdf_file(tmp_path: Path) -> None:
    text_path = tmp_path / "manual.txt"
    text_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(DocumentProcessingError, match="Expected a PDF"):
        PdfTextExtractor().extract(text_path)
