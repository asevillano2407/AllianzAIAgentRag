"""Tests for ingestion failure modes."""

from pathlib import Path

import pytest

from allianz_rag.ingestion import extract_pdf_pages


def test_missing_pdf_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="Manual not found"):
        extract_pdf_pages(path)
