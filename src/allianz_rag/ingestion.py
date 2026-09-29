"""PDF ingestion pipeline."""

from pathlib import Path

from pypdf import PdfReader

from allianz_rag.models import DocumentChunk
from allianz_rag.text import chunk_pages
from allianz_rag.vector_store import VectorStore


def extract_pdf_pages(path: Path) -> list[str]:
    """Extract all PDF pages and fail clearly on unreadable or empty documents."""

    if not path.is_file():
        raise FileNotFoundError(f"Manual not found: {path}")
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported")
    pages = [page.extract_text() or "" for page in reader.pages]
    if not any(page.strip() for page in pages):
        raise ValueError("The PDF has no extractable text; OCR is required")
    return pages


def build_chunks(path: Path) -> list[DocumentChunk]:
    """Extract and chunk a manual with stable identifiers."""

    pages = extract_pdf_pages(path)
    return chunk_pages(pages, source=path.name)


def ingest_manual(path: Path, vector_store: VectorStore) -> int:
    """Build and persist all chunks from the supplied manual."""

    chunks = build_chunks(path)
    if not chunks:
        raise ValueError("No chunks were produced from the manual")
    return vector_store.upsert(chunks)
