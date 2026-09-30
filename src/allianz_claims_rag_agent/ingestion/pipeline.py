"""Orchestration for local document ingestion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from allianz_claims_rag_agent.domain import DocumentPage, SourceChunk
from allianz_claims_rag_agent.ingestion.chunking import PageChunker
from allianz_claims_rag_agent.ingestion.normalization import normalize_page
from allianz_claims_rag_agent.ingestion.pdf import PdfTextExtractor


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """In-memory result and quality signals from one ingestion run."""

    pages: tuple[DocumentPage, ...]
    chunks: tuple[SourceChunk, ...]
    pages_without_text: tuple[int, ...]
    pages_without_chunks: tuple[int, ...]


class PageExtractor(Protocol):
    """Contract for components that extract physical document pages."""

    def extract(self, document_path: Path) -> list[DocumentPage]:
        """Extract pages from a document path."""
        ...


class IngestionPipeline:
    """Extract, normalize, and chunk a PDF deterministically."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        extractor: PageExtractor | None = None,
    ) -> None:
        self._extractor = extractor or PdfTextExtractor()
        self._chunker = PageChunker(chunk_size, chunk_overlap)

    def run(self, document_path: Path) -> IngestionResult:
        """Process a PDF while keeping page provenance on every chunk."""
        extracted_pages = self._extractor.extract(document_path)
        normalized_pages = tuple(normalize_page(page) for page in extracted_pages)
        chunks = tuple(self._chunker.chunk(list(normalized_pages)))
        empty_pages = tuple(page.page for page in normalized_pages if not page.text)
        chunked_pages = {chunk.page for chunk in chunks}
        pages_without_chunks = tuple(
            page.page for page in normalized_pages if page.page not in chunked_pages
        )
        return IngestionResult(
            pages=normalized_pages,
            chunks=chunks,
            pages_without_text=empty_pages,
            pages_without_chunks=pages_without_chunks,
        )
