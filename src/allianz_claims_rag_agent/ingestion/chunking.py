"""Deterministic, page-aware text chunking."""

import hashlib
import re
from pathlib import Path

from allianz_claims_rag_agent.domain import DocumentPage, SourceChunk

_SECTION_HEADING = re.compile(r"^(?P<number>\d{1,2})\.\s+(?P<title>[^\n]+)", re.MULTILINE)


class PageChunker:
    """Split normalized pages without ever joining different page numbers."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap > chunk_size // 2:
            raise ValueError("chunk_overlap cannot exceed half of chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, pages: list[DocumentPage]) -> list[SourceChunk]:
        """Create deterministic chunks and carry section context across pages."""
        chunks: list[SourceChunk] = []
        current_section: str | None = None

        for page in pages:
            match = _SECTION_HEADING.search(page.text)
            if match:
                current_section = f"{match.group('number')}. {match.group('title')}"
            if not page.text:
                continue
            if match and not page.text[match.end() :].strip():
                continue

            for chunk_index, text in enumerate(self._split(page.text), start=1):
                chunks.append(
                    SourceChunk(
                        chunk_id=self._chunk_id(page, chunk_index, text),
                        text=text,
                        source=page.source,
                        page=page.page,
                        section=current_section,
                    )
                )
        return chunks

    def _split(self, text: str) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            hard_end = min(start + self._chunk_size, len(text))
            end = self._find_boundary(text, start, hard_end)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = self._next_start(text, start, end)
        return chunks

    def _find_boundary(self, text: str, start: int, hard_end: int) -> int:
        if hard_end >= len(text):
            return len(text)

        minimum_end = start + self._chunk_size // 2
        for separator in ("\n\n", ". ", "; ", " "):
            boundary = text.rfind(separator, minimum_end, hard_end)
            if boundary != -1:
                return boundary + len(separator)
        return hard_end

    def _next_start(self, text: str, previous_start: int, end: int) -> int:
        candidate = max(end - self._chunk_overlap, previous_start + 1)
        next_space = text.find(" ", candidate, end)
        if next_space != -1:
            candidate = next_space + 1
        return min(candidate, end)

    @staticmethod
    def _chunk_id(page: DocumentPage, chunk_index: int, text: str) -> str:
        identity = f"{page.source}\0{page.page}\0{chunk_index}\0{text}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        filename_stem = Path(page.source).stem
        source_stem = re.sub(r"[^a-z0-9]+", "-", filename_stem.lower()).strip("-")
        return f"{source_stem}-p{page.page:03d}-c{chunk_index:03d}-{digest}"
