"""Text cleaning and page-aware chunking."""

import hashlib
import re
import unicodedata
from collections.abc import Iterable

from allianz_rag.models import DocumentChunk

_WHITESPACE_RE = re.compile(r"[ \t]+")
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")
_SECTION_RE = re.compile(r"^\s*(\d{1,2})\.\s+(.{3,100}?)(?:\s*\(.*\))?\s*$", re.I)


def clean_text(text: str) -> str:
    """Normalize extracted PDF text without inventing missing characters."""

    normalized = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = _WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line or _PAGE_NUMBER_RE.fullmatch(line):
            continue
        # The legacy PDF maps some accented glyphs to U+FFFD. Removing only the
        # damaged glyph is deterministic and safer than guessing the source word.
        line = line.replace("\ufffd", "")
        lines.append(line)
    return "\n".join(lines)


def detect_section(text: str, fallback: str) -> str:
    """Find the first numbered manual heading in a page."""

    for line in text.splitlines()[:6]:
        match = _SECTION_RE.match(line)
        if match:
            return f"{match.group(1)}. {match.group(2).strip()}"
    return fallback


def _stable_chunk_id(source: str, page: int, index: int, text: str) -> str:
    value = f"{source}|{page}|{index}|{text}".encode()
    return hashlib.sha256(value).hexdigest()[:20]


def chunk_pages(
    pages: Iterable[str],
    *,
    source: str,
    max_characters: int = 1400,
    overlap_characters: int = 220,
) -> list[DocumentChunk]:
    """Split page text into deterministic chunks that never cross page boundaries."""

    if max_characters < 200:
        raise ValueError("max_characters must be at least 200")
    if overlap_characters < 0 or overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be between 0 and max_characters")

    chunks: list[DocumentChunk] = []
    current_section = "Introduccion"
    for page_number, raw_text in enumerate(pages, start=1):
        text = clean_text(raw_text)
        if not text:
            continue
        current_section = detect_section(text, current_section)
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + max_characters, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
                if boundary > start + max_characters // 2:
                    end = boundary + 1
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=_stable_chunk_id(
                            source, page_number, chunk_index, chunk_text
                        ),
                        source=source,
                        page=page_number,
                        section=current_section,
                        text=chunk_text,
                    )
                )
                chunk_index += 1
            if end >= len(text):
                break
            start = max(end - overlap_characters, start + 1)
    return chunks
