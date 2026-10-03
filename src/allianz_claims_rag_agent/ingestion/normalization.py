"""Conservative normalization for extracted Spanish text."""

import re
import unicodedata

from allianz_claims_rag_agent.domain import DocumentPage

_BLANK_LINES = re.compile(r"\n[ \t]*\n+")
_HORIZONTAL_SPACE = re.compile(r"[ \t]+")


def normalize_page(page: DocumentPage) -> DocumentPage:
    """Normalize extraction artifacts while retaining paragraph boundaries."""
    # Unicode puede representar una letra acentuada de varias formas internas.
    # NFC las transforma a una representación consistente y previene fallos
    # sutiles durante la búsqueda de texto.
    text = unicodedata.normalize("NFC", page.text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Normalizamos espacios no separables y retiramos guiones blandos invisibles.
    text = text.replace("\u00a0", " ").replace("\u00ad", "")

    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    lines = _remove_printed_page_number(lines, page.page)

    paragraphs: list[str] = []
    paragraph_lines: list[str] = []
    for line in lines:
        if line:
            paragraph_lines.append(line)
        elif paragraph_lines:
            paragraphs.append(" ".join(paragraph_lines))
            paragraph_lines = []
    if paragraph_lines:
        paragraphs.append(" ".join(paragraph_lines))

    normalized = "\n\n".join(paragraphs).strip()
    normalized = _BLANK_LINES.sub("\n\n", normalized)
    return page.model_copy(update={"text": normalized})


def _remove_printed_page_number(lines: list[str], pdf_page: int) -> list[str]:
    """Remove the manual's printed page number when it is the first text line."""
    first_content = next((index for index, line in enumerate(lines) if line), None)
    if first_content is None:
        return lines

    expected_printed_page = str(pdf_page + 1)
    if lines[first_content] == expected_printed_page:
        return lines[:first_content] + lines[first_content + 1 :]
    return lines
