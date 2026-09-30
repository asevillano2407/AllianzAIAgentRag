"""Page-preserving text extraction from PDF documents."""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from allianz_claims_rag_agent.domain import DocumentPage
from allianz_claims_rag_agent.errors import DocumentProcessingError


class PdfTextExtractor:
    """Extract text without mixing content from different physical pages."""

    def extract(self, document_path: Path) -> list[DocumentPage]:
        """Return every PDF page, including pages without extractable text.

        Parameters
        ----------
        document_path : Path
            The file system path pointing to the target PDF document. Supports
            user directory shortcuts (e.g., '~' notation).

        Returns
        -------
        list[DocumentPage]
            A chronological list of extracted page models, where each item contains
            the file source name, the 1-based page number, and its extracted text.
        """
        path = document_path.expanduser()
        if not path.is_file():
            raise DocumentProcessingError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise DocumentProcessingError(f"Expected a PDF file: {path}")

        try:
            reader = PdfReader(path)
        except (OSError, PdfReadError) as exc:
            raise DocumentProcessingError(f"Unable to read PDF: {path}") from exc

        if reader.is_encrypted:
            raise DocumentProcessingError(f"Encrypted PDFs are not supported: {path}")

        pages: list[DocumentPage] = []
        for page_number, pdf_page in enumerate(reader.pages, start=1):
            try:
                text = pdf_page.extract_text() or ""
            except Exception as exc:  # pypdf backends can expose different exceptions
                msg = f"Unable to extract page {page_number} from {path.name}"
                raise DocumentProcessingError(msg) from exc
            pages.append(DocumentPage(source=path.name, page=page_number, text=text))

        if not pages:
            raise DocumentProcessingError(f"PDF contains no pages: {path}")
        return pages
