"""Tests for ingestion orchestration."""

from pathlib import Path

from allianz_claims_rag_agent.domain import DocumentPage
from allianz_claims_rag_agent.ingestion.pipeline import IngestionPipeline


class StubExtractor:
    """Return controlled pages without reading an external document."""

    def extract(self, document_path: Path) -> list[DocumentPage]:
        return [
            DocumentPage(source=document_path.name, page=1, text=" 2\n1. Inicio\n\nTexto"),
            DocumentPage(source=document_path.name, page=2, text=" 3\n "),
        ]


def test_pipeline_reports_pages_without_text() -> None:
    pipeline = IngestionPipeline(
        chunk_size=200,
        chunk_overlap=20,
        extractor=StubExtractor(),
    )

    result = pipeline.run(Path("manual.pdf"))

    assert len(result.pages) == 2
    assert len(result.chunks) == 1
    assert result.pages_without_text == (2,)
    assert result.pages_without_chunks == (2,)
    assert result.chunks[0].page == 1
