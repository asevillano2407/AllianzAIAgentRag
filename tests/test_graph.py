"""Tests for routing, validation and safe fallback behavior."""

from allianz_rag.graph import (
    classify_route,
    insufficient_evidence_analysis,
    validate_citations,
)
from allianz_rag.models import Citation, ClaimAnalysis, RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        source="manual.pdf",
        page=9,
        section="3. Alcoholemia",
        text="La alcoholemia no es motivo de exclusion.",
        distance=0.2,
    )


def test_route_is_deterministic() -> None:
    assert classify_route("Describe el accidente entre vehiculo A y B") == "accident_case"
    assert classify_route("¿Que dice el manual sobre alcoholemia?") == "question"


def test_valid_citation_must_match_retrieved_metadata() -> None:
    answer = ClaimAnalysis(
        answer="Es aplicable.",
        citations=[
            Citation(source="manual.pdf", page=9, chunk_id="chunk-1", quote="texto")
        ],
    )

    valid, feedback = validate_citations(answer, [_chunk()])

    assert valid is True
    assert feedback == ""


def test_hallucinated_citation_is_rejected() -> None:
    answer = ClaimAnalysis(
        answer="Es aplicable.",
        citations=[Citation(source="manual.pdf", page=99, chunk_id="fake")],
    )

    valid, feedback = validate_citations(answer, [_chunk()])

    assert valid is False
    assert "fake" in feedback


def test_fallback_never_claims_responsibility() -> None:
    answer = insufficient_evidence_analysis("No relevant evidence")

    assert answer.confidence == "low"
    assert answer.citations == []
    assert answer.agreement_responsibility == "No determinada."
