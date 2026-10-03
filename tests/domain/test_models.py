"""Tests for core domain contracts."""

import pytest
from pydantic import ValidationError

from allianz_claims_rag_agent.domain import (
    AccidentAnalysisResponse,
    AnalysisRequest,
    AnalysisResponse,
    ConfidenceLevel,
    ConventionApplicability,
    ConventionResponsibility,
    DocumentPage,
    QueryType,
    SourceChunk,
)


def test_analysis_request_strips_surrounding_whitespace() -> None:
    request = AnalysisRequest(text="  ¿Quién tiene prioridad?  ")

    assert request.text == "¿Quién tiene prioridad?"


@pytest.mark.parametrize("text", ["", "  ", "a"])
def test_analysis_request_rejects_empty_or_too_short_text(text: str) -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(text=text)


def test_source_chunk_requires_a_positive_page() -> None:
    with pytest.raises(ValidationError):
        SourceChunk(
            chunk_id="manual:0:0",
            text="Contenido",
            source="manual.pdf",
            page=0,
        )


def test_document_page_allows_empty_text_for_image_only_pages() -> None:
    page = DocumentPage(source="manual.pdf", page=32, text="")

    assert page.text == ""


def test_response_collections_are_not_shared_between_instances() -> None:
    first = AnalysisResponse(
        query_type=QueryType.MANUAL_QUESTION,
        conclusion="Primera respuesta",
        confidence=ConfidenceLevel.LOW,
    )
    second = AnalysisResponse(
        query_type=QueryType.ACCIDENT_DESCRIPTION,
        conclusion="Segunda respuesta",
        confidence=ConfidenceLevel.MEDIUM,
    )

    first.facts.append("Hecho aislado")

    assert second.facts == []


def test_domain_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(text="Consulta válida", unexpected="not allowed")


def test_accident_decisions_use_explicit_domain_values() -> None:
    response = AnalysisResponse(
        query_type=QueryType.ACCIDENT_DESCRIPTION,
        conclusion="El convenio se aplica y el vehículo B es responsable.",
        convention_applicability=ConventionApplicability.APPLICABLE,
        convention_responsibility=ConventionResponsibility.VEHICLE_B,
        confidence=ConfidenceLevel.HIGH,
    )

    assert response.convention_applicability is ConventionApplicability.APPLICABLE
    assert response.convention_responsibility is ConventionResponsibility.VEHICLE_B


def test_accident_schema_requires_decisions_and_decision_citations() -> None:
    required_fields = set(AccidentAnalysisResponse.model_json_schema()["required"])

    assert {
        "convention_applicability",
        "convention_responsibility",
        "applicability_citations",
        "responsibility_citations",
    } <= required_fields
