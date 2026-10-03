"""Tests for grounded answer generation and citation validation."""

import json

import pytest

from allianz_claims_rag_agent.domain import QueryType, SourceChunk
from allianz_claims_rag_agent.errors import OutputValidationError
from allianz_claims_rag_agent.generation import AnswerGenerator, LlmGeneration


class FakeLlm:
    model_name = "fake-llm"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.received_schema: dict[str, object] | None = None

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
    ) -> LlmGeneration:
        self.received_schema = json_schema
        return LlmGeneration(
            content=json.dumps(self.payload),
            prompt_tokens=100,
            completion_tokens=25,
            total_duration_ms=2_000.0,
        )


def _chunk() -> SourceChunk:
    return SourceChunk(
        chunk_id="chunk-14",
        text="Los siniestros tendrán un año desde la fecha del accidente.",
        source="manual.pdf",
        page=14,
        section="Caducidad",
    )


def _alcohol_chunk() -> SourceChunk:
    return SourceChunk(
        chunk_id="chunk-9",
        text=(
            "No se considera la alcoholemia como motivo de exclusión por lo que en estos "
            "casos son de aplicación los Convenios."
        ),
        source="manual.pdf",
        page=9,
        section="Alcoholemia",
    )


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "query_type": "manual_question",
        "conclusion": "El plazo es de un año.",
        "facts": ["El cómputo comienza en la fecha del accidente."],
        "missing_information": [],
        "confidence": "high",
        "citations": [
            {
                "chunk_id": "chunk-14",
                "page": 14,
                "quote": "un año desde la fecha del accidente",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _accident_payload(**overrides: object) -> dict[str, object]:
    payload = _payload(
        query_type="accident_description",
        convention_applicability="undetermined",
        convention_responsibility="undetermined",
        applicability_citations=[],
        responsibility_citations=[],
    )
    payload.update(overrides)
    return payload


def test_generator_returns_validated_answer_and_usage() -> None:
    llm = FakeLlm(_payload())

    generated = AnswerGenerator(llm).generate(
        "¿Cuándo caduca?",
        QueryType.MANUAL_QUESTION,
        [_chunk()],
    )

    assert generated.response.conclusion == "El plazo es de un año."
    assert generated.model_name == "fake-llm"
    assert generated.prompt_tokens == 100
    assert generated.completion_tokens == 25
    assert llm.received_schema is not None
    assert llm.received_schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "payload, message",
    [
        (_payload(query_type="accident_description"), "query type"),
        (_payload(citations=[]), "requires citations"),
        (
            _payload(
                citations=[{"chunk_id": "invented", "page": 14, "quote": "un año"}]
            ),
            "not given",
        ),
        (
            _payload(citations=[{"chunk_id": "chunk-14", "page": 99, "quote": "un año"}]),
            "page",
        ),
        (
            _payload(
                citations=[{"chunk_id": "chunk-14", "page": 14, "quote": "texto inventado"}]
            ),
            "not present",
        ),
    ],
)
def test_generator_rejects_ungrounded_output(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(OutputValidationError, match=message):
        AnswerGenerator(FakeLlm(payload)).generate(
            "¿Cuándo caduca?",
            QueryType.MANUAL_QUESTION,
            [_chunk()],
        )


def test_generator_allows_low_confidence_without_citations() -> None:
    payload = _payload(
        confidence="low",
        citations=[],
        conclusion="La evidencia no permite responder.",
        missing_information=["Falta una regla aplicable."],
    )

    result = AnswerGenerator(FakeLlm(payload)).generate(
        "¿Cuál es la prima?",
        QueryType.MANUAL_QUESTION,
        [_chunk()],
    )

    assert not result.response.citations
    assert result.response.missing_information


def test_generator_rejects_invalid_json_schema_output() -> None:
    with pytest.raises(OutputValidationError, match="AnalysisResponse"):
        AnswerGenerator(FakeLlm({"unexpected": True})).generate(
            "¿Cuándo caduca?",
            QueryType.MANUAL_QUESTION,
            [_chunk()],
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"convention_applicability": None, "convention_responsibility": "vehicle_b"},
        {"convention_applicability": "applicable", "convention_responsibility": None},
    ],
)
def test_generator_rejects_missing_accident_decisions(
    overrides: dict[str, object],
) -> None:
    payload = _accident_payload(**overrides)

    with pytest.raises(OutputValidationError, match="does not match AnalysisResponse"):
        AnswerGenerator(FakeLlm(payload)).generate(
            "El vehículo B alcanza al vehículo A.",
            QueryType.ACCIDENT_DESCRIPTION,
            [_chunk()],
        )


def test_generator_degrades_applicability_without_valid_evidence() -> None:
    payload = _accident_payload(
        convention_applicability="applicable",
        convention_responsibility="undetermined",
        applicability_citations=[],
    )

    generated = AnswerGenerator(FakeLlm(payload)).generate(
        "El vehículo B alcanza al vehículo A.",
        QueryType.ACCIDENT_DESCRIPTION,
        [_chunk()],
    )

    assert generated.response.convention_applicability.value == "undetermined"
    assert generated.response.convention_responsibility.value == "undetermined"
    assert generated.response.confidence.value == "low"
    assert "applicability_downgraded_missing_valid_citation" in generated.adjustments


def test_generator_repairs_inconsistent_applicable_responsibility() -> None:
    citation = {
        "chunk_id": "chunk-9",
        "page": 9,
        "quote": "No se considera la alcoholemia como motivo de exclusión",
    }
    payload = _accident_payload(
        convention_applicability="applicable",
        convention_responsibility="not_applicable",
        applicability_citations=[citation],
        citations=[],
    )

    generated = AnswerGenerator(FakeLlm(payload)).generate(
        "El conductor se encontraba bajo los efectos del alcohol.",
        QueryType.ACCIDENT_DESCRIPTION,
        [_alcohol_chunk()],
    )

    assert generated.response.convention_applicability.value == "applicable"
    assert generated.response.convention_responsibility.value == "undetermined"
    assert "responsibility_repaired_from_inconsistent_decision" in generated.adjustments


def test_generator_aligns_responsibility_when_convention_is_not_applicable() -> None:
    chunk = SourceChunk(
        chunk_id="chunk-exclusion",
        text="La existencia de un tercer vehículo invalida la aplicación de los Convenios.",
        source="manual.pdf",
        page=10,
    )
    citation = {
        "chunk_id": "chunk-exclusion",
        "page": 10,
        "quote": "invalida la aplicación de los Convenios",
    }
    payload = _accident_payload(
        convention_applicability="not_applicable",
        convention_responsibility="vehicle_b",
        applicability_citations=[citation],
        responsibility_citations=[citation],
        citations=[],
    )

    generated = AnswerGenerator(FakeLlm(payload)).generate(
        "Intervienen tres vehículos.",
        QueryType.ACCIDENT_DESCRIPTION,
        [chunk],
    )

    assert generated.response.convention_responsibility.value == "not_applicable"
    assert generated.response.responsibility_citations == []
    assert (
        "responsibility_aligned_with_non_applicable_convention"
        in generated.adjustments
    )


def test_generator_accepts_applicability_supported_by_specific_citation() -> None:
    citation = {
        "chunk_id": "chunk-9",
        "page": 9,
        "quote": "No se considera la alcoholemia como motivo de exclusión",
    }
    payload = _accident_payload(
        convention_applicability="applicable",
        convention_responsibility="undetermined",
        applicability_citations=[citation],
        citations=[],
    )

    generated = AnswerGenerator(FakeLlm(payload)).generate(
        "El conductor se encontraba bajo los efectos del alcohol.",
        QueryType.ACCIDENT_DESCRIPTION,
        [_alcohol_chunk()],
    )

    assert generated.response.applicability_citations[0].page == 9


def test_generator_rejects_applicability_that_contradicts_its_citation() -> None:
    citation = {
        "chunk_id": "chunk-9",
        "page": 9,
        "quote": (
            "No se considera la alcoholemia como motivo de exclusión por lo que en estos "
            "casos son de aplicación los Convenios."
        ),
    }
    payload = _accident_payload(
        convention_applicability="not_applicable",
        convention_responsibility="not_applicable",
        applicability_citations=[citation],
        citations=[],
    )

    with pytest.raises(OutputValidationError, match="contradicts evidence"):
        AnswerGenerator(FakeLlm(payload)).generate(
            "El conductor se encontraba bajo los efectos del alcohol.",
            QueryType.ACCIDENT_DESCRIPTION,
            [_alcohol_chunk()],
        )


def test_generator_downgrades_responsibility_without_normative_evidence() -> None:
    citation = {
        "chunk_id": "chunk-9",
        "page": 9,
        "quote": (
            "No se considera la alcoholemia como motivo de exclusión por lo que en estos "
            "casos son de aplicación los Convenios."
        ),
    }
    payload = _accident_payload(
        conclusion="El relato describe alcoholemia y lesiones.",
        convention_applicability="applicable",
        convention_responsibility="vehicle_b",
        applicability_citations=[citation],
        responsibility_citations=[citation],
        citations=[],
    )

    generated = AnswerGenerator(FakeLlm(payload)).generate(
        "El conductor se encontraba bajo los efectos del alcohol.",
        QueryType.ACCIDENT_DESCRIPTION,
        [_alcohol_chunk()],
    )

    assert generated.response.convention_responsibility.value == "undetermined"
    assert generated.response.responsibility_citations == []
    assert generated.response.confidence.value == "low"
    assert generated.adjustments == (
        "responsibility_downgraded_missing_normative_evidence",
        "conclusion_rendered_from_structured_decisions",
    )
    assert "responsabilidad no puede determinarse" in generated.response.conclusion


def test_generator_rewrites_explicit_responsibility_without_normative_evidence() -> None:
    citation = {
        "chunk_id": "chunk-9",
        "page": 9,
        "quote": "No se considera la alcoholemia como motivo de exclusión",
    }
    payload = _accident_payload(
        conclusion="El vehículo B es responsable según el convenio.",
        convention_applicability="applicable",
        convention_responsibility="vehicle_b",
        applicability_citations=[citation],
        responsibility_citations=[citation],
        citations=[],
    )

    generated = AnswerGenerator(FakeLlm(payload)).generate(
        "El conductor se encontraba bajo los efectos del alcohol.",
        QueryType.ACCIDENT_DESCRIPTION,
        [_alcohol_chunk()],
    )

    assert generated.response.convention_responsibility.value == "undetermined"
    assert "responsabilidad no puede determinarse" in generated.response.conclusion
    assert generated.adjustments == (
        "responsibility_downgraded_missing_normative_evidence",
        "conclusion_rendered_from_structured_decisions",
    )


def test_generator_renders_conclusion_from_supported_responsibility() -> None:
    chunk = SourceChunk(
        chunk_id="chunk-75",
        text=(
            "En un alcance trasero se considerará responsable al conductor del vehículo "
            "que presente daños en la parte delantera."
        ),
        source="manual.pdf",
        page=75,
    )
    citation = {
        "chunk_id": "chunk-75",
        "page": 75,
        "quote": "se considerará responsable al conductor del vehículo",
    }
    payload = _accident_payload(
        conclusion="Texto libre potencialmente inconsistente.",
        convention_applicability="applicable",
        convention_responsibility="vehicle_b",
        applicability_citations=[citation],
        responsibility_citations=[citation],
        citations=[],
    )

    generated = AnswerGenerator(FakeLlm(payload)).generate(
        "El vehículo B choca por detrás contra A.",
        QueryType.ACCIDENT_DESCRIPTION,
        [chunk],
    )

    assert generated.response.conclusion.endswith(
        "El vehículo B resulta responsable según el convenio."
    )
    assert generated.adjustments == (
        "conclusion_rendered_from_structured_decisions",
    )


def test_generator_repairs_accident_citation_page_from_chunk_metadata() -> None:
    citation = {
        "chunk_id": "chunk-9",
        "page": 99,
        "quote": "No se considera la alcoholemia como motivo de exclusión",
    }
    payload = _accident_payload(
        convention_applicability="applicable",
        convention_responsibility="undetermined",
        applicability_citations=[citation],
        citations=[],
    )

    generated = AnswerGenerator(FakeLlm(payload)).generate(
        "El conductor se encontraba bajo los efectos del alcohol.",
        QueryType.ACCIDENT_DESCRIPTION,
        [_alcohol_chunk()],
    )

    assert generated.response.applicability_citations[0].page == 9
    assert "citation_pages_repaired" in generated.adjustments


def test_generator_degrades_only_responsibility_with_invalid_citation() -> None:
    applicability_citation = {
        "chunk_id": "chunk-9",
        "page": 9,
        "quote": "No se considera la alcoholemia como motivo de exclusión",
    }
    responsibility_citation = {
        "chunk_id": "chunk-9",
        "page": 9,
        "quote": "El vehículo B es culpable.",
    }
    payload = _accident_payload(
        convention_applicability="applicable",
        convention_responsibility="vehicle_b",
        applicability_citations=[applicability_citation],
        responsibility_citations=[responsibility_citation],
        citations=[],
    )

    generated = AnswerGenerator(FakeLlm(payload)).generate(
        "El conductor se encontraba bajo los efectos del alcohol.",
        QueryType.ACCIDENT_DESCRIPTION,
        [_alcohol_chunk()],
    )

    assert generated.response.convention_applicability.value == "applicable"
    assert generated.response.convention_responsibility.value == "undetermined"
    assert generated.response.applicability_citations
    assert generated.response.responsibility_citations == []
    assert "invalid_citations_removed" in generated.adjustments
    assert (
        "responsibility_downgraded_missing_valid_citation"
        in generated.adjustments
    )


@pytest.mark.parametrize(
    "chunk_text, quote",
    [
        (
            "El “vehículo” será de aplicación según el Convenio.",
            'El "vehiculo" sera de aplicacion segun el Convenio.',
        ),
        (
            "Los Convenios serán de aplica-\nción directa.",
            "Los Convenios serán de aplicación directa",
        ),
        (
            "La regla se aplica a CIDE/ASCIDE.",
            "La regla se aplica a CIDE - ASCIDE",
        ),
    ],
)
def test_generator_accepts_harmless_citation_formatting_differences(
    chunk_text: str,
    quote: str,
) -> None:
    chunk = SourceChunk(
        chunk_id="chunk-format",
        text=chunk_text,
        source="manual.pdf",
        page=10,
    )
    payload = _payload(
        citations=[{"chunk_id": "chunk-format", "page": 10, "quote": quote}]
    )

    generated = AnswerGenerator(FakeLlm(payload)).generate(
        "¿Qué indica la regla?",
        QueryType.MANUAL_QUESTION,
        [chunk],
    )

    assert generated.response.citations[0].quote == quote


def test_generator_rejects_citation_that_omits_preceding_negation() -> None:
    chunk = SourceChunk(
        chunk_id="chunk-negative",
        text="Los convenios no son de aplicación en este supuesto.",
        source="manual.pdf",
        page=10,
    )
    payload = _payload(
        citations=[
            {
                "chunk_id": "chunk-negative",
                "page": 10,
                "quote": "son de aplicación en este supuesto",
            }
        ]
    )

    with pytest.raises(OutputValidationError, match="not present"):
        AnswerGenerator(FakeLlm(payload)).generate(
            "¿Se aplica el convenio?",
            QueryType.MANUAL_QUESTION,
            [chunk],
        )
