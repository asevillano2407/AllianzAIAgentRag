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
