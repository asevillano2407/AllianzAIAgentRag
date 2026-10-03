"""Tests for deterministic query routing."""

import pytest

from allianz_claims_rag_agent.domain import QueryType
from allianz_claims_rag_agent.orchestration.routing import DeterministicQueryRouter


@pytest.mark.parametrize(
    "query",
    [
        "¿Cuándo caduca una reclamación CICOS?",
        "Explica qué significa CIDE",
        "Cuál es el plazo aplicable",
    ],
)
def test_router_identifies_manual_questions(query: str) -> None:
    assert DeterministicQueryRouter().route(query) is QueryType.MANUAL_QUESTION


@pytest.mark.parametrize(
    "query",
    [
        "El vehículo A estaba detenido y el vehículo B choca por detrás.",
        "Dos vehículos colisionan durante un cambio de carril.",
        "El vehículo A está detenido y B choca por detrás, ¿quién es responsable?",
        "El automóvil A roza lateralmente al automóvil B al cambiar de carril.",
        "El coche A impacta contra el coche B.",
    ],
)
def test_router_identifies_accident_descriptions(query: str) -> None:
    assert DeterministicQueryRouter().route(query) is QueryType.ACCIDENT_DESCRIPTION


def test_router_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        DeterministicQueryRouter().route("   ")
