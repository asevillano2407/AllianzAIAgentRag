"""Tests for deterministic accident-query vocabulary expansion."""

import pytest

from allianz_claims_rag_agent.domain import QueryType
from allianz_claims_rag_agent.retrieval.query_expansion import (
    build_retrieval_queries,
    expand_accident_query,
)


@pytest.mark.parametrize(
    "query, expected_term",
    [
        ("El vehículo B choca por detrás contra A", "alcance trasero"),
        ("El vehículo cambia de carril", "cambio de carril"),
        ("El conductor da marcha atrás", "marcha atrás"),
        ("El vehículo estaba aparcado", "vehículo estacionado"),
        ("El vehículo se incorpora a la circulación", "incorporación"),
        ("El vehículo, estacionado, presenta daños", "vehículo estacionado"),
        ("El automóvil A va a cambiar de carril", "cambio de carril"),
        ("Cinco vehículos sufren una colisión múltiple", "más de dos vehículos"),
        ("Un vehículo no identificado se da a la fuga", "identificación"),
        ("El conductor estaba bajo los efectos del alcohol", "alcoholemia"),
    ],
)
def test_expand_accident_query_adds_manual_vocabulary(
    query: str,
    expected_term: str,
) -> None:
    expanded = expand_accident_query(query)

    assert expanded is not None
    assert expected_term in expanded
    assert "CIDE ASCIDE" in expanded


def test_expand_accident_query_combines_distinct_matching_concepts() -> None:
    expanded = expand_accident_query(
        "El vehículo cambia de carril y el contrario dice que le golpea por detrás"
    )

    assert expanded is not None
    assert "cambio de carril" in expanded
    assert "alcance trasero" in expanded


def test_expand_accident_query_returns_none_without_a_supported_pattern() -> None:
    assert expand_accident_query("Dos vehículos sufren un accidente") is None


def test_accident_retrieval_queries_prioritize_expansion_and_preserve_report() -> None:
    query = "El vehículo B choca por detrás contra el vehículo A"

    queries = build_retrieval_queries(query, QueryType.ACCIDENT_DESCRIPTION)

    assert "alcance trasero" in queries[0]
    assert queries[1] == query


def test_manual_question_does_not_use_accident_expansion() -> None:
    query = "¿Qué significa chocar por detrás?"

    assert build_retrieval_queries(query, QueryType.MANUAL_QUESTION) == [query]


def test_accident_retrieval_queries_keep_distinct_manual_concepts() -> None:
    query = "Un vehículo, estacionado, fue golpeado y el contrario se da a la fuga"

    queries = build_retrieval_queries(query, QueryType.ACCIDENT_DESCRIPTION)

    assert len(queries) == 3
    assert "vehículos aparcados" in queries[0]
    assert "identificación" in queries[1]
    assert queries[2] == query
