"""Tests for deterministic accident-query vocabulary expansion."""

import pytest

from allianz_claims_rag_agent.retrieval.query_expansion import expand_accident_query


@pytest.mark.parametrize(
    "query, expected_term",
    [
        ("El vehículo B choca por detrás contra A", "alcance trasero"),
        ("El vehículo cambia de carril", "cambio de carril"),
        ("El conductor da marcha atrás", "marcha atrás"),
        ("El vehículo estaba aparcado", "vehículo estacionado"),
        ("El vehículo se incorpora a la circulación", "incorporación"),
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
