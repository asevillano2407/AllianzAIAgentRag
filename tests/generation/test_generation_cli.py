"""Tests for the manual grounded-generation command parser."""

import pytest

from allianz_claims_rag_agent.domain import QueryType
from allianz_claims_rag_agent.generation.cli import _build_retrieval_queries, build_parser


def test_parser_defaults_to_manual_question() -> None:
    args = build_parser().parse_args(["¿Cuándo caduca?"])

    assert args.query_type == "manual_question"
    assert args.top_k is None


def test_parser_accepts_accident_description() -> None:
    args = build_parser().parse_args(
        ["El vehículo A golpeó al B", "--query-type", "accident_description", "--top-k", "4"]
    )

    assert args.query_type == "accident_description"
    assert args.top_k == 4


def test_parser_rejects_invalid_query_type() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["consulta", "--query-type", "unknown"])


def test_accident_retrieval_queries_prioritize_expansion_and_preserve_report() -> None:
    query = "El vehículo B choca por detrás contra el vehículo A"

    queries = _build_retrieval_queries(query, QueryType.ACCIDENT_DESCRIPTION)

    assert "alcance trasero" in queries[0]
    assert queries[1] == query


def test_manual_question_does_not_use_accident_expansion() -> None:
    query = "¿Qué significa chocar por detrás?"

    assert _build_retrieval_queries(query, QueryType.MANUAL_QUESTION) == [query]
