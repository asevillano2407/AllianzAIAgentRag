"""Tests for the manual grounded-generation command parser."""

import pytest

from allianz_claims_rag_agent.generation.cli import build_parser


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


def test_parser_accepts_optional_reranking() -> None:
    args = build_parser().parse_args(
        ["consulta", "--rerank", "--candidate-k", "12"]
    )

    assert args.rerank is True
    assert args.candidate_k == 12


def test_parser_rejects_invalid_query_type() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["consulta", "--query-type", "unknown"])
