"""Tests for the bounded agent command parser."""

import pytest

from allianz_claims_rag_agent.orchestration.cli import build_parser


def test_agent_parser_requires_only_the_query() -> None:
    args = build_parser().parse_args(["¿Cuándo caduca CICOS?"])

    assert args.query == "¿Cuándo caduca CICOS?"
    assert args.top_k is None


def test_agent_parser_accepts_runtime_overrides() -> None:
    args = build_parser().parse_args(
        ["consulta", "--top-k", "4", "--llm-model", "llama3.2:3b"]
    )

    assert args.top_k == 4
    assert args.llm_model == "llama3.2:3b"


def test_agent_parser_accepts_optional_reranking() -> None:
    args = build_parser().parse_args(
        ["consulta", "--rerank", "--candidate-k", "12", "--top-k", "4"]
    )

    assert args.rerank is True
    assert args.candidate_k == 12


def test_agent_parser_rejects_invalid_top_k() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["consulta", "--top-k", "0"])
