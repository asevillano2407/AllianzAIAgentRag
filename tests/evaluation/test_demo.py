"""Tests for the versioned interview demo runner."""

from pathlib import Path

import pytest

from allianz_claims_rag_agent.domain import (
    AnalysisResponse,
    ConfidenceLevel,
    ConventionApplicability,
    ConventionResponsibility,
    QueryType,
)
from allianz_claims_rag_agent.evaluation.cases import (
    DemoCase,
    evaluate_business_response,
    page_group_recall,
)
from allianz_claims_rag_agent.evaluation.demo import build_parser, load_demo_cases


def test_load_demo_cases_validates_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        (
            '{"case_id":"a","query":"Consulta válida","expected_page_groups":[[75]],'
            '"expected_applicability":"applicable",'
            '"expected_responsibility":"vehicle_b"}\n'
        ),
        encoding="utf-8",
    )

    cases = load_demo_cases(dataset)

    assert cases[0].case_id == "a"
    assert cases[0].expected_page_groups == [[75]]


def test_load_demo_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            [
                (
                    '{"case_id":"a","query":"Primera consulta",'
                    '"expected_applicability":"applicable",'
                    '"expected_responsibility":"vehicle_b"}'
                ),
                (
                    '{"case_id":"a","query":"Segunda consulta",'
                    '"expected_applicability":"applicable",'
                    '"expected_responsibility":"vehicle_b"}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_demo_cases(dataset)


def test_demo_parser_uses_versioned_dataset_by_default() -> None:
    args = build_parser().parse_args([])

    assert args.cases.name == "demo_accident_cases.jsonl"
    assert args.output is None
    assert args.rerank is False


def test_demo_parser_accepts_optional_reranking() -> None:
    args = build_parser().parse_args(
        ["--rerank", "--reranker-model", "custom-reranker", "--candidate-k", "10"]
    )

    assert args.rerank is True
    assert args.reranker_model == "custom-reranker"
    assert args.candidate_k == 10


def test_page_group_recall_accepts_alternative_pages_per_concept() -> None:
    recall = page_group_recall([[56, 57, 58], [9]], [58, 9, 18])

    assert recall == 1.0


def test_page_group_recall_detects_missing_concept() -> None:
    recall = page_group_recall([[73], [33, 34]], [73, 46])

    assert recall == 0.5


def test_business_evaluation_scores_both_decisions() -> None:
    case = DemoCase(
        case_id="a",
        query="El vehículo B alcanza al vehículo A.",
        expected_applicability=ConventionApplicability.APPLICABLE,
        expected_responsibility=ConventionResponsibility.VEHICLE_B,
    )
    response = AnalysisResponse(
        query_type=QueryType.ACCIDENT_DESCRIPTION,
        conclusion="El convenio se aplica, pero se atribuye al vehículo equivocado.",
        convention_applicability=ConventionApplicability.APPLICABLE,
        convention_responsibility=ConventionResponsibility.VEHICLE_A,
        confidence=ConfidenceLevel.MEDIUM,
    )

    evaluation = evaluate_business_response(case, response)

    assert evaluation == {
        "applicability_correct": True,
        "responsibility_correct": False,
        "business_correct": False,
    }
