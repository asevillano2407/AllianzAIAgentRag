"""Tests for replay loading and live-result serialization."""

from pathlib import Path

import pytest

from allianz_claims_rag_agent.demo import (
    load_saved_demo_runs,
    serialize_agent_result,
)
from allianz_claims_rag_agent.demo.cli import main as demo_main
from allianz_claims_rag_agent.domain import (
    AnalysisResponse,
    ConfidenceLevel,
    QueryType,
    SourceChunk,
)
from allianz_claims_rag_agent.generation import GeneratedAnswer


def test_packaged_demo_results_contain_the_final_five_cases() -> None:
    runs = load_saved_demo_runs()

    assert len(runs) == 5
    assert {run["case_id"] for run in runs} == {
        "a_rear_end",
        "b_multi_vehicle",
        "c_parked_unknown",
        "d_lane_change",
        "e_alcohol_injuries",
    }
    assert sum(
        bool(run["business_evaluation"]["business_correct"]) for run in runs
    ) == 4


def test_saved_demo_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    result_path = tmp_path / "duplicate.jsonl"
    record = (
        '{"case_id":"same","query":"consulta válida",'
        '"query_type":"manual_question","status":"completed",'
        '"response":{"query_type":"manual_question","conclusion":"respuesta",'
        '"facts":[],"missing_information":[],"confidence":"low","citations":[]}}'
    )
    result_path.write_text(f"{record}\n{record}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate demo case_id"):
        load_saved_demo_runs(result_path)


def test_serialize_agent_result_builds_demo_view_model() -> None:
    response = AnalysisResponse(
        query_type=QueryType.MANUAL_QUESTION,
        conclusion="El plazo es de un año.",
        confidence=ConfidenceLevel.LOW,
    )
    generated = GeneratedAnswer(
        response=response,
        model_name="fake-model",
        prompt_tokens=10,
        completion_tokens=5,
        total_duration_ms=1_500,
    )
    chunk = SourceChunk(
        chunk_id="chunk-14",
        text="El plazo es de un año.",
        source="manual.pdf",
        page=14,
    )

    view = serialize_agent_result(
        "¿Cuál es el plazo?",
        {
            "response": response,
            "generated": generated,
            "chunks": [chunk],
            "query_type": QueryType.MANUAL_QUESTION,
            "status": "completed",
            "fallback_used": False,
            "retry_count": 0,
            "execution_path": ["route", "retrieve", "generate"],
            "retrieval_queries": ["¿Cuál es el plazo?"],
            "error": None,
        },
    )

    assert view["response"]["conclusion"] == "El plazo es de un año."
    assert view["retrieval"]["pages"] == [14]
    assert view["agent"]["execution_path"] == ["route", "retrieve", "generate"]
    assert view["generation"]["model_name"] == "fake-model"


def test_demo_launcher_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)

    assert demo_main() == 1
    assert "[demo]" in capsys.readouterr().err
