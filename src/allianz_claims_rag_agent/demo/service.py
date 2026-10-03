"""Pure data transformations shared by the Streamlit demo and its tests."""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from allianz_claims_rag_agent.domain import AnalysisResponse

DEFAULT_RESULTS_PATH = Path(__file__).with_name("data") / "qwen3_final_five_cases.jsonl"

CASE_TITLES = {
    "a_rear_end": "A · Alcance trasero",
    "b_multi_vehicle": "B · Colisión múltiple",
    "c_parked_unknown": "C · Vehículo aparcado",
    "d_lane_change": "D · Cambio de carril",
    "e_alcohol_injuries": "E · Alcoholemia y lesiones",
}


def load_saved_demo_runs(path: Path = DEFAULT_RESULTS_PATH) -> list[dict[str, Any]]:
    """Load and validate the frozen final benchmark used by replay mode."""
    if not path.is_file():
        raise FileNotFoundError(f"Demo results file not found: {path}")

    records: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid demo JSON on line {line_number}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Demo record on line {line_number} must be an object")
        record = _normalize_saved_record(raw, line_number)
        case_id = record["case_id"]
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate demo case_id: {case_id}")
        seen_case_ids.add(case_id)
        records.append(record)

    if not records:
        raise ValueError("Demo results file cannot be empty")
    return records


def serialize_agent_result(query: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a real LangGraph result to the same view model as replay mode."""
    response = result.get("response")
    if not isinstance(response, AnalysisResponse):
        raise ValueError("Agent result does not contain a validated response")
    generated = result.get("generated")
    chunks = result.get("chunks", [])
    query_type = result.get("query_type")
    generation = None
    if generated is not None:
        generation = {
            key: value for key, value in asdict(generated).items() if key != "response"
        }

    return {
        "case_id": "live",
        "title": "Análisis en directo",
        "source": "live_agent",
        "query": query,
        "response": response.model_dump(mode="json"),
        "agent": {
            "query_type": getattr(query_type, "value", query_type),
            "status": result.get("status"),
            "fallback_used": result.get("fallback_used", False),
            "retry_count": result.get("retry_count", 0),
            "execution_path": result.get("execution_path", []),
            "last_error": result.get("error"),
        },
        "generation": generation,
        "retrieval": {
            "queries": result.get("retrieval_queries", []),
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "pages": [chunk.page for chunk in chunks],
        },
        "business_evaluation": None,
    }


def _normalize_saved_record(raw: dict[str, Any], line_number: int) -> dict[str, Any]:
    case_id = raw.get("case_id")
    retrieval_queries = raw.get("retrieval_queries", [])
    query = raw.get("query")
    if not query and isinstance(retrieval_queries, list) and retrieval_queries:
        query = retrieval_queries[-1]
    response_payload = raw.get("response")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"Missing case_id on demo line {line_number}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Missing query on demo line {line_number}")
    if not isinstance(response_payload, dict):
        raise ValueError(f"Missing response on demo line {line_number}")
    response = AnalysisResponse.model_validate(response_payload)

    return {
        "case_id": case_id,
        "title": CASE_TITLES.get(case_id, case_id),
        "source": "saved_benchmark",
        "query": query,
        "response": response.model_dump(mode="json"),
        "agent": {
            "query_type": raw.get("query_type"),
            "status": raw.get("status"),
            "fallback_used": False,
            "retry_count": 0,
            "execution_path": [],
            "last_error": raw.get("error"),
        },
        "generation": raw.get("generation"),
        "retrieval": {
            "queries": retrieval_queries,
            "chunk_ids": raw.get("retrieved_chunk_ids", []),
            "pages": raw.get("retrieved_pages", []),
        },
        "business_evaluation": raw.get("business_evaluation"),
    }
