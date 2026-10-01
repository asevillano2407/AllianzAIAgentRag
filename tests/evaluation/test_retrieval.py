"""Tests for deterministic retrieval evaluation."""

from pathlib import Path

import pytest

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.errors import DocumentProcessingError
from allianz_claims_rag_agent.evaluation import (
    RetrievalCase,
    evaluate_rankings,
    read_retrieval_cases,
)


def _chunk(chunk_id: str, page: int) -> SourceChunk:
    return SourceChunk(chunk_id=chunk_id, text="evidence", source="manual.pdf", page=page)


def test_evaluate_rankings_calculates_macro_metrics() -> None:
    cases = [
        RetrievalCase(case_id="one", query="first question", relevant_pages={10}),
        RetrievalCase(case_id="two", query="second question", relevant_pages={20, 21}),
    ]
    rankings = {
        "one": [_chunk("a", 10), _chunk("b", 99)],
        "two": [_chunk("c", 99), _chunk("d", 20)],
    }

    metrics = evaluate_rankings(cases, rankings, k=2)

    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 0.75
    assert metrics.mrr == 0.75


def test_evaluate_rankings_requires_every_case() -> None:
    cases = [RetrievalCase(case_id="one", query="a question", relevant_pages={10})]

    with pytest.raises(ValueError, match="Missing ranking"):
        evaluate_rankings(cases, {}, k=1)


def test_evaluate_rankings_does_not_count_one_page_twice() -> None:
    cases = [RetrievalCase(case_id="one", query="a question", relevant_pages={10, 11})]
    rankings = {
        "one": [_chunk("a", 10), _chunk("b", 10), _chunk("c", 99)],
    }

    metrics = evaluate_rankings(cases, rankings, k=3)

    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.recall_at_k == 0.5
    assert metrics.mrr == 1.0


def test_read_retrieval_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"case_id":"one","query":"first query","relevant_pages":[1]}\n'
        '{"case_id":"one","query":"second query","relevant_pages":[2]}\n',
        encoding="utf-8",
    )

    with pytest.raises(DocumentProcessingError, match="Duplicate"):
        read_retrieval_cases(dataset)


def test_project_retrieval_dataset_is_valid() -> None:
    cases = read_retrieval_cases(Path("evaluation/datasets/retrieval_cases.jsonl"))

    assert len(cases) == 10
    assert len({case.case_id for case in cases}) == 10
