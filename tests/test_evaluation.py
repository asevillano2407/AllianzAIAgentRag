"""Tests for retrieval metrics."""

from allianz_rag.evaluation import evaluate
from allianz_rag.models import EvaluationCase, RetrievedChunk
from tests.conftest import FakeVectorStore


def test_evaluation_calculates_recall_and_mrr() -> None:
    store = FakeVectorStore(
        [
            RetrievedChunk(
                chunk_id="a",
                source="manual.pdf",
                page=20,
                section="Other",
                text="Other",
                distance=0.1,
            ),
            RetrievedChunk(
                chunk_id="b",
                source="manual.pdf",
                page=9,
                section="Alcohol",
                text="Relevant",
                distance=0.2,
            ),
        ]
    )
    cases = [
        EvaluationCase(
            case_id="alcohol",
            query="alcoholemia",
            relevant_pages=[9],
        )
    ]

    result = evaluate(cases, store, top_k=2)

    assert result.recall_at_k == 1.0
    assert result.mean_reciprocal_rank == 0.5
    assert result.citation_validity is None
