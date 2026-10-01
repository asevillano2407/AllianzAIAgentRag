"""Deterministic retrieval evaluation models and metrics."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from allianz_claims_rag_agent.domain.models import DomainModel, SourceChunk
from allianz_claims_rag_agent.errors import DocumentProcessingError


class RetrievalCase(DomainModel):
    """One query and the physical manual pages considered relevant."""

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=3)
    relevant_pages: set[int] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Macro-averaged retrieval metrics at a fixed cutoff."""

    cases: int
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float


def read_retrieval_cases(input_path: Path) -> tuple[RetrievalCase, ...]:
    """Read a validated JSONL retrieval dataset."""
    if not input_path.is_file():
        raise DocumentProcessingError(f"Evaluation file does not exist: {input_path}")

    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    try:
        with input_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    case = RetrievalCase.model_validate_json(line)
                except ValueError as exc:
                    raise DocumentProcessingError(
                        f"Invalid evaluation case at {input_path}:{line_number}"
                    ) from exc
                if case.case_id in seen_ids:
                    raise DocumentProcessingError(
                        f"Duplicate evaluation case_id at {input_path}:{line_number}"
                    )
                seen_ids.add(case.case_id)
                cases.append(case)
    except OSError as exc:
        raise DocumentProcessingError(f"Cannot read evaluation file: {input_path}") from exc

    if not cases:
        raise DocumentProcessingError(f"Evaluation file is empty: {input_path}")
    return tuple(cases)


def evaluate_rankings(
    cases: Sequence[RetrievalCase],
    rankings: Mapping[str, Sequence[SourceChunk]],
    k: int,
) -> RetrievalMetrics:
    """Calculate macro Precision@K, Recall@K and reciprocal rank."""
    if not cases:
        raise ValueError("At least one evaluation case is required")
    if k < 1:
        raise ValueError("k must be at least 1")

    precision_sum = 0.0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0

    for case in cases:
        if case.case_id not in rankings:
            raise ValueError(f"Missing ranking for case: {case.case_id}")
        retrieved_pages = [chunk.page for chunk in rankings[case.case_id]][:k]
        relevant_retrieved = len(set(retrieved_pages) & case.relevant_pages)

        precision_sum += relevant_retrieved / k
        recall_sum += relevant_retrieved / len(case.relevant_pages)
        reciprocal_rank_sum += _reciprocal_rank(retrieved_pages, case.relevant_pages)

    case_count = len(cases)
    return RetrievalMetrics(
        cases=case_count,
        k=k,
        precision_at_k=precision_sum / case_count,
        recall_at_k=recall_sum / case_count,
        mrr=reciprocal_rank_sum / case_count,
    )


def _reciprocal_rank(retrieved_pages: Sequence[int], relevant_pages: set[int]) -> float:
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return 1.0 / rank
    return 0.0
