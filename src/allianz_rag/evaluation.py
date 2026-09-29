"""Offline retrieval metrics and optional end-to-end evaluation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from allianz_rag.models import EvaluationCase
from allianz_rag.service import ClaimsService
from allianz_rag.vector_store import VectorStore


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate deterministic evaluation metrics."""

    examples: int
    recall_at_k: float
    mean_reciprocal_rank: float
    citation_validity: float | None = None
    structured_completeness: float | None = None

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a JSON-serializable metrics dictionary."""

        return asdict(self)


def load_dataset(path: Path) -> list[EvaluationCase]:
    """Load and validate a JSON Lines golden dataset."""

    cases: list[EvaluationCase] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvaluationCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid dataset row {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    return cases


def evaluate(
    cases: list[EvaluationCase],
    store: VectorStore,
    *,
    top_k: int,
    service: ClaimsService | None = None,
) -> EvaluationResult:
    """Measure retrieval and, when configured, response integrity."""

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    citation_scores: list[float] = []
    completeness_scores: list[float] = []

    for case in cases:
        results = store.search(case.query, top_k=top_k)
        retrieved_pages = [result.page for result in results]
        relevant = set(case.relevant_pages)
        recalls.append(float(bool(relevant.intersection(retrieved_pages))))
        rank = next(
            (index for index, page in enumerate(retrieved_pages, start=1) if page in relevant),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)

        if service is not None:
            response = service.analyze(case.query)
            citations = response.analysis.citations
            citation_scores.append(
                float(
                    bool(citations)
                    and all(citation.page in retrieved_pages for citation in citations)
                )
            )
            required = [
                response.analysis.answer,
                response.analysis.agreement_responsibility,
                response.analysis.confidence,
            ]
            completeness_scores.append(float(all(required)))

    count = len(cases)
    return EvaluationResult(
        examples=count,
        recall_at_k=sum(recalls) / count,
        mean_reciprocal_rank=sum(reciprocal_ranks) / count,
        citation_validity=(sum(citation_scores) / count if citation_scores else None),
        structured_completeness=(
            sum(completeness_scores) / count if completeness_scores else None
        ),
    )


def write_result(result: EvaluationResult, output_path: Path) -> None:
    """Persist metrics for reproducible comparison."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
