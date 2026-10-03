"""Validated datasets shared by demonstration and model-selection runs."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from allianz_claims_rag_agent.domain import (
    AnalysisResponse,
    ConventionApplicability,
    ConventionResponsibility,
)


class DemoCase(BaseModel):
    """One versioned interview scenario with expected evidence pages."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=3)
    expected_page_groups: list[list[int]] = Field(default_factory=list)
    expected_applicability: ConventionApplicability
    expected_responsibility: ConventionResponsibility


def load_demo_cases(path: Path) -> list[DemoCase]:
    """Load and validate a non-empty JSONL demonstration dataset."""
    cases = [
        DemoCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("The demo dataset cannot be empty")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Demo case IDs must be unique")
    return cases


def page_group_recall(
    expected_page_groups: list[list[int]],
    retrieved_pages: list[int],
) -> float | None:
    """Measure how many required evidence concepts have at least one matching page."""
    if not expected_page_groups:
        return None
    retrieved = set(retrieved_pages)
    matched_groups = sum(bool(retrieved.intersection(group)) for group in expected_page_groups)
    return matched_groups / len(expected_page_groups)


def evaluate_business_response(
    case: DemoCase,
    response: AnalysisResponse | None,
) -> dict[str, bool]:
    """Compare explicit convention decisions with the versioned case expectations."""
    applicability_correct = bool(
        response
        and response.convention_applicability is case.expected_applicability
    )
    responsibility_correct = bool(
        response
        and response.convention_responsibility is case.expected_responsibility
    )
    return {
        "applicability_correct": applicability_correct,
        "responsibility_correct": responsibility_correct,
        "business_correct": applicability_correct and responsibility_correct,
    }
