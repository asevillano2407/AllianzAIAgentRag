"""Domain and API models."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DocumentChunk(BaseModel):
    """A traceable chunk extracted from a source page."""

    chunk_id: str
    source: str
    page: int = Field(ge=1)
    section: str
    text: str = Field(min_length=1)


class RetrievedChunk(DocumentChunk):
    """A retrieved chunk and its vector distance."""

    distance: float = Field(ge=0.0)


class Citation(BaseModel):
    """Evidence reference emitted by the model."""

    source: str
    page: int = Field(ge=1)
    chunk_id: str
    quote: str = Field(default="", max_length=350)


class ClaimAnalysis(BaseModel):
    """Validated response schema for questions and accident descriptions."""

    answer: str = Field(min_length=1)
    case_type: str = "Consulta general"
    parties: list[str] = Field(default_factory=list)
    agreement_responsibility: str = (
        "No puede determinarse con la informacion disponible."
    )
    applicable_framework: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    limitations: list[str] = Field(default_factory=list)

    @field_validator("citations")
    @classmethod
    def unique_citations(cls, citations: list[Citation]) -> list[Citation]:
        """Remove duplicate chunk citations without changing their order."""

        seen: set[str] = set()
        result: list[Citation] = []
        for citation in citations:
            if citation.chunk_id not in seen:
                seen.add(citation.chunk_id)
                result.append(citation)
        return result


class AnalyzeRequest(BaseModel):
    """Request accepted by the analysis API."""

    query: str = Field(min_length=5, max_length=6000)


class AnalyzeResponse(BaseModel):
    """API response with answer and execution metadata."""

    analysis: ClaimAnalysis
    route: Literal["question", "accident_case"]
    retrieved_chunks: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    """One golden evaluation example."""

    case_id: str
    query: str
    relevant_pages: list[int]
    expected_topics: list[str] = Field(default_factory=list)
