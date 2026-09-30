"""Validated domain models for requests, retrieval, and responses."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DomainModel(BaseModel):
    """Common validation behaviour for domain models."""

    model_config = ConfigDict(
        extra="forbid", # Block unexpected fields
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class QueryType(StrEnum):
    """Kinds of user input supported by the application."""

    MANUAL_QUESTION = "manual_question"
    ACCIDENT_DESCRIPTION = "accident_description"


class ConfidenceLevel(StrEnum):
    """Qualitative confidence exposed to the user."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalysisRequest(DomainModel):
    """User input accepted by the analysis workflow."""

    text: str = Field(min_length=3, max_length=4_000)


class DocumentPage(DomainModel):
    """Text extracted from one physical page of a source document."""

    source: str = Field(min_length=1)
    page: int = Field(ge=1)
    text: str


class SourceChunk(DomainModel):
    """A fragment recovered from a source document."""

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int = Field(ge=1)
    section: str | None = None
    score: float | None = Field(default=None, allow_inf_nan=False)


class Citation(DomainModel):
    """Evidence linking an answer to a retrieved fragment."""

    chunk_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=1_000)


class AnalysisResponse(DomainModel):
    """Structured result returned by the analysis workflow."""

    query_type: QueryType
    conclusion: str = Field(min_length=1)
    facts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel
    citations: list[Citation] = Field(default_factory=list)
