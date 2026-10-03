"""Domain contracts shared by the application layers."""

from allianz_claims_rag_agent.domain.models import (
    AccidentAnalysisResponse,
    AnalysisRequest,
    AnalysisResponse,
    Citation,
    ConfidenceLevel,
    ConventionApplicability,
    ConventionResponsibility,
    DocumentPage,
    QueryType,
    SourceChunk,
)

__all__ = [
    "AccidentAnalysisResponse",
    "AnalysisRequest",
    "AnalysisResponse",
    "Citation",
    "ConfidenceLevel",
    "ConventionApplicability",
    "ConventionResponsibility",
    "DocumentPage",
    "QueryType",
    "SourceChunk",
]
