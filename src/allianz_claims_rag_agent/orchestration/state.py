"""Raw shared state for the claims LangGraph workflow."""

from typing import Literal, TypedDict

from allianz_claims_rag_agent.domain import AnalysisResponse, QueryType, SourceChunk
from allianz_claims_rag_agent.generation.service import GeneratedAnswer


class ClaimsAgentState(TypedDict, total=False):
    """Values persisted between small, single-purpose workflow nodes."""

    query: str
    query_type: QueryType
    retrieval_queries: list[str]
    chunks: list[SourceChunk]
    generated: GeneratedAnswer
    response: AnalysisResponse
    retry_count: int
    status: Literal["running", "completed", "fallback"]
    fallback_used: bool
    error: str | None
    failure_retriable: bool
    execution_path: list[str]
