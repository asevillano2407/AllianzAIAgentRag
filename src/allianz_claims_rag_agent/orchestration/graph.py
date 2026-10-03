"""LangGraph workflow with bounded retries and deterministic fallback."""

from dataclasses import dataclass
from typing import Literal, Protocol

from langgraph.graph import END, START, StateGraph

from allianz_claims_rag_agent.domain import (
    AnalysisRequest,
    AnalysisResponse,
    ConfidenceLevel,
    ConventionApplicability,
    ConventionResponsibility,
    QueryType,
    SourceChunk,
)
from allianz_claims_rag_agent.errors import (
    EmbeddingError,
    GenerationError,
    OutputValidationError,
    RetrievalError,
)
from allianz_claims_rag_agent.generation.service import GeneratedAnswer
from allianz_claims_rag_agent.orchestration.routing import DeterministicQueryRouter
from allianz_claims_rag_agent.orchestration.state import ClaimsAgentState
from allianz_claims_rag_agent.retrieval.query_expansion import build_retrieval_queries


class EvidenceRetriever(Protocol):
    """Minimum retrieval capability required by the graph."""

    def retrieve_many(self, queries: list[str], limit: int) -> list[SourceChunk]: ...


class GroundedGenerator(Protocol):
    """Minimum grounded-generation capability required by the graph."""

    def generate(
        self,
        query: str,
        query_type: QueryType,
        chunks: list[SourceChunk],
    ) -> GeneratedAnswer: ...


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    """Injected services and bounded workflow configuration."""

    retriever: EvidenceRetriever
    generator: GroundedGenerator
    router: DeterministicQueryRouter
    top_k: int = 3
    max_retries: int = 1

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 <= self.max_retries <= 3:
            raise ValueError("max_retries must be between 0 and 3")


def build_claims_graph(dependencies: AgentDependencies):
    """Compile the claims workflow with explicit success, retry, and fallback paths."""

    def route_query(state: ClaimsAgentState) -> ClaimsAgentState:
        request = AnalysisRequest(text=state["query"])
        query_type = dependencies.router.route(request.text)
        return {
            "query": request.text,
            "query_type": query_type,
            "retrieval_queries": build_retrieval_queries(request.text, query_type),
            "retry_count": 0,
            "status": "running",
            "fallback_used": False,
            "error": None,
            "execution_path": ["route"],
        }

    def retrieve(state: ClaimsAgentState) -> ClaimsAgentState:
        path = [*state.get("execution_path", []), "retrieve"]
        try:
            chunks = dependencies.retriever.retrieve_many(
                state["retrieval_queries"], dependencies.top_k
            )
        except (EmbeddingError, RetrievalError) as exc:
            return {
                "error": str(exc),
                "failure_retriable": True,
                "retry_count": state.get("retry_count", 0) + 1,
                "execution_path": path,
            }
        return {
            "chunks": chunks,
            "error": None,
            "failure_retriable": False,
            "execution_path": path,
        }

    def generate(state: ClaimsAgentState) -> ClaimsAgentState:
        path = [*state.get("execution_path", []), "generate"]
        try:
            generated = dependencies.generator.generate(
                state["query"], state["query_type"], state["chunks"]
            )
        except GenerationError as exc:
            return {
                "error": str(exc),
                "failure_retriable": True,
                "retry_count": state.get("retry_count", 0) + 1,
                "execution_path": path,
            }
        except OutputValidationError as exc:
            return {
                "error": str(exc),
                "failure_retriable": False,
                "execution_path": path,
            }
        return {
            "generated": generated,
            "response": generated.response,
            "status": "completed",
            "fallback_used": False,
            "error": None,
            "execution_path": path,
        }

    def fallback(state: ClaimsAgentState) -> ClaimsAgentState:
        is_accident = state["query_type"] is QueryType.ACCIDENT_DESCRIPTION
        response = AnalysisResponse(
            query_type=state["query_type"],
            conclusion="No se pudo obtener una respuesta validada con la evidencia disponible.",
            convention_applicability=(
                ConventionApplicability.UNDETERMINED if is_accident else None
            ),
            convention_responsibility=(
                ConventionResponsibility.UNDETERMINED if is_accident else None
            ),
            missing_information=[
                "Revise la consulta o inténtelo de nuevo si el servicio local no estaba disponible."
            ],
            confidence=ConfidenceLevel.LOW,
        )
        return {
            "response": response,
            "status": "fallback",
            "fallback_used": True,
            "execution_path": [*state.get("execution_path", []), "fallback"],
        }

    def can_retry(state: ClaimsAgentState) -> bool:
        return bool(
            state.get("failure_retriable")
            and state.get("retry_count", 0) <= dependencies.max_retries
        )

    def after_retrieval(state: ClaimsAgentState) -> Literal["generate", "retrieve", "fallback"]:
        if state.get("error") is None:
            return "generate"
        if can_retry(state):
            return "retrieve"
        return "fallback"

    def after_generation(state: ClaimsAgentState) -> Literal["complete", "generate", "fallback"]:
        if state.get("error") is None:
            return "complete"
        if can_retry(state):
            return "generate"
        return "fallback"

    workflow = StateGraph(ClaimsAgentState)
    workflow.add_node("route", route_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("fallback", fallback)
    workflow.add_edge(START, "route")
    workflow.add_edge("route", "retrieve")
    workflow.add_conditional_edges("retrieve", after_retrieval)
    workflow.add_conditional_edges(
        "generate",
        after_generation,
        {"complete": END, "generate": "generate", "fallback": "fallback"},
    )
    workflow.add_edge("fallback", END)
    return workflow.compile()
