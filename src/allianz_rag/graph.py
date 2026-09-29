"""Bounded LangGraph workflow for retrieval and grounded generation."""

from collections.abc import Callable
from typing import Literal, TypedDict

from allianz_rag.llm import AnalysisGenerator
from allianz_rag.models import ClaimAnalysis, RetrievedChunk
from allianz_rag.vector_store import VectorStore

Route = Literal["question", "accident_case"]


class GraphState(TypedDict, total=False):
    """Explicit data passed between graph nodes."""

    query: str
    route: Route
    retrieved: list[RetrievedChunk]
    answer: ClaimAnalysis
    warnings: list[str]
    validation_feedback: str
    retry_count: int
    is_valid: bool


_CASE_TERMS = (
    "accidente",
    "vehiculo a",
    "vehículo a",
    "coche a",
    "colision",
    "colisión",
    "alcance",
    "siniestro",
    "conductor",
)


def classify_route(query: str) -> Route:
    """Use deterministic routing when lexical business rules are sufficient."""

    normalized = query.casefold()
    return "accident_case" if any(term in normalized for term in _CASE_TERMS) else "question"


def validate_citations(
    answer: ClaimAnalysis, retrieved: list[RetrievedChunk]
) -> tuple[bool, str]:
    """Ensure every citation points to evidence supplied to the model."""

    if not retrieved:
        return False, "No evidence was retrieved."
    if not answer.citations:
        return False, "The answer contains no citations."
    allowed = {
        (chunk.source, chunk.page, chunk.chunk_id): chunk for chunk in retrieved
    }
    invalid = [
        citation.chunk_id
        for citation in answer.citations
        if (citation.source, citation.page, citation.chunk_id) not in allowed
    ]
    if invalid:
        return False, f"Unknown citation identifiers: {', '.join(invalid)}"
    return True, ""


def insufficient_evidence_analysis(reason: str) -> ClaimAnalysis:
    """Return a safe response when retrieval or generation cannot be trusted."""

    return ClaimAnalysis(
        answer=(
            "No hay evidencia suficiente en los fragmentos recuperados para emitir "
            "una conclusion fiable. Debe revisarlo una persona especialista."
        ),
        agreement_responsibility="No determinada.",
        missing_information=[reason],
        confidence="low",
        limitations=[
            "El manual aportado data de 2004.",
            "La respuesta no constituye una decision legal, de cobertura o pago.",
        ],
    )


def build_graph(
    vector_store: VectorStore,
    generator: AnalysisGenerator,
    *,
    top_k: int,
    max_distance: float,
) -> Callable[[GraphState], GraphState]:
    """Compile the workflow and return its invoke method."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("LangGraph is not installed") from exc

    def validate_input(state: GraphState) -> GraphState:
        query = state["query"].strip()
        if len(query) < 5:
            raise ValueError("The query must contain at least five characters")
        return {
            "query": query,
            "route": classify_route(query),
            "warnings": [],
            "retry_count": 0,
            "validation_feedback": "none",
        }

    def retrieve(state: GraphState) -> GraphState:
        chunks = vector_store.search(state["query"], top_k=top_k)
        accepted = [chunk for chunk in chunks if chunk.distance <= max_distance]
        warnings = list(state.get("warnings", []))
        if not accepted:
            warnings.append("No passage met the configured relevance threshold.")
        return {"retrieved": accepted, "warnings": warnings}

    def generate(state: GraphState) -> GraphState:
        chunks = state.get("retrieved", [])
        if not chunks:
            return {
                "answer": insufficient_evidence_analysis("No relevant passage was found."),
                "is_valid": True,
            }
        answer = generator.generate(
            state["query"],
            state["route"],
            chunks,
            feedback=state.get("validation_feedback", "none"),
        )
        return {"answer": answer}

    def validate_output(state: GraphState) -> GraphState:
        valid, feedback = validate_citations(
            state["answer"], state.get("retrieved", [])
        )
        return {"is_valid": valid, "validation_feedback": feedback}

    def retry_or_fallback(state: GraphState) -> GraphState:
        retries = state.get("retry_count", 0)
        if retries >= 1:
            warnings = list(state.get("warnings", []))
            warnings.append(f"Generation rejected: {state['validation_feedback']}")
            return {
                "answer": insufficient_evidence_analysis(state["validation_feedback"]),
                "warnings": warnings,
                "is_valid": True,
            }
        return {"retry_count": retries + 1}

    def after_validation(state: GraphState) -> Literal["done", "retry"]:
        return "done" if state.get("is_valid", False) else "retry"

    def after_retry(state: GraphState) -> Literal["generate", "done"]:
        return "done" if state.get("is_valid", False) else "generate"

    workflow = StateGraph(GraphState)
    workflow.add_node("validate_input", validate_input)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("validate_output", validate_output)
    workflow.add_node("retry_or_fallback", retry_or_fallback)
    workflow.add_edge(START, "validate_input")
    workflow.add_edge("validate_input", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate_output")
    workflow.add_conditional_edges(
        "validate_output", after_validation, {"done": END, "retry": "retry_or_fallback"}
    )
    workflow.add_conditional_edges(
        "retry_or_fallback", after_retry, {"generate": "generate", "done": END}
    )
    compiled = workflow.compile()
    return compiled.invoke
