"""Application composition and use-case service."""

from functools import lru_cache

from allianz_rag.config import Settings, get_settings
from allianz_rag.graph import GraphState, build_graph
from allianz_rag.llm import VertexAIAnalysisGenerator
from allianz_rag.models import AnalyzeResponse
from allianz_rag.vector_store import ChromaVectorStore


class ClaimsService:
    """Run one analysis through the compiled workflow."""

    def __init__(self, graph_invoke: object) -> None:
        if not callable(graph_invoke):
            raise TypeError("graph_invoke must be callable")
        self._graph_invoke = graph_invoke

    def analyze(self, query: str) -> AnalyzeResponse:
        """Return a typed response for a question or accident narrative."""

        state = self._graph_invoke({"query": query})
        if not isinstance(state, dict):
            raise RuntimeError("The graph returned an invalid state")
        typed_state = GraphState(**state)
        return AnalyzeResponse(
            analysis=typed_state["answer"],
            route=typed_state["route"],
            retrieved_chunks=len(typed_state.get("retrieved", [])),
            warnings=typed_state.get("warnings", []),
        )


def build_service(settings: Settings) -> ClaimsService:
    """Compose production adapters from validated settings."""

    store = ChromaVectorStore(
        settings.chroma_path,
        settings.chroma_collection,
        settings.embedding_model,
    )
    generator = VertexAIAnalysisGenerator(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )
    graph_invoke = build_graph(
        store,
        generator,
        top_k=settings.top_k,
        max_distance=settings.max_distance,
    )
    return ClaimsService(graph_invoke)


@lru_cache(maxsize=1)
def get_service() -> ClaimsService:
    """Return one service per application process."""

    return build_service(get_settings())
