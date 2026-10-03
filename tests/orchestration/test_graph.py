"""Tests for bounded LangGraph execution paths."""

from collections.abc import Sequence

from allianz_claims_rag_agent.domain import (
    AnalysisResponse,
    ConfidenceLevel,
    ConventionApplicability,
    ConventionResponsibility,
    QueryType,
    SourceChunk,
)
from allianz_claims_rag_agent.errors import GenerationError, OutputValidationError, RetrievalError
from allianz_claims_rag_agent.generation.service import GeneratedAnswer
from allianz_claims_rag_agent.orchestration import (
    AgentDependencies,
    DeterministicQueryRouter,
    build_claims_graph,
)


class FakeRetriever:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0
        self.received_queries: list[str] = []

    def retrieve_many(self, queries: Sequence[str], limit: int) -> list[SourceChunk]:
        self.calls += 1
        self.received_queries = list(queries)
        if self.calls <= self.failures:
            raise RetrievalError("temporary retrieval failure")
        return [
            SourceChunk(
                chunk_id="chunk-75",
                text="Se considera responsable el vehículo con daños delanteros.",
                source="manual.pdf",
                page=75,
            )
        ]


class FakeGenerator:
    def __init__(self, failures: Sequence[Exception] = ()) -> None:
        self.failures = list(failures)
        self.calls = 0

    def generate(
        self,
        query: str,
        query_type: QueryType,
        chunks: list[SourceChunk],
    ) -> GeneratedAnswer:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        response = AnalysisResponse(
            query_type=query_type,
            conclusion="Respuesta validada",
            confidence=ConfidenceLevel.LOW,
        )
        return GeneratedAnswer(
            response=response,
            model_name="fake-llm",
            prompt_tokens=10,
            completion_tokens=5,
            total_duration_ms=20.0,
        )


def _dependencies(
    retriever: FakeRetriever,
    generator: FakeGenerator,
    max_retries: int = 1,
) -> AgentDependencies:
    return AgentDependencies(
        retriever=retriever,
        generator=generator,
        router=DeterministicQueryRouter(),
        top_k=3,
        max_retries=max_retries,
    )


def test_graph_completes_happy_path_and_routes_accident() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator()
    graph = build_claims_graph(_dependencies(retriever, generator))

    result = graph.invoke({"query": "El vehículo B choca por detrás contra A."})

    assert result["query_type"] is QueryType.ACCIDENT_DESCRIPTION
    assert result["status"] == "completed"
    assert result["fallback_used"] is False
    assert result["execution_path"] == ["route", "retrieve", "generate"]
    assert "alcance trasero" in retriever.received_queries[0]
    assert generator.calls == 1


def test_graph_retries_one_transient_generation_failure() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator([GenerationError("temporary timeout")])
    graph = build_claims_graph(_dependencies(retriever, generator))

    result = graph.invoke({"query": "¿Cuándo caduca CICOS?"})

    assert result["status"] == "completed"
    assert result["retry_count"] == 1
    assert result["execution_path"].count("generate") == 2
    assert generator.calls == 2


def test_graph_retries_one_transient_retrieval_failure() -> None:
    retriever = FakeRetriever(failures=1)
    graph = build_claims_graph(_dependencies(retriever, FakeGenerator()))

    result = graph.invoke({"query": "¿Cuándo caduca CICOS?"})

    assert result["status"] == "completed"
    assert result["retry_count"] == 1
    assert retriever.calls == 2


def test_graph_falls_back_without_retrying_invalid_model_output() -> None:
    generator = FakeGenerator([OutputValidationError("invented citation")])
    graph = build_claims_graph(_dependencies(FakeRetriever(), generator))

    result = graph.invoke({"query": "¿Cuándo caduca CICOS?"})

    assert result["status"] == "fallback"
    assert result["fallback_used"] is True
    assert result["response"].confidence is ConfidenceLevel.LOW
    assert result["execution_path"][-1] == "fallback"
    assert generator.calls == 1


def test_accident_fallback_exposes_undetermined_convention_decisions() -> None:
    generator = FakeGenerator([OutputValidationError("invalid business decision")])
    graph = build_claims_graph(_dependencies(FakeRetriever(), generator))

    result = graph.invoke({"query": "El vehículo B choca por detrás contra A."})

    assert (
        result["response"].convention_applicability
        is ConventionApplicability.UNDETERMINED
    )
    assert (
        result["response"].convention_responsibility
        is ConventionResponsibility.UNDETERMINED
    )


def test_graph_stops_after_configured_retry_limit() -> None:
    generator = FakeGenerator(
        [GenerationError("first timeout"), GenerationError("second timeout")]
    )
    graph = build_claims_graph(_dependencies(FakeRetriever(), generator, max_retries=1))

    result = graph.invoke({"query": "¿Cuándo caduca CICOS?"})

    assert result["status"] == "fallback"
    assert result["retry_count"] == 2
    assert generator.calls == 2


def test_agent_dependencies_reject_invalid_limits() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator()

    try:
        AgentDependencies(retriever, generator, DeterministicQueryRouter(), top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected invalid top_k to fail")
