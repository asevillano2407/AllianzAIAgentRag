"""Tests for provider-independent two-stage retrieval."""

from collections.abc import Sequence

import pytest

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.retrieval import RerankingRetriever


def _chunk(chunk_id: str) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        text=f"Evidence {chunk_id}",
        source="manual.pdf",
        page=1,
    )


class RecordingCandidateRetriever:
    def __init__(self, chunks: list[SourceChunk]) -> None:
        self.chunks = chunks
        self.received_queries: list[str] = []
        self.received_limit: int | None = None

    def retrieve_many(self, queries: Sequence[str], limit: int) -> list[SourceChunk]:
        self.received_queries = list(queries)
        self.received_limit = limit
        return self.chunks[:limit]


class ReverseReranker:
    model_name = "fake-reranker"

    def __init__(self) -> None:
        self.received_queries: list[str] = []
        self.received_limits: list[int] = []

    def rerank(
        self,
        query: str,
        chunks: Sequence[SourceChunk],
        limit: int,
    ) -> list[SourceChunk]:
        self.received_queries.append(query)
        self.received_limits.append(limit)
        return list(reversed(chunks))[:limit]


def test_reranking_retriever_separates_candidate_and_context_limits() -> None:
    candidate_retriever = RecordingCandidateRetriever(
        [_chunk(str(index)) for index in range(12)]
    )
    reranker = ReverseReranker()
    retriever = RerankingRetriever(candidate_retriever, reranker, candidate_k=12)

    results = retriever.retrieve_many(
        ["consulta técnica expandida", "relato original"],
        limit=4,
    )

    assert candidate_retriever.received_limit == 12
    assert reranker.received_queries == ["consulta técnica expandida", "relato original"]
    assert reranker.received_limits == [12, 12]
    assert [chunk.chunk_id for chunk in results] == ["11", "10", "9", "8"]


class QueryAwareReranker(ReverseReranker):
    def __init__(self, rankings: dict[str, list[str]]) -> None:
        super().__init__()
        self.rankings = rankings

    def rerank(
        self,
        query: str,
        chunks: Sequence[SourceChunk],
        limit: int,
    ) -> list[SourceChunk]:
        self.received_queries.append(query)
        self.received_limits.append(limit)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        return [chunks_by_id[chunk_id] for chunk_id in self.rankings[query]][:limit]


def test_reranking_retriever_preserves_best_chunk_for_each_query() -> None:
    candidates = [_chunk(chunk_id) for chunk_id in ["a", "b", "c", "d"]]
    reranker = QueryAwareReranker(
        {
            "expansion one": ["b", "a", "c", "d"],
            "expansion two": ["c", "a", "b", "d"],
            "original": ["a", "d", "b", "c"],
        }
    )
    retriever = RerankingRetriever(
        RecordingCandidateRetriever(candidates),
        reranker,
        candidate_k=4,
    )

    results = retriever.retrieve_many(
        ["expansion one", "expansion two", "original"],
        limit=3,
    )

    assert [chunk.chunk_id for chunk in results] == ["b", "c", "a"]
    assert all(chunk.score is not None for chunk in results)


def test_reranking_retriever_rejects_final_limit_above_candidate_limit() -> None:
    retriever = RerankingRetriever(
        RecordingCandidateRetriever([_chunk("one")]),
        ReverseReranker(),
        candidate_k=3,
    )

    with pytest.raises(ValueError, match="exceed candidate_k"):
        retriever.retrieve_many(["consulta"], limit=4)


class InventingReranker(ReverseReranker):
    def rerank(
        self,
        query: str,
        chunks: Sequence[SourceChunk],
        limit: int,
    ) -> list[SourceChunk]:
        return [_chunk("invented")]


def test_reranking_retriever_rejects_chunks_outside_candidate_set() -> None:
    retriever = RerankingRetriever(
        RecordingCandidateRetriever([_chunk("known")]),
        InventingReranker(),
    )

    with pytest.raises(ValueError, match="outside the candidate set"):
        retriever.retrieve_many(["consulta"], limit=1)
